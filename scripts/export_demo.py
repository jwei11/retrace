"""Export agent trajectories for the demo viewer (public OSV-5M photos only).

  uv run python scripts/export_demo.py demo/viewer_template.html <out.html>

Picks dev300 photos where the arms differ most (labelled as selected), plus random ones, and
inlines photo thumbnails, trajectories, scores and a simplified world map into the template
(replacing /*DATA*/). Never reads data/private: the manifest is OSV-5M val.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from PIL import Image

from retrace.geo import haversine_km

DATA = Path(os.environ["RETRACE_DATA"])
P = "agent_qwen3-vl-2b-instruct_view+"
ARMS = {  # label -> run id (all on val dev300, temperature 0)
    "No skill": P + "similar+sun+gazetteer_val_dev300",
    "Playbook v1": P + "similar+sun+gazetteer_skill-playbook_v1_val_dev300",
    "v2 + agreement line": P + "similar_sum+sun+gazetteer_skill-playbook_v2_val_dev300",
}
NB = re.compile(r"\| (-?\d+\.\d+), (-?\d+\.\d+) \|")
# Photos with a camera overlay that prints capture date/time, GPS or an uploader handle into the
# pixels (D-051). Checked by eye on the exported set; never shown in the demo.
EXCLUDE = {"4ed6ac1cd6", "1be54e5660"}


def copy_class(steps) -> str:
    """How the answer relates to the first similar() call's lookalikes (as in analyze_skill.py)."""
    ans = next((s for s in steps if s["tool"] == "answer"), None)
    nb = next((s["nb"] for s in steps if s.get("nb")), None)
    if ans is None:
        return "no answer"
    if not nb:
        return "did not ask for lookalikes"
    d = [haversine_km(ans["args"]["lat"], ans["args"]["lon"], a, b) for a, b in nb]
    return ("copied lookalike #1" if d[0] <= 5 else "copied another lookalike" if min(d) <= 5 else
            "near a lookalike" if min(d) <= 50 else "own answer")


def thumb(path: Path, w: int = 560) -> str:
    im = Image.open(path).convert("RGB")
    im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def world() -> list[str]:
    gj = json.loads((DATA / "geo" / "ne_10m_admin_0_countries.geojson").read_text())
    out = []
    for f in gj["features"]:
        g = shapely.geometry.shape(f["geometry"]).simplify(0.25, preserve_topology=False)
        polys = getattr(g, "geoms", [g])
        d = []
        for p in polys:
            if p.is_empty or p.area < 0.3:
                continue
            xy = np.asarray(p.exterior.coords)
            d.append("M" + "L".join(f"{x:.1f} {-y:.1f}" for x, y in xy) + "Z")
        if d:
            out.append("".join(d))
    return out


def main(template: str, out: str) -> None:
    scores = {a: pd.read_parquet(DATA / "runs" / r / "scores.parquet").set_index("id") for a, r in ARMS.items()}
    traj = {a: {(t := json.loads(l))["id"]: t["steps"] for l in open(DATA / "runs" / r / "trajectories.jsonl")}
            for a, r in ARMS.items()}
    g = pd.DataFrame({a: s.geo_shaped for a, s in scores.items()}).dropna()
    n_all = len(g)
    answered = pd.concat([s.answered for s in scores.values()], axis=1).all(axis=1)
    g = g[answered.reindex(g.index).fillna(False) & ~g.index.str[:10].isin(EXCLUDE)]
    a0, a1, a2 = ARMS
    pick: dict[str, str] = {}
    def take(ids, why, n):
        for i in ids:
            if len([1 for v in pick.values() if v == why]) >= n:
                break
            pick.setdefault(i, why)
    take((g[a1] - g[a0]).sort_values(ascending=False).index, "Playbook v1 beats no skill", 5)
    take((g[a0] - g[a1]).sort_values(ascending=False).index, "No skill beats Playbook v1", 4)
    take((g[a2] - g[a1]).sort_values(ascending=False).index, "v2 + agreement line beats v1", 4)
    take(g.drop(index=list(pick)).sample(5, random_state=0).index, "Random", 5)

    m = pd.read_parquet(DATA / "manifests" / "osv5m_val_sanitized.parquet").set_index("id")
    eps = []
    for i, why in pick.items():
        r = m.loc[i]
        arms = {}
        for a in ARMS:
            steps = []
            for s in traj[a][i]:
                st = {"thought": s["thought"], "tool": s["action"]["tool"], "args": s["action"]["args"],
                      "obs": s.get("obs") or "", "tok": s.get("prompt_tokens"), "lat_s": s.get("model_latency_s")}
                if st["tool"] == "similar":
                    st["nb"] = [[float(x), float(y)] for x, y in NB.findall(st["obs"])]
                steps.append(st)
            sc = scores[a].loc[i]
            arms[a] = {"score": round(float(sc.geo_shaped), 3), "km": round(float(sc.dist_km), 1),
                       "in_radius": bool(sc.in_radius), "copy": copy_class(steps), "steps": steps}
        t = r.utc_time
        eps.append({"id": i[:10], "why": why, "img": thumb(DATA / r.sanitized_path),
                    "truth": {"lat": float(r.lat), "lon": float(r.lon),
                              "place": ", ".join(x for x in (r.city, r.region, r.country) if isinstance(x, str) and x),
                              "date": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC") if r.time_reliable and not pd.isna(t) else None},
                    "arms": arms})
    agg = {a: {"geo_shaped": round(float(s.geo_shaped.mean()), 3), "median_km": round(float(s.dist_km.median())),
               "acc200": round(float(s["acc@200"].mean()), 3), "tokens": int(s.tokens_in.mean())}
           for a, s in scores.items()}
    data = {"arms": list(ARMS), "agg": agg, "episodes": eps, "world": world(), "n": n_all}
    html = Path(template).read_text(encoding="utf-8").replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    Path(out).write_text(html, encoding="utf-8")
    print(f"{len(eps)} episodes, {len(html) / 1e6:.2f} MB -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
