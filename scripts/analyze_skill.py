"""Does the agent just copy kNN, and does it follow the playbook? (reads existing runs only)

  uv run python scripts/analyze_skill.py <run_id> [<run_id> ...]

Per photo, from trajectories.jsonl + scores.parquet (+ knn10_val's per-photo scores):
  copy class  copy#1      answer within 5 km of neighbor #1 from the first similar() call
              copy-other  within 5 km of another returned neighbor
              near        within 50 km of some neighbor
              own         farther from every neighbor (the model went its own way)
              no-similar  similar() never called
  compliance  similar first; gazetteer calls that were 'no match' or a name taken from the
              neighbor list rather than (possibly) read in the photo; radius vs neighbor spread.
Writes results/analysis/skill_<run_id>.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from retrace.geo import haversine_km

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]
NB = re.compile(r"\| (-?\d+\.\d+), (-?\d+\.\d+) \|")


def per_photo(run_id: str) -> pd.DataFrame:
    rd = DATA / "runs" / run_id
    sc = pd.read_parquet(rd / "scores.parquet").set_index("id")
    knn = pd.read_parquet(DATA / "runs" / "knn10_val" / "scores.parquet").set_index("id")
    rows = []
    for line in open(rd / "trajectories.jsonl", encoding="utf-8"):
        t = json.loads(line)
        steps = t["steps"]
        acts = [s["action"] for s in steps]
        ans = next((a["args"] for a in acts if a["tool"] == "answer"), None)
        sim = next((s for s in steps if s["action"]["tool"] == "similar"), None)
        nb = np.array([[float(a), float(b)] for a, b in NB.findall(sim["obs"])]) if sim and sim.get("obs") else None
        gz = [s for s in steps if s["action"]["tool"] == "gazetteer"]
        r = {"id": t["id"], "n_steps": len(steps), "similar_first": bool(acts) and acts[0]["tool"] == "similar",
             "called_similar": sim is not None, "n_gazetteer": len(gz),
             "gz_no_match": sum("no place named" in (s.get("obs") or "") for s in gz),
             "gz_from_neighbors": sum(bool(sim) and s["action"]["args"]["name"].casefold() in (sim.get("obs") or "").casefold()
                                      for s in gz)}
        if ans is not None and nb is not None and len(nb):
            d = haversine_km(ans["lat"], ans["lon"], nb[:, 0], nb[:, 1])
            r.update(d_nn1=float(d[0]), d_min=float(d.min()), spread_km=float(np.median(d)),
                     radius_km=float(ans["radius_km"]))
            r["copy"] = ("copy#1" if d[0] <= 5 else "copy-other" if d.min() <= 5 else
                         "near" if d.min() <= 50 else "own")
        else:
            r["copy"] = "no-similar" if ans is not None else "no-answer"
        rows.append(r)
    df = pd.DataFrame(rows).set_index("id")
    df["geo_shaped"] = sc.geo_shaped
    df["knn10_geo_shaped"] = knn.geo_shaped.reindex(df.index)
    df["uploader_seen"] = sc.uploader_seen
    return df


def report(run_id: str) -> dict:
    df = per_photo(run_id)
    g = df.groupby("copy").agg(n=("geo_shaped", "size"), agent=("geo_shaped", "mean"),
                               knn10=("knn10_geo_shaped", "mean"))
    g["share"] = g.n / len(df)
    gz = int(df.n_gazetteer.sum())
    has_r = df.dropna(subset=["radius_km"])
    comp = {
        "similar_first": float(df.similar_first.mean()),
        "called_similar": float(df.called_similar.mean()),
        "gazetteer_calls_per_photo": gz / len(df),
        "gazetteer_no_match_share": (df.gz_no_match.sum() / gz) if gz else None,
        "gazetteer_name_from_neighbors_share": (df.gz_from_neighbors.sum() / gz) if gz else None,
        "median_radius_over_spread": float((has_r.radius_km / has_r.spread_km.clip(lower=1)).median()) if len(has_r) else None,
    }
    by_first = df.groupby("similar_first").geo_shaped.agg(["size", "mean"]).to_dict("index")
    out = {"run_id": run_id, "n": len(df), "copy_classes": g.round(4).to_dict("index"),
           "compliance": comp, "geo_shaped_by_similar_first": by_first,
           "agent_vs_knn10_overall": [float(df.geo_shaped.mean()), float(df.knn10_geo_shaped.mean())]}
    p = ROOT / "results" / "analysis" / f"skill_{run_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))

    print(f"\n== {run_id} (n={len(df)})   agent {out['agent_vs_knn10_overall'][0]:.3f} vs knn10 "
          f"{out['agent_vs_knn10_overall'][1]:.3f}")
    print(g[["n", "share", "agent", "knn10"]].round(3).to_string())
    print("compliance:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in comp.items()})
    print("geo_shaped by similar-first:", {k: (v["size"], round(v["mean"], 3)) for k, v in by_first.items()})
    return out


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        report(rid)
