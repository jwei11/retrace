"""Audit #4: junk images (black, washed out, flat, blurry) among SANITIZED images, all splits.

Per image (on a 384x256 grayscale downsample of what the agent sees; retrace.sanitize.image_stats):
  brightness      mean luma, 0-255
  sky_brightness  mean luma of the top quarter
  contrast        std of luma
  sharpness       variance of the Laplacian (low = blurry)
Stats are shared with the night rescue (cache: $RETRACE_DATA/audits/image_stats.parquet).
Writes results/m1_junk_audit.json (quantiles per split) and data/audits/junk_images.html: the most
extreme images per measure plus random controls, for a human to judge. No thresholds are
applied here; that is the user's decision.
"""
from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path

import pandas as pd

from retrace.sanitize.image_stats import STAT_COLS, load_stats

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "audits" / "junk_images.html"
N_SHOW = 20
QS = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]


def main() -> None:
    mans = sorted((DATA / "manifests").glob("osv5m_*_sanitized.parquet"))
    cols = ["id", "sanitized_path", "split", "country", "time_label_status"]
    df = pd.concat([pd.read_parquet(m, columns=cols) for m in mans], ignore_index=True)
    df = df.merge(load_stats(df[["id", "sanitized_path"]], DATA), on="id", how="left")
    assert df[STAT_COLS].notna().all().all(), "missing stats"
    print(f"{len(df)} images; splits: {df.split.value_counts().to_dict()}")

    q_all = df[STAT_COLS].quantile(QS).round(1)
    print(q_all.to_string())
    report = {
        "n": len(df),
        "quantiles_all": {c: {str(k): v for k, v in q_all[c].items()} for c in STAT_COLS},
        "quantiles_by_split": {
            s: {c: {str(k): float(v) for k, v in g[c].quantile(QS).round(1).items()} for c in STAT_COLS}
            for s, g in df.groupby("split")},
    }
    (ROOT / "results" / "m1_junk_audit.json").write_text(json.dumps(report, indent=2))

    def card(r, reason: str) -> str:
        b64 = base64.b64encode((DATA / r.sanitized_path).read_bytes()).decode()
        return (f'<div class="card"><img loading="lazy" src="data:image/jpeg;base64,{b64}">'
                f'<div class="meta"><b>{html.escape(reason)}</b> · {r.split} · {html.escape(str(r.country))}'
                f' · {html.escape(str(r.time_label_status))}<br>'
                f'brightness {r.brightness:.0f} · sky {r.sky_brightness:.0f} · contrast {r.contrast:.0f}'
                f' · sharpness {r.sharpness:.0f}</div>'
                f'<div class="guess">Could a person locate this? <label><input type=radio name="{r.id}">yes</label>'
                f'<label><input type=radio name="{r.id}">maybe</label><label><input type=radio name="{r.id}">no</label></div></div>')

    sections = [
        ("Darkest", df.nsmallest(N_SHOW, "brightness"), "dark"),
        ("Brightest (washed out)", df.nlargest(N_SHOW, "brightness"), "bright"),
        ("Lowest contrast (flat)", df.nsmallest(N_SHOW, "contrast"), "flat"),
        ("Blurriest", df.nsmallest(N_SHOW, "sharpness"), "blurry"),
        ("Random controls", df.sample(N_SHOW, random_state=0), "random"),
    ]
    body = "".join(
        f"<h2>{t}</h2><div class=grid>{''.join(card(r, why) for r in s.itertuples())}</div>"
        for t, s, why in sections)
    qtable = q_all.rename(index=lambda x: f"{x * 100:g}th pct").to_html(classes="t")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Junk image review</title><style>
:root{{color-scheme:light;--bg:#fcfcfb;--card:#fff;--fg:#0b0b0b;--mut:#52514e;--line:#e4e3de}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#f0efec;--mut:#c3c2b7;--line:#3a3936}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#f0efec;--mut:#c3c2b7;--line:#3a3936}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 10px}} p{{color:var(--mut);max-width:80ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,260px),1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}}
.card img{{width:100%;border-radius:6px;display:block}} .meta{{font-size:12px;color:var(--mut);margin-top:6px}}
.guess{{font-size:13px;margin-top:4px}} .guess label{{margin-right:8px}}
.scroll{{overflow-x:auto}} table.t{{border-collapse:collapse;font-size:13px}}
.t td,.t th{{padding:2px 10px;border-bottom:1px solid var(--line);text-align:right}}
</style></head><body>
<h1>Junk image review ({len(df):,} sanitized OSV-5M images, all splits)</h1>
<p>The {N_SHOW} most extreme images on each measure, plus random controls. Decide which you'd call unlocatable;
that tells us whether a threshold is worth applying and where. Nothing has been removed.
The status after the country is the photo's time-label outcome (e.g. <i>rescued_night</i> = kept as a real night photo).</p>
<h2>Distribution (quantiles)</h2><div class=scroll>{qtable}</div>{body}</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
