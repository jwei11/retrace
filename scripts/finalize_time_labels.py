"""Apply the final time-label rule (D-012/D-013/D-016) to the sanitized OSV-5M manifests.

  1. compute image stats for every sanitized image (cached in $RETRACE_DATA/audits/image_stats.parquet)
  2. darkness cutoffs from day-labeled TRAIN photos only
  3. apply the rule, write `time_label_status` + `time_reliable` back into the manifests
  4. results/m1_night_rescue.json and data/audits/night_rescue.html (rescued photos to review,
     plus dark night photos blocked by the uploader rule)

Run after scripts/sanitize_osv5m.py. Idempotent.
"""
from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path

import pandas as pd

from retrace.ingest.time_labels import apply_time_rule, cutoffs_dict, dark_cutoffs
from retrace.sanitize.image_stats import STAT_COLS, load_stats

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def main() -> None:
    mans = {s: pd.read_parquet(DATA / "manifests" / f"osv5m_{s}_sanitized.parquet") for s in SPLITS}
    every = pd.concat(mans.values(), ignore_index=True)[["id", "sanitized_path"]]
    st = load_stats(every, DATA)
    mans = {s: m.drop(columns=[c for c in STAT_COLS if c in m]).merge(st, on="id", how="left")
            for s, m in mans.items()}

    cut = dark_cutoffs(mans["train"])
    report = {"cutoffs_from_train": cutoffs_dict(cut), "splits": {}}
    review = []
    for s, m in mans.items():
        out = apply_time_rule(m, cut)
        counts = out.time_label_status.value_counts().to_dict()
        night = out[out.sun_elev_label < -6]
        dark_night = night[(night.brightness < cut.brightness) & (night.sky_brightness < cut.sky_brightness)]
        report["splits"][s] = {
            "n": len(out),
            "status_counts": {k: int(v) for k, v in counts.items()},
            "time_reliable": int(out.time_reliable.sum()),
            "night_labeled": int(len(night)),
            "dark_night_labeled": int(len(dark_night)),
            "dark_night_blocked_by_uploader_rule": int((~dark_night.creator_time_ok).sum()),
        }
        review.append(out[out.time_label_status == "rescued_night"].assign(group="rescued"))
        review.append(dark_night[~dark_night.creator_time_ok].assign(group="blocked by uploader rule"))
        out.drop(columns=STAT_COLS).to_parquet(DATA / "manifests" / f"osv5m_{s}_sanitized.parquet", index=False)

    (ROOT / "results" / "m1_night_rescue.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    write_page(pd.concat(review, ignore_index=True), report)


def write_page(r: pd.DataFrame, report: dict) -> None:
    cards = []
    for g, part in r.groupby("group", sort=False):
        cards.append(f"<h2>{html.escape(g.capitalize())} ({len(part)})</h2><div class=grid>")
        for x in part.sort_values("brightness").itertuples():
            b64 = base64.b64encode((DATA / x.sanitized_path).read_bytes()).decode()
            local = pd.Timestamp(x.local_time).strftime("%Y-%m-%d %H:%M") if pd.notna(x.local_time) else "?"
            cards.append(
                f'<div class=card><img loading=lazy src="data:image/jpeg;base64,{b64}">'
                f'<div class=meta>{x.split} · {html.escape(str(x.country))} · local {local} · '
                f'sun {x.sun_elev_label:+.0f}°<br>brightness {x.brightness:.0f} · sky {x.sky_brightness:.0f}</div>'
                f'<div class=guess>Really night? <label><input type=radio name="{x.id}">yes</label>'
                f'<label><input type=radio name="{x.id}">no</label><label><input type=radio name="{x.id}">unsure</label></div></div>')
        cards.append("</div>")
    c = report["cutoffs_from_train"]
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Night rescue review</title><style>
:root{{color-scheme:light;--bg:#fcfcfb;--card:#fff;--fg:#0b0b0b;--mut:#52514e;--line:#e4e3de}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#f0efec;--mut:#c3c2b7;--line:#3a3936}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#f0efec;--mut:#c3c2b7;--line:#3a3936}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 10px}} p{{color:var(--mut);max-width:80ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,260px),1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}}
.card img{{width:100%;border-radius:6px;display:block}} .meta{{font-size:12px;color:var(--mut);margin-top:6px}}
.guess{{font-size:13px;margin-top:4px}} .guess label{{margin-right:8px}}
</style></head><body>
<h1>Night rescue review</h1>
<p><b>Rescued</b>: night label AND whole image darker than {c['brightness']:.0f} AND sky darker than {c['sky_brightness']:.0f}
(the darkest 2% of {c['n_day_photos']:,} day-labeled train photos). These keep their time label. Check they really look like night;
any that are daytime (tunnel, forest, storm) are false rescues.</p>
<p><b>Blocked by uploader rule</b>: same darkness test passed, but the uploader fails the 90% daylight rule, so the label is still dropped.</p>
{''.join(cards)}</body></html>"""
    out = ROOT / "data" / "audits" / "night_rescue.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
