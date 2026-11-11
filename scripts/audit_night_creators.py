"""Audit: are 'mostly-night' uploaders really cameras on local time mislabeled as UTC?

For each creator with >=20 photos and daylight_frac < 0.5:
  * find the clock shift h (in 0.5 h steps) that maximizes their daylight fraction;
  * compare it with the real UTC offset of their timezone (local-as-UTC predicts h == offset).
Then build a self-contained HTML page with sample photos (labels hidden until revealed),
including a control group from reliable creators.

Output: data/audits/night_creators.html   (label side; gitignored)
"""
from __future__ import annotations

import base64
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from retrace.ingest import osv5m_remote
from retrace.ingest.osv5m import creator_time_quality, load_slim, zone_at
from retrace.tools.sun import solar_position

DATA = Path(os.environ["RETRACE_DATA"])
OUT = Path(__file__).resolve().parents[1] / "data" / "audits" / "night_creators.html"
SHIFTS = np.arange(-14, 14.5, 0.5)
N_BAD_CREATORS, N_CONTROL = 24, 8


def best_shift(g: pd.DataFrame) -> tuple[float, float]:
    t = g.captured_at.to_numpy() / 1000
    fr = [(solar_position(g.lat, g.lon, t - h * 3600)[0] >= -6).mean() for h in SHIFTS]
    i = int(np.argmax(fr))
    return float(SHIFTS[i]), float(fr[i])


def tz_offset_h(lat: float, lon: float, ms: float) -> float | None:
    z = zone_at(lat, lon)
    if not z:
        return None
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(ZoneInfo(z))
    return dt.utcoffset().total_seconds() / 3600


def main() -> None:
    meta = DATA / "osv5m" / "meta"
    tr, te = load_slim(meta / "train_slim.parquet"), load_slim(meta / "test_slim.parquet")
    df = pd.concat([tr.assign(src="train"), te.assign(src="test")], ignore_index=True)
    q = creator_time_quality(df)
    bad = q[(q.n_photos >= 20) & (q.daylight_frac < 0.5)]

    rows = []
    for cid in bad.index:
        g = df[df.creator_id == cid]
        h, fr = best_shift(g)
        mid = g.iloc[len(g) // 2]
        off = tz_offset_h(mid.lat, mid.lon, mid.captured_at)
        rows.append(dict(creator_id=cid, n=len(g), daylight_as_utc=q.daylight_frac[cid],
                         best_shift_h=h, daylight_after_shift=fr, tz_offset_h=off,
                         country=g.country.mode().iat[0]))
    stats = pd.DataFrame(rows)
    stats["shift_matches_tz"] = (stats.best_shift_h - stats.tz_offset_h).abs() <= 1.0
    w = stats.n / stats.n.sum()
    summary = {
        "bad_creators": len(stats),
        "bad_photos": int(stats.n.sum()),
        "photo_weighted_frac_where_best_shift_matches_tz_offset_within_1h":
            round(float((w * stats.shift_matches_tz).sum()), 3),
        "photo_weighted_mean_daylight_before": round(float((w * stats.daylight_as_utc).sum()), 3),
        "photo_weighted_mean_daylight_after_best_shift": round(float((w * stats.daylight_after_shift).sum()), 3),
    }
    # Shifts 24 h apart give the same sun position, so wrap the residual to [-12, 12).
    resid = ((stats.best_shift_h - stats.tz_offset_h + 12) % 24 - 12).abs()
    stats["shift_matches_tz"] = resid <= 0.5
    summary = {
        "bad_uploaders": len(stats),
        "bad_photos": int(stats.n.sum()),
        "daylight_before_shift": summary["photo_weighted_mean_daylight_before"],
        "daylight_after_best_shift": summary["photo_weighted_mean_daylight_after_best_shift"],
        "share_of_photos_where_best_shift_vs_tz_offset_is": {
            name: round(float(w[(resid >= lo) & (resid < hi)].sum()), 3)
            for lo, hi, name in [(0, 0.6, "exact match (local time stored as UTC)"),
                                 (0.6, 1.6, "off by ~1h (daylight saving?)"),
                                 (1.6, 4.6, "off by 2-4h (unexplained)"),
                                 (4.6, 13, "off by 5h+ (small uploaders, noisy)")]},
    }
    print(json.dumps(summary, indent=2))

    # ---- sample photos for the visual audit
    rng = np.random.default_rng(0)
    pick_bad = stats.sample(min(N_BAD_CREATORS, len(stats)), random_state=0)
    samples = []
    for r in pick_bad.itertuples():
        g = df[df.creator_id == r.creator_id]
        samples.append(g.sample(1, random_state=int(rng.integers(1e9))).assign(
            group="suspect", best_shift_h=r.best_shift_h, tz_offset_h=r.tz_offset_h))
    good_ids = q[(q.n_photos >= 20) & (q.daylight_frac >= 0.98)].sample(N_CONTROL, random_state=0).index
    for cid in good_ids:
        samples.append(df[df.creator_id == cid].sample(1, random_state=0).assign(
            group="control", best_shift_h=np.nan, tz_offset_h=np.nan))
    s = pd.concat(samples).sample(frac=1, random_state=1).reset_index(drop=True)  # mix the groups

    imgs = {}
    for split in ("train", "test"):
        idx = osv5m_remote.build_index(split, DATA / "osv5m" / f"index_{split}.parquet")
        need = idx[idx.id.isin(s[s.src == split].id)]
        imgs.update(dict(osv5m_remote.fetch(need.itertuples(), workers=16)))

    cards = []
    for i, r in s.iterrows():
        el_label = float(solar_position(r.lat, r.lon, r.captured_at / 1000)[0])
        utc = pd.Timestamp(r.captured_at, unit="ms", tz="UTC")
        z = zone_at(r.lat, r.lon)
        local = utc.tz_convert(z).strftime("%Y-%m-%d %H:%M") if z else "?"
        extra = ""
        if r.group == "suspect":
            el_shift = float(solar_position(r.lat, r.lon, r.captured_at / 1000 - r.best_shift_h * 3600)[0])
            extra = (f"<br>best-fit clock shift: <b>{r.best_shift_h:+.1f} h</b>, "
                     f"tz offset: <b>{r.tz_offset_h:+.1f} h</b><br>"
                     f"sun after shift: <b>{el_shift:+.0f}°</b>")
        b64 = base64.b64encode(imgs[r.id]).decode()
        cards.append(f"""
<div class="card"><div class="num">#{i + 1}</div>
<img src="data:image/jpeg;base64,{b64}">
<div class="guess">Your call: <label><input type="radio" name="g{i}">day</label>
<label><input type="radio" name="g{i}">twilight</label><label><input type="radio" name="g{i}">night</label></div>
<div class="label">
<b>{html.escape(r.group.upper())}</b> · {html.escape(str(r.country))} · uploader {html.escape(str(r.creator_id))[:10]}<br>
labeled UTC: {utc:%Y-%m-%d %H:%M} · local: {local} ({html.escape(z or '?')})<br>
sun at labeled time: <b>{el_label:+.0f}°</b> {'(night)' if el_label < -6 else '(day/twilight)'}{extra}
</div></div>""")

    table = stats.sort_values("n", ascending=False).head(40).to_html(
        index=False, float_format=lambda x: f"{x:.2f}", classes="t")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Night uploader audit</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#2f6fdf}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecebe6;--mut:#a09f99;--card:#20201e;--line:#34332f;--acc:#7aa7ff}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:var(--mut);max-width:75ch}}
pre{{background:var(--card);border:1px solid var(--line);padding:12px;border-radius:8px;overflow:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;position:relative}}
.card img{{width:100%;border-radius:6px;display:block}} .num{{position:absolute;top:16px;left:16px;background:#000a;color:#fff;padding:0 8px;border-radius:4px}}
.guess{{margin:8px 0;font-size:14px}} .guess label{{margin-right:10px}}
.label{{font-size:13px;color:var(--mut);border-top:1px solid var(--line);padding-top:6px}}
body.hide .label{{display:none}}
button{{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
table.t{{border-collapse:collapse;font-size:13px;width:100%;overflow-x:auto;display:block}} .t td,.t th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}}
</style></head><body class="hide">
<h1>Audit: uploaders whose photos are “mostly at night”</h1>
<p>Hypothesis: these cameras recorded <i>local</i> time, and it was stored as if it were UTC.
If so, (1) the best-fit clock shift should equal the timezone offset, and (2) the photos should look like daytime.
{len(s)} photos below: {len(pick_bad)} from suspect uploaders, {N_CONTROL} controls from reliable uploaders, shuffled.
Judge each photo first, then reveal labels.</p>
<h2>1 · Numbers</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre>
<details><summary>Per-uploader table (top 40 by photos)</summary>{table}</details>
<h2>2 · Look at the photos</h2>
<p><button onclick="document.body.classList.toggle('hide')">Show / hide labels</button></p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    stats.to_csv(OUT.with_suffix(".csv"), index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
