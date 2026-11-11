"""Render results/m1_time_label_brightness.json as data/audits/time_brightness.html."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "m1_time_label_brightness.json").read_text())
OUT = ROOT / "data" / "audits" / "time_brightness.html"

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 520, 300, 44, 12, 14, 36
X0, X1, Y0, Y1 = -18, 15, 0, 255


def sx(e: float) -> float:
    return PAD_L + (e - X0) / (X1 - X0) * (W - PAD_L - PAD_R)


def sy(v: float) -> float:
    return PAD_T + (1 - (v - Y0) / (Y1 - Y0)) * (H - PAD_T - PAD_B)


def panel(name: str, r: dict) -> str:
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Sky brightness vs labeled sun elevation, {name} uploaders">']
    for v in (0, 64, 128, 192, 255):
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                     f'<text x="{PAD_L - 6}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v}</text>')
    for e in range(-18, 16, 6):
        parts.append(f'<text x="{sx(e):.1f}" y="{H - PAD_B + 16}" class="tick" text-anchor="middle">{e}°</text>')
    parts.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{PAD_T}" y2="{H - PAD_B}" class="horizon"/>'
                 f'<text x="{sx(0) + 4:.1f}" y="{PAD_T + 10}" class="tick">horizon</text>')
    parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2}" y="{H - 4}" class="axis" text-anchor="middle">'
                 f'sun elevation at the LABELED time</text>')
    for phase, cls in (("dawn", "s1"), ("dusk", "s2")):
        c = r["fits"][phase]["center_deg"]
        parts.append(f'<line x1="{sx(c):.1f}" x2="{sx(c):.1f}" y1="{PAD_T}" y2="{H - PAD_B}" class="center {cls}"/>')
        pts = sorted((float(k), v) for k, v in r["curve"][phase].items())
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(e + 0.5):.1f},{sy(v):.1f}" for i, (e, v) in enumerate(pts))
        parts.append(f'<path d="{d}" class="line {cls}"/>')
        for e, v in pts:
            tip = html.escape(f"{phase}: labeled sun {e:+.0f}° to {e + 1:+.0f}° · median sky brightness {v:.0f}")
            parts.append(f'<circle cx="{sx(e + 0.5):.1f}" cy="{sy(v):.1f}" r="4" class="dot {cls}" data-tip="{tip}"/>'
                         f'<circle cx="{sx(e + 0.5):.1f}" cy="{sy(v):.1f}" r="10" class="hit" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "".join(parts)


def stat_rows(r: dict) -> str:
    f = r["fits"]
    lo, hi = r["bias_95ci_hours"]
    if min(f["dawn"]["r2"], f["dusk"]["r2"]) < 0.2:
        return ('<tr><td colspan=2 class="warn"><b>Fit failed (R² &lt; 0.2): brightness barely changes with the labeled sun '
                f'angle, so the bias and spread below are NOT meaningful.</b> Fit R² dawn/dusk: '
                f'{f["dawn"]["r2"]:.2f} / {f["dusk"]["r2"]:.2f}.</td></tr>')
    return (f"<tr><td>Dawn transition at</td><td>{f['dawn']['center_deg']:+.1f}°</td></tr>"
            f"<tr><td>Dusk transition at</td><td>{f['dusk']['center_deg']:+.1f}°</td></tr>"
            f"<tr><td>Clock bias (labels late if +)</td><td><b>{r['bias_hours_labels_late_if_positive']:+.2f} h</b> "
            f"<span class=mut>(95% CI {lo:+.2f} to {hi:+.2f})</span></td></tr>"
            f"<tr><td>Spread, upper bound (1 SD)</td><td><b>≤ {r['spread_upper_bound_hours_sd']:.2f} h</b></td></tr>"
            f"<tr><td>Fit R² (dawn / dusk)</td><td>{f['dawn']['r2']:.2f} / {f['dusk']['r2']:.2f}</td></tr>"
            f"<tr><td>Photos (dawn / dusk)</td><td>{f['dawn']['n']:,} / {f['dusk']['n']:,}</td></tr>")


def table(r: dict) -> str:
    rows = "".join(
        f"<tr><td>{float(e):+.0f}°</td><td>{r['curve']['dawn'].get(e, '')}</td><td>{r['curve']['dusk'].get(e, '')}</td></tr>"
        for e in sorted(set(r["curve"]["dawn"]) | set(r["curve"]["dusk"]), key=float))
    return f"<table class=t><tr><th>labeled sun</th><th>dawn</th><th>dusk</th></tr>{rows}</table>"


cards = "".join(
    f"""<section class="card"><h3>{title}</h3>
<div class="legend"><span><i class="sw s1"></i>dawn (sun rising)</span><span><i class="sw s2"></i>dusk (sun setting)</span>
<span><i class="dash"></i>fitted transition</span></div>
{panel(key, RES[key])}<table class="stats">{stat_rows(RES[key])}</table>
<details><summary>Table view</summary>{table(RES[key])}</details></section>"""
    for key, title in (("reliable", "Reliable uploaders (kept for time scoring)"),
                       ("unreliable", "Unreliable uploaders (excluded) — positive control")))

page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Time label check</title><style>
:root{{color-scheme:light;--bg:#fcfcfb;--card:#fff;--fg:#0b0b0b;--mut:#52514e;--line:#e4e3de;--grid:#ecebe7;--s1:#2a78d6;--s2:#eb6834}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#f0efec;--mut:#c3c2b7;--line:#3a3936;--grid:#2e2d2a;--s1:#3987e5;--s2:#d95926}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#1a1a19;--card:#222220;--fg:#f0efec;--mut:#c3c2b7;--line:#3a3936;--grid:#2e2d2a;--s1:#3987e5;--s2:#d95926}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 6px}} h3{{font-size:16px;margin:0 0 6px}} p{{color:var(--mut);max-width:80ch}} .mut{{color:var(--mut)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}
svg{{width:100%;height:auto;display:block}} .grid{{stroke:var(--grid)}} .horizon{{stroke:var(--mut);stroke-dasharray:2 3}}
.tick{{fill:var(--mut);font-size:11px}} .axis{{fill:var(--mut);font-size:12px}}
.line{{fill:none;stroke-width:2}} .line.s1{{stroke:var(--s1)}} .line.s2{{stroke:var(--s2)}}
.dot{{stroke:var(--card);stroke-width:2}} .dot.s1{{fill:var(--s1)}} .dot.s2{{fill:var(--s2)}} .hit{{fill:transparent;cursor:crosshair}}
.center{{stroke-width:1.5;stroke-dasharray:5 4}} .center.s1{{stroke:var(--s1)}} .center.s2{{stroke:var(--s2)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--mut);margin-bottom:4px}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}} .sw.s1{{background:var(--s1)}} .sw.s2{{background:var(--s2)}}
.dash{{display:inline-block;width:16px;border-top:2px dashed var(--mut);margin-right:5px;vertical-align:3px}}
table.stats{{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}} .stats td{{padding:4px 0;border-bottom:1px solid var(--line)}} .stats td:last-child{{text-align:right}}
.warn{{color:var(--fg);background:color-mix(in srgb,var(--s2) 14%,transparent);padding:8px!important;border-radius:6px}}
table.t{{border-collapse:collapse;font-size:13px}} .t td,.t th{{padding:2px 10px;border-bottom:1px solid var(--line);text-align:right}}
#tip{{position:fixed;pointer-events:none;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:13px;display:none;box-shadow:0 2px 8px #0002}}
</style></head><body>
<h1>Are the “good” time labels actually right?</h1>
<p>Each line is the median brightness of the sky (top quarter of the photo) for photos whose <b>label</b> puts the sun at that angle.
If labels are correct, dawn and dusk brighten/darken at the same labeled angle. A gap between the two dashed lines means clocks are
systematically off; a blurry (shallow) transition means clocks scatter. The right panel is the group we already excluded — the method should flag it.</p>
<div class="grid2">{cards}</div><div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{{
 el.addEventListener('mousemove',e=>{{tip.textContent=el.dataset.tip;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'}});
 el.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script></body></html>"""
OUT.write_text(page, encoding="utf-8")
print("wrote", OUT)
