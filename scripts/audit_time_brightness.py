"""Audit: how wrong are OSV-5M time labels? Measured from sky brightness around sunrise/sunset.

Idea
  Sky brightness falls off as the sun sets and rises as it comes up. Plot sky brightness
  against the sun elevation implied by the LABEL and fit a sigmoid; the transition has a
  center c (deg) and a width w (deg).

  * Bias. If labels run late by D hours, at dusk the labeled sun is lower than the true sun
    (transition appears at lower labeled elevation), at dawn it is higher. The natural
    brightness curve is ~symmetric between dawn and dusk, so
        D ~= (c_dawn - c_dusk) / (2 * rate),   rate = |d elevation / dt| near the horizon.
  * Spread. Width combines natural variation (clouds, exposure, terrain) and label noise,
        w_obs^2 = w_nat^2 + (rate * sigma)^2   ->   sigma <= w_obs / rate  (upper bound).
  * Positive control: run the same fit on the UNRELIABLE uploaders; it should look broken.

Stage 1 (fetch) caches per-photo brightness in $RETRACE_DATA/audits/time_brightness.parquet.
Stage 2 (fit) writes results/m1_time_label_brightness.json and data/audits/time_brightness.html.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retrace.ingest import osv5m_remote
from retrace.ingest.osv5m import creator_time_quality, load_slim
from retrace.tools.sun import solar_position

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]
CACHE = DATA / "audits" / "time_brightness.parquet"
ELEV_BINS = np.arange(-18, 16, 1.0)  # 1-degree bins of labeled sun elevation
PER_BIN = 80                         # photos per (group, phase, bin)
PER_CREATOR_PER_BIN = 3


def sky_brightness(jpeg: bytes) -> tuple[float, float]:
    """Mean luma (0-255) of the top quarter of the frame (mostly sky) and of the whole frame."""
    with Image.open(io.BytesIO(jpeg)) as im:
        g = np.asarray(im.convert("L").resize((128, 77)), dtype=np.float32)
    return float(g[: g.shape[0] // 4].mean()), float(g.mean())


def sample(seed: int = 0) -> pd.DataFrame:
    meta = DATA / "osv5m" / "meta"
    tr, te = load_slim(meta / "train_slim.parquet"), load_slim(meta / "test_slim.parquet")
    df = pd.concat([tr.assign(src="train"), te.assign(src="test")], ignore_index=True)
    q = creator_time_quality(df)
    df["reliable"] = df.creator_id.map(q.time_reliable).fillna(False)
    t = df.captured_at.to_numpy() / 1000
    el, az = solar_position(df.lat.to_numpy(), df.lon.to_numpy(), t)
    el2, _ = solar_position(df.lat.to_numpy(), df.lon.to_numpy(), t + 600)
    df["elev"], df["rate_deg_per_h"] = el, np.abs(el2 - el) * 6
    df["phase"] = np.where(el2 > el, "dawn", "dusk")  # sun rising vs setting at labeled time
    df = df[(df.elev >= ELEV_BINS[0]) & (df.elev < ELEV_BINS[-1] + 1)].copy()
    df["bin"] = np.floor(df.elev).astype(int)
    df = df.sample(frac=1, random_state=seed)
    df = df.groupby(["reliable", "phase", "bin", "creator_id"]).head(PER_CREATOR_PER_BIN)
    return df.groupby(["reliable", "phase", "bin"]).head(PER_BIN)


def stage_fetch() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    s = sample()
    print(f"sampled {len(s)} photos", flush=True)
    out = {}
    for split in ("train", "test"):
        idx = osv5m_remote.build_index(split, DATA / "osv5m" / f"index_{split}.parquet")
        need = idx[idx.id.isin(s[s.src == split].id)]
        for i, (pid, data) in enumerate(osv5m_remote.fetch(need.itertuples())):
            if data is None:
                continue
            try:
                out[pid] = sky_brightness(data)
            except Exception:  # noqa: BLE001 - corrupt image: skip
                pass
            if i % 1000 == 0:
                print(f"  {split}: {i}/{len(need)}", flush=True)
    s = s[s.id.isin(out)].copy()
    s["sky"] = s.id.map(lambda i: out[i][0])
    s["whole"] = s.id.map(lambda i: out[i][1])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "creator_id", "country", "lat", "reliable", "phase", "elev", "bin",
            "rate_deg_per_h", "sky", "whole"]
    s[cols].to_parquet(CACHE, index=False)
    return s[cols]


def fit_sigmoid(e: np.ndarray, y: np.ndarray) -> dict:
    """y ~ lo + (hi-lo) * sigmoid((e - c)/s). Grid over (c, s), linear LSQ for (lo, hi)."""
    C, S = np.meshgrid(np.arange(-14, 8.01, 0.2), np.arange(0.3, 8.01, 0.2), indexing="ij")
    C, S = C.ravel()[:, None], S.ravel()[:, None]
    z = 1 / (1 + np.exp(-(e[None, :] - C) / S))          # (grid, n)
    u = 1 - z
    # 2x2 normal equations for (lo, hi), solved in closed form for every grid point at once.
    a, b, d = (u * u).sum(1), (u * z).sum(1), (z * z).sum(1)
    py, qy = (u * y).sum(1), (z * y).sum(1)
    det = a * d - b * b
    lo_all, hi_all = (d * py - b * qy) / det, (a * qy - b * py) / det
    sse_all = ((lo_all[:, None] * u + hi_all[:, None] * z - y[None, :]) ** 2).sum(1)
    k = int(np.nanargmin(sse_all))
    sse, c, s, lo, hi = float(sse_all[k]), float(C[k, 0]), float(S[k, 0]), lo_all[k], hi_all[k]
    return {"center_deg": round(c, 2), "scale_deg": round(s, 2),
            "width_sd_deg": round(s * np.pi / np.sqrt(3), 2),  # logistic -> std-equivalent
            "lo": round(lo, 1), "hi": round(hi, 1), "contrast": round(hi - lo, 1),
            "r2": round(1 - sse / float(((y - y.mean()) ** 2).sum()), 3), "n": int(len(e))}


def bootstrap(g: pd.DataFrame, n: int = 60, seed: int = 0) -> tuple[float, float]:
    """95% interval for the bias in hours (resampling photos within each phase)."""
    rng = np.random.default_rng(seed)
    vals = []
    dawn, dusk = g[g.phase == "dawn"], g[g.phase == "dusk"]
    rate = g.rate_deg_per_h.median()
    for _ in range(n):
        a = dawn.iloc[rng.integers(0, len(dawn), len(dawn))]
        b = dusk.iloc[rng.integers(0, len(dusk), len(dusk))]
        ca = fit_sigmoid(a.elev.to_numpy(), a.sky.to_numpy())["center_deg"]
        cb = fit_sigmoid(b.elev.to_numpy(), b.sky.to_numpy())["center_deg"]
        vals.append((ca - cb) / (2 * rate))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stage_fit(s: pd.DataFrame) -> dict:
    res = {}
    for rel, g in s.groupby("reliable"):
        name = "reliable" if rel else "unreliable"
        fits = {ph: fit_sigmoid(h.elev.to_numpy(), h.sky.to_numpy()) for ph, h in g.groupby("phase")}
        rate = float(g.rate_deg_per_h.median())
        bias_h = (fits["dawn"]["center_deg"] - fits["dusk"]["center_deg"]) / (2 * rate)
        width = (fits["dawn"]["width_sd_deg"] + fits["dusk"]["width_sd_deg"]) / 2
        lo, hi = bootstrap(g)
        res[name] = {
            "fits": fits, "median_rate_deg_per_h": round(rate, 2),
            "bias_hours_labels_late_if_positive": round(bias_h, 2),
            "bias_95ci_hours": [round(lo, 2), round(hi, 2)],
            "spread_upper_bound_hours_sd": round(width / rate, 2),
            "curve": {ph: h.groupby("bin").sky.median().round(1).to_dict() for ph, h in g.groupby("phase")},
        }
    return res


def main() -> None:
    s = stage_fetch()
    print(f"have brightness for {len(s)} photos", flush=True)
    res = stage_fit(s)
    out = ROOT / "results" / "m1_time_label_brightness.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
