"""Two trivial baselines (M6), using TRAIN labels only. They set the floor every agent must beat.

  constant      one fixed answer for every photo: the train photo with the smallest mean
                distance to the other train photos (a geographic "median"), radius = median
                distance to it, date = the median train capture time.
  random_train  copy the label (location and time) of a random train photo (seeded).

  python scripts/baseline_trivial.py <constant|random_train> <split>
Writes $RETRACE_DATA/runs/<name>_<split>/{run.json,answers.jsonl}; then run score_run.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from retrace.geo import haversine_km

DATA = Path(os.environ["RETRACE_DATA"])


def main() -> None:
    kind, split = sys.argv[1], sys.argv[2]
    train = pd.read_parquet(DATA / "manifests" / "osv5m_train_sanitized.parquet")
    train = train[~train.junk]
    target = pd.read_parquet(DATA / "manifests" / f"osv5m_{split}_sanitized.parquet")
    rng = np.random.default_rng(0)
    timed = train[train.time_reliable]

    if kind == "constant":
        cand = train.sample(500, random_state=0)
        ref = train.sample(5000, random_state=1)
        mean_d = [haversine_km(r.lat, r.lon, ref.lat.to_numpy(), ref.lon.to_numpy()).mean()
                  for r in cand.itertuples()]
        c = cand.iloc[int(np.argmin(mean_d))]
        radius = float(np.median(haversine_km(c.lat, c.lon, train.lat.to_numpy(), train.lon.to_numpy())))
        t = pd.Timestamp(timed.utc_time.sort_values().iloc[len(timed) // 2])  # median, unit-safe
        answers = [{"id": i, "lat": float(c.lat), "lon": float(c.lon), "radius_km": radius,
                    "utc_datetime": t.isoformat()} for i in target.id]
        notes = f"constant at ({c.lat:.3f}, {c.lon:.3f}) [{c.country}], radius {radius:.0f} km, time {t}"
    elif kind == "random_train":
        pick = train.iloc[rng.integers(0, len(train), len(target))]
        tpick = timed.utc_time.iloc[rng.integers(0, len(timed), len(target))]
        radius = 5000.0
        answers = [{"id": i, "lat": float(p.lat), "lon": float(p.lon), "radius_km": radius,
                    "utc_datetime": pd.Timestamp(t).isoformat()}
                   for i, p, t in zip(target.id, pick.itertuples(), tpick)]
        notes = "random train label per photo (seed 0), radius 5000 km"
    else:
        raise SystemExit(f"unknown baseline {kind}")

    run_id = f"{kind}_{split}"
    rdir = DATA / "runs" / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "run.json").write_text(json.dumps(
        {"run_id": run_id, "dataset": "osv5m", "split": split, "model": f"baseline:{kind}",
         "config": {}, "notes": notes}, indent=2))
    with open(rdir / "answers.jsonl", "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps({**a, "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0}) + "\n")
    print(run_id, "-", notes)


if __name__ == "__main__":
    main()
