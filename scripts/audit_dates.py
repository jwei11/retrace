"""Audit #1: are OSV-5M dates/years right? (The daylight checks only test the hour.)

Checks, on all train+test metadata (5.1M photos), then their impact on our subset:
  A. Calendar spikes: dates with far more photos than their neighbours (default/reset clocks),
     the 2011 year anomaly, and timestamps at exactly 00:00:00.000.
  B. Sequence span: a sequence is one capture session; spans > 1 day are suspicious.
  C. Impossible motion inside a sequence: consecutive photos implying > 250 km/h, or
     identical timestamps > 1 km apart.

Writes results/m1_date_audit.json and adds per-photo flags to
$RETRACE_DATA/audits/date_flags.parquet (label side).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from retrace.geo import haversine_km
from retrace.ingest.osv5m import load_slim

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]
MAX_KMH = 250.0
SPAN_DAYS = 1.0
SPIKE_FACTOR = 20.0  # a date with >= 20x the median count of the surrounding +-15 days


def main() -> None:
    meta = DATA / "osv5m" / "meta"
    df = pd.concat([load_slim(meta / "train_slim.parquet"), load_slim(meta / "test_slim.parquet")],
                   ignore_index=True)
    df["ts"] = pd.to_datetime(df.captured_at, unit="ms", utc=True)
    out: dict = {"n_photos": len(df)}

    # ---- A. calendar spikes -------------------------------------------------------------
    day = df.ts.dt.floor("D")
    per_day = day.value_counts().sort_index()
    full = per_day.reindex(pd.date_range(per_day.index.min(), per_day.index.max(), freq="D"), fill_value=0)
    baseline = full.rolling(31, center=True, min_periods=5).median().clip(lower=1)
    ratio = full / baseline
    spikes = ratio[(ratio >= SPIKE_FACTOR) & (full >= 200)].sort_values(ascending=False)
    spike_days = set(spikes.index)
    df["flag_spike_day"] = day.isin(spike_days)
    out["A_calendar"] = {
        "spike_days": [
            {"date": str(d.date()), "photos": int(full[d]), "x_neighbourhood_median": round(float(ratio[d]), 1),
             "uploaders": int(df.loc[day == d, "creator_id"].nunique())}
            for d in spikes.index[:15]],
        "photos_on_spike_days": int(df.flag_spike_day.sum()),
        "exact_midnight_frac": round(float((df.captured_at % 86_400_000 == 0).mean()), 6),
        "expected_midnight_frac_if_random": round(1 / 86_400_000, 9),
    }
    y2011 = df[df.ts.dt.year == 2011]
    out["A_calendar"]["year_2011"] = {
        "photos": len(y2011),
        "uploaders": int(y2011.creator_id.nunique()),
        "top_uploader_share": round(float(y2011.creator_id.value_counts(normalize=True).iloc[0]), 3)
        if len(y2011) else None,
        "distinct_days": int(y2011.ts.dt.floor("D").nunique()),
        "top_days": {str(k.date()): int(v) for k, v in y2011.ts.dt.floor("D").value_counts().head(5).items()},
    }

    # ---- B + C. sequence consistency ------------------------------------------------------
    df = df.sort_values(["sequence", "captured_at"]).reset_index(drop=True)
    g = df.groupby("sequence", sort=False)
    span_days = (g.captured_at.transform("max") - g.captured_at.transform("min")) / 86_400_000
    df["flag_long_sequence"] = span_days > SPAN_DAYS

    same_seq = df.sequence.eq(df.sequence.shift())
    dt_h = (df.captured_at - df.captured_at.shift()) / 3_600_000
    dist = haversine_km(df.lat, df.lon, df.lat.shift(), df.lon.shift())
    with np.errstate(divide="ignore", invalid="ignore"):
        kmh = np.where(dt_h > 0, dist / dt_h, np.inf)
    bad_step = same_seq & (((dt_h > 0) & (kmh > MAX_KMH) & (dist > 0.5)) | ((dt_h == 0) & (dist > 1.0)))
    # Flag both photos of an impossible step.
    df["flag_impossible_motion"] = bad_step | bad_step.shift(-1, fill_value=False)

    multi = g.size() > 1
    seq_span = span_days.groupby(df.sequence).first()[multi]
    out["B_sequence_span"] = {
        "multi_photo_sequences": int(multi.sum()),
        "span_quantiles_hours": {str(q): round(float(seq_span.quantile(q) * 24), 2)
                                 for q in (0.5, 0.9, 0.99, 0.999)},
        "sequences_over_1_day_frac": round(float((seq_span > SPAN_DAYS).mean()), 4),
        "photos_in_such_sequences": int(df.flag_long_sequence.sum()),
    }
    out["C_motion"] = {
        "consecutive_pairs": int(same_seq.sum()),
        "impossible_steps": int(bad_step.sum()),
        "impossible_step_frac": round(float(bad_step.sum() / max(same_seq.sum(), 1)), 5),
        "photos_flagged": int(df.flag_impossible_motion.sum()),
    }

    flags = ["flag_spike_day", "flag_long_sequence", "flag_impossible_motion"]
    df["any_date_flag"] = df[flags].any(axis=1)
    out["overall_flagged_frac"] = round(float(df.any_date_flag.mean()), 4)

    # ---- impact on our subset --------------------------------------------------------------
    sub = {}
    for split in ("train", "val", "test"):
        m = pd.read_parquet(DATA / "manifests" / f"osv5m_{split}_raw.parquet")
        j = m.merge(df[["id", *flags, "any_date_flag"]], left_on="source_id", right_on="id", how="left")
        rel = j[j.time_reliable]
        sub[split] = {
            "n": len(j), "time_reliable": int(len(rel)),
            **{f"{f}_among_reliable": int(rel[f].sum()) for f in flags},
            "any_flag_among_reliable_frac": round(float(rel.any_date_flag.mean()), 4),
        }
    out["subset_impact"] = sub

    (DATA / "audits").mkdir(exist_ok=True)
    df[["id", "creator_id", "sequence", *flags, "any_date_flag"]].to_parquet(
        DATA / "audits" / "date_flags.parquet", index=False)
    (ROOT / "results" / "m1_date_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
