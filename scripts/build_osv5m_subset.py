"""M1/M3: sample the OSV-5M subset and write label manifests (pre-sanitization).

  train/val: carved from the official train split, grouped by sequence, >=1 km apart.
  test:      from the official test split (already >=1 km from train by construction),
             1 image per sequence.

Outputs ($RETRACE_DATA/manifests/osv5m_{split}_raw.parquet) and results/m1_osv5m_subset.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from retrace.ingest.osv5m import add_time_labels, creator_time_quality, load_slim
from retrace.splits.group_split import cap_per_group, carve_train_val

ROOT = Path(__file__).resolve().parents[1]
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--n-test", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(os.environ["RETRACE_DATA"])
    meta = data / "osv5m" / "meta"
    train_all = load_slim(meta / "train_slim.parquet")
    test_all = load_slim(meta / "test_slim.parquet")

    log("loaded metadata")
    # Creator-level time QA on all of train+test (more photos per creator -> better estimate).
    quality = creator_time_quality(pd.concat([train_all, test_all], ignore_index=True))

    log("creator time QA done")
    train, val, rep = carve_train_val(train_all, args.n_train, args.n_val, seed=args.seed)
    test = cap_per_group(test_all, "group_id", 1, args.seed).sample(args.n_test, random_state=args.seed)
    test = test.assign(split="test")
    log(f"split done: {rep}")

    out_dir = data / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"split_report": asdict(rep)}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df = add_time_labels(df, quality)
        log(f"labels for {name}")
        df = df.rename(columns={"id": "source_id"}).assign(source="osv5m", id=None, sanitized_path=None)
        df.to_parquet(out_dir / f"osv5m_{name}_raw.parquet", index=False)
        summary[name] = {
            "n": len(df),
            "groups": int(df.group_id.nunique()),
            "countries": int(df.country.nunique()),
            "time_reliable_frac": round(float(df.time_reliable.mean()), 3),
            "label_sun_below_civil_frac": round(float((df.sun_elev_label < -6).mean()), 3),
            "top5_countries_pct": (df.country.value_counts(normalize=True).head(5) * 100).round(1).to_dict(),
        }
    q = quality
    summary["creator_qa"] = {
        "creators": len(q),
        "reliable_creators": int(q.time_reliable.sum()),
        "photos_with_reliable_time_frac": round(float((q.n_photos * q.time_reliable).sum() / q.n_photos.sum()), 3),
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "results" / "m1_osv5m_subset.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
