"""Score a run, and optionally compare it with another.

  python scripts/score_run.py <run_id> [--vs <other_run_id>]

Reads  $RETRACE_DATA/runs/<run_id>/{run.json,answers.jsonl} and the split's sanitized manifest.
Writes $RETRACE_DATA/runs/<run_id>/scores.parquet   (per photo; outside git)
       results/runs/<run_id>.json                   (summary; with --vs, the comparison too)
       results/runs.csv                             (one row per run; re-scoring replaces the row)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from retrace.splits.dev import subset_ids
from retrace.score.run import load_answers, paired_bootstrap, score_run, summarize

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]


def manifest(dataset: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / "manifests" / f"{dataset}_{split}_sanitized.parquet")


def score(run_id: str) -> tuple[dict, pd.DataFrame]:
    rdir = DATA / "runs" / run_id
    meta = json.loads((rdir / "run.json").read_text())
    if meta.get("private"):
        raise SystemExit("private-set runs are not written to results/ (hard rule 1); score them by hand")
    labels = manifest(meta["dataset"], meta["split"])
    if meta.get("subset"):  # frozen dev subset (e.g. dev300): score only those photos
        labels = labels[labels.id.isin(subset_ids(meta["dataset"], meta["split"], meta["subset"]))]
    train_creators = set(manifest("osv5m", "train").creator_id)  # the only training/reference set
    sc = score_run(load_answers(rdir / "answers.jsonl"), labels, train_creators)
    sc.to_parquet(rdir / "scores.parquet", index=False)
    return meta, sc


def csv_row(meta: dict, s: dict) -> dict:
    mi, ma = s["micro"], s["macro_by_stratum"]
    return {
        "run_id": meta["run_id"], "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "dataset": meta["dataset"], "split": meta["split"] + (f"/{meta['subset']}" if meta.get("subset") else ""), "model": meta.get("model", ""),
        "n": mi["n"], "answer_rate": round(s["answer_rate"], 4),
        "median_km": round(mi["median_km"], 1),
        **{k: round(mi[k], 4) for k in ("acc@1", "acc@25", "acc@200", "acc@750", "acc@2500", "geo_shaped")},
        "macro_geo_shaped": round(ma["geo_shaped"], 4),
        "in_radius": None if mi["in_radius"] is None else round(mi["in_radius"], 4),
        "n_time": mi["n_time"],
        "year_exact": round(mi["year_exact"], 4) if "year_exact" in mi else None,
        "doy_within_30d": round(mi["doy_within_30d"], 4) if "doy_within_30d" in mi else None,
        "new_uploader_geo_shaped": None if s["uploader"]["new"] is None else round(s["uploader"]["new"]["geo_shaped"], 4),
        "mean_tokens": None if s["cost"]["mean_tokens_in"] is None else
        round(s["cost"]["mean_tokens_in"] + (s["cost"]["mean_tokens_out"] or 0)),
        "mean_latency_s": s["cost"]["mean_latency_s"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--vs", help="run_id to compare against (paired bootstrap)")
    args = ap.parse_args()

    meta, sc = score(args.run_id)
    summary = {"run": meta, **summarize(sc)}
    if args.vs:
        meta_b, sc_b = score(args.vs)
        if (meta_b["dataset"], meta_b["split"]) != (meta["dataset"], meta["split"]):
            raise SystemExit("can only compare runs on the same dataset and split")
        if meta_b.get("subset") != meta.get("subset"):  # compare on this run's photos only
            if not set(sc.id) <= set(sc_b.id):
                raise SystemExit(f"{args.vs} does not cover this run's photos")
            sc_b = sc_b[sc_b.id.isin(set(sc.id))]
        summary["vs"] = {"run_id": args.vs, "metrics": paired_bootstrap(sc, sc_b)}

    out = ROOT / "results" / "runs" / f"{args.run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    csv = ROOT / "results" / "runs.csv"
    row = pd.DataFrame([csv_row(meta, summary)])
    if csv.exists():
        old = pd.read_csv(csv)
        row = pd.concat([old[old.run_id != meta["run_id"]], row], ignore_index=True)
    row.to_csv(csv, index=False)

    mi = summary["micro"]
    print(f"{args.run_id}: n={mi['n']} answered={summary['answer_rate']:.1%} "
          f"median={mi['median_km']:.0f} km geo_shaped={mi['geo_shaped']:.3f} "
          f"acc@25/200/750/2500 = " + " / ".join(f"{mi[f'acc@{k}']:.1%}" for k in (25, 200, 750, 2500)))
    if "year_exact" in mi:
        print(f"  time (n={mi['n_time']}): year exact {mi['year_exact']:.1%}, "
              f"date within 30 d {mi['doy_within_30d']:.1%}, median date error {mi['median_doy_err_d']:.0f} d")
    up = summary["uploader"]
    if up["new"] is not None and up["seen"] is not None:
        print(f"  uploader seen in train ({up['seen_frac']:.0%}): geo_shaped {up['seen']['geo_shaped']:.3f}; "
              f"new: {up['new']['geo_shaped']:.3f}")
    if args.vs:
        print(f"  vs {args.vs} (diff = this - other, 95% interval):")
        for k, v in summary["vs"]["metrics"].items():
            flag = "REAL" if v["significant"] else "could be luck"
            print(f"    {k:15s} {v['diff']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {flag}")
    print("wrote", out)


if __name__ == "__main__":
    main()
