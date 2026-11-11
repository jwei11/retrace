"""Apply the junk flag (D-017) to the sanitized OSV-5M manifests. Idempotent.

Inputs: image stats cache ($RETRACE_DATA/audits/image_stats.parquet, built by
finalize_time_labels.py / audit_junk_images.py) and the user's review, results/m1_junk_review.json.
Writes `junk` and `junk_reason` into each manifest and results/m1_junk_flags.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from retrace.ingest.junk import JUNK_CONTRAST_MAX, flag_junk
from retrace.sanitize.image_stats import load_stats

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    review = json.loads((ROOT / "results" / "m1_junk_review.json").read_text())["answers"]
    manual_no = {i for i, a in review.items() if a == "no"}
    report = {"rule": f"contrast < {JUNK_CONTRAST_MAX} OR reviewer answered 'no'",
              "manual_no": len(manual_no), "splits": {}}
    seen = set()
    for s in ("train", "val", "test"):
        p = DATA / "manifests" / f"osv5m_{s}_sanitized.parquet"
        m = pd.read_parquet(p).drop(columns=["junk", "junk_reason"], errors="ignore")
        st = load_stats(m[["id", "sanitized_path"]], DATA)[["id", "contrast"]]
        out = flag_junk(m.merge(st, on="id", how="left"), manual_no).drop(columns="contrast")
        assert len(out) == len(m)
        seen |= set(out.id) & manual_no
        report["splits"][s] = {
            "n": len(out), "junk": int(out.junk.sum()),
            "by_reason": {k: int(v) for k, v in out.junk_reason[out.junk].value_counts().items()},
            "junk_with_time_label": int((out.junk & out.time_reliable).sum()),
        }
        out.to_parquet(p, index=False)
    assert seen == manual_no, f"reviewed ids not found in manifests: {manual_no - seen}"
    (ROOT / "results" / "m1_junk_flags.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
