"""Write a run in the layout scripts/score_run.py reads ($RETRACE_DATA/runs/<run_id>/)."""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path


def run_dir(run_id: str) -> Path:
    return Path(os.environ["RETRACE_DATA"]) / "runs" / run_id


def write_run(run_id: str, meta: dict, answers: Iterable[dict]) -> Path:
    """meta: dataset, split, model, config, notes (run_id is added). answers: dicts with id and
    lat/lon/radius_km/utc_datetime/time_uncertainty_hours/tokens_in/tokens_out/latency_s/error."""
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"run_id": run_id, **meta}, indent=2))
    with open(d / "answers.jsonl", "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")
    return d
