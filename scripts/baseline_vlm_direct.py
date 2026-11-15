"""Baseline: the VLM alone, no tools. One look at the full sanitized photo, one JSON answer.

The model gets ONLY the pixels (768x512 sanitized JPEG) and a fixed instruction; no metadata.
vLLM's structured output (JSON schema) guarantees a parseable answer; the schema puts a short
free-text "clues" field first so the model can reason before committing to numbers.

  bash scripts/serve_vllm.sh &          # vLLM on 127.0.0.1:8000
  uv run python scripts/baseline_vlm_direct.py val [--limit N] [--concurrency 8]
  uv run python scripts/score_run.py vlm_direct_<model>_val
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from retrace.runs import write_run

DATA = Path(os.environ["RETRACE_DATA"])
URL = "http://127.0.0.1:8000/v1"

INSTRUCTION = """You are an expert at working out where and when a photo was taken, using only what is visible.
Look at road markings, signs and scripts, vehicles, vegetation, terrain, architecture, the sun and shadows, weather and season.
Then answer:
- clues: 1-3 sentences on the decisive visual clues
- country: your best guess
- lat, lon: your best single guess in decimal degrees
- radius_km: radius around your guess within which you think the true spot lies with about 40% probability
- utc_datetime: your best guess of the capture time in UTC, ISO 8601 (e.g. 2019-07-14T09:30:00Z)
- time_uncertainty_hours: how far off you think that time might be, in hours"""

SCHEMA = {
    "type": "object",
    "properties": {
        "clues": {"type": "string", "maxLength": 600},
        "country": {"type": "string", "maxLength": 60},
        "lat": {"type": "number", "minimum": -90, "maximum": 90},
        "lon": {"type": "number", "minimum": -180, "maximum": 180},
        "radius_km": {"type": "number", "minimum": 0.1, "maximum": 20000},
        "utc_datetime": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$"},
        "time_uncertainty_hours": {"type": "number", "minimum": 0, "maximum": 100000},
    },
    "required": ["clues", "country", "lat", "lon", "radius_km", "utc_datetime", "time_uncertainty_hours"],
    "additionalProperties": False,
}


def ask(model: str, path: Path, max_tokens: int) -> dict:
    img = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    body = {
        "model": model, "temperature": 0, "max_tokens": max_tokens, "seed": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img}},
            {"type": "text", "text": INSTRUCTION}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}},
    }
    t = time.time()
    r = requests.post(f"{URL}/chat/completions", json=body, timeout=600)
    lat_s = time.time() - t
    r.raise_for_status()
    j = r.json()
    out = json.loads(j["choices"][0]["message"]["content"])
    return {"answer": out, "latency_s": lat_s, "usage": j["usage"]}


def one(model: str, row, max_tokens: int, retries: int = 2) -> dict:
    rec = {"id": row.id}
    for attempt in range(retries + 1):
        try:
            res = ask(model, DATA / row.sanitized_path, max_tokens)
            a = res["answer"]
            rec.update(lat=a["lat"], lon=a["lon"], radius_km=a["radius_km"], utc_datetime=a["utc_datetime"],
                       time_uncertainty_hours=a["time_uncertainty_hours"], country_guess=a["country"],
                       clues=a["clues"], tokens_in=res["usage"]["prompt_tokens"],
                       tokens_out=res["usage"]["completion_tokens"], latency_s=round(res["latency_s"], 3))
            return rec
        except Exception as e:  # noqa: BLE001 - recorded as an error; scored as no answer
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--dataset", default="osv5m")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    model = requests.get(f"{URL}/models", timeout=30).json()["data"][0]["id"]
    m = pd.read_parquet(DATA / "manifests" / f"{args.dataset}_{args.split}_sanitized.parquet")
    m = m[~m.junk]
    if args.limit:
        m = m.sample(args.limit, random_state=0)
    t = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        answers = list(ex.map(lambda r: one(model, r, args.max_tokens), m.itertuples()))
    wall = time.time() - t
    n_err = sum("error" in a for a in answers)

    short = re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-")
    ds = "" if args.dataset == "osv5m" else f"{args.dataset}_"
    run_id = f"vlm_direct_{short}_{ds}{args.split}" + (f"_n{args.limit}" if args.limit else "")
    write_run(run_id, {
        "dataset": args.dataset, "split": args.split, "model": model,
        "config": {"runtime": "vllm 0.33.0 fp16", "image": "sanitized 768x512, full frame", "tools": None,
                   "temperature": 0, "max_tokens": args.max_tokens, "structured_output": "json_schema",
                   "concurrency": args.concurrency},
        "notes": f"model alone, one look, no tools. wall {wall:.0f}s for {len(answers)} photos, {n_err} errors",
    }, answers)
    print(f"{run_id}: {len(answers)} photos in {wall:.0f}s ({len(answers) / wall:.2f}/s), {n_err} errors")


if __name__ == "__main__":
    main()
