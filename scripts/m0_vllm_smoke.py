"""M0 (runtime, D-005): time a VLM served by vLLM (scripts/serve_vllm.sh) on the same request as
scripts/m0_vlm_smoke.py (same prompt, 384 px thumbnail, greedy, exactly 200 new tokens), alone
and with N concurrent requests. Appends to results/m0_vlm_smoke.jsonl.

  uv run python scripts/m0_vllm_smoke.py [--concurrency 8]
Uses public val images only.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PROMPT = (
    "Where and when was this photo taken? Reason briefly about visible clues "
    "(road markings, vegetation, signs, sun), then give a country, approximate "
    "latitude/longitude, season and time of day."
)
URL = "http://127.0.0.1:8000/v1/chat/completions"
DATA = Path(os.environ["RETRACE_DATA"])


def data_url(path: Path, thumb: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb, thumb))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(model: str, url_img: str, n_tok: int) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": n_tok, "min_tokens": n_tok,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url_img}},
                {"type": "text", "text": PROMPT}]}]}
    t = time.time()
    r = requests.post(URL, json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    return {"s": time.time() - t, "usage": j["usage"], "text": j["choices"][0]["message"]["content"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--thumb", type=int, default=384)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()
    model = requests.get("http://127.0.0.1:8000/v1/models", timeout=30).json()["data"][0]["id"]
    val = pd.read_parquet(DATA / "manifests" / "osv5m_val_sanitized.parquet").sample(
        args.concurrency + 1, random_state=0)
    imgs = [data_url(DATA / p, args.thumb) for p in val.sanitized_path]

    ask(model, imgs[0], 8)  # warm-up
    one = ask(model, imgs[0], args.max_new_tokens)
    t = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        many = list(ex.map(lambda u: ask(model, u, args.max_new_tokens), imgs[1:]))
    wall = time.time() - t
    n_many = sum(m["usage"]["completion_tokens"] for m in many)

    rec = {
        "model": model, "runtime": "vllm 0.33.0 server, fp16, TRITON_ATTN, max_model_len 6144",
        "device": "RTX 2070 SUPER", "thumb_px": args.thumb,
        "prompt_tokens": one["usage"]["prompt_tokens"],
        "new_tokens": one["usage"]["completion_tokens"],
        "gen_s": round(one["s"], 2),
        "tok_per_s": round(one["usage"]["completion_tokens"] / one["s"], 1),
        "concurrency": args.concurrency,
        "concurrent_wall_s": round(wall, 2),
        "concurrent_total_tok_per_s": round(n_many / wall, 1),
        "output": one["text"],
    }
    print(json.dumps(rec, indent=2))
    with (Path(__file__).resolve().parents[1] / "results" / "m0_vlm_smoke.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
