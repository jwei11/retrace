"""M0: load a small VLM in 4-bit on the 8 GB Turing GPU, run one image, record VRAM and tok/s.

Usage: uv run python scripts/m0_vlm_smoke.py --model Qwen/Qwen3-VL-2B-Instruct --image /tmp/probe.jpg
Appends a JSON line to results/m0_vlm_smoke.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

PROMPT = (
    "Where and when was this photo taken? Reason briefly about visible clues "
    "(road markings, vegetation, signs, sun), then give a country, approximate "
    "latitude/longitude, season and time of day."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--thumb", type=int, default=384, help="long side in px, matches harness")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # Turing: no native bf16 (DECISIONS D-003)
    )
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    if getattr(config, "audio_config", None) is not None:
        # Gemma 4 ships an audio tower we never use; dropping it saves VRAM and avoids
        # its lack of SDPA support (DECISIONS D-004).
        config.audio_config = None
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        config=config,
        quantization_config=quant if args.quant == "nf4" else None,
        dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa",  # no FlashAttention-2 on sm_75
    )
    load_s = time.time() - t0
    vram_after_load = torch.cuda.memory_allocated() / 2**30

    img = Image.open(args.image).convert("RGB")
    img.thumbnail((args.thumb, args.thumb))
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    n_in = inputs["input_ids"].shape[1]

    # Warm-up (CUDA kernels / caches), then timed run.
    model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    t1 = time.time()
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    new = out[0, n_in:]
    text = processor.decode(new, skip_special_tokens=True)

    rec = {
        "model": args.model,
        "quant": "bnb-nf4-double, fp16 compute, sdpa" if args.quant == "nf4" else "fp16, sdpa",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "load_s": round(load_s, 1),
        "vram_weights_gb": round(vram_after_load, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "vram_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "prompt_tokens": int(n_in),
        "new_tokens": int(new.numel()),
        "gen_s": round(gen_s, 2),
        "tok_per_s": round(new.numel() / gen_s, 1),
        "output": text,
    }
    print(json.dumps(rec, indent=2))
    out_path = Path(__file__).resolve().parents[1] / "results" / "m0_vlm_smoke.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
