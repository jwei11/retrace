"""D-051: find photos whose pixels carry burned-in label text (date, time, GPS) from camera overlays.

  uv run --with easyocr python scripts/audit_overlays.py <dataset> <split> [--ids id1,id2] [--limit N]

OCR (EasyOCR, English, CPU) on each sanitized image; regexes flag dates, clock times, coordinates
and speed readouts. Writes $RETRACE_DATA/audits/overlay_<dataset>_<split>.parquet
(id, text, has_date, has_time, has_coord, has_speed, has_brand, leak = date|time|coord).
Signs can contain digits too, so flags are candidates; a sample is checked by eye (see DECISIONS).
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA = Path(os.environ["RETRACE_DATA"])
RX = {
    "has_date": re.compile(r"\b(19|20)\d{2}\s?[./-]\s?\d{2}.?\d{2}\b|"   # OCR often garbles the 2nd separator
                           r"\b(19|20)\d{2}\s?[./-]\s?\d{1,2}\s?[./-]\s?\d{1,2}\b|\b\d{1,2}\s?[./-]\s?\d{1,2}\s?[./-]\s?(19|20)?\d{2}\b"),
    "has_time": re.compile(r"\b([01]?\d|2[0-3])\s?[:.]\s?[0-5]\d\s?[:.]\s?[0-5]\d\b|\b([01]?\d|2[0-3]):[0-5]\d\b"),
    "has_coord": re.compile(r"[NSEW]\s?-?\d{1,3}[.,]\d{3,}|\b-?\d{1,3}[.,]\d{4,}\s?[NSEW]?\b"),
    "has_speed": re.compile(r"\b\d{1,3}\s?(km\s?/?\s?h|kmh|mph)\b", re.I),
    "has_brand": re.compile(r"mivue|blackvue|garmin|nextbase|viofo|thinkware|papago|transcend|\bdod\b|lukas|"
                            r"vantrue|rexing|azdome|70mai|gopro|dashcam|dash cam|\bgps\b", re.I),
}
BANDS = ((0.0, 0.18), (0.78, 1.0))   # overlays sit in the top or bottom strip; OCR them upscaled


def ocr(reader, im: Image.Image) -> str:
    w, h = im.size
    parts = []   # bands only: overlays live there, and full-frame OCR mostly adds street signs
    for a, b in BANDS:
        band = im.crop((0, int(a * h), w, int(b * h)))
        parts.append(np.asarray(band.resize((band.width * 2, band.height * 2), Image.BICUBIC)))
    seen, out = set(), []
    for arr in parts:
        for _, t, conf in reader.readtext(arr):
            if conf > 0.2 and t not in seen:
                seen.add(t)
                out.append(t)
    return " | ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("split")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    m = pd.read_parquet(DATA / "manifests" / f"{args.dataset}_{args.split}_sanitized.parquet")
    if args.ids:
        m = m[m.id.str[:10].isin(args.ids.split(","))]
    if args.limit:
        m = m.head(args.limit)
    rows, t0 = [], time.time()
    for k, r in enumerate(m.itertuples()):
        text = ocr(reader, Image.open(DATA / r.sanitized_path).convert("RGB"))
        rows.append({"id": r.id, "text": text, **{f: bool(rx.search(text)) for f, rx in RX.items()}})
        if k % 200 == 0:
            print(f"{k}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["leak"] = df.has_date | df.has_time | df.has_coord
    df["overlay"] = df.leak | df.has_speed | df.has_brand   # candidates for review
    out = DATA / "audits" / f"overlay_{args.dataset}_{args.split}.parquet"
    if not (args.ids or args.limit):
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"wrote {out}")
    print(f"{len(df)} photos in {time.time() - t0:.0f}s; " +
          ", ".join(f"{c} {df[c].mean():.1%}" for c in ["has_date", "has_time", "has_coord", "has_speed", "has_brand", "leak", "overlay"]))
    if args.ids or args.limit:
        print(df.to_string(max_colwidth=120))


if __name__ == "__main__":
    main()
