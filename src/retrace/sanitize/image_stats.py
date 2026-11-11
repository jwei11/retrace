"""Cheap per-image statistics on a SANITIZED image (what the agent sees). Label side only:
these feed data audits and label rules, never the agent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

STAT_COLS = ["brightness", "sky_brightness", "contrast", "sharpness"]


def image_stats(path: Path) -> dict[str, float]:
    """brightness: mean luma 0-255; sky_brightness: mean luma of the top quarter (mostly sky);
    contrast: luma std; sharpness: variance of the Laplacian (low = blurry)."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((384, 256)), dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return {
        "brightness": float(g.mean()),
        "sky_brightness": float(g[: g.shape[0] // 4].mean()),
        "contrast": float(g.std()),
        "sharpness": float(lap.var()),
    }


def load_stats(ids_paths: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Stats for every (id, sanitized_path) row, cached in <data_root>/audits/image_stats.parquet.
    Only images missing from the cache are computed."""
    cache = data_root / "audits" / "image_stats.parquet"
    old = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["id", *STAT_COLS])
    if not set(STAT_COLS) <= set(old.columns):  # older cache without sky_brightness
        old = pd.DataFrame(columns=["id", *STAT_COLS])
    todo = ids_paths[~ids_paths.id.isin(old.id)]
    new = pd.DataFrame([{"id": i, **image_stats(data_root / p)}
                        for i, p in zip(todo.id, todo.sanitized_path)], columns=["id", *STAT_COLS])
    st = pd.concat([old, new], ignore_index=True).drop_duplicates("id")
    st = st.astype({c: "float64" for c in STAT_COLS})  # concat with an empty frame gives object dtype
    cache.parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(cache, index=False)
    return st
