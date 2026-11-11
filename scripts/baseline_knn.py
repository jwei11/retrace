"""M6 baseline #1 (no LLM): SigLIP kNN. Also builds the embedding index the similar() tool will use.

  embed:  uv run python scripts/baseline_knn.py embed            (all splits; GPU must be free)
  run:    uv run python scripts/baseline_knn.py run val          (writes two runs: knn1, knn10)

Embeddings: google/siglip2-so400m-patch14-384 image features of the SANITIZED photos, fp16,
L2-normalized -> $RETRACE_DATA/embeddings/siglip2-so400m/{split}.npy + {split}_ids.parquet.
Neighbors come from TRAIN only (non-junk), never from the query's own group.
  knn1   copy the nearest neighbor's location and time
  knn10  among the 10 nearest, pick the neighbor with the most similarity mass within 25 km of
         it (a vote that ignores one-off lookalikes); time from the most similar time-reliable
         neighbor; radius = median distance from the pick to the other 9 neighbors
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from retrace.geo import haversine_km
from retrace.runs import write_run

DATA = Path(os.environ["RETRACE_DATA"])
MODEL = "google/siglip2-so400m-patch14-384"
EMB = DATA / "embeddings" / "siglip2-so400m"
SPLITS = ("train", "val", "test")


def manifest(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / "manifests" / f"osv5m_{split}_sanitized.parquet")


def embed() -> None:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).vision_model.to("cuda").eval()
    EMB.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        m = manifest(s)
        out = np.zeros((len(m), model.config.hidden_size), np.float16)
        bs = 64
        for i in range(0, len(m), bs):
            imgs = [Image.open(DATA / p).convert("RGB") for p in m.sanitized_path.iloc[i:i + bs]]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
            with torch.no_grad():
                f = model(pixel_values=px).pooler_output.float()
            out[i:i + len(imgs)] = torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
            if i % (bs * 50) == 0:
                print(f"{s}: {i}/{len(m)}", flush=True)
        np.save(EMB / f"{s}.npy", out)
        m[["id"]].to_parquet(EMB / f"{s}_ids.parquet", index=False)
        print(f"{s}: {len(m)} embedded", flush=True)


def load(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    ids = pd.read_parquet(EMB / f"{split}_ids.parquet")
    m = manifest(split).set_index("id").loc[ids.id].reset_index()
    return m, np.load(EMB / f"{split}.npy").astype(np.float32)


def run(split: str, k: int = 10, vote_km: float = 25.0) -> None:
    tr, tr_e = load("train")
    keep = ~tr.junk.to_numpy()
    tr, tr_e = tr[keep].reset_index(drop=True), tr_e[keep]
    q, q_e = load(split)
    q_keep = ~q.junk.to_numpy()
    q, q_e = q[q_keep].reset_index(drop=True), q_e[q_keep]

    a1, a10 = [], []
    for i0 in range(0, len(q), 256):
        sims = q_e[i0:i0 + 256] @ tr_e.T
        for j, srow in enumerate(sims):
            r = q.iloc[i0 + j]
            srow[tr.group_id.to_numpy() == r.group_id] = -np.inf  # never the query's own group
            nn = np.argpartition(-srow, k)[:k]
            nn = nn[np.argsort(-srow[nn])]
            nb = tr.iloc[nn]
            lat, lon, s = nb.lat.to_numpy(), nb.lon.to_numpy(), srow[nn]
            d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
            pick = int(np.argmax(((d <= vote_km) * s[None, :]).sum(1)))
            others = np.delete(d[pick], pick)
            timed = nb[nb.time_reliable]
            t10 = timed.utc_time.iloc[0] if len(timed) else nb.utc_time.iloc[0]
            t1 = nb.utc_time.iloc[0]
            a1.append({"id": r.id, "lat": float(lat[0]), "lon": float(lon[0]),
                       "radius_km": float(max(np.median(d[0, 1:]), 1.0)),
                       "utc_datetime": None if pd.isna(t1) else pd.Timestamp(t1).isoformat(),
                       "tokens_in": 0, "tokens_out": 0})
            a10.append({"id": r.id, "lat": float(lat[pick]), "lon": float(lon[pick]),
                        "radius_km": float(max(np.median(others), 1.0)),
                        "utc_datetime": None if pd.isna(t10) else pd.Timestamp(t10).isoformat(),
                        "tokens_in": 0, "tokens_out": 0})
    cfg = {"embedder": MODEL, "index": "train, non-junk", "k": k, "vote_km": vote_km}
    write_run(f"knn1_{split}", {"dataset": "osv5m", "split": split, "model": "baseline:siglip-knn1",
                                "config": {**cfg, "k": 1}, "notes": "nearest train neighbor's label"}, a1)
    write_run(f"knn10_{split}", {"dataset": "osv5m", "split": split, "model": "baseline:siglip-knn10",
                                 "config": cfg, "notes": "25 km similarity vote among 10 nearest"}, a10)
    print(f"wrote knn1_{split}, knn10_{split} ({len(q)} photos)")


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed()
    else:
        run(sys.argv[2])
