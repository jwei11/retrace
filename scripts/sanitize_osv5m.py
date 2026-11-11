"""M2: fetch the sampled OSV-5M images (range reads), sanitize, write final manifests.

  raw bytes  -> $RETRACE_DATA/osv5m/raw/<source_id>.jpg       (label side; never agent-visible)
  sanitized  -> $RETRACE_DATA/images/<2hex>/<hash>.jpg        (the only dir the harness reads)
  manifest   -> $RETRACE_DATA/manifests/osv5m_<split>_sanitized.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from retrace.ingest import osv5m_remote
from retrace.ingest.osv5m import MANIFEST_COLS
from retrace.sanitize.sanitize import sanitize_bytes, write

DATA = Path(os.environ["RETRACE_DATA"])
SHARD_SPLIT = {"train": "train", "val": "train", "test": "test"}


def main(splits: list[str]) -> None:
    raw_dir = DATA / "osv5m" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        man = pd.read_parquet(DATA / "manifests" / f"osv5m_{split}_raw.parquet")
        index = osv5m_remote.build_index(SHARD_SPLIT[split],
                                         DATA / "osv5m" / f"index_{SHARD_SPLIT[split]}.parquet")
        need = index[index.id.isin(man.source_id) & ~index.id.map(
            lambda i: (raw_dir / f"{i}.jpg").exists())]
        missing = set(man.source_id) - set(index.id)
        if missing:
            print(f"[{split}] {len(missing)} ids not found in shards; dropping", file=sys.stderr)
        failed = 0
        for sid, data in tqdm(osv5m_remote.fetch(need.itertuples()), total=len(need), desc=f"fetch {split}"):
            if data is None:  # still failing after retries; a later re-run picks it up
                failed += 1
                continue
            (raw_dir / f"{sid}.jpg").write_bytes(data)
        if failed:
            print(f"[{split}] {failed} images failed to download; re-run to retry", file=sys.stderr)

        ids, paths = [], []
        for sid in tqdm(man.source_id, desc=f"sanitize {split}"):
            p = raw_dir / f"{sid}.jpg"
            if not p.exists():
                ids.append(None), paths.append(None)
                continue
            s = sanitize_bytes(p.read_bytes())
            ids.append(s.id), paths.append(write(s, DATA))
        man = man.assign(id=ids, sanitized_path=paths).dropna(subset=["id"])
        dup = man.id.duplicated(keep="first")
        if dup.any():
            print(f"[{split}] dropping {int(dup.sum())} pixel-identical duplicates", file=sys.stderr)
            man = man[~dup]
        man[MANIFEST_COLS].to_parquet(DATA / "manifests" / f"osv5m_{split}_sanitized.parquet", index=False)
        print(f"[{split}] wrote {len(man)} rows")


if __name__ == "__main__":
    main(sys.argv[1:] or ["val", "test", "train"])
