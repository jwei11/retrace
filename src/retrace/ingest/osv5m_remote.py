"""Fetch individual OSV-5M images out of the remote zip shards with HTTP range reads.

OSV-5M ships 98 train shards (~2.5 GB each, 246 GB total) and 5 test shards (9.4 GB).
We need ~26k images, so instead of downloading shards we:
  1. read each remote zip's central directory once (HfFileSystem, a few MB per shard)
     and record every member's byte offset and size -> a parquet index;
  2. fetch only the sampled members with parallel HTTP Range requests and parse the
     zip local header ourselves.
Total transfer is ~= the size of the images we keep. See DECISIONS.md D-010.
"""
from __future__ import annotations

import random
import struct
import time
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfFileSystem

REPO = "datasets/osv5m/osv5m"
RESOLVE = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/"
LOCAL_HEADER = struct.Struct("<4s5H3I2H")  # 30 bytes
EXTRA_SLACK = 1024  # local-header extra field may be longer than the central one


def shard_paths(split: str) -> list[str]:
    fs = HfFileSystem()
    return sorted(p for p in fs.ls(f"{REPO}/images/{split}", detail=False) if p.endswith(".zip"))


def build_index(split: str, out: Path) -> pd.DataFrame:
    """id -> (shard, header_offset, compress_size, compress_type). Cached as parquet."""
    if out.exists():
        return pd.read_parquet(out)
    fs = HfFileSystem()
    rows = []
    for path in shard_paths(split):
        rel = path.removeprefix(REPO + "/")
        with zipfile.ZipFile(fs.open(path, "rb", block_size=1 << 20)) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".jpg"):
                    rows.append((Path(info.filename).stem, rel, info.header_offset,
                                 info.compress_size, info.compress_type, len(info.filename)))
    df = pd.DataFrame(rows, columns=["id", "shard", "offset", "csize", "ctype", "name_len"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def _fetch_one(session: requests.Session, row) -> bytes:
    start = int(row.offset)
    end = start + LOCAL_HEADER.size + int(row.name_len) + EXTRA_SLACK + int(row.csize)
    r = session.get(RESOLVE + row.shard, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    buf = r.content
    sig, *_rest, n_len, e_len = LOCAL_HEADER.unpack_from(buf)
    if sig != b"PK\x03\x04":
        raise ValueError(f"bad local header for {row.id}")
    body = buf[LOCAL_HEADER.size + n_len + e_len: LOCAL_HEADER.size + n_len + e_len + int(row.csize)]
    if row.ctype == zipfile.ZIP_STORED:
        return body
    if row.ctype == zipfile.ZIP_DEFLATED:
        return zlib.decompress(body, -15)
    raise ValueError(f"unsupported compression {row.ctype}")


def fetch(rows: Iterable, workers: int = 12, retries: int = 8) -> Iterator[tuple[str, bytes | None]]:
    """Yield (id, jpeg bytes) for index rows (namedtuples from DataFrame.itertuples()).

    Hugging Face rate-limits anonymous range requests (HTTP 429), so we keep concurrency
    modest, back off exponentially (honoring Retry-After), and yield (id, None) for an
    image that still fails instead of aborting the whole batch.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)

    def task(row):
        for attempt in range(retries):
            try:
                return row.id, _fetch_one(session, row)
            except requests.HTTPError as e:
                wait = 2 ** attempt + random.random()
                if e.response is not None and e.response.status_code == 429:
                    ra = e.response.headers.get("Retry-After", "")
                    wait = max(wait, float(ra)) if ra.replace(".", "", 1).isdigit() else wait
                time.sleep(min(wait, 120))
            except Exception:  # noqa: BLE001 - transport/parse error: brief retry
                time.sleep(1 + attempt)
        return row.id, None

    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(task, r) for r in rows]):
            yield fut.result()
