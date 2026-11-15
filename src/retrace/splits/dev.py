"""Fixed dev subsets for fast iteration (e.g. val dev300). Final numbers still use the full split."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def subset_ids(dataset: str, split: str, name: str) -> set[str]:
    """name 'devN': N non-junk photos sampled once with seed 0 and frozen to disk.
    name 'not-devN': every other non-junk photo of the split (held-out confirmation set)."""
    data = Path(os.environ["RETRACE_DATA"])
    if name.startswith("not-"):
        m = pd.read_parquet(data / "manifests" / f"{dataset}_{split}_sanitized.parquet")
        return set(m[~m.junk].id) - subset_ids(dataset, split, name.removeprefix("not-"))
    path = data / "splits" / f"{dataset}_{split}_{name}.parquet"
    if not path.exists():
        n = int(name.removeprefix("dev"))
        m = pd.read_parquet(data / "manifests" / f"{dataset}_{split}_sanitized.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        m[~m.junk].sample(n, random_state=0)[["id"]].to_parquet(path, index=False)
    return set(pd.read_parquet(path).id)
