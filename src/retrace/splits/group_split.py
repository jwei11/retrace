"""Group-level splits with spatial separation (M3).

Rules (DECISIONS.md D-020):
  * the unit is the group (OSV-5M `sequence`, private `trip`); a group never spans splits;
  * at most `max_per_group` images per group, so long sequences can't dominate;
  * every train image is >= `min_sep_km` from every val image (OSV-5M's own test protocol
    uses 1 km), so the kNN tool can't return the same street corner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retrace.geo import min_dist_km


@dataclass
class SplitReport:
    n_train: int
    n_val: int
    dropped_for_separation: int
    train_groups: int
    val_groups: int
    creator_overlap_val_frac: float  # val images whose creator also appears in train


def cap_per_group(df: pd.DataFrame, group_col: str, k: int, seed: int) -> pd.DataFrame:
    return (df.sample(frac=1, random_state=seed)
              .groupby(group_col, sort=False).head(k))


def carve_train_val(
    df: pd.DataFrame, n_train: int, n_val: int, *, group_col: str = "group_id",
    max_per_group: int = 2, min_sep_km: float = 1.0, seed: int = 0, oversample: float = 1.15,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
    rng = np.random.default_rng(seed)
    capped = cap_per_group(df, group_col, max_per_group, seed)
    groups = np.asarray(capped[group_col].unique(), dtype=object)
    groups = groups[rng.permutation(len(groups))]  # (in-place shuffle of Arrow arrays is slow/unsafe)
    sizes = capped.groupby(group_col).size().reindex(groups).to_numpy()
    cum = np.cumsum(sizes)

    n_val_groups = int(np.searchsorted(cum, n_val) + 1)
    val_groups = set(groups[:n_val_groups])
    rest_cum = cum - cum[n_val_groups - 1]
    n_tr_groups = int(np.searchsorted(rest_cum, n_train * oversample) + 1)
    train_groups = set(groups[n_val_groups:n_val_groups + n_tr_groups])

    val = capped[capped[group_col].isin(val_groups)]
    train = capped[capped[group_col].isin(train_groups)]

    d = min_dist_km(train.lat.to_numpy(), train.lon.to_numpy(),
                    val.lat.to_numpy(), val.lon.to_numpy())
    keep = d >= min_sep_km
    dropped = int((~keep).sum())
    train = train[keep]
    # Truncate by whole groups so we never split a group.
    tcum = train.groupby(group_col, sort=False).size().cumsum()
    keep_groups = tcum.index[tcum <= n_train]
    train = train[train[group_col].isin(keep_groups)]

    assert not set(train[group_col]) & set(val[group_col])
    overlap = (val.creator_id.isin(set(train.creator_id)).mean()
               if "creator_id" in val else float("nan"))
    rep = SplitReport(len(train), len(val), dropped, train[group_col].nunique(),
                      val[group_col].nunique(), float(overlap))
    return train.assign(split="train"), val.assign(split="val"), rep
