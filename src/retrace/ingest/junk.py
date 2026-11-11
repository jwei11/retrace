"""Junk-image flags (D-017). Label side only: never agent-visible.

A photo is flagged when
  - its contrast (luma std on the sanitized image) is below JUNK_CONTRAST_MAX: fogged lens,
    near-blank frames, camera aimed at the road (rule; applies to any future data), or
  - a human reviewer marked it unlocatable ("no") in results/m1_junk_review.json (catches blurry
    and washed-out photos, which no single cutoff separates from usable ones).
Flagged photos stay on disk and in the manifests; scoring and training skip rows with junk=True.
"""
from __future__ import annotations

import pandas as pd

JUNK_CONTRAST_MAX = 15.0


def flag_junk(df: pd.DataFrame, manual_no: set[str]) -> pd.DataFrame:
    """Needs columns id, contrast. Adds junk (bool) and junk_reason ('', 'low_contrast',
    'manual', 'low_contrast+manual')."""
    out = df.copy()
    low = out.contrast < JUNK_CONTRAST_MAX
    man = out.id.isin(manual_no)
    reason = pd.Series("", index=out.index)
    reason[low] = "low_contrast"
    reason[man & ~low] = "manual"
    reason[man & low] = "low_contrast+manual"
    out["junk"] = low | man
    out["junk_reason"] = reason
    return out
