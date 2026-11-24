"""D-052: choose the skill per photo from retrieval confidence, before the episode starts.

The signal is the top SigLIP2 similarity between the photo and the labeled reference index (the
photo's own capture sequence excluded, as in similar()). No labels of the photo are used.
"""
from __future__ import annotations

import numpy as np


def top_similarity(emb: np.ndarray, groups: np.ndarray, query: np.ndarray, group_id: str) -> float:
    s = emb @ query
    s = np.where(groups == group_id, -np.inf, s)
    return float(s.max())


def parse_route(spec: str) -> tuple[str, float]:
    """'playbook_v1@0.80' -> ('playbook_v1', 0.80)."""
    skill, _, thr = spec.partition("@")
    if not skill or not thr:
        raise ValueError(f"route must look like skill@threshold, got {spec!r}")
    return skill, float(thr)


def choose(top: float, skill: str, threshold: float) -> str:
    """The skill when the photo resembles the index enough to trust its neighbors, else none."""
    return skill if top >= threshold else ""
