import numpy as np
import pytest

from retrace.harness.router import choose, parse_route, top_similarity


def test_parse_route():
    assert parse_route("playbook_v1@0.80") == ("playbook_v1", 0.80)
    with pytest.raises(ValueError):
        parse_route("playbook_v1")


def test_choose_threshold_inclusive():
    assert choose(0.80, "playbook_v1", 0.80) == "playbook_v1"
    assert choose(0.79, "playbook_v1", 0.80) == ""


def test_top_similarity_skips_own_sequence():
    emb = np.eye(3, dtype=np.float32)
    groups = np.array(["a", "b", "c"])
    q = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    assert top_similarity(emb, groups, q, "x") == 1.0
    assert top_similarity(emb, groups, q, "a") == 0.5   # its own sequence is never the evidence
