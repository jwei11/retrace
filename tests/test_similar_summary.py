"""similar() agreement line (D-050) and Placebo matching on Tool.name."""
import numpy as np
import pandas as pd

from retrace.middleware import Placebo
from retrace.tools.builtin import _summary
from retrace.tools.registry import Obs, Tool


class FakeIdx:
    def __init__(self, coords):
        self.train = pd.DataFrame({"lat": [c[0] for c in coords], "lon": [c[1] for c in coords],
                                   "country": "FR", "region": "Normandy"})


def test_summary_counts_densest_group():
    idx = FakeIdx([(49.0, 0.0), (49.1, 0.1), (49.2, -0.1), (10.0, 100.0), (-30.0, -60.0)])
    s = _summary(idx, np.arange(5), np.array([0.9] * 5))
    assert "3 of 5 references lie within 300 km of #1" in s and "typical" in s


def test_summary_disagree_and_low_similarity():
    idx = FakeIdx([(0, 0), (40, 40), (-40, 100), (60, -120)])
    s = _summary(idx, np.arange(4), np.array([0.7] * 4))
    assert "disagree" in s and "LOW" in s


def test_summary_too_few_references():
    s = _summary(FakeIdx([(0, 0)]), np.arange(1), np.array([0.95]))
    assert "too few" in s and "lie within" not in s


def test_placebo_matches_tool_name_not_factory_key():
    t = Tool("similar", "", {}, lambda ctx: Obs("real"), lambda ctx: Obs("fake"))
    out = Placebo({"similar"})(None, t, {}, lambda ctx, tool, args: tool.fn(ctx, **args))
    assert out.text == "fake"
