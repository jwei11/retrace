import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from retrace.score.score import (
    CALIBRATED_COVERAGE, Answer, Label, aggregate, score_answer, vmf_nll_per_km2,
)

PARIS = (48.8566, 2.3522)
LONDON = (51.5074, -0.1278)  # ~344 km from Paris


def lab(**kw):
    base = dict(lat=PARIS[0], lon=PARIS[1],
                utc_time=datetime(2019, 7, 4, 10, 0, tzinfo=timezone.utc),
                tz="Europe/Paris", time_reliable=True)
    base.update(kw)
    return Label(**base)


def test_perfect_answer():
    s = score_answer(Answer(*PARIS, radius_km=5, utc_datetime="2019-07-04T10:00:00Z"), lab())
    assert s["dist_km"] < 1e-6 and all(s[f"acc@{k}"] == 1 for k in (1, 25, 200, 750, 2500))
    assert s["geo_shaped"] == pytest.approx(1.0)
    assert s["hour_err_h"] == 0 and s["doy_err_d"] == 0 and s["year_err"] == 0


def test_paris_london_distance_and_thresholds():
    s = score_answer(Answer(*LONDON, radius_km=500), lab())
    assert 340 < s["dist_km"] < 347
    assert s["acc@200"] == 0 and s["acc@750"] == 1
    assert s["in_radius"] == 1.0


def test_hour_error_is_circular_and_in_true_local_zone():
    # 23:30 local vs 00:30 local next day = 1h, not 23h.
    true = datetime(2019, 7, 4, 22, 30, tzinfo=timezone.utc)  # 00:30 Paris (UTC+2), Jul 5
    s = score_answer(Answer(*PARIS, 1, "2019-07-04T21:30:00Z"), lab(utc_time=true))
    assert s["hour_err_h"] == pytest.approx(1.0)


def test_day_of_year_wraps_around_new_year():
    true = datetime(2019, 1, 1, 12, tzinfo=timezone.utc)
    s = score_answer(Answer(*PARIS, 1, "2018-12-31T12:00:00Z"), lab(utc_time=true))
    assert s["doy_err_d"] == pytest.approx(1.0, abs=0.01)
    assert s["year_err"] == 1


def test_time_ignored_when_label_unreliable():
    s = score_answer(Answer(*PARIS, 1, "2019-07-04T10:00:00Z"), lab(time_reliable=False))
    assert s["hour_err_h"] is None


def test_missing_answer_gets_worst_case():
    s = score_answer(None, lab())
    assert not s["valid"] and s["acc@2500"] == 0 and s["dist_km"] > 20000
    assert s["geo_nll"] > score_answer(Answer(*LONDON, 5000), lab())["geo_nll"]


def test_vmf_matches_flat_gaussian_for_small_radius():
    sigma, d = 10.0, 7.0
    flat = math.log(2 * math.pi * sigma**2) + d**2 / (2 * sigma**2)
    assert vmf_nll_per_km2(d, sigma) == pytest.approx(flat, abs=1e-4)


def test_vmf_is_proper_honest_radius_wins():
    """Sample errors from a 2-D Gaussian with sigma=50; reporting 50 should beat 10 and 250."""
    rng = np.random.default_rng(0)
    d = np.hypot(*rng.normal(0, 50, (2, 20000)))
    mean = {r: np.mean([vmf_nll_per_km2(x, r) for x in d]) for r in (10, 50, 250)}
    assert mean[50] < mean[10] and mean[50] < mean[250]
    assert np.mean(d <= 50) == pytest.approx(CALIBRATED_COVERAGE, abs=0.01)


def test_vmf_large_radius_tends_to_uniform():
    uniform = math.log(4 * math.pi * 6371.0088**2)
    assert vmf_nll_per_km2(10000, 1e6) == pytest.approx(uniform, rel=1e-4)


def test_aggregate_micro_and_macro():
    rows = [score_answer(Answer(*PARIS, 10), lab()) for _ in range(9)]
    rows.append(score_answer(Answer(*LONDON, 10), lab(lat=-33.87, lon=151.21)))  # Sydney truth
    df = pd.DataFrame(rows)
    strata = pd.Series(["FR"] * 9 + ["AU"])
    agg = aggregate(df, strata)
    assert agg["micro"]["acc@25"] == pytest.approx(0.9)
    assert agg["macro_by_stratum"]["acc@25"] == pytest.approx(0.5)  # one country is 0/1
