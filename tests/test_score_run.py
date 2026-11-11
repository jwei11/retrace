import json

import pandas as pd
import pytest

from retrace.score.run import load_answers, paired_bootstrap, score_run, summarize


def labels():
    t = pd.Timestamp("2020-06-01 12:00", tz="UTC")
    return pd.DataFrame({
        "id": ["a", "b", "c", "j"], "lat": [48.85, 40.71, -33.87, 0.0], "lon": [2.35, -74.0, 151.2, 0.0],
        "utc_time": [t, t, t, t], "tz": ["Europe/Paris", "America/New_York", "Australia/Sydney", "UTC"],
        "time_reliable": [True, True, False, True], "country": ["FR", "US", "AU", "XX"],
        "creator_id": ["u1", "u2", "u3", "u4"], "junk": [False, False, False, True]})


def ans(rows):
    return pd.DataFrame(rows).reindex(columns=["id", "lat", "lon", "radius_km", "utc_datetime",
                                               "time_uncertainty_hours", "tokens_in", "tokens_out",
                                               "latency_s", "error"])


def test_missing_answer_gets_worst_score_and_junk_is_excluded():
    a = ans([{"id": "a", "lat": 48.85, "lon": 2.35, "radius_km": 10, "utc_datetime": "2020-06-01T12:00:00Z"},
             {"id": "j", "lat": 0.0, "lon": 0.0, "radius_km": 1}])  # answer for a junk photo: ignored
    sc = score_run(a, labels(), train_creators={"u1"})
    assert list(sc.id) == ["a", "b", "c"]
    assert sc.answered.tolist() == [True, False, False]
    assert sc.loc[0, "dist_km"] < 0.01 and sc.loc[0, "year_exact"] == 1.0
    assert sc.loc[1, "dist_km"] > 20000 and sc.loc[1, "geo_nll"] is not None  # antipode, not skipped
    assert pd.isna(sc.loc[2, "year_exact"])  # time label not reliable -> no time score
    assert sc.uploader_seen.tolist() == [True, False, False]
    s = summarize(sc)
    assert s["answer_rate"] == pytest.approx(1 / 3)
    assert s["uploader"]["seen"]["n"] == 1 and s["uploader"]["new"]["n"] == 2


def test_unknown_id_is_an_error():
    with pytest.raises(ValueError, match="not in this split"):
        score_run(ans([{"id": "zzz", "lat": 0, "lon": 0}]), labels(), set())


def test_load_answers_rejects_duplicates(tmp_path):
    p = tmp_path / "answers.jsonl"
    p.write_text("\n".join(json.dumps({"id": "a", "lat": 1, "lon": 2}) for _ in range(2)))
    with pytest.raises(ValueError, match="duplicate"):
        load_answers(p)


def test_bootstrap_identical_runs_and_clear_win():
    perfect = ans([{"id": i, "lat": la, "lon": lo, "radius_km": 1}
                   for i, la, lo in [("a", 48.85, 2.35), ("b", 40.71, -74.0), ("c", -33.87, 151.2)]])
    sc_good = score_run(perfect, labels(), set())
    same = paired_bootstrap(sc_good, sc_good, n_boot=200)
    assert all(v["diff"] == 0 and not v["significant"] for v in same.values())
    sc_bad = score_run(ans([]), labels(), set())  # no answers at all
    win = paired_bootstrap(sc_good, sc_bad, n_boot=200)
    assert win["geo_shaped"]["diff"] > 0.9 and win["geo_shaped"]["significant"]
