import numpy as np
import pandas as pd
import pytest

from retrace.ingest.time_labels import DarkCutoffs, apply_time_rule, dark_cutoffs


def frame(rows):
    return pd.DataFrame(rows, columns=["sun_elev_label", "creator_time_ok", "brightness", "sky_brightness"])


CUT = DarkCutoffs(brightness=70.0, sky_brightness=60.0, n_day_photos=1000)


def test_statuses():
    df = frame([
        (30.0, True, 120, 150),    # day                         -> ok_day
        (-30.0, True, 40, 20),     # night label, dark image     -> rescued
        (-30.0, True, 130, 160),   # night label, bright image   -> dropped
        (-30.0, True, 40, 140),    # dark frame but bright sky (e.g. dark forest at noon) -> dropped
        (-30.0, False, 40, 20),    # unreliable uploader wins over everything
        (30.0, False, 120, 150),
    ])
    out = apply_time_rule(df, CUT)
    assert list(out.time_label_status) == [
        "ok_day", "rescued_night", "dropped_night", "dropped_night",
        "creator_unreliable", "creator_unreliable"]
    assert list(out.time_reliable) == [True, True, False, False, False, False]


def test_twilight_is_not_touched():
    # Between -6 and +6 degrees the night rule doesn't apply, even if the image is dark.
    out = apply_time_rule(frame([(-3.0, True, 30, 20)]), CUT)
    assert out.time_label_status.iat[0] == "ok_day"


def test_cutoffs_use_only_clear_day_photos_from_reliable_uploaders():
    rng = np.random.default_rng(0)
    day = frame([(40.0, True, b, b + 20) for b in rng.normal(120, 20, 2000)])
    noise = frame([(-40.0, True, 5, 5)] * 500 + [(40.0, False, 5, 5)] * 500)
    cut = dark_cutoffs(pd.concat([day, noise]))
    assert cut.n_day_photos == 2000
    assert cut.brightness == pytest.approx(np.quantile(day.brightness, 0.02))


def test_cutoffs_refuse_tiny_samples():
    with pytest.raises(ValueError):
        dark_cutoffs(frame([(40.0, True, 100, 100)] * 10))
