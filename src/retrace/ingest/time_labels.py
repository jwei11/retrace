"""Final time-label rule (D-012, D-013, D-016). Applied after sanitization, because the night
rescue needs image brightness.

  creator_unreliable  uploader fails the daylight rule (D-012)          -> no time label
  ok_day              labeled sun >= -6 deg                              -> time label kept
  rescued_night       labeled sun < -6 deg AND the image is genuinely dark
                      (whole frame AND sky darker than the cutoff)       -> time label kept
  dropped_night       labeled sun < -6 deg but the image looks like day  -> no time label

Cutoffs are percentiles of DAY-labeled TRAIN photos, so no eval data tunes the rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

NIGHT_ELEV_DEG = -6.0
DAY_ELEV_DEG = 6.0        # "clearly day" photos used to set the darkness cutoffs
DARK_PERCENTILE = 0.02    # darker than 98% of day photos


@dataclass(frozen=True)
class DarkCutoffs:
    brightness: float
    sky_brightness: float
    n_day_photos: int
    percentile: float = DARK_PERCENTILE


def dark_cutoffs(train: pd.DataFrame) -> DarkCutoffs:
    """train needs columns: sun_elev_label, creator_time_ok, brightness, sky_brightness."""
    day = train[(train.sun_elev_label >= DAY_ELEV_DEG) & train.creator_time_ok]
    if len(day) < 500:
        raise ValueError(f"only {len(day)} day photos to set cutoffs; need >= 500")
    return DarkCutoffs(float(day.brightness.quantile(DARK_PERCENTILE)),
                       float(day.sky_brightness.quantile(DARK_PERCENTILE)), len(day))


def apply_time_rule(df: pd.DataFrame, cut: DarkCutoffs) -> pd.DataFrame:
    out = df.copy()
    night = out.sun_elev_label < NIGHT_ELEV_DEG
    dark = (out.brightness < cut.brightness) & (out.sky_brightness < cut.sky_brightness)
    status = pd.Series("ok_day", index=out.index)
    status[night & dark] = "rescued_night"
    status[night & ~dark] = "dropped_night"
    status[~out.creator_time_ok] = "creator_unreliable"
    out["time_label_status"] = status
    out["time_reliable"] = status.isin(["ok_day", "rescued_night"])
    return out


def cutoffs_dict(cut: DarkCutoffs) -> dict:
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(cut).items()}
