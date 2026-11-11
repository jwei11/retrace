"""OSV-5M metadata -> label manifests (M1). Labels only; images are handled by sanitize/.

Time labels: `captured_at` is epoch ms. We found it is NOT uniformly true UTC (some
uploaders' clocks are wrong or local-as-UTC), so each row gets `time_reliable`, decided
per creator by a daylight test. See DECISIONS.md D-012 and results/m1_time_label_qa.md.
"""
from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from retrace.tools.sun import solar_position

DAYLIGHT_ELEV_DEG = -6.0          # civil twilight or brighter
CREATOR_DAYLIGHT_MIN = 0.90       # creator's photos that must be in daylight under UTC
CREATOR_MIN_PHOTOS = 20           # fewer than this -> can't judge -> unreliable


@cache
def _tf() -> TimezoneFinder:
    return TimezoneFinder()


def zone_at(lat: float, lon: float) -> str | None:
    """Timezone from coordinates (ported from PhotoTrails app/tz.py)."""
    return _tf().timezone_at(lat=lat, lng=lon)


def load_slim(path) -> pd.DataFrame:
    df = pd.read_parquet(path).rename(columns={"latitude": "lat", "longitude": "lon"})
    df["group_id"] = "osv-seq-" + df["sequence"].astype(str)
    return df


def creator_time_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Per creator: fraction of photos in daylight if captured_at is read as true UTC."""
    el, _ = solar_position(df.lat.to_numpy(), df.lon.to_numpy(), df.captured_at.to_numpy() / 1000)
    day = pd.Series(el >= DAYLIGHT_ELEV_DEG, index=df.index)
    q = day.groupby(df.creator_id).agg(["mean", "size"]).rename(
        columns={"mean": "daylight_frac", "size": "n_photos"})
    q["time_reliable"] = (q.daylight_frac >= CREATOR_DAYLIGHT_MIN) & (q.n_photos >= CREATOR_MIN_PHOTOS)
    return q


def add_time_labels(df: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    utc = pd.to_datetime(out.captured_at, unit="ms", utc=True)
    out["utc_time"] = utc
    out["tz"] = [zone_at(a, b) for a, b in zip(out.lat, out.lon)]
    out["local_time"] = [
        t.tz_convert(z).tz_localize(None) if z else pd.NaT for t, z in zip(utc, out.tz)
    ]
    el, _ = solar_position(out.lat.to_numpy(), out.lon.to_numpy(), out.captured_at.to_numpy() / 1000)
    out["sun_elev_label"] = np.round(el, 2)  # label-side QA only
    creator_ok = out.creator_id.map(quality.time_reliable).fillna(False).astype(bool)
    # Per-photo rule (D-013): a label putting the sun below civil twilight is almost always a
    # daytime photo with a wrong clock (results/m1_time_label_brightness.json), so drop it.
    out["time_reliable"] = creator_ok & (out.sun_elev_label >= DAYLIGHT_ELEV_DEG)
    return out


MANIFEST_COLS = [
    "id", "sanitized_path", "lat", "lon", "utc_time", "local_time", "tz", "group_id",
    "source", "split",
    # Extra label-side columns (never agent-visible): stratification + QA.
    "source_id", "country", "region", "city", "creator_id", "time_reliable", "sun_elev_label",
]
