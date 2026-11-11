"""Solar position (NOAA approximation), vectorized with numpy. Offline, no deps beyond numpy.

Accuracy: ~0.1 deg in elevation for 1900-2100, plenty for "is the sun up / where are shadows".
Used both as the agent's `sun` tool and for label QA (is this timestamp in daylight?).
Reference: NOAA Global Monitoring Division solar calculator equations.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


def solar_position(lat, lon, unix_seconds):
    """Return (elevation_deg, azimuth_deg) arrays. azimuth: 0=N, 90=E, clockwise."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    t = np.asarray(unix_seconds, dtype=np.float64)

    jd = t / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0  # Julian century

    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360)  # mean longitude
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)  # mean anomaly
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)  # orbit eccentricity
    mr = np.radians(m)
    c = (np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * mr) * (0.019993 - 0.000101 * jc) + np.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))
    decl = np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(app_long)))

    y = np.tan(np.radians(eps / 2)) ** 2
    l0r = np.radians(l0)
    eq_time = 4 * np.degrees(
        y * np.sin(2 * l0r) - 2 * e * np.sin(mr) + 4 * e * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * e * e * np.sin(2 * mr)
    )  # minutes

    minutes_utc = np.mod(t, 86400.0) / 60.0
    true_solar_time = np.mod(minutes_utc + eq_time + 4 * lon, 1440)
    hour_angle = np.radians(true_solar_time / 4 - 180)

    latr = np.radians(lat)
    cos_zen = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(hour_angle)
    zen = np.arccos(np.clip(cos_zen, -1, 1))
    elevation = 90 - np.degrees(zen)

    az = np.degrees(np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(latr) - np.tan(decl) * np.cos(latr),
    )) + 180
    return elevation, np.mod(az, 360)


def sun(lat: float, lon: float, utc_datetime: str) -> dict:
    """Agent tool: solar altitude/azimuth for a place and UTC time (ISO 8601)."""
    dt = datetime.fromisoformat(utc_datetime.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    el, az = solar_position(lat, lon, dt.timestamp())
    el, az = float(el), float(az)
    phase = ("day" if el > 6 else "golden_hour" if el > 0 else
             "civil_twilight" if el > -6 else "night")
    return {"altitude_deg": round(el, 2), "azimuth_deg": round(az, 2), "phase": phase}
