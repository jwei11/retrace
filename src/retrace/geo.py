"""Shared geodesy helpers (labels side only; never exposed to the agent)."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64))
                              for x in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def min_dist_km(lat_a, lon_a, lat_b, lon_b, chunk: int = 2048) -> np.ndarray:
    """For each point in A, distance to the nearest point in B (brute force, chunked)."""
    lat_a, lon_a = np.asarray(lat_a), np.asarray(lon_a)
    lat_b, lon_b = np.asarray(lat_b)[None, :], np.asarray(lon_b)[None, :]
    out = np.empty(len(lat_a))
    for i in range(0, len(lat_a), chunk):
        d = haversine_km(lat_a[i:i + chunk, None], lon_a[i:i + chunk, None], lat_b, lon_b)
        out[i:i + chunk] = d.min(axis=1)
    return out
