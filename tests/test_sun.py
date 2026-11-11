from datetime import datetime, timezone

import numpy as np
from astral import Observer
from astral.sun import azimuth, elevation

from retrace.tools.sun import solar_position, sun


def test_matches_astral_on_random_points():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-70, 70, 200)
    lon = rng.uniform(-180, 180, 200)
    t = rng.uniform(1.3e9, 1.7e9, 200)
    el, az = solar_position(lat, lon, t)
    for i in range(200):
        dt = datetime.fromtimestamp(t[i], tz=timezone.utc)
        obs = Observer(lat[i], lon[i])
        assert abs(el[i] - elevation(obs, dt, with_refraction=False)) < 0.3
        if el[i] > -10:  # azimuth is ill-conditioned near the zenith; keep loose
            d = abs((az[i] - azimuth(obs, dt)) % 360)
            assert min(d, 360 - d) < 1.0


def test_tool_phases():
    # Greenwich, summer solstice: high noon vs midnight.
    assert sun(51.48, 0.0, "2021-06-21T12:00:00Z")["phase"] == "day"
    assert sun(51.48, 0.0, "2021-06-21T00:00:00Z")["phase"] == "night"
    out = sun(51.48, 0.0, "2021-06-21T12:00:00+00:00")
    assert 60 < out["altitude_deg"] < 63 and 170 < out["azimuth_deg"] < 190
