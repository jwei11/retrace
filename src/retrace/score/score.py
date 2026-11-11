"""M4: scoring. One function per answer (`score_answer`), one aggregator per run (`aggregate`).

Location
  dist_km           haversine error
  acc@{1,25,200,750,2500}
  geo_shaped        exp(-d / tau_km)                                  (tau configurable)
Calibration (agent's radius_km)
  geo_nll           -log density (per km^2) of the truth under a von Mises-Fisher on the
                    sphere with kappa = (R / radius)^2  -- the spherical analogue of an
                    isotropic Gaussian with sigma = radius; a strictly proper scoring rule
                    that stays valid for continent-sized radii. (DECISIONS D-040)
  in_radius         d <= radius  (expect ~39.3% if calibrated, since P(r<=sigma)=1-e^-1/2)
Time (only where the label is reliable), all computed in the TRUE location's timezone
  PRIMARY (robust to few-hour clock errors in the labels):
  year_err          |year_pred - year_true| (integer years)
  doy_err_d         circular day-of-year error, in [0, 182.6]
  SECONDARY (label accuracy at this resolution is unverified, DECISIONS D-013/D-014):
  hour_err_h        circular hour-of-day error, in [0, 12]
  time_in_unc       |pred - true| <= time_uncertainty_hours
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from retrace.geo import EARTH_RADIUS_KM, haversine_km

ACC_KM = (1, 25, 200, 750, 2500)
DEFAULT_TAU_KM = 1492.7  # GeoGuessr's decay constant; makes geo_shaped familiar
MIN_RADIUS_KM = 0.01
DAYS_PER_YEAR = 365.2425
CALIBRATED_COVERAGE = 1 - math.exp(-0.5)  # 0.393


@dataclass(frozen=True)
class Answer:
    lat: float
    lon: float
    radius_km: float
    utc_datetime: str | None = None
    time_uncertainty_hours: float | None = None


@dataclass(frozen=True)
class Label:
    lat: float
    lon: float
    utc_time: datetime | None = None  # tz-aware UTC
    tz: str | None = None
    time_reliable: bool = False


def vmf_nll_per_km2(dist_km: float, radius_km: float) -> float:
    """-log p(truth) for a vMF on a sphere of radius R with kappa = (R/radius)^2, per km^2."""
    sigma = max(radius_km, MIN_RADIUS_KM)
    kappa = (EARTH_RADIUS_KM / sigma) ** 2
    cos_theta = math.cos(dist_km / EARTH_RADIUS_KM)
    if kappa < 1e-6:  # effectively uniform on the sphere
        log_p_sr = -math.log(4 * math.pi)
    else:
        # log[kappa / (4 pi sinh kappa)] + kappa cos(theta), rewritten to avoid overflow
        log_p_sr = (math.log(kappa) - math.log(2 * math.pi) - math.log1p(-math.exp(-2 * kappa))
                    + kappa * (cos_theta - 1))
    return -(log_p_sr - 2 * math.log(EARTH_RADIUS_KM))


def _circ(a: float, b: float, period: float) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _hour(dt: datetime) -> float:
    return dt.hour + dt.minute / 60 + dt.second / 3600


def _doy(dt: datetime) -> float:
    """Position in the year, scaled to DAYS_PER_YEAR so leap/non-leap years share one circle."""
    start = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    year_len = (start.replace(year=start.year + 1) - start).total_seconds()
    return (dt - start).total_seconds() / year_len * DAYS_PER_YEAR


def score_answer(ans: Answer | None, lab: Label, tau_km: float = DEFAULT_TAU_KM) -> dict:
    """Score one answer. `ans=None` (no answer / invalid) gets worst-case location scores."""
    out: dict = {}
    if ans is None or not (-90 <= ans.lat <= 90 and -180 <= ans.lon <= 180):
        d, valid = math.pi * EARTH_RADIUS_KM, False  # antipode
        radius = None
    else:
        d, valid = float(haversine_km(ans.lat, ans.lon, lab.lat, lab.lon)), True
        radius = ans.radius_km
    out["valid"] = valid
    out["dist_km"] = d
    for k in ACC_KM:
        out[f"acc@{k}"] = float(d <= k)
    out["geo_shaped"] = math.exp(-d / tau_km)
    if radius is not None and math.isfinite(radius) and radius > 0:
        out["geo_nll"] = vmf_nll_per_km2(d, radius)
        out["in_radius"] = float(d <= radius)
    else:
        out["geo_nll"] = vmf_nll_per_km2(math.pi * EARTH_RADIUS_KM, MIN_RADIUS_KM) if not valid else None
        out["in_radius"] = 0.0 if not valid else None

    out.update(hour_err_h=None, doy_err_d=None, year_err=None, time_in_unc=None)
    if lab.time_reliable and lab.utc_time is not None and ans is not None and ans.utc_datetime:
        try:
            pred = _parse_utc(ans.utc_datetime)
        except ValueError:
            pred = None
        if pred is not None:
            tz = ZoneInfo(lab.tz) if lab.tz else timezone.utc
            p_loc, t_loc = pred.astimezone(tz), lab.utc_time.astimezone(tz)
            out["hour_err_h"] = _circ(_hour(p_loc), _hour(t_loc), 24.0)
            out["doy_err_d"] = _circ(_doy(p_loc), _doy(t_loc), DAYS_PER_YEAR)
            out["year_err"] = abs(p_loc.year - t_loc.year)
            if ans.time_uncertainty_hours is not None:
                gap_h = abs((pred - lab.utc_time).total_seconds()) / 3600
                out["time_in_unc"] = float(gap_h <= ans.time_uncertainty_hours)
    return out


def label_from_row(row) -> Label:
    t = getattr(row, "utc_time", None)
    t = None if t is None or pd.isna(t) else pd.Timestamp(t).tz_convert("UTC").to_pydatetime()
    return Label(float(row.lat), float(row.lon), t, getattr(row, "tz", None),
                 bool(getattr(row, "time_reliable", False)))


def aggregate(scores: pd.DataFrame, strata: pd.Series | None = None) -> dict:
    """Run-level metrics. With `strata` (e.g. country), also a stratum-balanced macro view."""
    def summarize(s: pd.DataFrame) -> dict:
        r = {"n": int(len(s)), "valid_frac": float(s.valid.mean()),
             "median_km": float(s.dist_km.median())}
        for k in ACC_KM:
            r[f"acc@{k}"] = float(s[f"acc@{k}"].mean())
        r["geo_shaped"] = float(s.geo_shaped.mean())
        nll = s.geo_nll.dropna()
        r["geo_nll"] = float(nll.mean()) if len(nll) else None
        cov = s.in_radius.dropna()
        r["in_radius"] = float(cov.mean()) if len(cov) else None
        t = s.dropna(subset=["hour_err_h"])
        r["n_time"] = int(len(t))
        if len(t):
            # Primary time metrics (D-014): date and year are robust to the few-hour clock
            # errors we can't rule out in the labels.
            r["year_exact"] = float((t.year_err == 0).mean())
            r["median_year_err"] = float(t.year_err.median())
            r["median_doy_err_d"] = float(t.doy_err_d.median())
            r["doy_within_7d"] = float((t.doy_err_d <= 7).mean())
            r["doy_within_30d"] = float((t.doy_err_d <= 30).mean())
            # Secondary: label accuracy at the hour level is unverified (see D-013).
            r["secondary_median_hour_err_h"] = float(t.hour_err_h.median())
        return r

    out = {"micro": summarize(scores)}
    if strata is not None:
        per = {k: summarize(g) for k, g in scores.groupby(strata.values)}
        keys = ["median_km", "geo_shaped", *[f"acc@{k}" for k in ACC_KM]]
        out["macro_by_stratum"] = {
            k: float(np.mean([p[k] for p in per.values()])) for k in keys}
        out["macro_by_stratum"]["n_strata"] = len(per)
    out["calibrated_coverage_target"] = CALIBRATED_COVERAGE
    return out
