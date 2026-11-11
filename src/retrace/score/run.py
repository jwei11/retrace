"""Run-level scoring: answers.jsonl + label manifest -> per-photo scores, summary, comparisons.

A run lives in $RETRACE_DATA/runs/<run_id>/ (outside git: private-set runs must never be committed):
  run.json       {"run_id", "dataset", "split", "model", "config": {...}, "notes"}
  answers.jsonl  one line per photo:
                 {"id", "lat", "lon", "radius_km", "utc_datetime", "time_uncertainty_hours",
                  "tokens_in", "tokens_out", "latency_s", "error"}
                 lat/lon null (or the line missing) = no answer.
Rules (agreed with the user, 2026-11-11):
  - every non-junk photo of the split is scored; a photo with no answer gets the worst score
    (antipode), so skipping hard photos can't help; answer rate is reported separately.
  - junk photos (D-017) are excluded from scoring entirely.
  - views: micro, country-macro (D-043), seen vs new uploader (uploader also in train, L-12).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from retrace.score.score import Answer, aggregate, label_from_row, score_answer

ANSWER_COLS = ["id", "lat", "lon", "radius_km", "utc_datetime", "time_uncertainty_hours",
               "tokens_in", "tokens_out", "latency_s", "error"]
# Metrics compared by the paired bootstrap: (name, per-photo column, how, time-only)
COMPARE = [
    ("geo_shaped", "geo_shaped", "mean", False),
    ("acc@25", "acc@25", "mean", False),
    ("acc@200", "acc@200", "mean", False),
    ("acc@750", "acc@750", "mean", False),
    ("acc@2500", "acc@2500", "mean", False),
    ("median_km", "dist_km", "median", False),
    ("year_exact", "year_exact", "mean", True),
    ("doy_within_30d", "doy_within_30d", "mean", True),
]


def load_answers(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows).reindex(columns=ANSWER_COLS)
    if df.id.isna().any():
        raise ValueError("answer line without an id")
    dup = df.id[df.id.duplicated()]
    if len(dup):
        raise ValueError(f"duplicate answer ids, e.g. {dup.iloc[0]}")
    return df


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _answer(r) -> Answer | None:
    lat, lon = _num(r.lat), _num(r.lon)
    if lat is None or lon is None:
        return None
    rad = _num(r.radius_km)
    t = r.utc_datetime if isinstance(r.utc_datetime, str) and r.utc_datetime else None
    return Answer(lat, lon, rad if rad is not None else float("nan"), t, _num(r.time_uncertainty_hours))


def score_run(answers: pd.DataFrame, labels: pd.DataFrame, train_creators: set[str]) -> pd.DataFrame:
    """Per-photo scores for every non-junk label row. `labels` is a sanitized manifest.
    Answers for junk photos are ignored; answers for ids outside the split are an error."""
    extra = set(answers.id) - set(labels.id)
    if extra:
        raise ValueError(f"{len(extra)} answer ids are not in this split, e.g. {sorted(extra)[0]}")
    if "junk" in labels:
        labels = labels[~labels.junk.astype(bool)]
    labels = labels.reset_index(drop=True)
    ans = answers.set_index("id").reindex(labels.id).reset_index()  # aligned to labels; missing -> NaN
    rows = []
    for lab_row, ans_row in zip(labels.itertuples(), ans.itertuples()):
        s = score_answer(_answer(ans_row), label_from_row(lab_row))
        s["id"] = lab_row.id
        rows.append(s)
    sc = pd.DataFrame(rows)
    sc["answered"] = pd.to_numeric(ans.lat, errors="coerce").notna().to_numpy()
    sc["country"] = labels.country.to_numpy()
    sc["uploader_seen"] = labels.creator_id.isin(train_creators).to_numpy()
    for c in ("tokens_in", "tokens_out", "latency_s"):
        sc[c] = pd.to_numeric(ans[c], errors="coerce").to_numpy()
    sc["error"] = ans["error"].to_numpy()
    sc["year_exact"] = np.where(sc.year_err.notna(), (sc.year_err == 0).astype(float), np.nan)
    sc["doy_within_30d"] = np.where(sc.doy_err_d.notna(), (sc.doy_err_d <= 30).astype(float), np.nan)
    return sc


def summarize(sc: pd.DataFrame) -> dict:
    out = aggregate(sc, sc.country)
    out["answer_rate"] = float(sc.answered.mean())
    out["n_errors"] = int(sc.error.notna().sum())
    out["uploader"] = {
        "seen": aggregate(sc[sc.uploader_seen])["micro"] if sc.uploader_seen.any() else None,
        "new": aggregate(sc[~sc.uploader_seen])["micro"] if (~sc.uploader_seen).any() else None,
        "seen_frac": float(sc.uploader_seen.mean()),
    }
    tok = sc.tokens_in.fillna(0) + sc.tokens_out.fillna(0)
    out["cost"] = {
        "mean_tokens_in": float(sc.tokens_in.mean()) if sc.tokens_in.notna().any() else None,
        "mean_tokens_out": float(sc.tokens_out.mean()) if sc.tokens_out.notna().any() else None,
        "total_tokens": int(tok.sum()),
        "mean_latency_s": float(sc.latency_s.mean()) if sc.latency_s.notna().any() else None,
        "p95_latency_s": float(sc.latency_s.quantile(0.95)) if sc.latency_s.notna().any() else None,
    }
    return out


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference a - b on the same photos, with a 95% interval from resampling photos.
    'significant' = the interval excludes 0."""
    if set(a.id) != set(b.id):
        raise ValueError("runs must cover the same photos (same split and junk filter)")
    b = b.set_index("id").loc[a.id].reset_index()
    rng = np.random.default_rng(seed)
    out = {}
    for name, col, how, time_only in COMPARE:
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        keep = ~(np.isnan(x) | np.isnan(y)) if time_only else np.ones(len(x), bool)
        x, y = x[keep], y[keep]
        if len(x) == 0:
            continue
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        if how == "mean":
            diff = x.mean() - y.mean()
            boots = x[idx].mean(1) - y[idx].mean(1)
        else:
            diff = np.median(x) - np.median(y)
            boots = np.median(x[idx], 1) - np.median(y[idx], 1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = {"a": float(x.mean() if how == "mean" else np.median(x)),
                     "b": float(y.mean() if how == "mean" else np.median(y)),
                     "diff": float(diff), "ci95": [float(lo), float(hi)],
                     "significant": bool(lo > 0 or hi < 0), "n": int(len(x))}
    return out
