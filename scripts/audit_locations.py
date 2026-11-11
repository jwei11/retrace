"""Audit #3: are OSV-5M locations sane?

  * invalid / placeholder coordinates: out of range, exactly (0, 0), or one coordinate shared
    by many different sequences (a default position, not a real capture);
  * at sea: not inside any Natural Earth country polygon and > SEA_KM from the nearest one
    (coastline at 1:10M scale is only accurate to a few hundred metres, hence the tolerance);
  * country mismatch: the polygon containing the point disagrees with OSV-5M's `country`.
    Near a border this is just border ambiguity; far from the labeled country it suggests a
    bad coordinate or a bad country code. Note: lat/lon is the scored label; `country` is
    used only for stratified reporting.

Runs on our subset (all rows) plus a 300k random sample of all metadata for base rates.
Writes results/m1_location_audit.json and $RETRACE_DATA/audits/location_flags.parquet.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from retrace.ingest.osv5m import load_slim

DATA = Path(os.environ["RETRACE_DATA"])
ROOT = Path(__file__).resolve().parents[1]
SEA_KM = 2.0
FAR_MISMATCH_KM = 25.0
SHARED_COORD_MIN_SEQS = 20
KM_PER_DEG = 111.2


def load_countries():
    gj = json.loads((DATA / "geo" / "ne_10m_admin_0_countries.geojson").read_text())
    geoms = [shapely.geometry.shape(f["geometry"]) for f in gj["features"]]
    codes = np.array([f["properties"]["ISO_A2_EH"] for f in gj["features"]])
    return geoms, codes


def approx_km(deg_dist: np.ndarray, lat: np.ndarray) -> np.ndarray:
    # Degrees -> km, conservative (uses the smaller of lat/lon scale), fine for flagging.
    return deg_dist * KM_PER_DEG * np.maximum(np.cos(np.radians(lat)), 0.2)


def check(df: pd.DataFrame, geoms, codes, tree: STRtree, shared: set) -> pd.DataFrame:
    out = df[["id", "lat", "lon", "country", "sequence"]].copy()
    out["flag_invalid"] = (~out.lat.between(-90, 90) | ~out.lon.between(-180, 180)
                           | ((out.lat == 0) & (out.lon == 0)))
    out["flag_shared_coord"] = list(zip(out.lat.round(6), out.lon.round(6)))
    out["flag_shared_coord"] = out.flag_shared_coord.isin(shared)

    pts = shapely.points(out.lon.to_numpy(), out.lat.to_numpy())
    pt_idx, geom_idx = tree.query(pts, predicate="within")
    inside = pd.Series(codes[geom_idx], index=pt_idx).groupby(level=0).first()
    out["ne_country"] = pd.Series(inside.reindex(range(len(out))).to_numpy(), index=out.index)

    near_idx, near_dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    dist_land = np.full(len(out), np.nan)
    dist_land[near_idx] = near_dist
    out["km_to_land"] = np.where(out.ne_country.notna(), 0.0, approx_km(dist_land, out.lat.to_numpy()))
    out["flag_at_sea"] = out.km_to_land > SEA_KM

    # Distance to the polygon(s) of the LABELED country, for mismatches only.
    mism = out.ne_country.notna() & (out.ne_country != out.country)
    out["flag_country_mismatch"] = mism
    km_to_labeled = np.full(len(out), np.nan)
    by_code = {c: shapely.union_all([g for g, k in zip(geoms, codes) if k == c]) for c in set(codes)}
    for i in np.flatnonzero(mism.to_numpy()):
        g = by_code.get(out.country.iat[i])
        if g is not None:
            km_to_labeled[i] = approx_km(np.array([shapely.distance(pts[i], g)]), out.lat.to_numpy()[i:i + 1])[0]
    out["km_to_labeled_country"] = km_to_labeled
    out["flag_far_mismatch"] = mism & ~(out.km_to_labeled_country <= FAR_MISMATCH_KM)
    return out


def summarize(c: pd.DataFrame) -> dict:
    n = len(c)
    r = {"n": n}
    for f in ["flag_invalid", "flag_shared_coord", "flag_at_sea", "flag_country_mismatch", "flag_far_mismatch"]:
        r[f] = {"count": int(c[f].sum()), "frac": round(float(c[f].mean()), 5)}
    r["at_sea_km_quantiles"] = (c.loc[c.flag_at_sea, "km_to_land"].quantile([.5, .9, 1]).round(1).to_dict()
                                if c.flag_at_sea.any() else {})
    fm = c[c.flag_far_mismatch]
    top = fm.groupby(["country", "ne_country"]).size().sort_values(ascending=False).head(10)
    r["far_mismatch_top_pairs"] = {f"{a}->{b}": int(v) for (a, b), v in top.items()}
    return r


def main() -> None:
    meta = DATA / "osv5m" / "meta"
    full = pd.concat([load_slim(meta / "train_slim.parquet"), load_slim(meta / "test_slim.parquet")],
                     ignore_index=True)
    coords = pd.Series(list(zip(full.lat.round(6), full.lon.round(6))))
    seqs_per_coord = full.groupby(coords.values).sequence.nunique()
    shared = set(seqs_per_coord[seqs_per_coord >= SHARED_COORD_MIN_SEQS].index)

    geoms, codes = load_countries()
    tree = STRtree(geoms)
    out = {"shared_coordinates": {"distinct": len(shared),
                                  "photos": int(coords.isin(shared).sum())}}

    sample = full.sample(300_000, random_state=0).reset_index(drop=True)
    out["all_metadata_sample"] = summarize(check(sample, geoms, codes, tree, shared))

    flags = []
    out["subset"] = {}
    for split in ("train", "val", "test"):
        m = pd.read_parquet(DATA / "manifests" / f"osv5m_{split}_raw.parquet")
        m = m.rename(columns={"id": "_id"}).rename(columns={"source_id": "id"})
        m["sequence"] = m.group_id
        c = check(m.reset_index(drop=True), geoms, codes, tree, shared)
        out["subset"][split] = summarize(c)
        flags.append(c.assign(split=split))
    pd.concat(flags).to_parquet(DATA / "audits" / "location_flags.parquet", index=False)
    (ROOT / "results" / "m1_location_audit.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
