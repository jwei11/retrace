"""M1/M2 for the Flickr benchmarks: sanitize images and write a scorable manifest.

  uv run python scripts/prepare_flickr_bench.py im2gps3k

Im2GPS3k: 3,000 Flickr photos with GPS (raw/im2gps3k/images, im2gps3k_places365.csv). No capture
time in the labels, so the benchmark is location-only (time_reliable = False everywhere).
Writes $RETRACE_DATA/manifests/im2gps3k_test_sanitized.parquet; sanitized images go to the same
hashed images/ dir as OSV-5M (the file name, which holds the Flickr id and author, never reaches
the agent). Country = Natural Earth polygon containing the point, else the nearest one.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree
from tqdm import tqdm

from retrace.ingest.osv5m import MANIFEST_COLS
from retrace.sanitize.sanitize import sanitize_bytes, write

DATA = Path(os.environ["RETRACE_DATA"])


def countries(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    gj = json.loads((DATA / "geo" / "ne_10m_admin_0_countries.geojson").read_text())
    geoms = [shapely.geometry.shape(f["geometry"]) for f in gj["features"]]
    codes = np.array([f["properties"]["ISO_A2_EH"] for f in gj["features"]])
    tree = STRtree(geoms)
    pts = shapely.points(lon, lat)
    near = tree.query_nearest(pts, all_matches=False)   # [point idx, geom idx]; distance 0 if inside
    out = np.full(len(pts), "", dtype=object)
    out[near[0]] = codes[near[1]]
    return out


def im2gps3k() -> None:
    raw = DATA / "flickr_bench" / "raw" / "im2gps3k"
    lab = pd.read_csv(raw / "im2gps3k_places365.csv")
    ids, paths, bad = [], [], 0
    for f in tqdm(lab.IMG_ID, desc="sanitize im2gps3k"):
        p = raw / "images" / f
        try:
            s = sanitize_bytes(p.read_bytes())
            ids.append(s.id), paths.append(write(s, DATA))
        except Exception:  # noqa: BLE001 - unreadable file: dropped and counted
            ids.append(None), paths.append(None)
            bad += 1
    m = pd.DataFrame({
        "id": ids, "sanitized_path": paths, "lat": lab.LAT.astype(float), "lon": lab.LON.astype(float),
        "utc_time": pd.Series(pd.NaT, index=lab.index, dtype="datetime64[us, UTC]"),
        "local_time": pd.Series(pd.NaT, index=lab.index, dtype="datetime64[us]"), "tz": "",
        "group_id": "flickr-" + lab.AUTHOR.astype(str), "source": "im2gps3k", "split": "test",
        "source_id": lab.IMG_ID, "region": "", "city": "", "creator_id": "flickr-" + lab.AUTHOR.astype(str),
        "creator_time_ok": False, "time_reliable": False, "sun_elev_label": np.nan,
    }).dropna(subset=["id"])
    m["country"] = countries(m.lat.to_numpy(), m.lon.to_numpy())
    dup = m.id.duplicated()
    m = m[~dup]
    m = m[MANIFEST_COLS].assign(time_label_status="no_time", junk=False, junk_reason="")
    out = DATA / "manifests" / "im2gps3k_test_sanitized.parquet"
    m.to_parquet(out, index=False)
    print(f"im2gps3k: {len(m)} photos ({bad} unreadable, {int(dup.sum())} pixel duplicates dropped); "
          f"{m.creator_id.nunique()} authors; top countries {m.country.value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    {"im2gps3k": im2gps3k}[sys.argv[1]]()
