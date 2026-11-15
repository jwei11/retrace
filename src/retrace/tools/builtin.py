"""The agent's tools (M5 v1): view, similar, sun, gazetteer. `answer` is handled by the env.

Each factory returns a Tool with a real function and a placebo. Outputs are short plain text
(plus one image for view) to keep the 6k-token context budget.
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retrace.tools.registry import Ctx, Obs, Tool
from retrace.tools.sun import solar_position

DATA = Path(os.environ.get("RETRACE_DATA", "."))
VIEW_BOX = (768, 512)   # zoomed crops are scaled to fit this box (~same token cost as the photo)


# ---------------------------------------------------------------- view

def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def view_tool() -> Tool:
    def crop(ctx: Ctx, x0: float, y0: float, x1: float, y1: float) -> Obs:
        img = Image.open(DATA / ctx.photo.image_path).convert("RGB")
        w, h = img.size
        x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
        y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
        x1, y1 = max(x1, x0 + 0.05), max(y1, y0 + 0.05)   # at least 5% of each side
        box = (int(x0 * w), int(y0 * h), min(int(x1 * w), w), min(int(y1 * h), h))
        c = img.crop(box)
        s = min(VIEW_BOX[0] / c.width, VIEW_BOX[1] / c.height)
        c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)
        return Obs(f"zoomed view of x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f}", _jpeg(c))

    def placebo(ctx: Ctx, x0: float, y0: float, x1: float, y1: float) -> Obs:
        img = Image.open(DATA / ctx.photo.image_path).convert("RGB")   # whole photo again: no new detail
        return Obs(f"zoomed view of x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f}", _jpeg(img))

    frac = {"type": "number", "minimum": 0, "maximum": 1}
    return Tool("view", "zoom into a region of the photo to read small details (signs, plates, "
                "markings). x0,y0 = top-left, x1,y1 = bottom-right, as fractions 0-1 of width/height.",
                {"x0": frac, "y0": frac, "x1": frac, "y1": frac}, crop, placebo)


# ---------------------------------------------------------------- similar

class _Index:
    """SigLIP2 embeddings of train (non-junk) + query embeddings for val/test (baseline_knn.py)."""

    def __init__(self, emb_dir: Path):
        def load(split, dataset="osv5m"):
            n = split if dataset == "osv5m" else f"{dataset}_{split}"
            ids = pd.read_parquet(emb_dir / f"{n}_ids.parquet").id
            m = pd.read_parquet(DATA / "manifests" / f"{dataset}_{split}_sanitized.parquet").set_index("id").loc[ids]
            return m.reset_index(), np.load(emb_dir / f"{n}.npy").astype(np.float32)

        tr, e = load("train")
        keep = ~tr.junk.to_numpy()
        self.train, self.emb = tr[keep].reset_index(drop=True), e[keep]
        self.groups = self.train.group_id.to_numpy()
        self.query: dict[str, np.ndarray] = {}
        for s, ds in (("val", "osv5m"), ("test", "osv5m"), ("test", "im2gps3k")):
            if (emb_dir / f"{s if ds == 'osv5m' else f'{ds}_{s}'}.npy").exists():
                q, qe = load(s, ds)
                self.query.update(zip(q.id, qe))

    def line(self, i: int, sim: float) -> str:
        r = self.train.iloc[i]
        place = ", ".join(str(x) for x in (r.country, r.region, r.city) if isinstance(x, str) and x)
        t = pd.Timestamp(r.utc_time).strftime("%Y-%m-%d %H:%M UTC") if r.time_reliable and not pd.isna(r.utc_time) \
            else "time unknown"
        return f"sim {sim:.2f} | {place} | {r.lat:.3f}, {r.lon:.3f} | {t}"


def similar_tool(emb_dir: Path | None = None) -> Tool:
    idx = _Index(emb_dir or DATA / "embeddings" / "siglip2-so400m")

    def ranked(ctx: Ctx, k: int):
        s = idx.emb @ idx.query[ctx.photo.id]
        s[idx.groups == ctx.photo.group_id] = -np.inf   # never the photo's own capture sequence
        k = int(min(max(k, 1), 10))
        nn = np.argpartition(-s, k)[:k]
        return nn[np.argsort(-s[nn])], s

    def fmt(rows, sims) -> Obs:
        return Obs("most similar reference photos (lookalikes; often nearby, sometimes misleading):\n" +
                   "\n".join(f"{j + 1}. " + idx.line(i, v) for j, (i, v) in enumerate(zip(rows, sims))))

    def real(ctx: Ctx, k: int) -> Obs:
        nn, s = ranked(ctx, k)
        return fmt(nn, s[nn])

    def placebo(ctx: Ctx, k: int) -> Obs:   # real similarity values, random photos' labels
        nn, s = ranked(ctx, k)
        return fmt(ctx.rng.choice(len(idx.train), len(nn), replace=False), s[nn])

    return Tool("similar", "find the k (1-10) most visually similar photos in a labeled reference "
                "collection, with their place, coordinates and capture time.",
                {"k": {"type": "integer", "minimum": 1, "maximum": 10}}, real, placebo)


# ---------------------------------------------------------------- sun

def _sun_text(lat: float, lon: float, t: datetime) -> str:
    el, az = solar_position([lat], [lon], [t.timestamp()])
    el, az = float(el[0]), float(az[0])
    mins = (t.hour * 60 + t.minute + 4 * lon) % 1440   # mean solar time (ignores equation of time)
    state = "up" if el > 0 else ("twilight" if el > -6 else "below the horizon (night)")
    return (f"at {lat:.3f}, {lon:.3f} on {t:%Y-%m-%d %H:%M} UTC: sun elevation {el:.1f} deg, azimuth "
            f"{az:.0f} deg (0=N, 90=E); shadows point toward {(az + 180) % 360:.0f} deg; "
            f"local solar time ~{int(mins // 60):02d}:{int(mins % 60):02d}; sun is {state}")


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def sun_tool() -> Tool:
    def real(ctx: Ctx, lat: float, lon: float, utc_datetime: str) -> Obs:
        return Obs(_sun_text(lat, lon, _parse_utc(utc_datetime)))

    def placebo(ctx: Ctx, lat: float, lon: float, utc_datetime: str) -> Obs:   # a random place/time
        t = _parse_utc(utc_datetime)
        rt = datetime(t.year, 1, 1, tzinfo=timezone.utc).timestamp() + ctx.rng.uniform(0, 365 * 86400)
        text = _sun_text(float(ctx.rng.uniform(-55, 65)), float(ctx.rng.uniform(-180, 180)),
                         datetime.fromtimestamp(rt, timezone.utc))
        return Obs(f"at {lat:.3f}, {lon:.3f} on {t:%Y-%m-%d %H:%M} UTC:" + text.split("UTC:", 1)[1])

    return Tool("sun", "sun elevation and direction at a place and UTC time, to check a time guess "
                "against shadows and light.",
                {"lat": {"type": "number", "minimum": -90, "maximum": 90},
                 "lon": {"type": "number", "minimum": -180, "maximum": 180},
                 "utc_datetime": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$"}},
                real, placebo, per_photo=False)


# ---------------------------------------------------------------- gazetteer

class _Gazetteer:
    """GeoNames cities500 (places with population >= 500), matched on any of their names."""

    def __init__(self, path: Path):
        g = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2, 3, 4, 5, 8, 14], quoting=3,
                        names=["name", "ascii", "alt", "lat", "lon", "cc", "pop"], keep_default_na=False,
                        dtype={"name": str, "ascii": str, "alt": str, "cc": str})
        self.g = g
        self.by_name: dict[str, list[int]] = {}
        for i, (a, b, alt) in enumerate(zip(g.name, g.ascii, g.alt)):
            for n in {a.casefold(), b.casefold(), *(x.casefold() for x in alt.split(",") if x)}:
                self.by_name.setdefault(n, []).append(i)

    def find(self, name: str, cc: str) -> list[int]:
        rows = self.by_name.get(name.strip().casefold(), [])
        if cc:
            rows = [i for i in rows if self.g.cc.iat[i] == cc]
        return sorted(rows, key=lambda i: -self.g["pop"].iat[i])[:5]

    def fmt(self, rows: list[int], name: str) -> Obs:
        if not rows:
            return Obs(f"no place named '{name}' found")
        r = self.g.iloc[rows]
        return Obs("\n".join(f"{x.name} ({x.cc}, pop {x.pop:,}): {x.lat:.3f}, {x.lon:.3f}" for x in r.itertuples()))


def gazetteer_tool(path: Path | None = None) -> Tool:
    gz = _Gazetteer(path or DATA / "gazetteer" / "cities500.txt")

    def real(ctx: Ctx, name: str, country_code: str) -> Obs:
        return gz.fmt(gz.find(name, country_code), name)

    def placebo(ctx: Ctx, name: str, country_code: str) -> Obs:   # same number of random places
        n = len(gz.find(name, country_code))
        return gz.fmt(list(ctx.rng.choice(len(gz.g), n, replace=False)) if n else [], name)

    return Tool("gazetteer", "look up a town or city name (e.g. read on a sign) and get its coordinates. "
                "country_code: ISO 2-letter code to filter, or \"\" for any.",
                {"name": {"type": "string", "minLength": 2, "maxLength": 60},
                 "country_code": {"type": "string", "pattern": r"^([A-Z]{2})?$"}},
                real, placebo, per_photo=False)


FACTORIES = {"view": view_tool, "similar": similar_tool, "sun": sun_tool, "gazetteer": gazetteer_tool}
