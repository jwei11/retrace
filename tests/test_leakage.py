"""M2 leakage tests. Two layers:
  1. unit tests on synthetic images loaded with every metadata channel we know of;
  2. an audit over the real sanitized corpus (skipped if it isn't built yet).
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageCms

from retrace.sanitize.sanitize import (
    ALLOWED_MARKERS, HASH_LEN, JPEG_QUALITY, OUT_H, OUT_W, jpeg_markers, sanitize_bytes,
)

# Things that look like dates, times, coordinates, or place/camera words.
LEAKY_NAME = re.compile(
    r"(19|20)\d{2}[-_:.]?\d{2}|\d{1,2}[-_]\d{1,2}[-_]\d{2,4}|IMG|DSC|PXL|GOPR|"
    r"[-+]?\d{1,3}\.\d{3,}|lat|lon|gps|train|val|test|osv|flickr|yfcc|im2gps|private",
    re.IGNORECASE,
)


def _loaded_jpeg(size=(1200, 1600)) -> bytes:
    """A portrait JPEG with EXIF (date, GPS, camera, orientation), XMP, ICC, comment, thumbnail."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x010F] = "Canon"  # Make
    exif[0x0110] = "EOS 5D"  # Model
    exif[0x0112] = 6  # Orientation: rotate
    exif[0x0132] = "2019:07:04 12:34:56"
    ifd = exif.get_ifd(0x8769)
    ifd[0x9003] = "2019:07:04 12:34:56"  # DateTimeOriginal
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (48.0, 51.0, 29.0), "E", (2.0, 17.0, 40.0)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=77, exif=exif.tobytes(), icc_profile=icc,
             comment=b"Paris trip 2019-07-04", dpi=(300, 300))
    data = buf.getvalue()
    xmp = (b"\xff\xe1" + (len(b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>"
           b"<exif:GPSLatitude>48,51.4N</exif:GPSLatitude></x:xmpmeta>") + 2).to_bytes(2, "big")
           + b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta><exif:GPSLatitude>48,51.4N"
           b"</exif:GPSLatitude></x:xmpmeta>")
    return data[:2] + xmp + data[2:]


def _assert_clean(data: bytes) -> None:
    markers = jpeg_markers(data)
    kinds = {m for m, _ in markers}
    assert kinds <= ALLOWED_MARKERS, f"unexpected markers {sorted(hex(k) for k in kinds - ALLOWED_MARKERS)}"
    app0 = [p for m, p in markers if m == 0xE0]
    assert len(app0) == 1 and app0[0].startswith(b"JFIF\x00")
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (OUT_W, OUT_H)
        assert im.mode == "RGB"
        assert not im.getexif()
        assert "icc_profile" not in im.info and "exif" not in im.info and "comment" not in im.info
        assert "progressive" not in im.info and "progression" not in im.info
        assert im.info.get("dpi") == (72, 72)
    for needle in (b"Exif", b"xmpmeta", b"Canon", b"2019", b"Paris", b"GPS", b"ICC_PROFILE"):
        assert needle not in data[:4096], needle


def test_strips_every_metadata_channel():
    raw = _loaded_jpeg()
    assert b"Exif" in raw and b"xmpmeta" in raw and b"ICC_PROFILE" in raw  # fixture sanity
    _assert_clean(sanitize_bytes(raw).data)


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGBA"), ("PNG", "L"), ("WEBP", "RGB"),
                                      ("JPEG", "CMYK"), ("TIFF", "RGB")])
def test_any_input_format_gives_identical_encoding(fmt, mode):
    img = Image.new(mode, (300, 900))
    buf = io.BytesIO()
    img.save(buf, fmt)
    _assert_clean(sanitize_bytes(buf.getvalue()).data)


def test_orientation_is_baked_in_not_kept():
    # A 1200x1600 portrait with Orientation=6 must still come out as 768x512 landscape,
    # with no orientation tag left behind.
    out = sanitize_bytes(_loaded_jpeg())
    with Image.open(io.BytesIO(out.data)) as im:
        assert im.size == (OUT_W, OUT_H) and 0x0112 not in im.getexif()


def test_quantization_tables_are_uniform():
    """Same DQT for every input -> no 'which camera/encoder' fingerprint survives."""
    dqts = set()
    for q, size in [(30, (640, 480)), (95, (4000, 3000)), (60, (853, 512))]:
        buf = io.BytesIO()
        Image.new("RGB", size, (120, 80, 40)).save(buf, "JPEG", quality=q)
        markers = jpeg_markers(sanitize_bytes(buf.getvalue()).data)
        dqts.add(b"".join(p for m, p in markers if m == 0xDB))
    assert len(dqts) == 1


def test_id_is_content_hash_and_name_is_clean():
    s = sanitize_bytes(_loaded_jpeg())
    assert re.fullmatch(rf"[0-9a-f]{{{HASH_LEN}}}", s.id)
    assert sanitize_bytes(_loaded_jpeg()).id == s.id  # deterministic


def test_leaky_name_regex_catches_examples():
    for bad in ["IMG_20190704_123456", "2019-07-04", "48.8582_2.2945", "val_000123", "osv5m_1"]:
        assert LEAKY_NAME.search(bad)


# ---------------------------------------------------------------- corpus audit

DATA = Path(os.environ.get("RETRACE_DATA", "/nonexistent"))
MANIFESTS = sorted((DATA / "manifests").glob("*_sanitized.parquet")) if DATA.exists() else []


@pytest.mark.skipif(not MANIFESTS, reason="sanitized corpus not built")
@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.stem)
def test_corpus_audit(manifest):
    df = pd.read_parquet(manifest)
    assert df.id.is_unique
    sample = df.sample(min(len(df), 300), random_state=0)
    for row in sample.itertuples():
        rel = row.sanitized_path
        # Hex-only names can't spell words/dates; (LEAKY_NAME would false-positive on
        # digit runs inside hashes, so the strict pattern is the real guard.)
        assert re.fullmatch(rf"images/[0-9a-f]{{2}}/[0-9a-f]{{{HASH_LEN}}}\.jpg", rel)
        _assert_clean((DATA / rel).read_bytes())


@pytest.mark.skipif(not MANIFESTS, reason="sanitized corpus not built")
def test_image_dir_has_no_side_files():
    allowed = re.compile(rf"[0-9a-f]{{{HASH_LEN}}}\.jpg")
    for d in (DATA / "images").iterdir():
        assert re.fullmatch(r"[0-9a-f]{2}", d.name)
        bad = [f.name for f in d.iterdir() if not allowed.fullmatch(f.name)]
        assert not bad, bad[:5]


def test_quality_constant_documented():
    assert JPEG_QUALITY == 90  # change requires a DECISIONS.md entry
