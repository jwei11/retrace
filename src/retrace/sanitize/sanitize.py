"""M2: turn any input image into pixels-only, uniformly encoded JPEG named by content hash.

Policy (DECISIONS.md D-030..D-033, LEAKAGE.md):
  * decode -> apply EXIF orientation -> convert embedded ICC profile to sRGB -> RGB
  * scale-to-cover + center-crop to a fixed OUT_W x OUT_H (removes resolution/aspect,
    i.e. device fingerprints such as 4:3 phone vs 5:3 OSV vs portrait)
  * re-encode baseline JPEG, fixed quality/subsampling, JFIF APP0 with fixed density,
    no EXIF / XMP / ICC / IPTC / COM / thumbnails
  * file name = sha256(sanitized bytes)[:24].jpg, stored under images/<2 hex>/ -- the
    path carries no split, source, original name or folder.
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

OUT_W, OUT_H = 768, 512
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 2  # 4:2:0
HASH_LEN = 24

# Only these JPEG markers may appear in a sanitized file.
ALLOWED_MARKERS = {
    0xD8,  # SOI
    0xE0,  # APP0 (JFIF)
    0xDB,  # DQT
    0xC0,  # SOF0 baseline
    0xC4,  # DHT
    0xDA,  # SOS
    0xD9,  # EOI
}

_SRGB = ImageCms.createProfile("sRGB")


@dataclass(frozen=True)
class Sanitized:
    id: str
    data: bytes


def _to_srgb(img: Image.Image) -> Image.Image:
    icc = img.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            img = ImageCms.profileToProfile(img.convert("RGB"), src, _SRGB, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError):
            pass  # unreadable profile: fall back to treating pixels as sRGB
    return img.convert("RGB")


def _cover_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    return ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def sanitize_bytes(raw: bytes) -> Sanitized:
    with Image.open(io.BytesIO(raw)) as src:
        src.load()
        img = ImageOps.exif_transpose(src)  # bake orientation into pixels, then drop EXIF
        img = _to_srgb(img)
    # Rebuild from raw pixel data so no `info` dict (exif, icc, comment, dpi...) survives.
    clean = Image.frombytes("RGB", img.size, img.tobytes())
    clean = _cover_crop(clean, OUT_W, OUT_H)
    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING,
               optimize=False, progressive=False, dpi=(72, 72))
    data = buf.getvalue()
    return Sanitized(hashlib.sha256(data).hexdigest()[:HASH_LEN], data)


def relpath_for(image_id: str) -> str:
    return f"images/{image_id[:2]}/{image_id}.jpg"


def write(s: Sanitized, root: Path) -> str:
    rel = relpath_for(s.id)
    p = root / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(s.data)
        tmp.replace(p)
    return rel


def jpeg_markers(data: bytes) -> list[tuple[int, bytes]]:
    """Parse JPEG segments up to SOS: [(marker, payload)]. Independent of PIL, for audits."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    out, i = [(0xD8, b"")], 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"bad marker at {i}")
        while data[i] == 0xFF:
            i += 1
        m = data[i]
        i += 1
        if m == 0xD9:
            out.append((m, b""))
            break
        (seg_len,) = struct.unpack(">H", data[i:i + 2])
        out.append((m, data[i + 2:i + seg_len]))
        i += seg_len
        if m == 0xDA:  # entropy-coded data follows; skip to EOI
            if not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
                raise ValueError("missing EOI")
            out.append((0xD9, b""))
            break
    return out
