# Leakage channels

Every way the agent could learn the answer from something other than the scene itself.
Status: **closed** (fixed and tested), **mitigated** (reduced, not eliminated), **open** (known,
not handled yet), **n/a yet** (the feature that creates it isn't built).
Update this file whenever a new tool, memory source or data source is added.

## Image file

| # | Channel | Status | How |
|---|---|---|---|
| L-01 | EXIF (date, GPS, camera, serial, orientation) | closed | Re-encoded from raw pixels; test asserts no APP1 / empty `getexif()`. |
| L-02 | XMP | closed | No APP1 segments allowed (marker whitelist test). |
| L-03 | ICC profile (can identify device/software) | closed | Converted to sRGB, then dropped; no APP2. |
| L-04 | IPTC / Photoshop / comments | closed | No APP13 / COM segments allowed. |
| L-05 | Embedded thumbnail | closed | Lives inside EXIF; never written. |
| L-06 | Resolution / aspect ratio (device -> year) | closed | Fixed 768x512 (D-030). |
| L-07 | JPEG quantization tables / encoder fingerprint | closed | Identical DQT for every output (tested). |
| L-08 | Progressive vs baseline, DPI field | closed | Always baseline, always 72 dpi (tested). |
| L-09 | File name / folder name | closed | Content hash under `images/<2 hex>/`; regex test on real corpus. |
| L-10 | Pixel-identical duplicates across splits | closed | Same hash -> dropped at manifest build. |

## Dataset structure

| # | Channel | Status | How |
|---|---|---|---|
| L-11 | Same place in train and val (kNN returns the answer) | closed | Group split + 1 km separation (D-020). |
| L-12 | Same uploader in train and eval (same car/camera look) | open | Mirrors the official benchmark (98% overlap). Planned: report seen/unseen-uploader slices. |
| L-13 | Dataset-source fingerprint (upscaled Flickr vs OSV look) | mitigated | Sources are scored separately, never mixed in one eval. |
| L-14 | Double-JPEG / resampling traces of the source pipeline | mitigated | Resize + re-encode largely destroys block alignment; not tested directly. |

## Pixels that genuinely contain the answer

| # | Channel | Status | Notes |
|---|---|---|---|
| L-15 | Burned-in timestamps (dashcam overlays, camera date stamps) | open | Real visual evidence, but makes time trivial. Need to measure how often it occurs (OCR scan) and decide whether to report a "no overlay" slice. |
| L-16 | Watermarks / uploader names in pixels | open | Same treatment as L-15. |

## Harness (M5, not built yet)

| # | Channel | Status | Notes |
|---|---|---|---|
| L-20 | Agent reads manifests or label-side folders | n/a yet | Sandbox must expose pixels only via tools. |
| L-21 | `similar(k)` returns the query's own group | n/a yet | Must exclude the query's group; index built from train only. |
| L-22 | Tool outputs echo ids or paths | n/a yet | Tool outputs must never contain file paths or source ids. |
| L-23 | Private photos in any index, memory or prompt | n/a yet | Hard rule 1; add a test once the private pipeline exists. |

## Label side (not agent-visible, listed so it stays that way)

`~/retrace-data/osv5m/` (metadata, raw originals, shard indexes), `manifests/`, `audits/`,
`flickr_bench/`, `data/audits/` in the repo. The harness may read only `~/retrace-data/images/`.
