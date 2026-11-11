# Decisions

Each entry: what, why, alternatives considered. Evidence files are referenced where they exist.
"Status: open" means the user has not decided yet.

## Environment

**D-001 Storage layout.** Code in `F:\Github\retrace` (git). Python venv, HF cache and all data in
WSL's ext4 disk (`~/retrace-data`, `~/.venvs/retrace`), whose vhdx was moved to `F:\WSL\Ubuntu`.
*Why:* C: had 12 GB free; reading tens of thousands of small files through `/mnt/f` (9P) is slow.
*Alternatives:* everything on `/mnt/f` (simple, slow); everything inside WSL (code invisible to
Windows tools).

**D-002 PyTorch build.** torch 2.14.1 from the CUDA 12.6 index, versions pinned by `uv.lock`.
*Why:* verified `torch.cuda.get_arch_list()` includes `sm_75`. *Alternatives:* newer CUDA builds,
which may drop older architectures.

**D-003 fp16 everywhere, never bf16.** torch reports `is_bf16_supported() == True` on Turing, but
it is emulated (no bf16 tensor cores), so it would be slow. SDPA attention, no FlashAttention-2.

**D-004 Drop Gemma 4's audio tower at load.** It has no SDPA support in transformers and we never
use audio; dropping it avoids the error and saves VRAM. *Alternative:* per-submodule eager attention.

**D-005 Local inference runtime.** Status: open. Plain transformers runs at 7-15 tok/s, so a
~1,200-token episode takes ~2 min. Candidates: vLLM, llama.cpp (prebuilt Windows CUDA binaries,
no compiler needed). Gemma 4 E2B in nf4 uses 6.3 GB because its per-layer embeddings are not
quantized; moving them to CPU is a known option.

## Data

**D-010 Fetch OSV-5M images by HTTP range reads.** Read each remote zip's central directory once,
record every member's offset, then fetch only sampled members. ~1 GB transferred instead of
246 GB. Concurrency 12 with exponential backoff honoring `Retry-After`, because Hugging Face
rate-limits anonymous requests (HTTP 429 seen at 48 workers). *Alternative:* download whole shards.

**D-011 Flickr benchmarks from a Hugging Face mirror** (`kinghorton42/geo-benchmarks`) rather than
the official Mediafire links in `lugiavn/revisiting-im2gps`, which can't be fetched reliably by
script. Image counts match the official release (3,000 / 4,536). Image contents not yet verified
against the official zips.

**D-012 Time labels: per-uploader daylight rule.** `captured_at` is epoch ms and assumed UTC, but
~12% of photos would be in civil night under that reading. An uploader's labels are reliable if
they have >= 20 photos and >= 90% fall at sun elevation >= -6 deg. Keeps 72% of photos.
*Evidence:* `results/m1_osv5m_subset.json`, `data/audits/night_creators.html` (night uploaders are
daytime photos with wrong clocks; ~64% are local time stored as UTC). Threshold sensitivity:
80% keeps 81%, 95% keeps 64%. *Alternatives:* repair local-as-UTC uploaders by their timezone
offset (rejected: ~1/3 of bad uploaders aren't explained by that, so repairs would inject wrong
labels); per-photo test only; no filtering.

**D-013 Time labels: per-photo night rule.** Additionally drop the time label of any photo whose
labeled sun elevation is < -6 deg. *Evidence:* `results/m1_time_label_brightness.json`: sky
brightness of night-labeled photos equals daytime brightness for reliable and unreliable
uploaders alike, so night labels are almost always wrong. Cost: ~1.5 pts of time-labeled data.
*Limitation:* "reliable" means "not grossly wrong"; hour-level accuracy is unmeasured.
*Consequence:* the dataset has essentially no genuine night photos; see PROGRESS.md "Future
iterations".

**D-014 Date and year are the primary time metrics; hour-of-day is secondary.** A clock off by a
few hours rarely changes the date and almost never the year, so those labels are trustworthy;
hour labels are not verified (D-013).

## Splits

**D-020 Group splits with spatial separation.** Group = OSV-5M `sequence`. A group never spans
splits; max 2 photos per sequence (train/val), 1 per sequence (test); every train photo >= 1 km
from every val photo (204 train photos dropped). Test is drawn from the official test split,
which OSV-5M built >= 1 km from train. Seeded (`seed=0`); rebuilding reproduces the same photos.
*Note:* 87% of our val photos share an uploader with train; the official test is 98%, so this
mirrors the benchmark. Creator overlap is a shortcut risk for the kNN tool (see LEAKAGE.md L-12).

## Sanitization

**D-030 Fixed output geometry: scale-to-cover + center-crop to 768x512 (3:2).** Removes resolution
and aspect ratio, which identify the device (phone 4:3, OSV 5:3, portrait), and device implies
year. This is arguably legitimate forensics, but we choose to close it so the agent reasons from
scene content. *Cost:* portrait photos lose much of their frame; small Flickr photos get upscaled.
*Alternatives:* letterbox padding (keeps content but padding reveals aspect); keep native size.

**D-031 Uniform JPEG encoding.** Baseline JPEG, quality 90, 4:2:0, JFIF APP0 with 72 dpi, no other
segments. Identical quantization tables for every image (tested), so source encoder/camera can't
be read from the file. ICC profiles are converted to sRGB before being dropped, and EXIF
orientation is baked into pixels.

**D-032 Content-hash names.** `images/<2 hex>/<sha256[:24]>.jpg`. No split, source or original name
in the path.

## Scoring

**D-040 Calibration via a von Mises-Fisher density.** The agent's `radius_km` is scored with the
NLL of the true location under a vMF on the sphere with kappa = (R / radius)^2: the spherical
analogue of an isotropic Gaussian with sigma = radius. Strictly proper, matches the flat Gaussian
for small radii (tested), and stays valid for continent-sized radii, where a flat Gaussian breaks.
Coverage (truth inside radius) is reported against the calibrated target 39.3%.

**D-041 `geo_shaped = exp(-d / tau)`, default tau = 1492.7 km** (GeoGuessr's decay), configurable.

**D-042 Time errors computed in the true location's timezone**, so hour error measures time of day
rather than mixing in location error. Clock time vs solar time: status open (PROGRESS.md Q1).

**D-043 Stratified view = country-macro average**, so "always guess the US" can't look good.
