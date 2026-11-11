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

**D-005 Local inference runtime: vLLM** (chosen by the user, 2026-11-11). vLLM 0.33.0 serves the
VLM over an OpenAI-compatible API on 127.0.0.1:8000 (`scripts/serve_vllm.sh`); the harness talks
HTTP. It lives in its own venv (`~/.venvs/vllm`, torch 2.14.0+cu130) because it pins its own torch.
On Turing it uses the TRITON_ATTN backend (no FlashAttention), fp16. Needs a C compiler for
Triton (`build-essential`, installed as root via `wsl -u root`). Settings that fit 8 GB next to
Windows' VRAM use: `--gpu-memory-utilization 0.85 --max-model-len 6144` (KV cache ~7k tokens).
*Evidence* (`results/m0_vlm_smoke.jsonl`, Qwen3-VL-2B fp16, 384 px, 200 tokens): 97 tok/s alone vs
14.7 with transformers (6.6x); 487 tok/s total with 8 concurrent requests (33x).
*Why vLLM also suits later work:* it serves LoRA adapters and batches many sampled answers, which
fine-tuning evaluation and RL rollouts need. *Alternatives:* plain transformers (works, too slow
for evaluation); llama.cpp (no compiler needed, newer VLM support). Gemma 4 E2B on vLLM not tested.

**D-006 shapely for geometry.** Added for point-in-polygon checks against Natural Earth borders
(location audit); it will also serve the gazetteer and region tools. Standard, wheel-installable.

## Data

**D-010 Fetch OSV-5M images by HTTP range reads.** Read each remote zip's central directory once,
record every member's offset, then fetch only sampled members. ~1 GB transferred instead of
246 GB. Hugging Face limits anonymous `resolve` calls to 3,000 per 5 minutes, so each shard's
`resolve/` URL is turned into its signed CDN URL once (refreshed after 40 min or on 401/403) and
byte ranges go to the CDN, which has no such limit. Concurrency 12, exponential backoff honoring
`Retry-After`. *Alternative:* download whole shards; log in with an HF token for higher limits.

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
*Known bias:* the rule assumes street photos are taken in daylight, so an uploader who genuinely
shoots at night fails it just like one with a broken clock. It can't tell the two apart. Measured
cost on our subset: 59 genuinely dark night-labeled photos blocked (45 train / 12 val / 2 test);
see D-016.

**D-013 Time labels: per-photo night rule.** Additionally drop the time label of any photo whose
labeled sun elevation is < -6 deg. *Evidence:* `results/m1_time_label_brightness.json`: sky
brightness of night-labeled photos equals daytime brightness for reliable and unreliable
uploaders alike, so night labels are almost always wrong. Cost: ~1.5 pts of time-labeled data.
*Limitation:* "reliable" means "not grossly wrong"; hour-level accuracy is unmeasured.
*Consequence:* the dataset has essentially no genuine night photos; see PROGRESS.md "Future
iterations". *Known cost:* ~4% of night-labeled photos are genuinely dark (real night photos with
correct labels) and would be dropped too (junk-image audit); D-016 rescues them.

**D-016 Night rescue.** A night label (sun < -6 deg) is kept when the image is genuinely dark:
whole-frame brightness AND sky (top quarter) brightness both below the 2nd percentile of
day-labeled (sun >= +6 deg) train photos from reliable uploaders. *Why both:* dark daytime scenes
(tunnels, dense forest, storms) usually still show bright sky; requiring a dark sky too cuts false
rescues. *Why train-only cutoffs:* setting them on val/test would tune a label rule on eval data.
*Scope:* overrides the per-photo night rule only; the uploader rule (D-012) still applies, and
the count of dark night photos it blocks is reported. Each photo's outcome is stored in
`time_label_status` (`ok_day`, `rescued_night`, `dropped_night`, `creator_unreliable`).
Implemented in `src/retrace/ingest/time_labels.py`, applied by `scripts/finalize_time_labels.py`.
*Alternatives:* drop all night labels (loses the real ones); whole-frame brightness only (more
false rescues).

**D-016a The rescue does NOT override the uploader rule** (decided with the user, 2026-11-11).
The 59 dark night photos from uploaders failing D-012 stay dropped.
*Why:* (1) "dark image + night label" confirms *night*, not the *hour*: a clock 3 h off can turn
23:00 into 02:00 and both still look like night, so a broken-clock uploader's dark photos can
agree with the pixels and still have the wrong hour (and sometimes the wrong date). (2) Scale:
even a full override gives 13 val / 73 train night photos, too few for a meaningful night score
or for teaching anything about night. No option produces a usable night set from OSV-5M.
(3) Every kept label stays defensible; the cost is ~60 of 24,000 photos.
*Alternatives considered:* override for all dark night photos (more night data, some hours
wrong by hours); a pixel-agreement uploader check, i.e. judge each uploader by whether their
photos' brightness agrees with their labels (dark<->night, bright<->day) instead of assuming
daylight (fixes the real flaw, but needs ~600 extra downloads and new code for ~60 photos;
deferred to the night-data iteration, where it will be needed anyway, see PROGRESS.md);
use night photos for date/year scoring only (special case in scoring for 13 val photos).

**D-017 Junk images are flagged, not deleted** (decided by the user, 2026-11-11). `junk=True` when
contrast (luma std of the sanitized image) < 15, OR the user marked the photo unlocatable in the
manual review (`results/m1_junk_review.json`). Flagged: 14 train / 6 val / 0 test (20 of 24,000).
Images and rows stay; scoring and training must skip `junk` rows. Reason in `junk_reason`.
*Evidence:* user's review of the 100 most extreme images (`data/audits/junk_images.html`): all 20
random controls locatable; darkest and brightest nearly all locatable (darkness/brightness are not
junk signals); contrast < 15 catches 14 photos the user rated 7 no / 6 maybe / 1 yes; blur has no
clean cutoff (the user's "no" blurry photos share sharpness values with "yes" ones), so those are
caught by hand only. *Why both:* the rule is reproducible on new data (Flickr); the hand marks
catch what the rule can't. *Side effect:* removes several val dashcam frames with a burned-in
text strip (LEAKAGE L-15). *Limitation:* only 100 photos were reviewed; unreviewed blurry junk
remains (rare: 0 of 20 random photos). *Alternatives:* leave all in (0.25% of val, small effect);
hand marks only (not reproducible); also cut on blur (would drop usable photos).
Implemented in `src/retrace/ingest/junk.py`, applied by `scripts/flag_junk.py`.

**D-015 Date/year and location labels are kept as-is.** Audits found no default-clock dates,
consistent sequences (0.06% impossible jumps) and clean coordinates (no placeholders; ~0.1% a few
km offshore). The few flagged rows are recorded in `audits/*_flags.parquet` but not removed, since
the flagged share (< 0.3%) is too small to change results and removal rules add their own risk.
Status: open if the user prefers to drop flagged rows.

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

**D-044 Run scoring rules** (decided by the user, 2026-11-11; `src/retrace/score/run.py`,
`scripts/score_run.py`).
- A photo with no answer (crash, timeout, unparseable) gets the worst score (antipode), so
  skipping hard photos can't raise a score; answer rate is reported alongside.
  *Alternative:* score answered photos only (rejected: rewards flaky agents).
- Junk photos (D-017) are excluded; answers for them are ignored. Answers for ids outside the
  split are an error.
- Views: micro, country-macro, and uploader seen-in-train vs new (L-12).
- Comparisons (`--vs`): paired bootstrap over photos, 2,000 resamples, 95% interval; a difference
  is called real only if the interval excludes 0. Required for every ablation (hard rule 7).
- Run answers and per-photo scores stay in `$RETRACE_DATA/runs/` (outside git); summaries go to
  `results/runs/<run_id>.json` and one row per run in `results/runs.csv`. Runs marked
  `"private": true` are refused, so private-set results never land in the repo.
