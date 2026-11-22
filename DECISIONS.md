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

**D-045 Harness v1 design (M5)** (choices by the user, 2026-11-11; `src/retrace/env`,
`src/retrace/harness/agent.py`, `src/retrace/tools/`, `src/retrace/middleware`, `scripts/run_agent.py`).
- **Actions are schema-forced JSON**: each turn vLLM constrains the reply to
  `{"thought", "action": {"tool", "args"}}`, and the schema lists only the actions the budgets still
  allow. A 2B model never makes a malformed call; trajectories are clean fine-tuning data.
  *Alternative:* the model's native tool-call format (rejected: small models drop calls).
  The server sets `disable_any_whitespace` (xgrammar): without it the model sometimes filled the
  JSON with spaces until max_tokens (2 of 8 smoke episodes).
- **Tools v1**: `view` (zoom crop of the sanitized photo, scaled to <=768x512), `similar(k)` (SigLIP
  train index; place, coordinates, UTC time of the k nearest; the photo's own capture sequence
  excluded, as in the kNN baseline), `sun` (NOAA solar position), `gazetteer` (GeoNames cities500,
  exact name match, top 5 by population), `answer`. OCR deferred: the VLM reads text via `view`.
- **Budgets**: 6 turns, 4 images, and a forced answer when the context would exceed 6,144 - 1,000
  tokens (the server's limit; the KV cache holds ~1.15 full-length requests, so concurrency is low).
- **Middleware**, outermost first: Trace (per-call log), Fault (injected tool failures, default
  off), Cache (identical calls), Placebo. Placebo outputs: `similar` = real similarity values with
  random train photos' labels; `sun` = values for a random place and time; `gazetteer` = as many
  random places; `view` = the whole photo again.
- **Eval**: fixed 300-photo val subset (`dev300`, seed 0, non-junk) for iteration, full val for
  final numbers. `score_run.py --vs` compares on the subset's photos when the other run is full val.
- Tools never see labels or metadata of the query photo; the env never holds them (hard rule 5).
  `similar` gets the photo's capture-sequence id env-side only, to exclude it.

**D-046 Skills are prompt files** (`skills/<name>.md`, `run_agent.py --skill`), appended to the
instruction, so each is versioned and ablated on its own. `playbook_v1` encodes the failures seen in
harness v1 (guessed gazetteer calls, ignored neighbors, overconfident radius): +0.066 geo_shaped on
dev300 and calibrated radius. Risk: written after looking at dev300 results, so dev300 overstates it;
full-val confirmation required before it counts. *Alternative:* skills as callable tools that return
instructions (deferred: needs the model to know when to call them).
D-046 controls (dev300): an equal-length adjacent skill (true geolocation tips, no procedure) scores
0.633 (-0.037 vs no skill, real) and a garbage skill 0.654 (no real change). So skills are kept only
if they beat both controls; the playbook's gain is the procedure itself.

**D-047 Pre-registered test of `playbook_v1`** (written before the run, 2026-11-13).
Set: val non-junk minus dev300 (`not-dev300`, 1,694 photos), never looked at while writing the
playbook. Arms: harness v1 without skill vs with `playbook_v1`, same settings as dev300.
**Pass** = geo_shaped gain >= +0.03 AND the paired-bootstrap 95% interval excludes 0.
Also reported, not part of pass/fail: in_radius, year exact, new-uploader geo_shaped, vs knn10.
D-047 result (2026-11-13): **pass**. +0.0615 geo_shaped [+0.049, +0.074] on not-dev300; agent 0.735
vs knn10 0.748 on the same photos (-0.013, real). `playbook_v1` is adopted as the harness default
for further layers; beating kNN is left to later layers and fine-tuning.

D-047 addendum (2026-11-14), check 4: a reworded playbook (`playbook_v1_para`, 0.721) and three
temperature-0.7 seeds (0.723 / 0.709 / 0.710) all beat no skill (+0.039 to +0.052, each real).
The effect is the rules, not the exact wording or one lucky decode; its honest size under sampling
is about +0.045, below the greedy +0.066. Kept temperature 0 as the default (best and reproducible).

## D-048: Im2GPS3k as an out-of-distribution check (2026-11-14)

What: ran kNN, the model alone and the playbook agent on Im2GPS3k (Flickr, location only), keeping
the OSV-5M train index. Agent on a frozen 500-photo subset (dev500, seed 0) to save ~4 h.
Why: on OSV-5M kNN beat everything; test whether that holds off street view.
Result: model alone 0.601 > agent+playbook 0.536 > kNN-10 0.445 (all gaps real). kNN's OSV-5M lead
is domain-specific. playbook_v1 is tuned to OSV-5M: its "trust the neighbors" rule costs -0.067 here.
Implication: the next skill or layer must judge whether the neighbors are trustworthy (e.g. top
similarity, agreement, photo type) instead of always deferring to them.
Alternatives: full 2,997-photo agent run (only if a final number is needed); YFCC4k as a second
OOD set; adding a Flickr reference index (would change what the agent can retrieve).

## D-049: No frontier/API model baseline (2026-11-15)

What: M6 baseline #5 (API frontier VLM) is dropped; all models run locally.
Why: user decision. Keeps the project self-contained and cost-free, and removes any path for data to
leave the machine. Alternatives: an API baseline on public photos only (rejected by user).

## D-050: playbook_v2 and the similar() agreement summary (2026-11-15, pre-registered)

What: (1) `skills/playbook_v2.md`: own guess first (anti-anchoring), a 3-band trust table keyed on
neighbor agreement and similarity, radius from the band, gazetteer only on text read in the photo.
(2) `similar_sum`: the same `similar` tool plus one computed line: size of the densest group of
neighbors within 300 km of each other, its median spread, and a LOW flag when top similarity < 0.80.
Why: on Im2GPS3k, playbook_v1 lost -0.067 to the model alone. Neighbor agreement predicts kNN
accuracy on both datasets (within 50 km: 88% / 73% of nn1 within 200 km; over 1,000 km: 17% / 8%),
but the 2B model cannot compute it (radius/spread 0.11; copied #1 on 34% of photos). Top similarity
separates the datasets (median 0.905 vs 0.701). Thresholds 300 km and 0.80 were set from OSV val
statistics. Im2GPS3k statistics have been looked at, so Im2GPS3k dev500 is a dev set from now on.
Dev arms: v2 and v2+similar_sum, each on OSV dev300 and Im2GPS3k dev500 (temperature 0).
Held-out pass/fail for the chosen arm (run after dev, not looked at before):
  OSV not-dev300: vs playbook_v1, 95% interval lower bound >= -0.02 (no real loss on street view);
  Im2GPS3k not-dev500 (2,497 photos): vs model alone, lower bound >= -0.02 (no longer hurts).
  Both must hold. Also reported: vs kNN-10, in_radius, tokens.
Alternatives: scene-type rule ("is it a street scene?", judged by the model; less reliable); a
fixed similarity cutoff only (does not predict accuracy within a dataset); G3-style candidates
plus verification (later layer).
D-050 dev result (2026-11-15): neither v2 arm is sent to held-out. The agreement line helps on
Im2GPS3k (+0.029, real) and is neutral on OSV; the v2 text loses ~0.025 on OSV. Next candidate:
playbook_v1 text + `similar_sum` (the untested cell), possibly with one added rule for the LOW flag.
v1 + similar_sum (2026-11-16, OSV dev300): 0.721, -0.015 vs v1 [-0.037, +0.005]. The lower bound misses
the -0.02 bar, so no held-out run. Im2GPS3k dev500 for this arm not run. Conclusion: on street view the
line adds nothing and v1 stays the shipped skill; the line's value is out of distribution only.
Im2GPS3k dev500 (2026-11-18): v1+sum 0.550 (+0.015 vs v1, luck). No-skill agent 0.593 (= model alone,
-0.010 luck; +0.057 real over v1): the out-of-distribution loss comes from the v1 skill, not the tools.
Open question for the next skill: gate the playbook on the agreement line (follow it only when the
references agree) instead of always pointing the agent at similar().

## D-051: Burned-in camera overlays leak labels through pixels (2026-11-16; LEAKAGE.md L-15/L-16, now measured)

What: some OSV-5M dashcam photos have an overlay that prints the capture date/time, sometimes GPS
coordinates and an uploader handle, into the image (e.g. "MiVue 788 2019/06/21 15:32", "W80.989250
N35.046212 NETWORKLANMAN"). Sanitizing strips file metadata, not pixels. Trajectories show the agent
reads these timestamps ("the timestamp in the image is 2019-06-21 15:32:17"); no case of it reading
the GPS text was seen.
Impact: time metrics (and possibly a few location answers) are inflated on these photos for every
run and baseline that sees pixels; kNN is unaffected by text. Size unknown.
Done now: the demo excludes such photos by id (scripts/export_demo.py EXCLUDE).
To do: OCR audit of val/test for overlay text (dates, times, coordinates); then either junk-flag
them or crop the overlay band, and re-score time metrics with and without them.
D-051 result (2026-11-16): OCR audit flags 77 of 2,000 OSV val photos (3.9%) as printing a date,
time or GPS; all 88 flagged texts checked by eye are real overlays (recall not measured). Location
conclusions are unchanged on clean photos (all paired differences move by <=0.003, same verdicts).
Time metrics are inflated: model alone year-exact 5.1% -> 1.1% clean. Policy for now: no data
change (frozen splits stay frozen); report time metrics on clean photos alongside all photos, and
exclude these photos from demos. Alternatives: junk-flag them (changes dev300/not-dev300 sizes),
or crop the bands for every photo (also removes legitimate signs near the edges).
