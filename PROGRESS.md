# Progress

Last updated: 2026-11-11

## Phase A status

| Milestone | Status | Notes |
|---|---|---|
| M0 Environment | Done | WSL2 Ubuntu (vhdx on F:), torch 2.14.1+cu126, sm_75 OK. See VRAM table below. |
| M1 Data: OSV-5M | Done | 20k train / 2k val / 2k test; all 24k images sanitized; final time rule applied (D-016); junk flagged (D-017). |
| M1 Data: Im2GPS3k, YFCC4k | Downloaded + extracted | Manifests, sanitization and timestamp audit not started. |
| M1 Data: private set | n/a | No private photos yet. |
| M2 Sanitization + leakage | Code + tests done | Corpus audit passes on val. LEAKAGE.md drafted. |
| M3 Splits | Done for OSV-5M | Stratified (country-macro) view implemented in scoring. Label-concentration report not written yet. |
| M4 Scoring | Done | `scripts/score_run.py` (D-044): worst score for missing answers, junk excluded, seen/new uploader view, `--vs` paired bootstrap. 14 unit tests. |
| M5 Harness | v1 done | Env, schema-forced JSON actions, tools view/similar/sun/gazetteer, middleware (trace, fault, cache, placebo), budgets, trajectory log (D-045). No memory/RAG/compaction yet. |
| M6 Baselines | Pre-harness part done | `results/baselines.md`: SigLIP kNN 0.745 geo_shaped, Qwen3-VL-2B alone 0.675, constant 0.229. #3 tools 0.670 / #4 placebo 0.501 on dev300 (kNN 0.732 there). #5 API model needs the user's OK. |

## M0 measurements (RTX 2070 Super, 8 GB, one 384 px image, 200 new tokens, greedy)

| Model | Precision | VRAM peak | tok/s | Notes |
|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | bnb nf4, fp16 compute | 1.53 GB | 10.2 | |
| Qwen3-VL-2B-Instruct | fp16 | 4.01 GB | 14.7 | faster than 4-bit on Turing |
| Gemma 4 E2B-it | bnb nf4, fp16 compute | 6.27 GB | 7.0 | audio tower dropped; per-layer embeddings stay fp16 |

| Qwen3-VL-2B-Instruct | fp16, **vLLM 0.33** (D-005) | 85% of card reserved | **97.2** | 487 tok/s total at 8 concurrent |

Rows 1-3 via plain transformers `generate`. vLLM chosen as the runtime (D-005).

## Audits done

- **Night uploaders** (`scripts/audit_night_creators.py`, `data/audits/night_creators.html`):
  352 uploaders whose photos are mostly "at night" under the UTC reading. For ~64% of their
  photos the best clock fix equals their timezone offset (local time stored as UTC) or is
  within 1 h of it; ~31% are off by 2-4 h, unexplained. Visual check: the photos are daytime,
  so the labels are wrong. Excluding them is correct; repairing them is not safe.
- **Brightness at sunset** (`scripts/audit_time_brightness.py`, `results/m1_time_label_brightness.json`):
  intended to measure clock bias/spread for "reliable" uploaders. **Failed to measure**: sky
  brightness is flat across labeled sun angles because street-view photos are almost never
  taken at dusk/night, so night-labeled photos are mislabeled daytime photos. Led to D-013.

- **Date/year sanity** (`scripts/audit_dates.py`, `results/m1_date_audit.json`): no default-clock
  dates (no many-uploader spikes on one day), sequences are internally consistent (median span
  2.4 min, 99.9% < 2 h, 0.01% > 1 day), impossible jumps in 0.06% of consecutive pairs. Subset:
  0-0.3% of time-labeled photos flagged. Limitation: a clock that is consistently wrong (e.g. wrong
  year) passes every metadata check.
- **Location sanity** (`scripts/audit_locations.py`, `results/m1_location_audit.json`): no invalid,
  (0,0) or placeholder coordinates; ~0.1% "at sea", typically 3-4 km offshore (ferries, bridges,
  small islands); 1.2% country mismatch, mostly near borders. "Far" mismatches (0.25%) are mainly
  a coding convention: OSV-5M codes Reunion/Guadeloupe/Martinique/French Guiana separately, Natural
  Earth files them under France. Likely real errors < 0.1%.
- **Junk images** (`scripts/audit_junk_images.py`, `results/m1_junk_audit.json`,
  `data/audits/junk_images.html`): all 24k. No pitch-black frames (0.1th percentile brightness 52/255).
  First look: the darkest and blurriest photos are mostly underexposed or soft but still locatable;
  the lowest-contrast list holds the real junk (fogged frame, near-blank frame, camera aimed at
  the road surface) plus a cluster of dark Indian dashcam night frames in val with a burned-in
  text strip (LEAKAGE L-15). **User reviewed 98 cards** (65 yes / 20 maybe / 13 no;
  `results/m1_junk_review.json`). **Decision (D-017):** flag contrast < 15 plus the user's "no"
  photos as `junk` (14 train / 6 val / 0 test; `results/m1_junk_flags.json`). Scoring and training
  skip junk rows; `scripts/score_run.py` must apply this filter when written. Side finding: ~4% of night-labeled photos (about 20 of 559 in
  val+test) are genuinely dark, i.e. real night photos with correct labels that D-013 drops.

- **Night rescue** (D-016, `scripts/finalize_time_labels.py`, `results/m1_night_rescue.json`,
  `data/audits/night_rescue.html`): cutoffs from 12,948 day-labeled train photos (brightness < 74,
  sky < 76). Rescued 28 train / 1 val / 1 test; spot-check looks like real night (headlights,
  streetlights). **But the uploader rule blocks more real night photos than we rescue**: 45 train /
  12 val / 2 test dark night-labeled photos belong to uploaders failing the 90% daylight rule.
  Likely cause: uploaders who genuinely drive at night fail a "should be daylight" test, so D-012
  can't tell "wrong clock" from "shoots at night". One blocked photo has a burned-in dashcam
  timestamp (LEAKAGE L-15 observed in our data).
  **Decision (D-016a):** leave the blocked photos dropped. A dark photo with a night label proves
  "night", not the hour, and even a full override would give only 13 val night photos.

## Open questions (for the user)

1. Hour-of-day metric: wall-clock time, solar time, or both? (Recommendation: both, solar primary.)
2. ~~Local runtime~~: vLLM (D-005). Gemma 4 E2B on vLLM still untested.
3. Rent a GPU (e.g. RunPod) for full-val baselines and training? Private set stays local regardless.
4. Crop policy: fixed 768x512 crop loses much of portrait photos. Accept, or pad?
5. Flickr timestamps: Im2GPS3k has none in its labels (originals may have EXIF); YFCC4k `DateTaken`
   is camera-local and often junk. Audit before scoring time on either.
6. Hugging Face token would remove the rate-limit slowdown on image downloads.

## Direction (from the user, 2026-11-11)

1. Baselines and measurement first (M5 harness, M6 baselines): the "before" numbers.
2. Build the harness out: memory, retrieval (RAG), context management, tools, skills, safeguards.
   Each layer ships with an ablation via `scripts/score_run.py --vs` (hard rule 7).
3. Then fine-tune the agent (the user's main interest). This replaces the spec's original "no model
   training in either phase" rule; candidates: supervised fine-tuning (photo -> answer), fine-tuning
   on successful tool-use trajectories, RL with `geo_shaped` as reward. QLoRA on the 2070 Super for
   a first small run; larger runs likely on a rented GPU. Private photos stay out of all training.

## Future iterations

- **Train and evaluate on genuine night photos.** OSV-5M has almost none (street-view capture
  stops before dusk), and our label rules (D-012, D-013) drop night-labeled photos because they
  are nearly always mislabeled daytime shots. So the agent will be weak at night and at
  twilight, which is exactly where sun-based time reasoning changes (no shadows; streetlights,
  sky glow, stars, moon). A future data iteration should add a source with verified night
  timestamps, e.g. photos with GPS time (not camera clock) or a curated night subset, and
  report night/twilight as its own eval slice.
  A cheap first step inside OSV-5M is done (D-016 night rescue: 30 photos kept).
  **Replace the uploader daylight rule with a pixel-agreement clock check** when night data
  arrives: judge each uploader by whether image brightness agrees with the labeled sun angle
  (dark<->night, bright<->day) rather than assuming daylight. The current rule (D-012) punishes
  genuine night shooters (59 dark night photos blocked in our subset, D-016a), and it would
  reject most of a real night dataset outright.
- Measure hour-level label accuracy with a manual shadow-direction audit (~40 photos), since the
  automated brightness check can't do it on this data.
- Creator-disjoint eval view: 98% of official OSV-5M test images share an uploader with train;
  report scores split by seen/unseen uploader once the kNN tool exists.
