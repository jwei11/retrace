# Progress

Last updated: 2026-11-10

## Phase A status

| Milestone | Status | Notes |
|---|---|---|
| M0 Environment | Done | WSL2 Ubuntu (vhdx on F:), torch 2.14.1+cu126, sm_75 OK. See VRAM table below. |
| M1 Data: OSV-5M | Labels done; images in progress | 20k train / 2k val / 2k test sampled. Val images sanitized; test/train fetching (HF rate limits). |
| M1 Data: Im2GPS3k, YFCC4k | Downloaded + extracted | Manifests, sanitization and timestamp audit not started. |
| M1 Data: private set | n/a | No private photos yet. |
| M2 Sanitization + leakage | Code + tests done | Corpus audit passes on val. LEAKAGE.md drafted. |
| M3 Splits | Done for OSV-5M | Stratified (country-macro) view implemented in scoring. Label-concentration report not written yet. |
| M4 Scoring | Done | 10 unit tests. Run-level script (`scripts/score_run.py`) not written yet. |
| M5 Harness | Not started | |
| M6 Baselines | Not started | |

## M0 measurements (RTX 2070 Super, 8 GB, one 384 px image, 200 new tokens, greedy)

| Model | Precision | VRAM peak | tok/s | Notes |
|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | bnb nf4, fp16 compute | 1.53 GB | 10.2 | |
| Qwen3-VL-2B-Instruct | fp16 | 4.01 GB | 14.7 | faster than 4-bit on Turing |
| Gemma 4 E2B-it | bnb nf4, fp16 compute | 6.27 GB | 7.0 | audio tower dropped; per-layer embeddings stay fp16 |

All via plain transformers `generate`, the slowest option. A serving engine (vLLM / llama.cpp)
is expected to be much faster; not measured yet.

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

## Open questions (for the user)

1. Hour-of-day metric: wall-clock time, solar time, or both? (Recommendation: both, solar primary.)
2. Local runtime for M5/M6: try vLLM or llama.cpp before deciding on Gemma vs Qwen.
3. Rent a GPU (e.g. RunPod) for full-val baselines and training? Private set stays local regardless.
4. Crop policy: fixed 768x512 crop loses much of portrait photos. Accept, or pad?
5. Flickr timestamps: Im2GPS3k has none in its labels (originals may have EXIF); YFCC4k `DateTaken`
   is camera-local and often junk. Audit before scoring time on either.
6. Hugging Face token would remove the rate-limit slowdown on image downloads.

## Future iterations

- **Train and evaluate on genuine night photos.** OSV-5M has almost none (street-view capture
  stops before dusk), and our label rules (D-012, D-013) drop night-labeled photos because they
  are nearly always mislabeled daytime shots. So the agent will be weak at night and at
  twilight, which is exactly where sun-based time reasoning changes (no shadows; streetlights,
  sky glow, stars, moon). A future data iteration should add a source with verified night
  timestamps, e.g. photos with GPS time (not camera clock) or a curated night subset, and
  report night/twilight as its own eval slice.
- Measure hour-level label accuracy with a manual shadow-direction audit (~40 photos), since the
  automated brightness check can't do it on this data.
- Creator-disjoint eval view: 98% of official OSV-5M test images share an uploader with train;
  report scores split by seen/unseen uploader once the kNN tool exists.
