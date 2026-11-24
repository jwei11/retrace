# M6 baselines (OSV-5M val, 1,994 photos after junk removal)

Scored with `scripts/score_run.py` (D-044). Test split untouched. Rows 3-5 of the spec need the M5
harness and are not run yet.

| # | Baseline | median km | acc@25 | acc@200 | acc@750 | acc@2500 | geo_shaped | country-macro | in_radius (target 0.39) | year exact | date ±30 d | new-uploader geo_shaped | tokens/ep | latency/ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | Random train label | 7,822 | 0.0% | 0.8% | 6.8% | 22.2% | 0.120 | 0.083 | 0.31 | 14.9% | 16.5% | 0.135 | 0 | - |
| 2 | Prior only (constant, NL, Oct 2018) | 5,748 | 0.0% | 0.9% | 16.0% | 41.0% | 0.229 | 0.189 | 0.52 | 23.5% | 16.7% | 0.298 | 0 | - |
| 1 | SigLIP kNN, k=1 | 289 | 15.3% | 42.0% | 76.5% | 93.5% | 0.743 | 0.627 | 0.64 | 41.6% | 44.8% | 0.675 | 0 | <0.01 s |
| 1 | SigLIP kNN, k=10 vote (25 km) | 295 | 17.8% | 42.1% | 76.2% | 94.4% | **0.745** | **0.635** | 0.60 | **41.9%** | **45.1%** | 0.683 | 0 | <0.01 s |
| extra | Qwen3-VL-2B alone, no tools (vLLM fp16) | 404 | 6.5% | 26.4% | 70.7% | 89.8% | 0.675 | 0.542 | **0.04** | 4.5% | 40.8% | **0.697** | 736 | 5.0 s* |
| 3 | Qwen3-VL-2B in harness, tools on (dev300) | 385 | 10.7% | 31.7% | 69.0% | 88.7% | 0.670 | 0.567 | 0.05 | 14.7% | 41.6% | 0.688 | 7,388 | 17.9 s† |
| 4 | Same, placebo tools (dev300) | 850 | 3.7% | 16.0% | 49.0% | 70.0% | 0.501 | 0.429 | 0.02 | 6.6% | 42.1% | 0.563 | 8,871 | 20.7 s† |
| 5 | Frontier API model (needs the user's OK) | | | | | | | | | | | | | |

\* 8 requests in parallel; throughput 1.6 photos/s (1,994 photos in 21 min).
† 4 episodes in parallel; 0.22 photos/s. Rows 3-4 are on the frozen 300-photo val subset `dev300`
(D-045). On the same 300 photos: model alone geo_shaped 0.650, kNN-10 0.732.

Embedder: `google/siglip2-so400m-patch14-384`; index = train, non-junk, own group excluded.

## What the numbers say

- **kNN beats the model alone** on every location metric and by a lot on year (41.9% vs 4.5%;
  all differences significant by paired bootstrap, `results/runs/knn10_val.json`).
- **kNN leans on uploader overlap**: 0.755 when the uploader is also in train vs 0.683 when new.
  The model alone shows no such gap (0.671 vs 0.697). Matching an uploader's camera/road style is
  a shortcut (LEAKAGE L-12); report the new-uploader column for anything that uses `similar()`.
- **The model is badly overconfident**: its 40%-radius contains the truth only 4% of the time.
- **The model can't date photos**: 4.5% year exact, below "always guess 2018" (23.5%); it tends to
  guess recent years. Date-of-year (season) is decent (40.8% within 30 days).
- Targets for the harness and fine-tuning: beat 0.745 geo_shaped overall and 0.683 on new uploaders,
  and fix calibration and year.

## Harness v1 (dev300, D-045)

- **Tools help the model a bit**: 0.670 vs 0.650 alone (could be luck), acc@200 +8 pts and year exact
  +10 pts (both real). Still **below kNN** (0.732, real gap), even though `similar` hands it the kNN
  neighbors.
- **The model does use tool output**: placebo tools drop it to 0.501 (-0.17, real), far below the
  model alone, so it trusts what tools say, including wrong answers.
- **Tool use is poor**: 610 `gazetteer` calls in 300 episodes, mostly famous places it guessed rather
  than names it read; `similar` only 210 times. ~10x the tokens of the model alone (7.4k vs 0.7k).
- Calibration still broken (in_radius 0.05). These are the targets for prompting/skills and fine-tuning.

## Skill: playbook_v1 (dev300, D-046)

`skills/playbook_v1.md` (6 rules: similar first, trust agreeing neighbors, gazetteer only on read
names, dates from neighbors, radius from neighbor spread, answer early) vs harness v1:

| Run | geo_shaped | median km | acc@200 | in_radius (target 0.39) | year exact | new-uploader | tokens/ep | latency/ep |
|---|---|---|---|---|---|---|---|---|
| Harness v1 | 0.670 | 385 | 31.7% | 0.05 | 14.7% | 0.688 | 7,388 | 17.9 s |
| + playbook_v1 | **0.736** | 299 | 41.3% | **0.40** | 29.6% | 0.676 | 8,405 | 33.1 s |
| kNN-10 (same photos) | 0.732 | | | | 39.7% | | 0 | <0.01 s |

- +0.066 geo_shaped over v1 (real); every location metric and year improve (real).
- **Ties kNN** (+0.004, could be luck); still behind on year (-10 pts, real).
- **Calibration fixed**: radius now holds the truth 40% of the time (was 5%).
- Caveat: the rules were written after seeing v1's failures on dev300, so confirm on full val.
- Latency doubled (33 s/photo at 4 parallel; 1 error in 300).

### Skill control arms (dev300)

Same harness, same length (~150 words) appended to the prompt:

| Skill | geo_shaped | median km | acc@200 | in_radius | year exact | vs v1 | vs playbook |
|---|---|---|---|---|---|---|---|
| none (harness v1) | 0.670 | 385 | 31.7% | 0.05 | 14.7% | | -0.066 real |
| `playbook_v1` (tool procedure) | **0.736** | 299 | 41.3% | **0.40** | 29.6% | +0.066 real | |
| `adjacent_v1` (true geo tips, no procedure) | 0.633 | 485 | 24.3% | 0.02 | 13.2% | **-0.037 real** | -0.103 real |
| `garbage_v1` (cooking, laundry...) | 0.654 | 413 | 33.0% | 0.07 | 13.2% | -0.017 luck | -0.083 real |

- The gain is the **procedure**, not extra text: garbage text does nothing measurable.
- Correct domain knowledge **hurts**: the model leans on generic visual rules (e.g. guesses from
  road-line colors) instead of the neighbors, and calls `similar` less (187 vs 302 times).
- Latency varies run to run (18-43 s/photo) with the same settings; not a reliable cost signal yet.

### Held-out confirmation (D-047, pre-registered): **PASS**

1,694 val photos not in dev300 (never looked at while writing the playbook):

| Run | geo_shaped | median km | acc@200 | in_radius | year exact | new-uploader | tokens/ep |
|---|---|---|---|---|---|---|---|
| Harness v1, no skill | 0.674 | | | | 12.7% | | |
| + playbook_v1 | **0.735** | 298 | 39.7% | 0.43 | 34.8% | 0.683 | 8,631 |
| kNN-10 (same photos) | 0.748 | | | | 42.1% | | 0 |

- Gain +0.0615 [+0.049, +0.074]: passes the bar (>= +0.03, interval excludes 0). Every location and
  time metric improves (real). The dev300 estimate (+0.066) held up.
- Still **below kNN** on held-out (-0.013, real) and on year (-7 pts, real); new uploaders tie kNN
  (0.683 vs 0.683 on full val). The playbook brings the agent up to kNN, not past it.

### Robustness: wording and sampling (check 4, dev300)

| Arm | geo_shaped | year exact | vs no skill (0.670) | vs playbook_v1 (0.736) |
|---|---|---|---|---|
| `playbook_v1_para` (same rules, reworded and reordered) | 0.721 | 24.9% | +0.051 real | -0.016 luck |
| `playbook_v1`, temperature 0.7, seed 0 | 0.723 | 33.0% | +0.052 real | -0.014 luck |
| `playbook_v1`, temperature 0.7, seed 1 | 0.709 | 27.9% | +0.039 real | -0.027 real |
| `playbook_v1`, temperature 0.7, seed 2 | 0.710 | 32.0% | +0.040 real | -0.027 real |

- Every arm beats no skill (real), so the gain survives rewording and sampling noise.
- The size is +0.04 to +0.05 under sampling; the greedy dev300 run (+0.066) was at the lucky end.
  Greedy decoding (temperature 0) stays the default: it scores best and is reproducible.
- Seed 2 used `view` much more (574 calls vs 63-255) and took 5.3 steps; same score as seed 1.

## Out-of-distribution check: Im2GPS3k (D-048)

2,997 Flickr photos (tourist, landmark, indoor), location only (no timestamps). None of the 198
authors are in our training set. kNN's reference index is still OSV-5M train (street view).

| Run | photos | geo_shaped | median km | acc@25 | acc@200 | acc@750 | acc@2500 |
|---|---|---|---|---|---|---|---|
| kNN-1 | 2,997 | 0.418 | 1,366 | 7.0% | 16.1% | 36.2% | 61.2% |
| kNN-10 | 2,997 | 0.445 | 1,184 | 9.1% | 18.7% | 39.3% | 63.3% |
| Model alone (Qwen3-VL-2B) | 2,997 | **0.601** | 462 | 21.9% | 36.2% | 58.3% | 75.8% |
| Agent + playbook_v1 | 500 (dev500) | 0.536 | 760 | 20.2% | 31.6% | 49.6% | 71.0% |

Paired on the same photos: model alone beats kNN-10 by +0.156 (real, full set); on dev500 the
agent beats kNN-10 by +0.096 (real) but loses to the model alone by -0.067 (real).

- kNN's lead on OSV-5M was the index matching the test photos' kind, not general skill.
- The playbook ("trust agreeing neighbors") hurts when the neighbors are the wrong kind of photo:
  the agent drifts from its own better guess toward street-view lookalikes.

## playbook_v2 and the similar() agreement line (D-050, dev sets)

| Arm | OSV dev300 | vs v1 | Im2GPS3k dev500 | vs model alone (0.603) | s/photo (Im2GPS3k) |
|---|---|---|---|---|---|
| playbook_v1 | 0.736 | | 0.536 | -0.067 real | 5.6 |
| playbook_v2 | 0.712 | -0.025 real | 0.548 | -0.054 real | 4.6 |
| playbook_v2 + `similar_sum` | 0.706 | -0.031 real | **0.578** | -0.025 luck | 15.3 |
| playbook_v1 + `similar_sum` | 0.721 | -0.015 luck | 0.550 | -0.053 real | 6.8 |
| no skill (reference) | 0.670 | -0.066 real | 0.593 | -0.010 luck | 4.0 |

- The agreement line is the part that works: +0.029 real on Im2GPS3k (v2 vs v2+sum), no effect on OSV
  (-0.006 luck). Its cost: the Im2GPS3k run was 3x slower (cause not yet checked).
- The v2 text costs ~0.025 on street view: "own guess first" makes the agent go its own way more
  often there (14% of photos, scoring 0.56 vs kNN's 0.72 on them).
- Compliance is still partial: 43-68% of gazetteer names still come from the neighbor list.
- Not run on held-out: neither arm meets D-050 on dev (both lose >0.02 to v1 on OSV).
- v1 + `similar_sum` (2026-11-16, OSV only): 0.721, -0.015 vs v1 [-0.037, +0.005], still +0.051 real
  vs no skill. acc@750 drops 4.7 points (real). Im2GPS3k (2026-11-18): 0.550, +0.015 vs v1 (luck).
  With v1's text the agreement line helps about half as much as with v2's (+0.029); the agent still
  calls similar first on 58% of photos, and those score 0.43 vs 0.72.
- No skill on Im2GPS3k (2026-11-18): 0.593, level with the model alone (-0.010, luck) and +0.057 real
  over v1. So the tools alone do not hurt out of distribution; the v1 skill does. v1 is a trade-off:
  +0.066 on street view, -0.057 on Flickr.
  Its OSV lower bound (-0.037) already misses the D-050 bar (>= -0.02), so it is not sent to held-out.

## Skill router by retrieval confidence (D-052, held-out)

`--route playbook_v1@0.80`: playbook_v1 when top similarity >= 0.80, else no skill.

| Set | Router | Routed to v1 | Comparison | Diff [95% CI] | Bar | Verdict |
|---|---|---|---|---|---|---|
| OSV not-dev300 (1,694) | 0.733 | 97% | vs playbook_v1 | -0.002 [-0.006, +0.002] | >= -0.02 | pass |
| Im2GPS3k not-dev500 (2,497) | 0.591 | 9% | vs model alone | -0.010 [-0.022, +0.002] | >= -0.02 | fail (by 0.002) |

- Also: OSV +0.060 real vs no skill, -0.015 real vs kNN-10; Im2GPS3k +0.145 real vs kNN-10.
- Routing keeps v1's street-view gain and removes most of the Flickr loss (v1: -0.067 vs model alone on
  dev500). The residual -0.010 matches the no-skill agent vs one look (-0.010 on dev500).
- Cost: 4.0 s/photo Im2GPS3k, 5.8 s/photo OSV (concurrency 4).

## Burned-in overlay audit (D-051, LEAKAGE.md L-15/L-16)

`scripts/audit_overlays.py`: OCR of the top and bottom bands of every photo, regexes for dates,
clock times and coordinates. All 88 flagged OSV val texts were read by eye: every one is a real
camera overlay (dashcam timestamps, GPS readouts, road-log captions with town names such as
"BRATTLEBORO"). Recall is not measured (small or stylized overlays can be missed).

| Set | Photos | Leak (date, time or GPS printed) | Any overlay |
|---|---|---|---|
| OSV-5M val | 2,000 | 77 (3.9%) | 88 (4.4%) |
| Im2GPS3k | 2,997 | 5 (0.2%; film date stamps, no time labels there) | 5 |

Impact on dev300 (14 leak photos) and held-out not-dev300 (63):

| Run | geo all | geo clean | year exact all | year exact clean | year exact on leak photos |
|---|---|---|---|---|---|
| kNN-10 | 0.733 | 0.722 | 40.1% | 39.6% | 50.0% |
| Model alone | 0.650 | 0.642 | 5.1% | **1.1%** | 80.0% |
| Agent, no skill | 0.670 | 0.665 | 14.7% | 11.2% | 80.0% |
| Agent + playbook_v1 | 0.736 | 0.734 | 29.6% | 27.3% | 77.8% |
| playbook_v1, held-out | 0.735 | 0.732 | 34.8% | 32.6% | 91.1% |

Paired differences, all photos vs leak photos removed:

| Comparison | All | Clean |
|---|---|---|
| playbook_v1 − no skill (dev300) | +0.066 [+0.037, +0.095] | +0.068 [+0.037, +0.100] |
| no skill − placebo (dev300) | +0.169 | +0.174 |
| playbook_v1 − kNN-10 (held-out) | −0.013 [−0.023, −0.002] | −0.011 [−0.022, −0.001] |
| playbook_v1 − model alone (held-out) | +0.056 | +0.054 |

- **Location conclusions are unchanged** (every difference moves by ≤0.003 and keeps its verdict).
- **Time is where the leak matters:** the model alone's year accuracy is almost all overlays (5.1% →
  1.1% clean). The agent's year gains hold on clean photos (no skill 11.2% → playbook 27.3%).
- kNN also scores high on overlay photos (0.939) without reading text: overlay photos come from a few
  heavy contributors with distinctive cameras, so lookalikes are often the same uploader's photos.
