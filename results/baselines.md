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
| 3 | Qwen3-VL-2B in harness, tools on | | | | | | | | | | | | | |
| 4 | Same, placebo tools | | | | | | | | | | | | | |
| 5 | Frontier API model (needs the user's OK) | | | | | | | | | | | | | |

\* 8 requests in parallel; throughput 1.6 photos/s (1,994 photos in 21 min).
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
