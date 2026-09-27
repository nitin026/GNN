# Tuning log

Every tuning decision, what it was chosen on, and whether any test case was involved.
Splits are locked (pipeline/splits.py):
- train: cyc_01, cyc_02, heat_01, heat_02, cold_01, cold_02
- val: cyc_03, heat_03, cold_03
- test: cyc_04, heat_04, cold_04, amphan_replay

| # | when | what was tuned | chosen on | value | test cases involved? |
|---|---|---|---|---|---|
| 1 | BRIEF Phase 5 | baseline z thresholds (MSLP -2.0, T2m +/-1.75), min object size, 25 km smoothing | a first look at heat_01 (train) | as stated in BASELINE_RESULTS.md | no (they were set once and then applied to every case) |
| 2 | BRIEF2 Ph.1 | synthetic T2m background soft clip 1.0 sigma (was 1.5), Tibet/IMD domain exclusion | diagnosis of heat_01 / cold_01 (train) | 1.0 sigma; > 2500 m excluded | no |
| 3 | BRIEF2 Ph.1 | rain quantile map (option A global vs option B intensity-aware) | TRAIN fit (cyc_01, cyc_02); A vs B chosen on train+val p99 | option A | no. The per-case table also shows the test cases, but they were not used for the choice |
| 4 | BRIEF2 Ph.1 | tracker v2 grid: high/low/min_area per hazard | TRAIN+VAL (8 of 20 members) | reports/tracker2_params.json | no |
| 5 | BRIEF2 Ph.1 | IMD-domain crest threshold 2500 -> 4500 m | a Srinagar sanity check (geography, not skill) | 4500 m | no |
| 6 | BRIEF2 Ph.2 | GNN checkpoint (best val AP node+edge) | VAL | best epoch per variant | no |
| 7 | BRIEF2 Ph.2 | GNN decoding tau_n, min_len per hazard/variant | VAL by CSI | reports/gnn_results.json "decode" | no |
| 8 | BRIEF2 Ph.2 | GNN main-track rule: largest area -> largest summed probability | VAL cyc_03 (truth-run IoU 0.00) | summed probability | no |
| 9 | BRIEF2 Ph.3 | downscaler size (base 16, 96 px crops, 20/25 epochs) | CPU time budget, not skill | as in models/downscale/*/config.json | no |
| 10 | BRIEF2 Ph.3 | downscaler checkpoints (best val loss) | VAL patches | best epoch | no |
| 11 | BRIEF3 A | alert probability cut-offs 0.2 / 0.5 | set by hand once | docs/ALERT_RULES.md | no |
| 12 | BRIEF3 A | **per-cell -> 50 km neighbourhood probability** | **amphan_replay (TEST)**: a super cyclone could never reach severe (peak cell P = 0.45) | 50 km | **YES**. This decision was made after looking at a TEST case. Both probabilities are shown on every alert, and the change is disclosed in docs/ALERT_RULES.md. |
| 13 | BRIEF3 A | 4-D box extent definition (cyclone: closed 4 hPa ring-median area; heat/cold: >= 3 C core component) | amphan_replay (TEST) bbox spanning the whole domain | as described | **YES** (a definitional change, made after looking at a test case; disclosed here) |

Entries from BRIEF4 onwards are appended below as they are made.

## BRIEF4

| # | phase | what was tuned | chosen on | value | test cases involved? |
|---|---|---|---|---|---|
| 14 | Phase 0 | evaluation design (leads every 24 h, 8 patches/lead, half on the event, seed 2026, 8 DDIM samples x 25 steps) | CPU budget, fixed before any results | as stated | no; evaluation only, nothing tuned |
