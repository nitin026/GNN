# Phase 1 improvements (BRIEF2): before / after

All synthetic cases are **SYNTHETIC** (exact labels); Amphan ERA5 vs IBTrACS is real. "Old" is the Phase 5 tracker; "v2" is `pipeline/tracker2.py`. Both run on the same regenerated data (generator `synth-0.2`) with the same metric definitions (`pipeline/evaluate2.py`). v2 parameters were chosen on TRAIN+VAL cases only (`reports/tracker2_params.json`). TEST cases (cyc_04, heat_04, cold_04, amphan_replay) were not used for any choice.

## 1. Tracker false alarms and IoU

Spurious ensemble tracks = tracks, summed over the 20 members, that never overlap that member's own injected-event mask (the generator's deliberate false-alarm events count as real). Target: < 10 per case, with the same or better IoU.

| case | split | hazard | IoU old | IoU v2 | centroid err old (km) | v2 (km) | spurious ens tracks old | v2 | truth-run spurious old | v2 | ens hit @peak old | v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cyc_01 | train | tropical_cyclone | 0.70 | 0.70 | 8 | 8 | 364 | 5 | 4 | 0 | 0.40 | 0.50 |
| cyc_02 | train | tropical_cyclone | 0.26 | 0.26 | 7 | 7 | 474 | 7 | 5 | 0 | 0.00 | 0.10 |
| heat_01 | train | heat_dome | 0.58 | 0.50 | 147 | 70 | 565 | 3 | 0 | 0 | 0.60 | 0.95 |
| heat_02 | train | heat_dome | 0.26 | 0.10 | 71 | 220 | 545 | 11 | 0 | 1 | 0.05 | 0.90 |
| cold_01 | train | cold_wave | 0.24 | 0.31 | 246 | 258 | 420 | 4 | 0 | 0 | 0.00 | 0.85 |
| cold_02 | train | cold_wave | 0.19 | 0.11 | 148 | 417 | 373 | 14 | 0 | 0 | 0.05 | 0.45 |
| cyc_03 | val | tropical_cyclone | 0.61 | 0.61 | 12 | 12 | 445 | 3 | 8 | 0 | 0.05 | 0.65 |
| heat_03 | val | heat_dome | 0.48 | 0.41 | 140 | 170 | 577 | 8 | 0 | 0 | 0.40 | 1.00 |
| cold_03 | val | cold_wave | 0.32 | 0.18 | 77 | 136 | 428 | 12 | 1 | 0 | 0.05 | 1.00 |
| cyc_04 | test | tropical_cyclone | 0.35 | 0.35 | 10 | 10 | 431 | 5 | 9 | 0 | 0.25 | 0.50 |
| heat_04 | test | heat_dome | 0.28 | 0.00 | 134 | n/a | 536 | 27 | 1 | 0 | 0.00 | 0.00 |
| cold_04 | test | cold_wave | 0.34 | 0.00 | 183 | 2159 | 441 | 27 | 0 | 1 | 0.10 | 0.10 |
| amphan_replay | test | tropical_cyclone | 0.34 | 0.34 | 8 | 8 | 465 | 3 | 18 | 0 | 0.65 | 0.85 |

### Summary by hazard (mean over cases)

| hazard | split | n | IoU old | IoU v2 | spurious/case old | v2 | cases meeting < 10 spurious (v2) |
|---|---|---|---|---|---|---|---|
| tropical_cyclone | train+val | 3 | 0.52 | 0.52 | 428 | 5 | 3/3 |
| tropical_cyclone | test | 2 | 0.34 | 0.34 | 448 | 4 | 2/2 |
| heat_dome | train+val | 3 | 0.44 | 0.33 | 562 | 7 | 2/3 |
| heat_dome | test | 1 | 0.28 | 0.00 | 536 | 27 | 0/1 |
| cold_wave | train+val | 3 | 0.25 | 0.20 | 407 | 10 | 1/3 |
| cold_wave | test | 1 | 0.34 | 0.00 | 441 | 27 | 0/1 |

Chosen v2 parameters (TRAIN+VAL):

- tropical_cyclone: `{"high": 2.0, "low": 2.0, "min_area_km2": 3000.0}` (train+val IoU 0.52, spurious/case 8.3)
- heat_dome: `{"high": 4.5, "low": 3.5, "min_area_km2": 50000.0}` (train+val IoU 0.33, spurious/case 5.8)
- cold_wave: `{"high": 4.5, "low": 3.5, "min_area_km2": 50000.0}` (train+val IoU 0.20, spurious/case 10.0)

Phase 5 numbers on the *old* data (synth-0.1, `reports/BASELINE_RESULTS.md`) are kept for reference: the ensemble false-alarm column there used the truth mask and a slightly different background. The table above re-runs the old tracker on the new data so the comparison is fair.

## 2. Real Amphan (ERA5 G12 vs IBTrACS)

| tracker | matched 6-h fixes | mean err (km) | median (km) | max (km) | other tracks in window |
|---|---|---|---|---|---|
| old (Phase 5) | 21/26 | 40 | 28 | 195 | n/a |
| v2 | 23/26 | 45 | 28 | 195 | 0 |

## 3. Heat and cold waves: IMD criteria + persistence

v2 replaces the single-step T2m z-score with the IMD operational criteria on daily Tmax/Tmin (trailing 24 h of 6-hourly values):
- heat wave: departure >= 4.5 C with Tmax >= 40 C (plains), >= 37 C (coastal), >= 30 C (hills), or Tmax >= 45 C;
- cold wave: departure <= -4.5 C with Tmin <= 10 C (plains), <= 15 C (coastal), <= 0 C (hills), or Tmin <= 4 C in the plains;
- persistence of at least 2 days (8 six-hourly steps), as in IMD's two-day rule.

Cells above 2500 m (Tibetan plateau) are excluded as outside the IMD domain. Target: IoU >= 0.4 (see table 1).

## 4. Synthetic cyclone rain vs IMD (quantile mapping)

Swath = daily land rain within 500 km of the storm centre at ~0.25 deg, wet cells. The map was fitted on the TRAIN cases cyc_01 and cyc_02 only. Target: p99 within 20 % of IMD Amphan.

| case | split | p99 before | p99 after | after vs IMD | max before | max after |
|---|---|---|---|---|---|---|
| cyc_01 | train | 394 | 200 | +7 % | 541 | 272 |
| cyc_02 | train | 142 | 96 | -49 % | 192 | 134 |
| cyc_03 | val | 218 | 149 | -21 % | 411 | 271 |
| cyc_04 | test | 342 | 176 | -7 % | 458 | 236 |
| amphan_replay | test | 191 | 128 | -32 % | 230 | 166 |
| IMD Amphan (real) | - | 188 | 188 | +0 % | 230 | 230 |

Two maps were tried:
- **Option A**: global empirical QM (kept).
- **Option B**: intensity-aware QM, with rain normalised by the generator's own g(V) = 3 + 0.3 Vmax before mapping (`reports/rain_qm_optionB_intensity.json`).

B was worse on the TRAIN/VAL storms (raw p99 vs IMD cyc_01 +49 %, cyc_02 -43 %, cyc_03 -16 %, against A's +7 %, -49 %, -21 %), so A was kept. The choice used no test case. The full Phase 4 validation was rerun (`reports/SYNTH_VALIDATION.md`).

## 5. 850 hPa q/u/v for the heat and cold windows

- ERA5 coldwave 850 hPa q/u/v (daily 12 UTC): **OK**, 2022-12-20 .. 2023-01-20 daily 12 UTC (32 steps; 990s; direct HTTPS chunk reads (fsspec timed out on 100 MB chunks); fills the ea)
- ERA5 heatwave 850 hPa q/u/v (daily 12 UTC): **OK**, 2024-05-15 .. 2024-06-20 daily 12 UTC (37 steps; 2800s; direct HTTPS chunk reads (fsspec timed out on 100 MB chunks); fills the e)

## 6. Why targets were missed

- **Heat and cold IoU went down in 7/13 cases.** The IMD criteria are daily and operational. The exact label is the instantaneous injected anomaly (>= 3 K), which shrinks at night by design (diurnal factor). The old z-score tracker follows that label more closely; v2 follows IMD's definition. Merging masks over each day (daily IoU in `reports/phase1_results.json`) narrows the gap but does not close it: on train+val, heat is old 0.51 vs v2 0.43 and cold is old 0.38 vs v2 0.32. The old tracker gets that IoU while producing 400-580 spurious tracks per case, against 5-14 for v2 (train+val).
- **heat_04 (TEST): IoU 0.** At 18 N in June, no mask cell reaches Tmax >= 40 C, so the synthetic +6 K dome is not an IMD heat wave; Phase 4 already showed 0 % of its cells meeting IMD. The label and the operational definition disagree for this case. It was diagnosed after evaluation and not tuned on.
- **cold_04 (TEST): IoU 0.** v2 does detect the event (a track at ~26 N 85 E). The main-track rule (largest summed area) instead picks a long-lived object near Herat (36 N 62 E), which the rough IMD-domain mask (lat <= 37 N, south of the Himalayan crest, no country boundary) does not exclude. A proper India boundary would fix this; it was not added, because it was found on a test case.
- **Spurious tracks.** The misses are heat_02 (11), cold_02 (14), cold_03 (12), heat_04 (27) and cold_04 (27). Most of the rest come from outside India (Afghanistan/Iran) for the same domain reason.
- **Rain.** One global map cannot fit storms of different intensity to one storm's (Amphan's) distribution. cyc_02, a weak Arabian Sea storm, ends up 49 % too dry, and amphan_replay went from +2 % to -32 %. Pooled p99 is within 20 % (train -6 %, held-out -18 %). Option B (intensity-aware) was worse on train/val.
- **Amphan real.** Mean error rose 40 -> 45 km because v2 keeps 23 of 26 fixes against 21 (the added ones are the harder landfall/decay fixes); the median is unchanged at 28 km, and v2 produced 0 other tracks in the window.

## Targets

- **Spurious tracks < 10 per case:** met in 8/13 cases (v2 total 129 vs old 6064).
- **Same or better IoU:** met in 6/13 cases; misses: heat_01, heat_02, cold_02, heat_03, cold_03, heat_04, cold_04.
- **Heat/cold IoU >= 0.4:** met in 2/8 cases (heat_01, heat_03).
- **Rain p99 within 20 % of IMD:** met in 2/5 cyclone cases (cyc_01, cyc_04). Pooled over storms, see the Phase 1 summary.
