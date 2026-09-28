# Object GNN v3 (BRIEF4 Phase 3)

All cases are **SYNTHETIC**. Heat/cold cases are now labelled and verified with IMD-consistent **daily** labels (scripts/imd_labels.py; reports/imd_labels.json), so the numbers below are not directly comparable with reports/GNN_RESULTS.md for heat and cold (cyclones use the same labels as before). TEST is only reported; decoding thresholds, calibrators and the early-warning threshold were chosen on VAL.

## Mean over TEST cases

| method | IoU | CSI | POD | FAR | spurious / case | cases with < 10 spurious |
|---|---|---|---|---|---|---|
| v2 | 0.19 | 0.36 | 0.48 | 0.52 | 11.0 | 10/13 (all splits) |
| te_style | 0.34 | 0.32 | 0.42 | 0.56 | 8.8 | 10/13 (all splits) |
| gnn_full | 0.03 | 0.56 | 0.68 | 0.24 | 5.8 | 12/13 (all splits) |
| gnn_temporal | 0.04 | 0.57 | 0.70 | 0.25 | 6.5 | 10/13 (all splits) |
| gnn_cross | 0.04 | 0.56 | 0.73 | 0.28 | 9.5 | 11/13 (all splits) |
| gnn_none | 0.03 | 0.59 | 0.70 | 0.23 | 6.0 | 9/13 (all splits) |

BRIEF2 object GNN (old edges, old labels) on TEST for reference: iou 0.21, csi 0.65, far 0.16

## Do the links help now? (edge ablations, TEST and VAL means)

| variant | TEST CSI | TEST FAR | TEST spurious | VAL CSI | VAL spurious | val AP node+edge | params |
|---|---|---|---|---|---|---|---|
| full | 0.56 | 0.24 | 5.8 | 0.77 | 4.3 | 1.957 | 170402 |
| temporal | 0.57 | 0.25 | 6.5 | 0.79 | 3.0 | 1.957 | 170402 |
| cross | 0.56 | 0.28 | 9.5 | 0.75 | 5.3 | 1.952 | 170402 |
| none | 0.59 | 0.23 | 6.0 | 0.80 | 3.7 | 1.949 | 170402 |

## Per case (TEST)

| case | v2 | te_style | gnn_full | gnn_temporal | gnn_cross | gnn_none |
|---|---|---|---|---|---|---|
| cyc_04 | CSI 0.35 / IoU 0.35 / spur 5 | CSI 0.22 / IoU 0.71 / spur 0 | CSI 0.55 / IoU 0.07 / spur 1 | CSI 0.53 / IoU 0.12 / spur 0 | CSI 0.53 / IoU 0.12 / spur 1 | CSI 0.54 / IoU 0.08 / spur 0 |
| heat_04 | CSI 0.01 / IoU 0.00 / spur 23 | CSI 0.00 / IoU 0.00 / spur 19 | CSI 0.37 / IoU 0.00 / spur 13 | CSI 0.37 / IoU 0.00 / spur 16 | CSI 0.34 / IoU 0.00 / spur 24 | CSI 0.40 / IoU 0.00 / spur 14 |
| cold_04 | CSI 0.42 / IoU 0.08 / spur 13 | CSI 0.34 / IoU 0.00 / spur 16 | CSI 0.62 / IoU 0.02 / spur 9 | CSI 0.65 / IoU 0.02 / spur 9 | CSI 0.69 / IoU 0.02 / spur 12 | CSI 0.66 / IoU 0.02 / spur 10 |
| amphan_replay | CSI 0.67 / IoU 0.34 / spur 3 | CSI 0.71 / IoU 0.65 / spur 0 | CSI 0.71 / IoU 0.02 / spur 0 | CSI 0.73 / IoU 0.02 / spur 1 | CSI 0.69 / IoU 0.02 / spur 1 | CSI 0.74 / IoU 0.03 / spur 0 |

## Calibration (GNN full, node event probability vs the truth)

Calibrator per lead band chosen on VAL by leave-one-case-out Brier: 0-72h: temperature (T = 2.20), 78-168h: temperature (T = 2.45), 174-240h: temperature (T = 1.43)

| lead band | n (TEST nodes) | reliability slope raw | slope calibrated | Brier raw | Brier cal | BSS cal vs climatology | BSS cal vs raw ensemble frequency | target slope 0.8-1.2 |
|---|---|---|---|---|---|---|---|---|
| 0-72h | 5239 | 0.67 | 0.72 | 0.0476 | 0.0431 | 0.51 | 0.91 | NO |
| 78-168h | 7353 | 0.69 | 0.76 | 0.0592 | 0.0520 | 0.51 | 0.86 | NO |
| 174-240h | 4477 | 0.60 | 0.62 | 0.0376 | 0.0360 | 0.29 | 0.88 | NO |

## Early-warning mode (threshold 0.8 chosen on VAL)

Alert at lead t when the calibrated probability that an event exists within [t, t+72 h] exceeds the threshold. Warning = hours between the first alert and the truth onset (valid time); v2 and EFI-only use the same 72 h look-ahead (v2: >= 50 % of members track an object; EFI-only: domain-max EFI >= 0.8).

| case | split | onset lead (h) | warning GNN-EW (h) | v2 (h) | EFI-only (h) | gained vs v2 | gained vs EFI |
|---|---|---|---|---|---|---|---|
| cyc_03 | val | 12 | 12 | 12 | 12 | 0 | 0 |
| heat_03 | val | 0 | 0 | 0 | 0 | 0 | 0 |
| cold_03 | val | 60 | 60 | 60 | 60 | 0 | 0 |
| cyc_04 | test | 36 | 36 | 36 | 36 | 0 | 0 |
| heat_04 | test | 78 | 78 | 30 | 78 | 48 | 0 |
| cold_04 | test | 42 | 18 | 42 | 42 | -24 | -24 |
| amphan_replay | test | 6 | 6 | 6 | 6 | 0 | 0 |

Every synthetic case contains an event, so false-alarm behaviour of the early-warning mode cannot be measured here; the onset in several cases is at lead 0-6 h, where no warning time is possible.

## Heat/cold labels and heat_04

| case | old label cells | IMD label cells | truth peak (C) | peak departure (C) | plains cells >= 40 C |
|---|---|---|---|---|---|
| heat_01 | 103024 | 88184 | 51.5 | 10.8 | 5586 |
| heat_02 | 15905 | 8900 | 47.5 | 9.3 | 1145 |
| cold_01 | 24721 | 1901 | 1.6 | 7.1 | - |
| cold_02 | 25560 | 1358 | 1.3 | 6.8 | - |
| heat_03 | 75521 | 50676 | 48.6 | 10.5 | 5564 |
| cold_03 | 13124 | 2479 | 0.8 | 7.5 | - |
| heat_04 | 34320 | 142 | 40.5 | 7.3 | 10 |
| cold_04 | 18753 | 898 | 1.6 | 7.1 | - |

heat_04 meets the IMD heat-wave criteria only marginally (peak Tmax 40.5 C, 10 plains cells >= 40 C), which is why tracker v2 (IMD 40 C rule) never fires on it; the case is kept as generated rather than edited after seeing TEST results.


## Simplification (links add little)

Because the edge ablations show no benefit from message passing, a node-only model with hidden 32 (`none_small`, 40402 parameters vs 170402 for full) was trained on the same graphs: TEST CSI 0.60, FAR 0.19, spurious 4.2 per case; inference over the VAL+TEST graphs 0.39 s vs 2.43 s for full. Its own lead-band calibration (fitted on VAL, calibration_none_small.json) gives TEST reliability slopes 0-72h 0.76, 78-168h 0.82, 174-240h 0.60. **The operational pipeline therefore uses the simplified model** (pipeline/run_operational.py, scripts/export_extras.py); whether cross-member links help on real NEPS-G ensembles is untested.

## What did not work

- Calibration slope target missed in 0-72h: 0.72.
- Calibration slope target missed in 78-168h: 0.76.
- Calibration slope target missed in 174-240h: 0.62.
- Spurious-track target (< 10 per case in 13/13) missed in: heat_04.
- No lead-time gain over v2 on TEST cases cyc_04, cold_04, amphan_replay.
- Links still add little on TEST (full CSI 0.56 vs node-only 0.59).
- The TempestExtremes warm-core criterion needs 300-500 hPa temperature, which the synthetic cases do not have; a moist-core proxy (RH850 >= 80 %) is used for cyclones.
