# Results: headline table (BRIEF2 Phase 6)

Built by `scripts/make_results.py` from `reports/*.json`. **Synthetic** rows are SYNTHETIC TEST cases (cyc_04, heat_04, cold_04, amphan_replay; exact labels, never used for training or tuning). **Real** rows are ERA5 reanalysis against the IBTrACS best track or IMD gridded rain.

## 1. Tracking (SYNTHETIC TEST, mean of 4 cases)

| method | IoU | CSI | POD | FAR | centroid err (km) | spurious tracks / case |
|---|---|---|---|---|---|---|
| baseline (BRIEF Phase 5: z-score + label + Hungarian) | 0.33 | – | – | – | 84 | 468 |
| tracker v2 (hysteresis + Kalman + IMD criteria) | 0.17 | 0.32 | 0.48 | 0.58 | 725 | 15.5 |
| TempestExtremes-style | 0.00 | 0.09 | 0.44 | 0.89 | 2599 | 176.8 |
| **GNN tracker (full)** | 0.21 | 0.65 | 0.74 | 0.16 | 148 | 6.0 |
| GNN (temporal edges only) | 0.21 | 0.66 | 0.74 | 0.15 | 154 | 2.5 |
| node-only MLP (ablation) | 0.19 | 0.65 | 0.74 | 0.16 | 122 | 3.5 |

Baseline POD/FAR/CSI were not computed in BRIEF Phase 5 (it scored IoU and centroid error only).

## 2. Real cyclone track: Amphan 2020, ERA5 G12 vs IBTrACS (REAL)

| method | matched 6-h fixes | mean err (km) | median (km) | max (km) |
|---|---|---|---|---|
| tracker v2 | 23 | 45 | 28 | 195 |
| TempestExtremes-style | 25 | 55 | 30 | 216 |
| GNN full | 23 | 45 | 28 | 195 |
| **GNN temporal** (like-for-like: ERA5 is one run) | 22 | 38 | 28 | 133 |

## 3. Downscaling 12 -> 5 km (SYNTHETIC TEST, rain)

| model | RMSE (mm/6h) | CRPS | p99 ratio | 10 km power ratio | patch-peak ratio | conservation err |
|---|---|---|---|---|---|---|
| bicubic+lapse | 1.577 | 0.352 | 0.988 | 0.064 | 0.812 | 1.6e-05 |
| unet | 1.674 | 0.396 | 0.996 | 0.245 | 0.860 | 2.1e-05 |
| diffusion_mean | 1.639 | 0.316 | 0.993 | 0.152 | 0.869 | 1.8e-05 |
| diffusion_sample | 1.995 | 0.616 | 1.001 | 0.614 | 1.051 | 2.0e-05 |

REAL IMD 0.25 deg perfect-model check (coarsened 3x, downscaled back, JJAS 2020): RMSE bicubic+lapse 4.36, unet 4.39, diffusion_mean 4.36, diffusion_sample 4.45 mm/day (out of distribution for all learned models).

## 4. Inference time per forecast cycle (CPU)

| case | members x leads | total (s) | slowest stage | peak RSS (MB) |
|---|---|---|---|---|
| amphan_replay | 20 x 41 | 122 | 3 candidate objects + graph (38 s) | 5844.9 |

Measured by `python scripts/demo.py --case <case>` (`reports/demo_timing.json`).

## 5. Public SIH-26078 repositories (their self-reported numbers)

Copied from `research/PRIOR_WORK.md` (section d; every link fetched). They are **not** directly comparable: different cases, data and metric definitions. The 'data' column says whether their numbers come from real or synthetic data.

| repo | data | tracking (reported) | downscaling (reported) | caveat |
|---|---|---|---|---|
| [AERO-TRACK 4D (krishnendukoley2007-arch)](https://github.com/krishnendukoley2007-arch/aero-track-4d) | REAL ERA5 + IBTrACS, Amphan only (13 hourly steps); Fani/Yaas on 'ERA5-equivalent synthetic' data | mean track error 50.9 km (Amphan), landfall error 8.7 km | peak wind 83.9 % of ERA5; CRPS 7.26 km/h; FSS 0.072 | one storm, 13 steps; the '5 km target' is 25 km ERA5, so no true 12->5 km pair; no conservation check |
| [Avarta / avarta_test (yadavayush834, diiviikk5)](https://github.com/yadavayush834/avarta_test) | REAL NOAA GEFS 0.5 deg (control + 4 members, one init), IMD 0.25 deg, CHIRPS | heavy-rain IoU 0.00 (missed); ensemble-mean peak 44.6 vs IMD 469.2 mm/day | CNN vs bilinear peak error 61.5 vs 84.9 mm/day; IoU 0.37 vs 0.26 | 5 members, one case; GNN and DDPM untrained |
| [ChakraNet (ErrGuhan)](https://github.com/ErrGuhan/Chakra-Net) | SYNTHETIC 50-member ensemble built from the Phailin best track + IMERG | no IoU or km-error tracking benchmark reported | high-frequency power 98.7 % vs 50.5 % bilinear; p99 bias 5.72 % vs 4.87 % bilinear; CRPS 3.52 vs 4.21 mm | one storm; conservation losses are stubs |
| [SIH26078-weather-anomaly-ai (sauravrajput2124-cmyk)](https://github.com/sauravrajput2124-cmyk/SIH26078-weather-anomaly-ai) | SYNTHETIC 10-day demo JSON | none reported | none reported | no ML model, no real data |
| [MEGHA-DRISHTI (Srujanmirji)](https://github.com/Srujanmirji/MEGHA-DRISHTI) | 'DEMO DATA - illustrative' | none reported | none reported | front end only |

Like for like, only AERO-TRACK 4D reports a real Amphan track error (50.9 km over 13 hourly steps). Ours is 38 km over 22 six-hourly fixes of the ERA5 run; different time sampling and a different matching rule, so the two numbers are indicative only.

## What did not work

- Area overlap: GNN TEST IoU 0.21 (v2 0.17) is BELOW the original Phase 5 baseline's truth-run IoU 0.33 (centroid error 84 km vs 148 km). The baseline gets there with ~470 spurious ensemble tracks per case, but on overlap alone it is better. The GNN links candidate objects and does not re-segment them (a mesh-GNN segmentation head is BRIEF4 Phase 2).
- Edges add little: the node-only MLP matches the full GNN on TEST CSI (0.65 vs 0.65).
- Downscaling p99 contrast: diffusion rain p99 ratio 1.001 vs U-Net 0.996; the advantage is only in 10 km power and patch peaks.
- GNN probabilities are over-confident against the truth at days 7-10 (reports/GNN_RESULTS.md).
- No real ensemble is scored in this table: tracking skill on real data is one ERA5 run of one storm. The multi-storm real evaluation and the real IFS ENS / GenCast ensembles are BRIEF4 Phase 1.
