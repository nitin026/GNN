# Real-data evaluation (BRIEF4 Phase 1)

Everything in this report is **REAL**: WeatherBench 2 public ensemble forecasts, ERA5 reanalysis and IBTrACS best tracks. The trackers were trained or tuned on SYNTHETIC data only. REAL-TEST events are locked in data/real/SPLITS.json and were NOT evaluated here.

Amphan landfall (first IBTrACS fix with DIST2LAND <= 0 after peak): 2020-05-20 12:00:00.

## A. Real ensemble forecasts of Amphan

The forecasts are regridded bilinearly to G12 and tracked member by member. A member "detects" Amphan when one of its tracks comes within 500 km of IBTrACS at a common time; the closest such track is used.
- `p_detect_any`: the fraction of members that detect Amphan at any lead.
- Per-lead columns: `p` = fraction of members with a matched position at that lead; errors (km) of the ensemble-mean position and of the median member.
- GenCast has 12-hourly leads (odd 6-h steps are empty).

### ifs_ens_1p5 (50 members, 6 h steps)

#### Track error by lead time (km)

Error of the ensemble-mean position of the matched member tracks, averaged over all inits that have an IBTrACS fix at that valid time; p = mean fraction of members with a matched position; n = inits:

| method | 24 h | 48 h | 72 h | 120 h | 168 h | 240 h |
|---|---|---|---|---|---|---|
| v2 | 77 (p 1.00, n=6) | 118 (p 0.99, n=6) | 129 (p 0.94, n=6) | 153 (p 0.84, n=6) | 236 (p 0.69, n=6) | 346 (p 0.34, n=4) |
| gnn_full | 77 (p 0.92, n=6) | 120 (p 0.90, n=6) | 131 (p 0.85, n=6) | 158 (p 0.72, n=6) | 247 (p 0.48, n=6) | 350 (p 0.15, n=4) |
| gnn_temporal | 77 (p 0.93, n=6) | 119 (p 0.90, n=6) | 130 (p 0.85, n=6) | 169 (p 0.65, n=6) | 244 (p 0.45, n=6) | 354 (p 0.15, n=4) |
| te_style | 78 (p 1.00, n=6) | 119 (p 0.98, n=6) | 130 (p 0.96, n=6) | 155 (p 0.86, n=6) | 252 (p 0.71, n=6) | 365 (p 0.38, n=4) |

#### P ≥ 0.5 first-flag lead time

The earliest forecast init at which at least half of the members detect Amphan (a track within 500 km of IBTrACS at a common time), in hours before landfall and before the first IBTrACS fix (genesis):

| method | earliest init flagged | hours before landfall | hours before first IBTrACS fix | P at that init |
|---|---|---|---|---|
| v2 | 2020-05-11 00:00:00 | 228 | 102 | 0.74 |
| gnn_full | 2020-05-11 00:00:00 | 228 | 102 | 0.56 |
| gnn_temporal | 2020-05-11 00:00:00 | 228 | 102 | 0.56 |
| te_style | 2020-05-11 00:00:00 | 228 | 102 | 0.78 |

Detection fraction by init:

| init | v2 | gnn_full | gnn_temporal | te_style |
|---|---|---|---|---|
| 2020-05-08 00:00:00 | 0.10 | 0.02 | 0.04 | 0.14 |
| 2020-05-09 00:00:00 | 0.44 | 0.22 | 0.22 | 0.46 |
| 2020-05-10 00:00:00 | 0.36 | 0.32 | 0.30 | 0.38 |
| 2020-05-11 00:00:00 | 0.74 | 0.56 | 0.56 | 0.78 |
| 2020-05-12 00:00:00 | 0.96 | 0.92 | 0.90 | 0.96 |
| 2020-05-13 00:00:00 | 0.98 | 0.98 | 0.98 | 0.98 |
| 2020-05-14 00:00:00 | 1.00 | 1.00 | 0.98 | 1.00 |
| 2020-05-15 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-16 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-17 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-18 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-19 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-20 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |

### gencast_1p5 (56 members, 12 h steps)

#### Track error by lead time (km)

Error of the ensemble-mean position of the matched member tracks, averaged over all inits that have an IBTrACS fix at that valid time; p = mean fraction of members with a matched position; n = inits:

| method | 24 h | 48 h | 72 h | 120 h | 168 h | 240 h |
|---|---|---|---|---|---|---|
| v2 | 45 (p 0.83, n=6) | 94 (p 1.00, n=6) | 154 (p 0.96, n=6) | 251 (p 0.86, n=6) | 269 (p 0.72, n=6) | 328 (p 0.56, n=4) |
| gnn_full | 58 (p 0.97, n=6) | 94 (p 0.99, n=6) | 152 (p 0.94, n=6) | 243 (p 0.78, n=6) | 323 (p 0.58, n=6) | 368 (p 0.42, n=4) |
| gnn_temporal | 58 (p 0.98, n=6) | 94 (p 0.99, n=6) | 153 (p 0.95, n=6) | 250 (p 0.77, n=6) | 365 (p 0.57, n=6) | 363 (p 0.38, n=4) |
| te_style | 58 (p 1.00, n=6) | 94 (p 1.00, n=6) | 152 (p 0.95, n=6) | 233 (p 0.83, n=6) | 267 (p 0.68, n=6) | 328 (p 0.56, n=4) |

#### P ≥ 0.5 first-flag lead time

The earliest forecast init at which at least half of the members detect Amphan (a track within 500 km of IBTrACS at a common time), in hours before landfall and before the first IBTrACS fix (genesis):

| method | earliest init flagged | hours before landfall | hours before first IBTrACS fix | P at that init |
|---|---|---|---|---|
| v2 | 2020-05-09 00:00:00 | 276 | 150 | 0.66 |
| gnn_full | 2020-05-09 00:00:00 | 276 | 150 | 0.59 |
| gnn_temporal | 2020-05-09 00:00:00 | 276 | 150 | 0.57 |
| te_style | 2020-05-09 00:00:00 | 276 | 150 | 0.68 |

Detection fraction by init:

| init | v2 | gnn_full | gnn_temporal | te_style |
|---|---|---|---|---|
| 2020-05-08 00:00:00 | 0.36 | 0.21 | 0.21 | 0.36 |
| 2020-05-09 00:00:00 | 0.66 | 0.59 | 0.57 | 0.68 |
| 2020-05-10 00:00:00 | 0.71 | 0.68 | 0.52 | 0.73 |
| 2020-05-11 00:00:00 | 0.95 | 0.88 | 0.82 | 0.95 |
| 2020-05-12 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-13 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-14 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-15 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-16 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-17 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-18 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-19 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2020-05-20 00:00:00 | 0.00 | 0.88 | 0.88 | 0.98 |

### ifs_ens_0p25 (50 members, 6 h steps)

#### Track error by lead time (km)

Error of the ensemble-mean position of the matched member tracks, averaged over all inits that have an IBTrACS fix at that valid time; p = mean fraction of members with a matched position; n = inits:

| method | 24 h | 48 h | 72 h | 120 h | 168 h | 240 h |
|---|---|---|---|---|---|---|
| v2 | 26 (p 1.00, n=1) | 82 (p 1.00, n=1) | 106 (p 1.00, n=1) | 160 (p 0.90, n=1) | n/a | n/a |
| gnn_full | 21 (p 0.94, n=1) | 79 (p 0.94, n=1) | 103 (p 0.94, n=1) | 137 (p 0.80, n=1) | n/a | n/a |
| gnn_temporal | 22 (p 0.96, n=1) | 79 (p 0.96, n=1) | 103 (p 0.96, n=1) | 112 (p 0.78, n=1) | n/a | n/a |
| te_style | 26 (p 1.00, n=1) | 82 (p 1.00, n=1) | 106 (p 1.00, n=1) | 114 (p 1.00, n=1) | n/a | n/a |

#### P ≥ 0.5 first-flag lead time

The earliest forecast init at which at least half of the members detect Amphan (a track within 500 km of IBTrACS at a common time), in hours before landfall and before the first IBTrACS fix (genesis):

| method | earliest init flagged | hours before landfall | hours before first IBTrACS fix | P at that init |
|---|---|---|---|---|
| v2 | 2020-05-16 00:00:00 | 108 | -18 | 1.00 |
| gnn_full | 2020-05-16 00:00:00 | 108 | -18 | 1.00 |
| gnn_temporal | 2020-05-16 00:00:00 | 108 | -18 | 1.00 |
| te_style | 2020-05-16 00:00:00 | 108 | -18 | 1.00 |

Detection fraction by init:

| init | v2 | gnn_full | gnn_temporal | te_style |
|---|---|---|---|---|
| 2020-05-16 00:00:00 | 1.00 | 1.00 | 1.00 | 1.00 |

## B. REAL-VAL cyclones in ERA5 reanalysis (single run)

| event | method | matched fixes / IBTrACS fixes | mean err (km) | median err (km) | tracks |
|---|---|---|---|---|---|
| amphan_2020 | v2 | 23/26 | 45 | 28 | 1 |
| amphan_2020 | te_style | 25/26 | 55 | 30 | 23 |
| amphan_2020 | gnn_full | 23/26 | 45 | 28 | 1 |
| amphan_2020 | gnn_temporal | 22/26 | 38 | 28 | 1 |
| fani_2019 | v2 | 36/36 | 50 | 37 | 1 |
| fani_2019 | te_style | 32/36 | 43 | 34 | 18 |
| fani_2019 | gnn_full | 31/36 | 43 | 34 | 3 |
| fani_2019 | gnn_temporal | 30/36 | 44 | 35 | 2 |
| gulab_2021 | v2 | 15/43 | 38 | 33 | 1 |
| gulab_2021 | te_style | 36/43 | 49 | 40 | 22 |
| gulab_2021 | gnn_full | 5/43 | 18 | 16 | 4 |
| gulab_2021 | gnn_temporal | 6/43 | 57 | 52 | 1 |
| nisarga_2020 | v2 | 10/15 | 28 | 20 | 1 |
| nisarga_2020 | te_style | 13/15 | 42 | 40 | 12 |
| nisarga_2020 | gnn_full | 12/15 | 32 | 34 | 1 |
| nisarga_2020 | gnn_temporal | 12/15 | 32 | 34 | 1 |
| nivar_2020 | v2 | 16/22 | 45 | 38 | 1 |
| nivar_2020 | te_style | 20/22 | 48 | 52 | 10 |
| nivar_2020 | gnn_full | 17/22 | 45 | 38 | 4 |
| nivar_2020 | gnn_temporal | 17/22 | 45 | 38 | 4 |
| tauktae_2021 | v2 | 23/25 | 40 | 26 | 1 |
| tauktae_2021 | te_style | 23/25 | 40 | 26 | 15 |
| tauktae_2021 | gnn_full | 23/25 | 199 | 29 | 2 |
| tauktae_2021 | gnn_temporal | 19/25 | 36 | 25 | 3 |
| vayu_2019 | v2 | 32/39 | 37 | 26 | 1 |
| vayu_2019 | te_style | 35/39 | 37 | 27 | 21 |
| vayu_2019 | gnn_full | 33/39 | 41 | 27 | 2 |
| vayu_2019 | gnn_temporal | 30/39 | 36 | 26 | 1 |
| yaas_2021 | v2 | 17/20 | 47 | 49 | 1 |
| yaas_2021 | te_style | 19/20 | 59 | 51 | 14 |
| yaas_2021 | gnn_full | 19/20 | 59 | 51 | 3 |
| yaas_2021 | gnn_temporal | 16/20 | 56 | 51 | 2 |

## C. REAL-VAL heat and cold waves in ERA5 reanalysis (single run)

Reference = IMD operational criteria (departure >= 4.5 C from the ERA5 1990-2019 normal + regional absolute Tmax/Tmin threshold) applied to ERA5 daily Tmax/Tmin, persisting >= 2 days and covering >= 50,000 km2. **This reference comes from the same reanalysis the trackers read, so it is not an independent truth**; tracker v2 thresholds the same field and is close to the reference by construction. Step-level scores (event present anywhere in the domain), IoU on hit steps.

| event | method | ref steps / steps | POD | FAR | CSI | IoU | tracks |
|---|---|---|---|---|---|---|---|
| cold_2019 | v2 | 28/108 | 1.00 | 0.48 | 0.52 | 0.36 | 3 |
| cold_2019 | gnn_temporal | 28/108 | 0.18 | 0.89 | 0.07 | 0.13 | 11 |
| cold_2019 | te_style | 28/108 | 1.00 | 0.12 | 0.88 | 0.65 | 1 |
| heat_2019 | v2 | 88/88 | 1.00 | 0.00 | 1.00 | 0.62 | 4 |
| heat_2019 | gnn_temporal | 88/88 | 0.51 | 0.00 | 0.51 | 0.25 | 5 |
| heat_2019 | te_style | 88/88 | 1.00 | 0.00 | 1.00 | 0.83 | 8 |

ERA5 vs IMD gridded temperature inside the reference event (independent check of the input):

| event | IMD check | cell-days | bias ERA5 - IMD (C) | MAE (C) |
|---|---|---|---|---|
| cold_2019 | ok | 13715 | 0.35 | 1.78 |
| heat_2019 | ok | 79461 | -0.72 | 1.43 |

Peak RSS 3217 MB.

## What did not work

- gencast_1p5_2020050800 gnn_full: only 0.21 of members detect Amphan.
- gencast_1p5_2020050800 v2: only 0.36 of members detect Amphan.
- gencast_1p5_2020052000 v2: only 0.00 of members detect Amphan.
- ifs_ens_1p5_2020050800 gnn_full: only 0.02 of members detect Amphan.
- ifs_ens_1p5_2020050800 v2: only 0.10 of members detect Amphan.
- ifs_ens_1p5_2020050900 gnn_full: only 0.22 of members detect Amphan.
- ifs_ens_1p5_2020050900 v2: only 0.44 of members detect Amphan.
- ifs_ens_1p5_2020051000 gnn_full: only 0.32 of members detect Amphan.
- ifs_ens_1p5_2020051000 v2: only 0.36 of members detect Amphan.
- gulab_2021 v2: matched 15/43 IBTrACS fixes.
- gulab_2021 gnn_full: matched 5/43 IBTrACS fixes.
- gulab_2021 gnn_temporal: matched 6/43 IBTrACS fixes.
- cold_2019 gnn_temporal: POD 0.18, FAR 0.89 against the ERA5 IMD-criteria reference.
