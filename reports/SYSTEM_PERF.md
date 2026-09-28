# System performance (BRIEF3 Phase D)

Measured 2026-09-28T15:25:41 on Intel64 Family 6 Model 186 Stepping 3, GenuineIntel (Windows), CPU only, no GPU by `scripts/measure_perf.py` (API served by uvicorn, called over real HTTP from the same machine).

## API latency

| endpoint | n | p50 (ms) | p95 (ms) | max (ms) | response size |
|---|---|---|---|---|---|
| GET /health | 40 | 1.1 | 1.3 | 1.5 | 0.1 kB |
| GET /cases | 40 | 1.2 | 1.4 | 1.8 | 3.8 kB |
| GET /cases/{id} | 40 | 2.0 | 2.3 | 2.3 | 15.0 kB |
| GET /cases/{id}/tracks?lead=96 | 40 | 3.8 | 5.2 | 19.9 | 16.5 kB |
| GET /cases/{id}/bbox4d | 40 | 1.1 | 1.7 | 6.2 | 0.6 kB |
| GET /cases/{id}/fields/wind (PNG 12 km) | 40 | 1.6 | 3.2 | 4.2 | 51.8 kB |
| GET /cases/{id}/fields/wind (PNG 5 km) | 40 | 2.6 | 3.6 | 4.5 | 263.7 kB |
| GET /alerts?case&lead | 40 | 1.5 | 1.6 | 2.1 | 2.5 kB |
| GET /alerts?lat&lon (point query) | 40 | 3.2 | 4.0 | 4.2 | 21.6 kB |
| GET /alerts (all cases) | 40 | 12.2 | 15.8 | 31.4 | 119.7 kB |
| GET /alerts/districts?lead | 40 | 2.6 | 3.3 | 3.5 | 0.2 kB |
| GET /alerts/districts (all leads) | 40 | 52.7 | 60.3 | 63.2 | 25.5 kB |

Worst endpoint p95: **60 ms**.

## End-to-end inference through the API (POST /run)

Job on `cyc_04` (20 members x 41 leads, 12 km): status **done**, job time **94 s** (wall incl. polling 95 s), 27.5 MB of products written.

| stage | seconds |
|---|---|
| setup | 0.9 |
| load | 10.3 |
| load | 0.1 |
| candidate graph | 23.2 |
| GNN tracking | 2.5 |
| downscaling + images | 44.8 |
| alerts | 9.6 |

## Forecast cycle without image export (scripts/demo.py)

| case | members x leads | total (s) | peak RSS (MB) | stages |
|---|---|---|---|---|
| amphan_replay | 20 x 41 | 122 | 5844.9 | setup: climatology, DEM, masks, models 2s; 1 load 12 km ensemble NetCDF 15s; 2 z-score anomaly + EFI (all leads) 16s; 3 candidate objects + graph 38s; 4 GNN inference + track decoding 1s; 5 U-Net downscaling 12->5 km (member 0, 41 leads) 37s; 6 alerts (ensemble probability x IMD thresholds) 14s |

## What did not work

- Everything runs on one CPU (no GPU); a Colab T4 timing is BRIEF4 Phase 5.
- Peak RAM of a forecast cycle is several GB because the whole 20-member ensemble is held in memory; streaming per member/lead is BRIEF4 Phase 5.
- The API runs one pipeline job at a time; a second POST /run queues.
