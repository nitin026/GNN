# SUMMARY — SIH PS 26078 research + data foundation

Scope: the research and data groundwork for tracking extreme-weather anomalies in 3–10 day
ensemble forecasts, plus 12 km → 5 km amplitude-preserving downscaling (India box 0–40N,
60–100E). No models were trained. `make all` (or `python scripts/run_all.py all` on Windows
without make) reproduces everything with fixed seeds. `python scripts/verify_deliverables.py` checks
every deliverable.

## What was done
1. **Prior work** (`research/PRIOR_WORK.md`): 37 verified entries in 4 groups: benchmarks 12,
   tracking/EFI/NEPS-G 12, downscaling 7, public SIH-26078 repos 6. Every link was fetched by
   `scripts/check_links.py`, and all 37 return 2xx after redirects (`research/LINKS_CHECK.csv`).
   - 4 DOI links that returned 403 to scripts were replaced by their Crossref records.
   - 1 Zenodo entry was removed because it returned 403.
   - Brief items that could not be verified as stated were corrected: the IOP paper is 2026,
     "Ashish et al." NEPS-G does not exist, and the Zenodo IMD dataset is from Aug 2026.
2. **Real data** (`data/real/`, `data/SOURCES.md`), on grids G12 = 0.12° and G5 = 0.04°:
   - ERA5 for Amphan, the cold wave and the heatwave. Amphan came from WB2 and includes 850 hPa
     q/u/v/RH; the cold wave and heatwave came from ARCO-ERA5, because WB2 ends 2023-01-10.
   - The 1990–2019 ERA5 climatology.
   - IBTrACS v04r01 for the North Indian basin (Amphan plus a track library of 337 storms).
   - The Copernicus GLO-90 DEM (979 tiles).
   - IMD 0.25° rain for 2020 and 2024.
3. **Synthetic generator** (`synth/`) and **13 cases** (`data/synthetic/`): 4 cyclones on jittered
   real IBTrACS tracks, 4 heat domes, 4 cold waves, and an Amphan replay on the real ERA5 background.
   - Each case has 20 members, leads 0–240 h every 6 h, a 5 km truth and exact labels (masks,
     track, 4-D box, peak, member roles).
   - avgpool(5 km) == 12 km exactly.
   - Every file carries `synthetic="true"`.
4. **Realism check** (`reports/SYNTH_VALIDATION.md`) and **baseline** (`reports/BASELINE_RESULTS.md`),
   both with figures and tables. The test suite has 11 tests (`pytest -q`).

## Real data obtained vs skipped
| obtained (OK) | skipped / failed |
|---|---|
| ERA5 Amphan 2020-05-10..25 (sfc + 850 hPa q/u/v/RH) | 850 hPa for cold/heat windows (SKIPPED: ARCO 3-D chunks are ~100 MB/step) |
| ERA5 cold wave 2022-12-20..2023-01-20, heatwave 2024-05-15..06-20 (sfc, ARCO) | the brief's WB2 path (FAILED 404; the `wb13-` store was used instead) |
| ERA5 climatology mean (WB2) + std (1.5° WB2, ±15 d) | gcsfs anonymous access (FAILED: hangs on Windows; used HTTPS to the same bucket) |
| IBTrACS NI v04r01, Copernicus DEM GLO-90, IMD rain 2020 & 2024 | TIGGE NEPS-G, GPM IMERG, CDS ERA5-Land (SKIPPED: no keys in .env) |

## Dataset inventory (on disk, 11.3 GB total, under the 15 GB limit)
- **data/real**: 2.5 GB in total.
  - ERA5 events: 1.6 GB (0.25° native plus G12/G5, int16 packed).
  - Climatology: 0.69 GB.
  - IMD: 0.10 GB, DEM: 0.07 GB, IBTrACS: 0.03 GB.
- **data/synthetic**: 8.8 GB, 13 cases of 0.58–0.79 GB each (truth_5km ~0.2 GB, fcst_12km ~0.4–0.6 GB).
- NetCDF files are gitignored because they are reproducible; `labels.json` files and small CSVs are committed.

## Baseline numbers (z-score + threshold + label + Hungarian; EFI)
| | IoU (mean) | centroid / track error | EFI AUC @peak |
|---|---|---|---|
| synthetic cyclones (5 cases) | 0.45 | 9 km | 0.85 |
| synthetic heat domes (4) | 0.31 | 148 km | 0.81 |
| synthetic cold waves (4) | 0.19 | 252 km | 0.93 |
| **real Amphan: ERA5 G12 vs IBTrACS** | – | **mean 40 km, median 28 km (21 fixes)** | – |

## Known limitations
- The synthetic data is **not** real. Fine scales are spectral noise, not dynamics.
- Backgrounds are soft-clipped to 1.5σ, which makes the labels exact but the tails lighter.
- Cyclones are analytic Holland vortices. Local rain maxima are about 2× too intense (627 mm/day
  vs an IMD maximum of 318).
- Heat and cold events are surface-only 2-D blobs.
- The ensembles are prescribed perturbations, not a model.
- ERA5 winds are much weaker than the observed Amphan winds (26 vs 66 m/s): at 0.25°, real
  "truth" for extremes is itself smoothed.
- The baseline thresholds were set by hand once. The per-member tracker locks onto background-noise
  objects, giving hundreds of false-alarm tracks per case.
- No real ensemble forecast (NEPS-G) is available yet, because there is no TIGGE key.
- Python 3.12 was used instead of 3.11 (3.11 is not installed; an Application Control policy on
  this machine blocks newly built binary wheels, so the binary packages are pinned to the
  system-approved versions). G5 is 0.04° instead of 0.045°, to keep the block average exact.

## Next steps
1. Get a TIGGE key and pull NEPS-G (origin `dems`) for the three events; its variable names already
   match `fcst_12km.nc`.
2. **GNN tracker**: build an icosahedral or lat-lon mesh GNN over (member, lead, space) anomalies. Train
   it on the synthetic masks and 4-D boxes, then fine-tune on ERA5 plus IBTrACS or IMD event
   catalogues. Aim to beat this baseline on IoU, false-alarm tracks and track error.
3. **Diffusion downscaler** (12 → 5 km): train a CorrDiff-style residual diffusion model with a hard
   avgpool-conservation constraint (the property the data now guarantees). Validate against IMD
   0.0625° rain and against the spectra and p99 checks here.
4. Make the synthetic data more realistic: dynamical small scales (e.g. from a km-scale model), rain
   maxima calibrated to IMD, vertical structure for heat domes, and ensemble perturbations taken from
   real NEPS-G spread.
