# BRIEF — SIH PS 26078 (MoES / NCMRWF)
AI-driven spatio-temporal tracking of extreme weather anomalies in 3–10 day ensemble forecasts,
plus amplitude-preserving downscaling 12 km -> 5 km. Region: 0–40N, 60–100E (India box).
This session builds the RESEARCH + DATA foundation only (no heavy model training yet).

## Rules
- Work phase by phase in order. Commit to git after each phase with a clear message.
- Never present synthetic data as real. Every synthetic file carries the global attribute
  `synthetic = "true"` and lives under data/synthetic/. Real data lives under data/real/.
- Never invent a citation or URL. Every link in research/ must be fetched and checked;
  record the HTTP status. If a paper can't be verified, drop it.
- Never commit secrets or .env. Keep total downloaded data under 15 GB (India box, 6-hourly).
- If a real source fails twice (auth, 403, timeout), log it in data/SOURCES.md with the
  error and move on. Fall back to synthetic for that piece; do not stall.
- Fix all random seeds. Everything must reproduce with `make all`.
- Python 3.11, xarray, dask, zarr, gcsfs, s3fs, netCDF4, numpy, scipy, pandas, matplotlib,
  cartopy (optional), pytest. Pin versions in requirements.txt.

## Phase 1 — Prior-work research  -> research/PRIOR_WORK.md
Search the web and build a table of at least 20 verified entries in 4 groups, each with:
title, year, authors/org, link, what they did, data used, key metric/result,
what we can reuse, and what gap it leaves for us.
  a) Competitions / benchmarks: WeatherBench 2, ExtremeWeatherBench (Brightband),
     ClimateNet and ClimateNetLarge, ECMWF AI Weather Quest, plus any Kaggle, NeurIPS,
     AGU or ISRO/IMD/NCMRWF hackathons on extreme weather, cyclone tracking or downscaling.
  b) Papers, tracking and ensembles: GraphCast (icosahedral GNN), GenCast, the ECMWF Extreme
     Forecast Index (Lalaurette 2003; Zsoter 2006), TempestExtremes / cyclone-tracking
     algorithms, NEPS-G verification papers from NCMRWF.
  c) Papers, downscaling: CorrDiff (Mardani et al. 2025), DDPM and DDIM, diffusion or GAN
     downscaling over India (e.g. Chandel et al. 2025 JGR; the IOP 2025 "robust deep
     learning downscaling ... Indian subcontinent" paper; the Zenodo 0.0625° IMD diffusion
     dataset), and physics-informed losses.
  d) Existing public SIH-26078 repos on GitHub (search "SIH26078", "SIH 26078",
     "26078 NCMRWF"; known examples: krishnendukoley2007-arch/aero-track-4d,
     yadavayush834/avarta_test, sauravrajput2124-cmyk/SIH26078-weather-anomaly-ai).
     For each one record its data (real or synthetic), models, reported metrics and weak
     spots. Do NOT copy their code.
End the file with "Our differentiation": 5–8 bullets on what we will do that none of them do.
Also write research/LINKS_CHECK.csv (url, http_status, checked_at).

## Phase 2 — Real data (no-login sources first)  -> data/real/, data/SOURCES.md
Try in this order, and log each attempt in SOURCES.md (source, url, status, size, time range):
  1. ERA5 via WeatherBench 2 (anonymous GCS, token="anon"):
     gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-1440x721_with_derived_variables.zarr
     Slice the India box for: Cyclone Amphan (2020-05-10..2020-05-25), N-India heatwave
     (2024-05-15..2024-06-20; if WB2 ends before 2024, use ARCO-ERA5
     gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3), and the cold
     wave (2022-12-20..2023-01-20). Variables: 2m_temperature, 10m u/v wind,
     mean_sea_level_pressure, total_precipitation (6h), plus 850 hPa specific humidity, u, v.
  2. ERA5 climatology (anonymous): gs://weatherbench2/datasets/era5-hourly-climatology/
     1990-2019_6h_1440x721.zarr -> India box -> data/real/clim/.
  3. IBTrACS v4 North Indian basin CSV from NOAA NCEI -> extract Amphan (2020).
  4. Copernicus DEM GLO-90 (anonymous AWS s3://copernicus-dem-90m) -> mosaic the India box ->
     regrid to 5 km and 12 km.
  5. IMD 0.25° gridded rainfall for 2020 and 2024 via the `imdlib` Python package.
  6. Optional, only if keys exist in .env: TIGGE NEPS-G (origin "dems", 11+1 members,
     from Aug 2017) for the three events; GPM IMERG daily 0.1°; CDS ERA5-Land.
     If a key is missing, write "SKIPPED – no key" and continue.
Convert everything to CF-compliant NetCDF on common grids: G12 = 0.12° (~12 km) and
G5 = 0.045° (~5 km) over the India box. Write data/real/README.md.

## Phase 3 — Synthetic generator  -> synth/ package, data/synthetic/
Goal: physically plausible events with EXACT ground truth, merged onto real backgrounds.
  - Background: real ERA5 fields (from Phase 2) interpolated to G5. If Phase 2 failed, use a
    synthetic climatology (seasonal cycle + latitude gradient + monsoon trough) and flag it.
  - Event injectors (each a documented function with parameters):
    * Tropical cyclone: Holland (1980) wind profile, MSLP depression, spiral rain bands, eye;
      the track is sampled from real IBTrACS NI tracks, with jitter; intensity lifecycle
      (genesis -> peak -> landfall decay).
    * Heat dome: slow-moving T2m anomaly of +4 to +8 °C with a realistic spatial shape,
      lasting 5–12 days.
    * Cold wave: T2m anomaly of −4 to −8 °C over N-India plains, advected from the NW.
  - Fine-scale realism: add multiscale noise with a k^(−5/3) spectrum, plus orographic
    rain enhancement from the DEM (upslope wind · ∇terrain > 0).
  - Physics consistency: rain placed only where 850 hPa moisture-flux convergence > 0;
    rain ≥ 0; RH ≤ 100%.
  - Resolutions: generate the 5 km "truth", then derive 12 km by CONSERVATIVE block averaging
    (so avgpool(5 km) == 12 km exactly; test this).
  - Ensembles: 20 members per case; the spread of track and intensity grows with lead time
    (lead 0–240 h, 6-hourly). Include a few "miss" and "false alarm" members.
  - Labels saved with every case: per-timestep event mask, centroid track, 4D bounding box
    (lat, lon, level, time), peak value, and hazard type.
  - Size: at least 3 hazard types × 4 cases each (≥ 12 cases), plus 1 case that is a
    synthetic replay of Amphan on real ERA5 background (the "merged real+synthetic" case).
  - Output: data/synthetic/{case_id}/{fcst_12km.nc, truth_5km.nc, labels.json}, with variable
    names matching NEPS-G conventions so real data can drop in later.

## Phase 4 — Realism check  -> reports/SYNTH_VALIDATION.md (+ PNG figures)
Compare synthetic against real (ERA5/IMD/IBTrACS) on: value histograms, 99th percentile,
radially averaged power spectrum, cyclone radial wind profile vs. the Amphan observed
radius of maximum wind / Vmax, and heatwave anomaly magnitude vs. IMD criteria.
State plainly where synthetic data is unrealistic.

## Phase 5 — Smoke-test pipeline  -> pipeline/, tests/
  - anomaly.py: z-score against the climatology; efi.py: ECMWF EFI formula
    (unit test: an ensemble drawn from climate gives EFI ≈ 0; one shifted +3σ gives EFI > 0.8).
  - track.py: threshold + scipy.ndimage.label + Hungarian matching -> tracks and 4D boxes.
  - Evaluate the tracker on all synthetic cases (IoU, centroid error km) and on real Amphan
    ERA5 vs. IBTrACS (track error km). Write reports/BASELINE_RESULTS.md.
  - scripts/verify_deliverables.py checks every deliverable in this brief and prints
    "ALL CHECKS PASSED" or lists what is missing.
  - Makefile targets: research, data-real, data-synth, validate, baseline, test, all.

## Final summary  -> SUMMARY.md
One page: what was done, what real data was obtained vs. skipped, dataset inventory (sizes),
baseline numbers, known limitations, and next steps (GNN tracker, diffusion downscaler).