# data/real — real, no-login data (India box 0–40N, 60–100E)

Every file here is **real** (global attribute `synthetic = "false"` on NetCDF files). Fetched by
`scripts/fetch_real.py` (`make data-real`); the attempt log is `data/SOURCES.md`.

## Grids
- **0p25**: native ERA5 0.25° box (161 × 161, latitude ascending).
- **G12**: 0.12° (~12 km, NEPS-G native), 333 × 333 cell centres 0.06…39.90N, 60.06…99.90E.
- **G5**: 0.04° (~4.4 km), 999 × 999 cell centres 0.02…39.94N, 60.02…99.94E.
  Exactly 3 × 3 G5 cells per G12 cell, so avgpool(G5) = G12 holds exactly. The brief suggested
  0.045°, but 0.12/0.045 is not an integer and an exact block average would be impossible.
- ERA5 → G12/G5 is **bilinear interpolation** (it adds no information below ~30 km). The DEM is a
  **conservative block average** of a 0.01° mosaic.
- Regridded G12/G5 ERA5 files are packed as int16 (scale/offset): 0.01 K, 0.01 m/s, 1 Pa,
  0.01 mm, 1e-6 kg/kg, 0.01 %.

## Variables (CF names/units)
`t2m` K, `u10`/`v10` m s-1, `msl` Pa, `tp` mm accumulated over the previous 6 h (ERA5
`total_precipitation_6hr` for WB2; the sum of 6 hourly ARCO accumulations for 2022–24), and for Amphan
only `q850` kg/kg, `u850`/`v850` m/s and `r850` % (WB2-derived RH, which can exceed 100 %).

## Contents
| file | MB | dims / vars |
|---|---|---|
| `clim/era5_clim_0p25.nc` | 139.8 | hour=4, dayofyear=94, latitude=161, longitude=161; vars: t2m_mean, msl_mean, tp_mean, t2m_std, msl_std, tp_std |
| `clim/era5_clim_g12.nc` | 554.3 | hour=4, dayofyear=94, latitude=333, longitude=333; vars: t2m_mean, msl_mean, tp_mean, t2m_std, msl_std, tp_std |
| `dem/dem_g12.nc` | 0.2 | latitude=333, longitude=333; vars: orog |
| `dem/dem_g5.nc` | 1.7 | latitude=999, longitude=999; vars: orog |
| `era5/amphan_era5_0p25.nc` | 41.3 | time=64, latitude=161, longitude=161; vars: t2m, u10, v10, msl, tp, q850, u850, v850, r850 |
| `era5/amphan_era5_g12.nc` | 66.4 | time=64, latitude=333, longitude=333; vars: t2m, u10, v10, msl, tp, q850, u850, v850, r850 |
| `era5/amphan_era5_g5.nc` | 434.3 | time=64, latitude=999, longitude=999; vars: t2m, u10, v10, msl, tp, q850, u850, v850, r850 |
| `era5/coldwave_era5_0p25.nc` | 39.7 | time=128, latitude=161, longitude=161; vars: t2m, u10, v10, msl, tp |
| `era5/coldwave_era5_g12.nc` | 60.4 | time=128, latitude=333, longitude=333; vars: t2m, u10, v10, msl, tp |
| `era5/coldwave_era5_g5.nc` | 361.9 | time=128, latitude=999, longitude=999; vars: t2m, u10, v10, msl, tp |
| `era5/heatwave_era5_0p25.nc` | 49.3 | time=148, latitude=161, longitude=161; vars: t2m, u10, v10, msl, tp |
| `era5/heatwave_era5_g12.nc` | 76.7 | time=148, latitude=333, longitude=333; vars: t2m, u10, v10, msl, tp |
| `era5/heatwave_era5_g5.nc` | 468.5 | time=148, latitude=999, longitude=999; vars: t2m, u10, v10, msl, tp |
| `ibtracs/amphan_2020_ibtracs.csv` | 0.0 |  |
| `ibtracs/ibtracs.NI.list.v04r01.csv` | 27.9 |  |
| `ibtracs/ni_tracks_1990_2023.csv` | 0.9 |  |
| `imd/imd_rain_2020_0p25.nc` | 3.0 | time=366, latitude=129, longitude=135; vars: rain |
| `imd/imd_rain_2020_g12.nc` | 23.9 | time=366, latitude=333, longitude=333; vars: rain |
| `imd/imd_rain_2024_0p25.nc` | 2.9 | time=366, latitude=129, longitude=135; vars: rain |
| `imd/imd_rain_2024_g12.nc` | 22.4 | time=366, latitude=333, longitude=333; vars: rain |

## Known gaps
- The ERA5 cold-wave (2022-12-20..2023-01-20) and heatwave (2024-05-15..06-20) windows come from
  ARCO-ERA5, because WB2 ends 2023-01-10. They have surface fields only: 850 hPa was skipped for
  bandwidth (see SOURCES.md).
- Climatology: the mean is from WB2 1990–2019 (6-hourly, 0.25°) for the event day-of-year windows
  plus 10 days. The std is computed from 1.5° WB2 ERA5 1990–2019 with a ±15-day window and
  interpolated; it is smoother than a true 0.25° std.
- IMD rainfall is daily 0.25° (IMD Pune via imdlib) for 2020 and 2024, stored native and on G12.
- TIGGE NEPS-G, GPM IMERG and CDS ERA5-Land were SKIPPED (no keys in .env).
