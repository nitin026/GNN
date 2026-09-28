# How to plug in NCMRWF NEPS-G

No TIGGE/ECMWF key is in `.env`, so the real-ensemble results in this project come from the public
WeatherBench 2 ensembles:
- IFS ENS: 50 members, 0.25 deg and 1.5 deg;
- GenCast: 1.5 deg.

NEPS-G drops in through `pipeline/loaders.py`, which gives every source the same interface.

## 1. Get the data
- TIGGE archive (ECMWF). NEPS-G is origin `dems` (NCMRWF), with 11 perturbed members + 1 control, 12 km,
  archived from August 2017.
- Retrieval needs an ECMWF account key (`ECMWF_API_KEY` in `.env`, never committed).
- Request `msl`, `10u`, `10v`, `2t`, `tp` for one init, all members, steps 0-240 h every 6 h, area 40/60/0/100
  (N/W/S/E).

## 2. Convert to the project format
- The files arrive as GRIB2. Either install `cfgrib` + `eccodes` (they may be blocked by Windows Application
  Control on this machine), or convert elsewhere, e.g. `grib_to_netcdf -o nepsg_2020051600.nc in.grib`.
- The NetCDF needs dims `(number, step, latitude, longitude)`, `step` in hours, variables `msl` (Pa),
  `u10`, `v10` (m/s), and a global attribute `init_time`.

## 3. Run
```
python -c "from pipeline.loaders import load; e = load('nepsg_2020051600.nc', 'nepsg'); print(e.members)"
```
Put the file in `data/real/ensembles/` with a `nepsg_12km_YYYYMMDDHH.nc` name. `python -m pipeline.real_eval ens "nepsg_*"`
then tracks it exactly like IFS ENS and GenCast. The data are already on a 12 km grid, but they are still
bilinearly regridded to G12, which is harmless.

## What is not done
The GRIB branch of `NEPSGLoader` is a stub. No NEPS-G file was available, so the mapping from
TIGGE short names to `msl/u10/v10` inside cfgrib has not been tested.
