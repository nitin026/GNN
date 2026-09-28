"""BRIEF4 Phase 1: ERA5 1990-2019 climatology for the day-of-year windows of the new real events
that data/real/clim/era5_clim_0p25.nc does not cover.

Mean: WB2 hourly climatology (6-hourly, 0.25 deg). Std: 1.5 deg WB2 ERA5 1990-2019, +-15 day
window, interpolated (same method as scripts/fetch_real.py clim()). Output (0.25 deg only; G12 is
interpolated on demand by pipeline.anomaly.Climatology):
    data/real/clim/era5_clim_extra_0p25.nc
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_real import (BOX, CLIM_VARS, GCS, WB2_CLIM, WB2_COARSE, attempt, dir_mb, log,  # noqa: E402
                                open_store)
from scripts.fetch_real_events import CYCLONES, HEATCOLD, ibtracs_track  # noqa: E402

REAL = ROOT / "data" / "real"


def windows():
    w = []
    for eid, (name, season) in CYCLONES.items():
        tr = ibtracs_track(name, season)
        w.append((pd.to_datetime(tr.ISO_TIME.iloc[0]) - pd.Timedelta("3D"),
                  pd.to_datetime(tr.ISO_TIME.iloc[-1]) + pd.Timedelta("3D")))
    for eid, (hz, s, e) in HEATCOLD.items():
        w.append((pd.Timestamp(s) - pd.Timedelta("2D"), pd.Timestamp(e) + pd.Timedelta("2D")))
    # valid times of the real Amphan ensemble forecasts (inits 2020-05-08..20, leads to 240 h)
    w.append((pd.Timestamp("2020-05-06"), pd.Timestamp("2020-06-01")))
    return w


def main():
    t0 = time.time()
    have = set(xr.open_dataset(REAL / "clim" / "era5_clim_0p25.nc").dayofyear.values.tolist())
    extra_p = REAL / "clim" / "era5_clim_extra_0p25.nc"
    prev = xr.open_dataset(extra_p).load() if extra_p.exists() else None
    if prev is not None:
        prev.close()
        have |= set(prev.dayofyear.values.tolist())
    need = sorted({d.dayofyear for s, e in windows() for d in pd.date_range(s.normalize(), e)} - have)
    print(f"missing day-of-year values: {len(need)}", flush=True)
    if not need:
        return
    ds = open_store(WB2_CLIM)
    mean = ds[list(CLIM_VARS)].sel(dayofyear=need, **BOX).rename(CLIM_VARS)
    mean = attempt(lambda: mean.load())
    mean["tp"] = (mean["tp"] * 1000).clip(min=0)
    co = open_store(WB2_COARSE)[list(CLIM_VARS)].rename(CLIM_VARS)
    co = co.sel(time=slice("1990-01-01", "2019-12-31T18"), latitude=slice(-3, 43), longitude=slice(57, 103))
    co = attempt(lambda: co.load())
    co["tp"] = co["tp"] * 1000
    tdoy, thour = co.time.dt.dayofyear.values, co.time.dt.hour.values
    std = {}
    for v in CLIM_VARS.values():
        a = co[v].transpose("time", "latitude", "longitude").values
        out = np.full((4, len(need), a.shape[1], a.shape[2]), np.nan, np.float32)
        for hi, h in enumerate((0, 6, 12, 18)):
            for di, d in enumerate(need):
                dd = np.abs(((tdoy - d + 183) % 366) - 183)
                out[hi, di] = a[(thour == h) & (dd <= 15)].std(axis=0)
        std[v] = (("hour", "dayofyear", "latitude", "longitude"), out)
    std = xr.Dataset(std, coords={"hour": [0, 6, 12, 18], "dayofyear": need,
                                  "latitude": co.latitude.values, "longitude": co.longitude.values})
    std = std.sortby("latitude").interp(latitude=mean.latitude, longitude=mean.longitude)
    out = xr.Dataset({f"{v}_mean": mean[v] for v in CLIM_VARS.values()} |
                     {f"{v}_std": std[v].astype("float32") for v in CLIM_VARS.values()}).sortby("latitude")
    out.attrs = {"Conventions": "CF-1.8", "synthetic": "false",
                 "title": "ERA5 1990-2019 climatology (extra day-of-year windows for BRIEF4 real events)",
                 "mean_source": f"{GCS}/{WB2_CLIM}", "std_source": f"{GCS}/{WB2_COARSE} (1.5 deg, +-15 d)"}
    p = REAL / "clim" / "era5_clim_extra_0p25.nc"
    if prev is not None:                          # merge with the earlier extra days
        out = xr.concat([prev, out], dim="dayofyear").sortby("dayofyear")
        out.attrs = prev.attrs
    out.to_netcdf(p, encoding={v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out.data_vars})
    log("ERA5 climatology extra doy (mean+std)", f"{GCS}/{WB2_CLIM}", "OK", dir_mb(p),
        f"{len(need)} extra day-of-year values", f"{time.time() - t0:.0f}s; 0.25 deg only (G12 on demand)")


if __name__ == "__main__":
    main()
