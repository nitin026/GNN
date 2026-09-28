"""Ensemble-forecast loaders with one interface, so any real ensemble can drive the pipeline.

    ens = load(path_or_spec)          # -> EnsembleForecast
    ens.msl (M, T, y, x) Pa, ens.u10 / ens.v10 (optional), ens.times, ens.init, ens.lat, ens.lon

Implemented:
  WB2Loader   - files written by scripts/fetch_ensembles.py (IFS ENS, GenCast; public, no key)
Stub (no key available in this project):
  NEPSGLoader - NCMRWF NEPS-G via TIGGE (origin "dems"). See docs/NEPS_G.md for how to plug it
                in. It raises a clear error until a GRIB reader and data are present.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class EnsembleForecast:
    msl: np.ndarray
    times: pd.DatetimeIndex
    init: pd.Timestamp
    lat: np.ndarray
    lon: np.ndarray
    u10: np.ndarray = None
    v10: np.ndarray = None
    source: str = ""
    synthetic: bool = False

    @property
    def members(self):
        return self.msl.shape[0]


class WB2Loader:
    """data/real/ensembles/{model}_{res}_{init}.nc (dims number, step [h], latitude, longitude)."""

    def load(self, path):
        ds = xr.open_dataset(path)
        init = pd.Timestamp(ds.attrs["init_time"])
        steps = ds.step.values.astype(int)
        out = EnsembleForecast(
            msl=ds.msl.values.astype(np.float32),
            times=pd.DatetimeIndex([init + pd.Timedelta(hours=int(h)) for h in steps]),
            init=init, lat=ds.latitude.values, lon=ds.longitude.values,
            u10=ds.u10.values.astype(np.float32) if "u10" in ds else None,
            v10=ds.v10.values.astype(np.float32) if "v10" in ds else None,
            source=ds.attrs.get("source", ""), synthetic=ds.attrs.get("synthetic") == "true")
        ds.close()
        return out


class NEPSGLoader:
    """NCMRWF NEPS-G (TIGGE origin 'dems', 11+1 members at 12 km since Aug 2017) - STUB.

    Expected input: TIGGE GRIB2 files with msl, 10u, 10v, 2t, tp for one init, all members and
    steps 0-240 h (6-hourly), retrieved with the ECMWF API (needs an account key; none is present
    in this project's .env). Reading GRIB needs cfgrib/eccodes; if Windows Application Control
    blocks eccodes, convert to NetCDF elsewhere (e.g. `grib_to_netcdf`) and use WB2Loader-style
    NetCDF with the same variable names (msl, u10, v10) and dims (number, step, latitude, longitude).
    """

    def load(self, path):
        path = Path(path)
        if path.suffix in (".nc", ".nc4"):
            return WB2Loader().load(path)             # NetCDF converted from TIGGE GRIB works as is
        try:
            import cfgrib  # noqa: F401
        except ImportError as e:
            raise RuntimeError("NEPS-G GRIB needs cfgrib/eccodes (not installed here); convert the GRIB "
                               "to NetCDF with dims (number, step, latitude, longitude) and variables "
                               "msl/u10/v10, then pass the .nc file. See docs/NEPS_G.md.") from e
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"filter_by_keys": {"dataType": "pf"}})
        raise NotImplementedError(f"cfgrib is available; map {list(ds.data_vars)} to msl/u10/v10 here "
                                  "(see docs/NEPS_G.md). No NEPS-G data was available to test this path.")


def load(path, kind="wb2"):
    return {"wb2": WB2Loader, "nepsg": NEPSGLoader}[kind]().load(path)
