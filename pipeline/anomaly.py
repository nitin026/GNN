"""Standardised anomalies (z-scores) against the ERA5 1990-2019 climatology."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
CLIM_G12 = ROOT / "data" / "real" / "clim" / "era5_clim_g12.nc"


def zscore(x, mean, std, min_std=1e-6):
    """z = (x - mean) / std, with std floored to avoid division by zero."""
    return (np.asarray(x, float) - mean) / np.maximum(std, min_std)


class Climatology:
    """Mean/std on G12 for a (valid_time) -> nearest (hour, day-of-year)."""

    def __init__(self, path=CLIM_G12):
        self.ds = xr.open_dataset(path).load()
        self.doys = self.ds.dayofyear.values

    def get(self, var, valid_time):
        t = pd.Timestamp(valid_time)
        doy = int(self.doys[np.argmin(np.abs(((self.doys - t.dayofyear + 183) % 366) - 183))])
        hour = int(6 * round(t.hour / 6)) % 24
        c = self.ds.sel(dayofyear=doy, hour=hour)
        return c[f"{var}_mean"].values, c[f"{var}_std"].values

    def z(self, var, x, valid_time):
        m, s = self.get(var, valid_time)
        return zscore(x, m, s)
