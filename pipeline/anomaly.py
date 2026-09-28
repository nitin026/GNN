"""Standardised anomalies (z-scores) against the ERA5 1990-2019 climatology."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
CLIM_G12 = ROOT / "data" / "real" / "clim" / "era5_clim_g12.nc"
CLIM_EXTRA = ROOT / "data" / "real" / "clim" / "era5_clim_extra_0p25.nc"
MAX_DOY_GAP = 1          # days; beyond this the climatology is missing, and we fail loudly


def zscore(x, mean, std, min_std=1e-6):
    """z = (x - mean) / std, with std floored to avoid division by zero."""
    return (np.asarray(x, float) - mean) / np.maximum(std, min_std)


def _gap(doys, d):
    return np.abs(((doys - d + 183) % 366) - 183)


class Climatology:
    """Mean/std on G12 for a valid time, at the nearest (hour, day-of-year).

    Uses data/real/clim/era5_clim_g12.nc. Day-of-year values missing there are taken from
    era5_clim_extra_0p25.nc and bilinearly interpolated to G12 on demand (cached). If neither
    file has a day within MAX_DOY_GAP of the request, a KeyError is raised (previously the
    nearest day was used silently, however far away)."""

    def __init__(self, path=CLIM_G12, extra=CLIM_EXTRA):
        self.ds = xr.open_dataset(path)            # lazy: one (doy, hour) slice read per request
        self.doys = self.ds.dayofyear.values
        self.extra = xr.open_dataset(extra) if Path(extra).exists() else None
        self.edoys = self.extra.dayofyear.values if self.extra is not None else np.array([], int)
        self._cache = {}
        self._rg = None

    def _regridder(self):
        if self._rg is None:
            from synth.background import Regridder
            from synth.grids import LAT12, LON12
            self._rg = Regridder(self.extra.latitude.values, self.extra.longitude.values, LAT12, LON12)
        return self._rg

    def get(self, var, valid_time):
        t = pd.Timestamp(valid_time)
        hour = int(6 * round(t.hour / 6)) % 24
        g = _gap(self.doys, t.dayofyear)
        if g.min() <= MAX_DOY_GAP:
            key = ("base", var, int(self.doys[np.argmin(g)]), hour)
            if key not in self._cache:
                if len(self._cache) > 400:                 # bounded cache (~350 KB per entry)
                    self._cache.pop(next(iter(self._cache)))
                c = self.ds.sel(dayofyear=key[2], hour=hour)
                self._cache[key] = (c[f"{var}_mean"].values, c[f"{var}_std"].values)
            return self._cache[key]
        ge = _gap(self.edoys, t.dayofyear) if len(self.edoys) else np.array([999])
        if ge.min() > MAX_DOY_GAP:
            raise KeyError(f"no climatology within {MAX_DOY_GAP} d of day-of-year {t.dayofyear} "
                           "(run scripts/fetch_clim_extra.py)")
        d = int(self.edoys[np.argmin(ge)])
        key = (var, d, hour)
        if key not in self._cache:
            c = self.extra.sel(dayofyear=d, hour=hour)
            rg = self._regridder()
            self._cache[key] = (rg(c[f"{var}_mean"].values), rg(c[f"{var}_std"].values))
        return self._cache[key]

    def z(self, var, x, valid_time):
        m, s = self.get(var, valid_time)
        return zscore(x, m, s)
