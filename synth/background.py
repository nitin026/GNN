"""Backgrounds for synthetic cases.

Real background (preferred): ERA5 0.25 deg India box from Phase 2, with extremes removed so
the injected event is the only extreme (labels stay exact):
  * anomalies w.r.t. the 1990-2019 ERA5 climatology are soft-clipped,
        x_bg = mean + c*std*tanh((x - mean) / (c*std)),  c = 1.5   (t2m, msl, tp)
  * in the Amphan window the real ERA5 vortex is removed by blending all fields toward a
    heavily smoothed (5 deg, applied twice) version within ~1000 km of the IBTrACS position (vortex removal, as in
    TC bogussing).
Fallback (only if Phase 2 data are missing): an analytic synthetic climatology (seasonal
cycle + latitude gradient + monsoon trough), flagged background="synthetic_climatology".
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from .events import local_xy

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
CLIP_C = 1.5
# t2m is clipped harder (1.0 sigma ~ 3 K) so a real background heat/cold wave cannot meet the
# IMD 4.5 C departure criterion on its own (it would be an unlabelled event)
CLIP_VAR = {"t2m": 1.0, "msl": 1.5, "tp": 1.5}


def interp_matrix(src, dst):
    """Linear-interpolation matrix W (len(dst) x len(src)) for ascending 1-D coords."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    i = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2)
    w = np.clip((dst - src[i]) / (src[i + 1] - src[i]), 0, 1)
    W = np.zeros((len(dst), len(src)), np.float64)
    W[np.arange(len(dst)), i] = 1 - w
    W[np.arange(len(dst)), i + 1] = w
    return W


class Regridder:
    def __init__(self, src_lat, src_lon, dst_lat, dst_lon):
        self.Wy = interp_matrix(src_lat, dst_lat)
        self.Wx = interp_matrix(src_lon, dst_lon)

    def __call__(self, a):
        return (self.Wy @ np.asarray(a, np.float64) @ self.Wx.T).astype(np.float32)


def softclip(x, mean, std, c=CLIP_C):
    s = np.maximum(c * std, 1e-6)
    return mean + s * np.tanh((x - mean) / s)


class RealBackground:
    """Serves cleaned 0.25-deg background fields for a valid time."""

    VARS = ("t2m", "u10", "v10", "msl", "tp")

    def __init__(self, event, with_850=False):
        self.event = event
        self.ds = xr.open_dataset(REAL / "era5" / f"{event}_era5_0p25.nc").load()
        self.clim = xr.open_dataset(REAL / "clim" / "era5_clim_0p25.nc").load()
        self.lat = self.ds.latitude.values
        self.lon = self.ds.longitude.values
        self.with_850 = with_850 and "q850" in self.ds
        self.track = None
        if event == "amphan":
            tr = pd.read_csv(REAL / "ibtracs" / "amphan_2020_ibtracs.csv")
            tr["time"] = pd.to_datetime(tr.ISO_TIME)
            self.track = tr[["time", "LAT", "LON"]]
        self.lat2d, self.lon2d = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.source = (f"ERA5 {event} (real, anomalies soft-clipped: t2m {CLIP_VAR['t2m']} sigma, "
                       f"msl/tp {CLIP_VAR['msl']} sigma)")

    def times(self):
        return pd.to_datetime(self.ds.time.values)

    def _clim(self, var, t):
        doy, hour = t.dayofyear, t.hour
        c = self.clim.sel(dayofyear=doy, hour=hour)
        return c[f"{var}_mean"].values, c[f"{var}_std"].values

    def fields(self, t):
        t = pd.Timestamp(t)
        snap = self.ds.sel(time=t)
        out = {}
        names = list(self.VARS) + (["q850", "u850", "v850", "r850"] if self.with_850 else [])
        for v in names:
            if v not in snap:
                continue
            x = snap[v].values.astype(np.float64)
            if v in ("t2m", "msl", "tp"):
                m, s = self._clim(v, t)
                x = softclip(x, m, s, CLIP_VAR[v])
            out[v] = x
        if "tp" in out:
            out["tp"] = np.maximum(out["tp"], 0)
        if self.track is not None:
            out = self._remove_vortex(out, t)
        return out

    def _remove_vortex(self, f, t):
        tr = self.track
        if t < tr.time.min() - pd.Timedelta("12h") or t > tr.time.max() + pd.Timedelta("12h"):
            return f
        ts = ((tr.time - pd.Timestamp("2000-01-01")).dt.total_seconds()).values
        t0 = (t - pd.Timestamp("2000-01-01")).total_seconds()
        la = np.interp(t0, ts, tr.LAT.values)
        lo = np.interp(t0, ts, tr.LON.values)
        x, y = local_xy(self.lat2d, self.lon2d, la, lo)
        w = np.exp(-((x ** 2 + y ** 2) / 900.0 ** 2))       # 1 at centre, ~0 beyond ~1500 km
        for v in f:
            for _ in range(2):                               # 20 px = 5 deg smoothing, twice
                smooth = gaussian_filter(f[v], sigma=20, mode="nearest")
                f[v] = (1 - w) * f[v] + w * smooth
        return f


class SyntheticClimatology:
    """Analytic fallback background (used only if real ERA5 is unavailable)."""

    VARS = ("t2m", "u10", "v10", "msl", "tp")

    def __init__(self, event):
        self.event = event
        self.lat = np.arange(0, 40.01, 0.25)
        self.lon = np.arange(60, 100.01, 0.25)
        self.lat2d, self.lon2d = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.source = "synthetic_climatology (analytic fallback, not real data)"

    def times(self):
        start = {"amphan": "2020-05-10", "coldwave": "2022-12-20", "heatwave": "2024-05-15"}
        return pd.date_range(start[self.event], periods=4 * 32, freq="6h")

    def fields(self, t):
        t = pd.Timestamp(t)
        season = np.cos(2 * np.pi * (t.dayofyear - 196) / 365.25)     # +1 mid-July
        lat = self.lat2d
        t2m = 300.0 - 0.35 * (lat - 20) * (1.4 - season) + 3 * season \
            + 2.5 * np.cos(2 * np.pi * (t.hour - 9) / 24)
        trough = np.exp(-((lat - (24 + 3 * season)) / 4.0) ** 2)
        msl = 101000 + 60 * (20 - lat) * (-season) - 400 * trough * max(season, 0)
        u10 = 3.0 * np.tanh((lat - 18) / 6) * (1 - season) - 5 * max(season, 0) * np.exp(-((lat - 14) / 6) ** 2)
        v10 = 1.5 * max(season, 0) * np.ones_like(lat)
        tp = 1.5 * max(season, 0) * trough
        return {"t2m": t2m, "u10": u10, "v10": v10, "msl": msl, "tp": tp}


def load_background(event, with_850=False):
    try:
        return RealBackground(event, with_850)
    except (FileNotFoundError, OSError, KeyError) as e:
        print(f"real background for {event} unavailable ({e}); using synthetic climatology")
        return SyntheticClimatology(event)
