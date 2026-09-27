"""Cyclone tracks from the real IBTrACS North Indian archive (Phase 2 download)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IBT = ROOT / "data" / "real" / "ibtracs"
KT = 0.514444          # knots -> m/s
ONE_TO_TEN_MIN = 0.88  # 1-min sustained -> 10-min mean wind (WMO guidance factor)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def ni_library(min_fixes=20, min_wind_kt=64, exclude=("AMPHAN",)):
    """6-hourly tracks of NI storms (1990-2023) that reached >= min_wind_kt."""
    df = pd.read_csv(IBT / "ni_tracks_1990_2023.csv", keep_default_na=False, low_memory=False)
    df["time"] = pd.to_datetime(df.ISO_TIME)
    df = df[df.time.dt.hour % 6 == 0]
    df["wind"] = _num(df.USA_WIND)
    out = []
    for sid, g in df.groupby("SID"):
        if g.NAME.iloc[0] in exclude or len(g) < min_fixes or g.wind.max() < min_wind_kt:
            continue
        lat0, lon0 = _num(g.LAT).iloc[0], _num(g.LON).iloc[0]
        if not (4 <= lat0 <= 22 and 62 <= lon0 <= 95):   # genesis inside the NI box (not W. Pacific)
            continue
        out.append({"sid": sid, "name": g.NAME.iloc[0], "lat": _num(g.LAT).values,
                    "lon": _num(g.LON).values, "time": g.time.values})
    return out


def sample_track(rng, lib, basin=None):
    """Pick a real NI track and jitter it: constant offset (+-0.75 deg) + AR(1) wiggle."""
    cands = lib
    if basin == "BoB":
        cands = [t for t in lib if np.nanmean(t["lon"][:4]) > 80]
    elif basin == "AS":
        cands = [t for t in lib if np.nanmean(t["lon"][:4]) < 78]
    tr = cands[rng.integers(len(cands))]
    n = len(tr["lat"])
    off = rng.uniform(-0.75, 0.75, 2)
    wig = np.zeros((n, 2))
    for i in range(1, n):
        wig[i] = 0.7 * wig[i - 1] + rng.normal(0, 0.08, 2)
    return {"sid": tr["sid"], "name": tr["name"], "lat": tr["lat"] + off[0] + wig[:, 0],
            "lon": tr["lon"] + off[1] + wig[:, 1], "offset_deg": off.round(3).tolist()}


def amphan_track():
    """Amphan 2020 best track: 6-hourly positions, 10-min Vmax [m/s], RMW [km], Pc [Pa]."""
    df = pd.read_csv(IBT / "amphan_2020_ibtracs.csv", keep_default_na=False)
    df["time"] = pd.to_datetime(df.ISO_TIME)
    df = df[df.time.dt.hour % 6 == 0].reset_index(drop=True)
    return {"time": df.time.values, "lat": _num(df.LAT).values, "lon": _num(df.LON).values,
            "vmax": _num(df.USA_WIND).values * KT * ONE_TO_TEN_MIN,
            "vmax_1min_kt": _num(df.USA_WIND).values,
            "rmw_km": _num(df.USA_RMW).values * 1.852,
            "pres_pa": _num(df.USA_PRES).values * 100.0}
