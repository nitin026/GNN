"""Cyclone-rain calibration against IMD (BRIEF2 Phase 1): intensity-aware quantile mapping.

Comparison sample ("swath"): daily rain on land within 500 km of the storm centre, on a
~0.25 deg grid, for days the storm is active; wet cells only (>= 1 mm/day).
  IMD  : real IMD 0.25 deg daily rain, Amphan 2020 (IBTrACS centre; IMD day = D-1 03 UTC..D 03 UTC)
  synth: 5 km truth tp, 4 six-hourly accumulations per day, block-averaged 6x6 (0.24 deg)

Intensity-aware map (mode "intensity"): the generator's eyewall rain peak scales as
g(V) = 3 + 0.3 V (V = 10-min Vmax, m/s). Each swath day is normalised by g(V_day), for IMD with
the IBTrACS Amphan Vmax of that day (USA_WIND x 0.88). The empirical quantile map T is fitted on
normalised values, and the generator applies  rain' = rain * g(V) T(D / g(V)) / D  per vortex.
Fitted on TRAIN cyclone cases only (pipeline/splits.py), from UNCALIBRATED rain
(SIH_RAIN_QM=off regeneration into a scratch folder).

    python -m synth.rain_calibration fit [raw_dir]   # fit on train cases from uncalibrated runs
    python -m synth.rain_calibration report [dir]    # raw and normalised quantiles vs IMD
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
REAL = ROOT / "data" / "real"
QM = Path(__file__).with_name("rain_qm.json")
PROBS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97,
                  0.98, 0.99, 0.995, 0.999])
WET, RADIUS = 1.0, 500.0
KT, TEN_MIN = 0.514444, 0.88


def g(v):
    """Intensity scale of the generator's rain peak (mm/h): 3 + 0.3 Vmax."""
    return 3.0 + 0.3 * np.asarray(v, float)


def _dist_km(lat2d, lon2d, la, lo):
    return np.hypot((lat2d - la) * 111.2, (lon2d - lo) * 111.2 * np.cos(np.deg2rad(la)))


def imd_amphan_sample(with_v=False):
    imd = xr.open_dataset(REAL / "imd" / "imd_rain_2020_0p25.nc").rain
    ib = pd.read_csv(REAL / "ibtracs" / "amphan_2020_ibtracs.csv")
    ib["time"] = pd.to_datetime(ib.ISO_TIME)
    ib["v"] = pd.to_numeric(ib.USA_WIND, errors="coerce") * KT * TEN_MIN
    vals, vs = [], []
    la2, lo2 = np.meshgrid(imd.latitude.values, imd.longitude.values, indexing="ij")
    for day in pd.date_range(ib.time.min().normalize(), ib.time.max().normalize()):
        mid = day - pd.Timedelta("9h")                       # centre of the IMD day window
        r = ib.iloc[(ib.time - mid).abs().argmin()]
        if abs(r.time - mid) > pd.Timedelta("6h"):
            continue
        win = ib[(ib.time > day - pd.Timedelta("21h")) & (ib.time <= day + pd.Timedelta("3h"))]
        v = float(np.nanmean(win.v)) if win.v.notna().any() else np.nan
        a = imd.sel(time=day).values
        m = (_dist_km(la2, lo2, r.LAT, r.LON) <= RADIUS) & np.isfinite(a) & (a >= WET)
        if not np.isfinite(v):
            continue                                          # no IBTrACS intensity that day
        vals.append(a[m])
        vs.append(np.full(m.sum(), v))
    x = np.concatenate(vals)
    return (x, np.concatenate(vs)) if with_v else x


def synth_sample(case, land5=None, base=None, with_v=False):
    base = Path(base) if base else SYN
    lab = json.loads((base / case / "labels.json").read_text())
    ds = netCDF4.Dataset(base / case / "truth_5km.nc")
    v = ds.variables["tp"]
    v.set_auto_maskandscale(False)
    lat, lon = ds.variables["latitude"][:], ds.variables["longitude"][:]
    if land5 is None:
        land5 = xr.open_dataset(REAL / "dem" / "dem_g5.nc").orog.values > 1.0
    times = pd.to_datetime(lab["valid_times"])
    n6 = (len(lat) // 6) * 6
    la_b = lat[:n6].reshape(-1, 6).mean(1)
    lo_b = lon[:n6].reshape(-1, 6).mean(1)
    land_b = land5[:n6, :n6].reshape(n6 // 6, 6, n6 // 6, 6).mean((1, 3)) > 0.5
    la2, lo2 = np.meshgrid(la_b, lo_b, indexing="ij")
    vals, vs = [], []
    for d0 in range(1, len(times) - 3, 4):
        idx = list(range(d0, d0 + 4))
        cen = [lab["truth_track"][i] for i in idx if lab["truth_track"][i]["center_lat"] is not None]
        if len(cen) < 2:
            continue
        la = np.mean([c["center_lat"] for c in cen])
        lo = np.mean([c["center_lon"] for c in cen])
        vday = float(np.mean([c["vmax_ms"] for c in cen]))
        tot = sum(v[i].astype(np.float64) * float(v.scale_factor) for i in idx)
        blk = tot[:n6, :n6].reshape(n6 // 6, 6, n6 // 6, 6).mean((1, 3))
        m = land_b & (_dist_km(la2, lo2, la, lo) <= RADIUS) & (blk >= WET)
        vals.append(blk[m])
        vs.append(np.full(m.sum(), vday))
    ds.close()
    x = np.concatenate(vals) if vals else np.array([])
    vv = np.concatenate(vs) if vs else np.array([])
    return (x, vv) if with_v else x


def quantiles(x):
    return {"n": int(len(x)), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "p999": float(np.percentile(x, 99.9)),
            "max": float(x.max())} if len(x) else {"n": 0}


def fit(train_cases, raw_dir):
    obs, vo = imd_amphan_sample(with_v=True)
    parts = [synth_sample(c, base=raw_dir, with_v=True) for c in train_cases]
    sim = np.concatenate([p[0] for p in parts])
    vs = np.concatenate([p[1] for p in parts])
    xq = np.quantile(sim / g(vs), PROBS)
    yq = np.quantile(obs / g(vo), PROBS)
    table = {"mode": "intensity", "probs": PROBS.tolist(),
             "x_norm": xq.tolist(), "y_norm": yq.tolist(),
             "g": "3 + 0.3 * Vmax (mm/h per m/s); rain normalised by g(V) before mapping",
             "fit_cases": train_cases, "fit_source": "uncalibrated regeneration (SIH_RAIN_QM=off)",
             "target": "IMD 0.25 deg daily rain, Amphan 2020 swath",
             "imd_v_range_ms": [float(vo.min()), float(vo.max())]}
    QM.write_text(json.dumps(table, indent=1))
    return table, quantiles(obs), quantiles(sim)


def _map(z, x, y):
    T = np.interp(z, x, y)
    T = np.where(z > x[-1], z * (y[-1] / x[-1]), T)
    return np.where(z < x[0], z * (y[0] / max(x[0], 1e-6)), T)


def apply_factor(daily_equiv_mm, table, vmax=None):
    """Multiplicative factor for a daily-equivalent rain D [mm/day]."""
    D = np.asarray(daily_equiv_mm, float)
    if table.get("mode") == "intensity":
        s = g(vmax)
        T = s * _map(D / s, np.asarray(table["x_norm"]), np.asarray(table["y_norm"]))
    else:
        T = _map(D, np.asarray(table["x_daily_mm"]), np.asarray(table["y_daily_mm"]))
    return np.clip(T / np.maximum(D, 1e-6), 0.1, 10.0)


def report(base=None):
    sys.path.insert(0, str(ROOT))
    from pipeline.splits import TEST, TRAIN, VAL
    obs, vo = imd_amphan_sample(with_v=True)
    qo, qon = quantiles(obs), quantiles(obs / g(vo))
    rows = [("IMD Amphan (real)", "-", qo, qon)]
    for c in [c for c in TRAIN + VAL + TEST if c.startswith(("cyc", "amphan"))]:
        x, v = synth_sample(c, base=base, with_v=True)
        split = "train" if c in TRAIN else "val" if c in VAL else "test"
        rows.append((c, split, quantiles(x), quantiles(x / g(v))))
    for name, sp, q, qn in rows:
        print(f"{name:18s} {sp:5s} n={q['n']:5d} p50={q['p50']:6.1f} p99={q['p99']:6.1f} "
              f"({(q['p99'] / qo['p99'] - 1) * 100:+4.0f}%) max={q['max']:6.1f} | normalised p99 "
              f"{qn['p99']:.2f} ({(qn['p99'] / qon['p99'] - 1) * 100:+4.0f}%)")
    return rows


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from pipeline.splits import TRAIN
    if sys.argv[1:2] == ["fit"]:
        raw = sys.argv[2] if len(sys.argv) > 2 else None
        t, o, s = fit([c for c in TRAIN if c.startswith("cyc")], raw)
        print("IMD", o)
        print("train synth (uncalibrated)", s)
        print("x_norm", np.round(t["x_norm"], 2).tolist())
        print("y_norm", np.round(t["y_norm"], 2).tolist())
    else:
        report(sys.argv[2] if len(sys.argv) > 2 else None)
