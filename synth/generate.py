"""Generate synthetic ensemble-forecast cases with exact ground truth.

    python -m synth.generate            # all cases
    python -m synth.generate cyc_01 ... # selected cases

Per case, data/synthetic/{case_id}/ gets:
  truth_5km.nc  - G5 (0.04 deg) "truth" at the 41 valid times (lead 0..240 h, 6-hourly)
  fcst_12km.nc  - G12 (0.12 deg) 20-member ensemble, the truth block-averaged to G12
                  (<var>_truth, so avgpool(truth_5km) == <var>_truth exactly), and masks
  labels.json   - event masks summary, centroid track, 4-D box, peak value, hazard, roles
Every file carries the global attribute synthetic="true".
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from . import events as ev
from .background import Regridder, load_background
from .grids import LAT5, LON5, LAT12, LON12, avgpool
from .noise import AR1Noise, spectral_noise
from .physics import cap_rh, moisture_flux_convergence, orographic_factor, place_rain
from .tracks import amphan_track, ni_library, sample_track

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
VERSION = "synth-0.1"
LEADS = np.arange(0, 241, 6)
N_MEMBERS = 20
KM = ev.KM_PER_DEG

# int16 packing (scale, offset); scale/offset stored as float64 so decoding is exact
PACK = {"t2m": (0.01, 273.15), "u10": (0.01, 0.0), "v10": (0.01, 0.0), "msl": (1.0, 100000.0),
        "tp": (0.02, 0.0), "r850": (0.01, 0.0)}
# coarser packing for ensemble members (well below ensemble spread) to bound disk use
PACK_MEMBER = {"t2m": (0.05, 273.15), "u10": (0.05, 0.0), "v10": (0.05, 0.0), "msl": (5.0, 100000.0),
               "tp": (0.05, 0.0), "r850": (0.1, 0.0)}
META = {  # CF units/long_name + NEPS-G (TIGGE) parameter short names
    "t2m": ("K", "2 metre temperature", "2t"), "u10": ("m s-1", "10 metre U wind component", "10u"),
    "v10": ("m s-1", "10 metre V wind component", "10v"),
    "msl": ("Pa", "Mean sea level pressure", "msl"),
    "tp": ("kg m-2", "Total precipitation accumulated over the previous 6 h", "tp"),
    "r850": ("%", "Relative humidity at 850 hPa", "r")}
# fine-scale noise amplitudes (truth) - k^-5/3 spectrum, wavelengths < ~500 km
FINE = {"t2m": 0.35, "u10": 0.6, "v10": 0.6, "msl": 20.0, "tp": 0.35, "r850": 2.0}

CASES = {
    "cyc_01": dict(hazard="tropical_cyclone", window="amphan", init="2020-05-10T00", seed=101, basin="BoB"),
    "cyc_02": dict(hazard="tropical_cyclone", window="amphan", init="2020-05-11T12", seed=102, basin="AS"),
    "cyc_03": dict(hazard="tropical_cyclone", window="amphan", init="2020-05-13T00", seed=103, basin="BoB"),
    "cyc_04": dict(hazard="tropical_cyclone", window="amphan", init="2020-05-14T12", seed=104, basin="BoB"),
    "heat_01": dict(hazard="heat_dome", window="heatwave", init="2024-05-15T00", seed=201, center=(26.5, 73.5)),
    "heat_02": dict(hazard="heat_dome", window="heatwave", init="2024-05-22T00", seed=202, center=(22.5, 79.0)),
    "heat_03": dict(hazard="heat_dome", window="heatwave", init="2024-05-29T00", seed=203, center=(25.5, 82.5)),
    "heat_04": dict(hazard="heat_dome", window="heatwave", init="2024-06-05T00", seed=204, center=(18.0, 79.5)),
    "cold_01": dict(hazard="cold_wave", window="coldwave", init="2022-12-20T00", seed=301, start=(31.5, 72.0)),
    "cold_02": dict(hazard="cold_wave", window="coldwave", init="2022-12-26T00", seed=302, start=(32.5, 73.5)),
    "cold_03": dict(hazard="cold_wave", window="coldwave", init="2023-01-01T00", seed=303, start=(31.0, 71.0)),
    "cold_04": dict(hazard="cold_wave", window="coldwave", init="2023-01-08T00", seed=304, start=(32.0, 74.5)),
    "amphan_replay": dict(hazard="tropical_cyclone", window="amphan", init="2020-05-15T06", seed=401,
                          replay="AMPHAN"),
}
LEVELS = {"tropical_cyclone": [1013, 850], "heat_dome": [1013, 1013], "cold_wave": [1013, 1013]}
MASK_DEF = {"tropical_cyclone": "injected MSLP depression <= -400 Pa (4 hPa)",
            "heat_dome": "injected 2-m temperature anomaly >= +3 K",
            "cold_wave": "injected 2-m temperature anomaly <= -3 K"}


# =========================================================================== static fields
def static_fields():
    p5, p12 = ROOT / "data/real/dem/dem_g5.nc", ROOT / "data/real/dem/dem_g12.nc"
    if p5.exists():
        o5 = xr.open_dataset(p5).orog.values.astype(np.float32)
        src = "Copernicus DEM GLO-90 (real)"
    else:
        o5 = np.zeros((len(LAT5), len(LON5)), np.float32)
        src = "flat (DEM unavailable)"
    o12 = avgpool(o5)
    return o5, o12, src


def plains_weight(orog, lat):
    land = gaussian_filter((orog > 1.0).astype(np.float32), 2)
    low = np.clip((900.0 - orog) / 500.0, 0, 1)
    north = 1 / (1 + np.exp(-(lat[:, None] - 21.5) / 0.8)) * np.ones_like(orog)
    return gaussian_filter(land * low * north, 2).astype(np.float32)


def land_weight(orog):
    return gaussian_filter((orog > 1.0).astype(np.float32), 3).astype(np.float32)


# =========================================================================== event states
def cyclone_truth_state(cfg, rng, land12):
    """Per-lead centre/intensity for the truth. Returns dict of arrays over LEADS."""
    n = len(LEADS)
    if cfg.get("replay") == "AMPHAN":
        tr = amphan_track()
        init = pd.Timestamp(cfg["init"])
        tt = (pd.to_datetime(tr["time"]) - init) / pd.Timedelta("1h")
        tt = np.asarray(tt, float)
        ok = np.isfinite(tr["vmax"])                  # final fixes lack USA_WIND -> not used
        tt, tr = tt[ok], {k: np.asarray(v)[ok] for k, v in tr.items()}
        act = (LEADS >= tt.min()) & (LEADS <= tt.max())

        def interp(a):
            a = pd.Series(a).interpolate(limit_direction="both").values
            return np.where(act, np.interp(LEADS, tt, a), np.nan)
        rmw = tr["rmw_km"]
        st = {"lat": interp(tr["lat"]), "lon": interp(tr["lon"]), "vmax": interp(tr["vmax"]),
              "rmw": interp(rmw), "dp": interp(np.maximum(1010e2 - tr["pres_pa"], 200.0)),
              "active": act}
        meta = {"track_source": "IBTrACS v04r01 AMPHAN 2020 (exact best-track positions, "
                                "USA_WIND x0.88 for 10-min, USA_RMW, USA_PRES)"}
        return st, meta
    lib = ni_library()
    tr = sample_track(rng, lib, cfg.get("basin"))
    t_gen = float(rng.choice([6, 12, 18, 24, 36]))
    k = ((LEADS - t_gen) / 6).astype(int)
    act = (LEADS >= t_gen) & (k < len(tr["lat"]))
    lat = np.where(act, tr["lat"][np.clip(k, 0, len(tr["lat"]) - 1)], np.nan)
    lon = np.where(act, tr["lon"][np.clip(k, 0, len(tr["lon"]) - 1)], np.nan)
    inbox = (lat > 1) & (lat < 39) & (lon > 61) & (lon < 99)
    act &= inbox
    # landfall = first active lead with the centre over land
    landfall = None
    for i in np.where(act)[0]:
        ii = int((lat[i] - 0) / 0.12)
        jj = int((lon[i] - 60) / 0.12)
        if land12[ii, jj] > 0.5 and LEADS[i] > t_gen + 24:
            landfall = float(LEADS[i])
            break
    vpeak = float(rng.uniform(35, 62))
    t_peak = t_gen + float(rng.uniform(60, 132))
    vm = ev.lifecycle_vmax(LEADS, t_gen, t_peak, 15.0, vpeak, landfall)
    rmw0 = float(rng.uniform(25, 45))
    rmw = rmw0 * (1.35 - 0.35 * vm / vpeak)
    if landfall is not None:
        rmw = rmw * np.where(LEADS > landfall, 1.4, 1.0)
    act &= vm > 12.0
    st = {"lat": lat, "lon": lon, "vmax": np.where(act, vm, np.nan),
          "rmw": np.where(act, rmw, np.nan), "dp": np.full(n, np.nan), "active": act}
    meta = {"track_source": f"IBTrACS NI {tr['sid']} ({tr['name']}) jittered by "
                            f"{tr['offset_deg']} deg + AR(1) wiggle",
            "t_genesis_h": t_gen, "t_peak_h": round(t_peak, 1), "vmax_peak_ms": round(vpeak, 2),
            "landfall_lead_h": landfall, "rmw0_km": round(rmw0, 2)}
    return st, meta


def blob_truth_state(cfg, rng):
    hz = cfg["hazard"]
    n = len(LEADS)
    if hz == "heat_dome":
        la0, lo0 = cfg["center"]
        amp = float(rng.uniform(4, 8))
        dur = float(rng.uniform(5, 12)) * 24
        t0 = float(rng.uniform(-max(dur - 5 * 24, 0), 48))   # may already be under way
        drift = rng.uniform(-1.0, 1.0, 2)                 # m/s, slow-moving
        sx, sy = float(rng.uniform(350, 600)), float(rng.uniform(250, 450))
        ang = float(rng.uniform(0, 180))
    else:
        la0, lo0 = cfg["start"]
        amp = float(rng.uniform(4, 8))
        dur = float(rng.uniform(5, 9)) * 24
        t0 = float(rng.uniform(0, 36))
        spd = float(rng.uniform(1.5, 3.0))                # m/s toward the ESE (from the NW)
        bearing = np.deg2rad(rng.uniform(110, 130))       # degrees clockwise from north
        drift = np.array([spd * np.sin(bearing), spd * np.cos(bearing)])
        sx, sy = float(rng.uniform(600, 850)), float(rng.uniform(250, 380))
        ang = float(rng.uniform(-30, -15))                # aligned WNW-ESE (plains axis)
    hrs = LEADS - t0
    dx_km = drift[0] * 3.6 * np.maximum(hrs, 0)
    dy_km = drift[1] * 3.6 * np.maximum(hrs, 0)
    lat = la0 + dy_km / KM
    lon = lo0 + dx_km / (KM * np.cos(np.deg2rad(la0)))
    env = ev.amplitude_envelope(LEADS, t0, dur)
    st = {"lat": lat, "lon": lon, "amp": amp * env, "active": env > 0.0}
    meta = {"amp_peak_K": round(amp, 3), "t_start_h": round(t0, 1), "duration_h": round(dur, 1),
            "drift_ms": np.round(drift, 3).tolist(), "sx_km": round(sx, 1), "sy_km": round(sy, 1),
            "angle_deg": round(ang, 1)}
    return st, meta, dict(sx=sx, sy=sy, ang=ang)


# =========================================================================== member perturbations
def sigma_pos_km(lead):
    return 20.0 + 1.6 * lead            # ~404 km at 240 h


def sigma_int(lead):
    return 0.03 + 0.25 * lead / 240.0


def member_plans(rng):
    """Roles: 2 'miss' members and 1-2 'false_alarm' members, rest 'normal'."""
    roles = ["normal"] * N_MEMBERS
    idx = rng.permutation(N_MEMBERS)
    for i in idx[:2]:
        roles[i] = "miss"
    for i in idx[2:2 + int(rng.integers(1, 3))]:
        roles[i] = "false_alarm"
    plans = []
    for m in range(N_MEMBERS):
        r = np.random.default_rng([int(rng.integers(1 << 30)), m])
        z = r.standard_normal(2)
        wig = np.zeros((len(LEADS), 2))
        for i in range(1, len(LEADS)):
            wig[i] = 0.85 * wig[i - 1] + r.normal(0, 0.35, 2)
        plans.append({"number": m, "role": roles[m], "z_pos": z, "wiggle": wig,
                      "z_int": float(r.standard_normal()), "miss_lead": float(r.uniform(24, 96)),
                      "rng": r})
    return plans


def perturb_state(st, plan, hazard):
    """Member state from the truth state: position error and intensity error grow with lead."""
    s = {k: np.array(v, dtype=float, copy=True) for k, v in st.items()}
    sp = sigma_pos_km(LEADS)
    dxy = sp[:, None] * (plan["z_pos"][None, :] * 0.8 + 0.45 * plan["wiggle"])
    s["lat"] = st["lat"] + dxy[:, 1] / KM
    s["lon"] = st["lon"] + dxy[:, 0] / (KM * np.cos(np.deg2rad(np.nan_to_num(st["lat"], nan=20))))
    fac = np.exp(sigma_int(LEADS) * plan["z_int"])
    if plan["role"] == "miss":
        fac = fac * np.clip(1 - (LEADS - plan["miss_lead"]) / 24.0, 0, 1)
    key = "vmax" if hazard == "tropical_cyclone" else "amp"
    s[key] = st[key] * fac
    if hazard == "tropical_cyclone":
        s["dp"] = st["dp"] * fac ** 2
        s["active"] = st["active"] & (np.nan_to_num(s["vmax"]) > 12)
    else:
        s["active"] = st["active"] & (np.nan_to_num(s["amp"]) > 0.05)
    return s


def false_alarm_state(cfg, rng):
    """A spurious event only this member has (a 'false alarm')."""
    hz = cfg["hazard"]
    t0 = float(rng.uniform(48, 120))
    dur = float(rng.uniform(60, 108))
    env = ev.amplitude_envelope(LEADS, t0, dur, ramp_h=24)
    if hz == "tropical_cyclone":
        la0, lo0 = (15.0, 66.0) if cfg.get("basin") != "AS" else (14.0, 88.0)
        lat = la0 + 0.012 * np.maximum(LEADS - t0, 0)
        lon = lo0 + (-0.01 if lo0 < 78 else -0.006) * np.maximum(LEADS - t0, 0)
        v = 30.0 * env
        return {"lat": lat, "lon": lon, "vmax": np.where(env > 0, np.maximum(v, 0), np.nan),
                "rmw": np.full(len(LEADS), 45.0), "dp": np.full(len(LEADS), np.nan),
                "active": v > 12}
    la0, lo0 = (20.0, 76.0) if hz == "heat_dome" else (25.0, 80.0)
    if hz == "heat_dome" and abs(cfg["center"][0] - 20) < 3 and abs(cfg["center"][1] - 76) < 4:
        la0, lo0 = 27.0, 71.5
    amp = float(rng.uniform(3.5, 5.0))
    return {"lat": np.full(len(LEADS), la0), "lon": np.full(len(LEADS), lo0),
            "amp": amp * env, "active": env > 0}


# =========================================================================== field synthesis
def cyclone_increments(state, i, lat2d, lon2d, land):
    if not state["active"][i]:
        return None
    la, lo = state["lat"][i], state["lon"][i]
    j0, j1 = max(i - 1, 0), min(i + 1, len(LEADS) - 1)
    if state["active"][j0] and state["active"][j1] and j1 > j0:
        dx = (state["lon"][j1] - state["lon"][j0]) * KM * 1000 * np.cos(np.deg2rad(la))
        dy = (state["lat"][j1] - state["lat"][j0]) * KM * 1000
        motion = (dx / ((j1 - j0) * 6 * 3600), dy / ((j1 - j0) * 6 * 3600))
    else:
        motion = (0.0, 0.0)
    vmax, rmw = float(state["vmax"][i]), float(state["rmw"][i])
    dp = state["dp"][i]
    if np.isfinite(dp):
        B = ev.holland_B(vmax, dp)
    else:
        B, dp = 1.8, ev.holland_dp(vmax, 1.8)
    return ev.tropical_cyclone(lat2d, lon2d, la, lo, vmax, rmw, dp=float(dp), B=B,
                               motion=motion, land=land > 0.5)


def blob_increments(hazard, state, i, lat2d, lon2d, hour, shp, distortion, weight):
    if not state["active"][i]:
        return None
    if hazard == "heat_dome":
        return ev.heat_dome(lat2d, lon2d, state["lat"][i], state["lon"][i], state["amp"][i],
                            shp["sx"], shp["sy"], shp["ang"], hour, distortion,
                            dmsl_amp=60.0 * state["amp"][i], land_w=weight)
    return ev.cold_wave(lat2d, lon2d, state["lat"][i], state["lon"][i], state["amp"][i],
                        shp["sx"], shp["sy"], shp["ang"], hour, weight, distortion,
                        dmsl_amp=80.0 * state["amp"][i])


def compose(bg, incs, hazard, lat, lon, orog, dxkm, dykm):
    """Merge background (dict of 2-D arrays) with event increments; apply physics rules.
    Returns (fields, event_signal) where event_signal is the injected quantity used for masks."""
    f = {k: v.astype(np.float64).copy() for k, v in bg.items()}
    lat2d = lat[:, None] * np.ones((1, len(lon)))
    signal = np.zeros_like(f["t2m"])
    for inc in incs:
        if inc is None:
            continue
        if hazard == "tropical_cyclone":
            f["msl"] += inc["dmsl"]
            f["u10"] += inc["du10"]
            f["v10"] += inc["dv10"]
            q = f.get("q850", np.full_like(f["t2m"], 0.012))
            u8 = f.get("u850", f["u10"]) + 1.1 * inc["du10"]
            v8 = f.get("v850", f["v10"]) + 1.1 * inc["dv10"]
            mfc = moisture_flux_convergence(q, u8, v8, lat, lon)
            oro = orographic_factor(f["u10"], f["v10"], orog, lat, lon)
            f["tp"] += place_rain(inc["rain_rate"], mfc, oro)
            if "r850" in f:
                f["r850"] += (97.0 - f["r850"]) * inc["core"]
            signal = np.minimum(signal, inc["dmsl"])
        else:
            f["t2m"] += inc["dt2m"]
            f["msl"] += inc["dmsl"]
            du, dv = ev.geostrophic_wind(inc["dmsl"], lat2d, dxkm, dykm)
            f["u10"] += du
            f["v10"] += dv
            f["tp"] *= inc["rain_factor"]
            signal = signal + inc["dt2m"]
    f["tp"] = np.maximum(f["tp"], 0.0)
    if "r850" in f:
        f["r850"] = cap_rh(f["r850"])
    return f, signal


def mask_from_signal(hazard, signal):
    if hazard == "tropical_cyclone":
        return signal <= -400.0
    if hazard == "heat_dome":
        return signal >= 3.0
    return signal <= -3.0


def quantize(v, x, pack=PACK):
    sc, off = pack[v]
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite values in {v} before packing")
    lo, hi = -32767, 32767
    q = np.clip(np.round((x - off) / sc), lo, hi).astype(np.int16)
    if v in ("tp", "r850"):
        q = np.maximum(q, 0)
    return q


def dequant(v, q):
    sc, off = PACK[v]
    return q.astype(np.float64) * sc + off


# =========================================================================== netCDF writers
def _nc(path, dims, lat, lon, init, attrs):
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    for d, n in dims.items():
        ds.createDimension(d, n)
    v = ds.createVariable("latitude", "f8", ("latitude",))
    v[:] = lat
    v.units, v.standard_name = "degrees_north", "latitude"
    v = ds.createVariable("longitude", "f8", ("longitude",))
    v[:] = lon
    v.units, v.standard_name = "degrees_east", "longitude"
    v = ds.createVariable("step", "i4", ("step",))
    v[:] = LEADS
    v.long_name, v.units = "forecast lead time", "hours"
    epoch = "hours since 1970-01-01 00:00:00"
    v = ds.createVariable("time", "f8", ())
    v[:] = netCDF4.date2num(init.to_pydatetime(), epoch)
    v.units, v.standard_name, v.long_name = epoch, "forecast_reference_time", "initial time"
    v = ds.createVariable("valid_time", "f8", ("step",))
    v[:] = [netCDF4.date2num((init + pd.Timedelta(hours=int(h))).to_pydatetime(), epoch)
            for h in LEADS]
    v.units, v.standard_name = epoch, "time"
    ds.setncatts(attrs)
    return ds


def _packed(ds, name, dims, chunks, pack=PACK):
    u, ln, short = META[name]
    sc, off = pack[name]
    v = ds.createVariable(name, "i2", dims, zlib=True, complevel=4, shuffle=True,
                          chunksizes=chunks, fill_value=np.int16(-32768))
    v.set_auto_maskandscale(False)
    v.setncattr("scale_factor", np.float64(sc))
    v.setncattr("add_offset", np.float64(off))
    v.units, v.long_name, v.neps_g_shortName = u, ln, short
    if name == "r850":
        v.level_hPa = 850
    return v


# =========================================================================== main case builder
def build_case(case_id):
    t_start = time.time()
    cfg = CASES[case_id]
    hazard = cfg["hazard"]
    rng = np.random.default_rng(cfg["seed"])
    init = pd.Timestamp(cfg["init"])
    valid = [init + pd.Timedelta(hours=int(h)) for h in LEADS]
    out = OUT / case_id
    out.mkdir(parents=True, exist_ok=True)

    cyclone = hazard == "tropical_cyclone"
    bg = load_background(cfg["window"], with_850=cyclone)
    variables = ["t2m", "u10", "v10", "msl", "tp"] + (["r850"] if cyclone and
                                                        getattr(bg, "with_850", False) else [])
    orog5, orog12, dem_src = static_fields()
    rg5 = Regridder(bg.lat, bg.lon, LAT5, LON5)
    lat5_2d, lon5_2d = np.meshgrid(LAT5, LON5, indexing="ij")
    lat12_2d, lon12_2d = np.meshgrid(LAT12, LON12, indexing="ij")
    land5, land12 = land_weight(orog5), land_weight(orog12)
    dx5 = 0.04 * KM * np.cos(np.deg2rad(LAT5))[:, None]
    dx12 = 0.12 * KM * np.cos(np.deg2rad(LAT12))[:, None]

    # ---- truth event state
    shp, distortion5 = None, None
    if cyclone:
        st, meta = cyclone_truth_state(cfg, rng, land12)
        weight5 = weight12 = None
    else:
        st, meta, shp = blob_truth_state(cfg, rng)
        distortion5 = spectral_noise((len(LAT5), len(LON5)), rng, slope=-3.0, kmin=3)
        weight5 = land5 if hazard == "heat_dome" else plains_weight(orog5, LAT5)
        weight12 = avgpool(weight5)
    distortion12 = None if distortion5 is None else avgpool(distortion5)

    attrs = {"Conventions": "CF-1.8", "synthetic": "true", "case_id": case_id, "hazard": hazard,
             "title": f"SYNTHETIC {hazard} case {case_id} (NOT real observations/forecasts)",
             "background": bg.source, "dem": dem_src, "generator": VERSION, "seed": cfg["seed"],
             "institution": "SIH 26078 research prototype",
             "history": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ} synth.generate"}

    # ---- truth at G5
    tds = _nc(out / "truth_5km.nc", {"step": len(LEADS), "latitude": len(LAT5),
                                    "longitude": len(LON5)}, LAT5, LON5, init,
              dict(attrs, grid="G5 0.04 deg"))
    tv = {v: _packed(tds, v, ("step", "latitude", "longitude"), (1, 333, 333)) for v in variables}
    tmask = tds.createVariable("event_mask", "u1", ("step", "latitude", "longitude"), zlib=True,
                               complevel=4, chunksizes=(1, 333, 333))
    tmask.long_name, tmask.definition = "exact event mask", MASK_DEF[hazard]

    fds = _nc(out / "fcst_12km.nc", {"number": N_MEMBERS, "step": len(LEADS),
                                    "latitude": len(LAT12), "longitude": len(LON12)},
              LAT12, LON12, init, dict(attrs, grid="G12 0.12 deg"))
    num = fds.createVariable("number", "i4", ("number",))
    num[:] = np.arange(N_MEMBERS)
    num.long_name = "ensemble member number"
    fv = {v: _packed(fds, v, ("number", "step", "latitude", "longitude"), (1, 1, 333, 333),
                     PACK_MEMBER) for v in variables}
    ftruth = {}
    for v in variables:
        u, ln, short = META[v]
        x = fds.createVariable(f"{v}_truth", "f4", ("step", "latitude", "longitude"), zlib=True,
                               complevel=4, chunksizes=(1, 333, 333))
        x.units, x.long_name = u, f"{ln}: truth_5km block-averaged 3x3 to G12 (exact)"
        ftruth[v] = x
    fmask_t = fds.createVariable("event_mask_truth", "u1", ("step", "latitude", "longitude"),
                                 zlib=True, chunksizes=(1, 333, 333))
    fmask_t.definition = "fraction of G5 truth mask in the G12 cell >= 0.5"
    fmask = fds.createVariable("event_mask", "u1", ("number", "step", "latitude", "longitude"),
                               zlib=True, chunksizes=(1, 1, 333, 333))
    fmask.definition = MASK_DEF[hazard] + " (member's own injected event incl. false alarms)"

    noise5 = {v: AR1Noise((len(LAT5), len(LON5)), np.random.default_rng([cfg["seed"], 7, k]),
                          kmin=9, rho=0.7) for k, v in enumerate(variables)}
    bg12_all = {v: np.zeros((len(LEADS), len(LAT12), len(LON12)), np.float32)
                for v in variables + (["q850", "u850", "v850"] if cyclone else [])}
    truth_track = []
    for i, t in enumerate(valid):
        raw = bg.fields(t)
        b5 = {k: rg5(v) for k, v in raw.items()}
        for k in bg12_all:
            if k in b5:
                bg12_all[k][i] = avgpool(b5[k])
        if cyclone:
            inc = [cyclone_increments(st, i, lat5_2d, lon5_2d, land5)]
        else:
            inc = [blob_increments(hazard, st, i, lat5_2d, lon5_2d, t.hour, shp, distortion5,
                                   weight5)]
        f, sig = compose(b5, inc, hazard, LAT5, LON5, orog5, dx5, 0.04 * KM)
        for v in variables:
            n = noise5[v].step()
            if v == "tp":
                f[v] = f[v] * np.exp(FINE[v] * n - 0.5 * FINE[v] ** 2)
            else:
                f[v] = f[v] + FINE[v] * n
        if "r850" in f:
            f["r850"] = cap_rh(f["r850"])
        mask = mask_from_signal(hazard, sig)
        for v in variables:
            q = quantize(v, f[v])
            tv[v][i] = q
            ftruth[v][i] = avgpool(dequant(v, q)).astype(np.float32)
        tmask[i] = mask.astype(np.uint8)
        fmask_t[i] = (avgpool(mask.astype(np.float32)) >= 0.5).astype(np.uint8)
        truth_track.append(track_entry(i, t, st, mask, sig, hazard, f))
    tds.close()
    t_truth = time.time() - t_start

    # ---- ensemble at G12
    plans = member_plans(rng)
    members_lab = []
    for p in plans:
        r = p["rng"]
        ms = perturb_state(st, p, hazard)
        fa = false_alarm_state(cfg, r) if p["role"] == "false_alarm" else None
        big = {v: AR1Noise((len(LAT12), len(LON12)), r, slope=-3.0, rho=0.9, kmin=1)
               for v in variables}
        fine = {v: AR1Noise((len(LAT12), len(LON12)), r, rho=0.7, kmin=9) for v in variables}
        mtrack = []
        for i, t in enumerate(valid):
            g = LEADS[i] / 240.0
            b12 = {k: bg12_all[k][i].astype(np.float64) for k in bg12_all}
            amp_bg = {"t2m": 0.2 + 1.3 * g, "u10": 0.3 + 2.2 * g, "v10": 0.3 + 2.2 * g,
                      "msl": 30 + 220 * g, "tp": 0.15 + 0.6 * g, "r850": 2 + 8 * g}
            for v in variables:
                n = big[v].step()
                if v == "tp":
                    b12[v] = b12[v] * np.exp(amp_bg[v] * n - 0.5 * amp_bg[v] ** 2)
                else:
                    b12[v] = b12[v] + amp_bg[v] * n
            if cyclone:
                inc = [cyclone_increments(ms, i, lat12_2d, lon12_2d, land12)]
                if fa is not None:
                    inc.append(cyclone_increments(fa, i, lat12_2d, lon12_2d, land12))
            else:
                inc = [blob_increments(hazard, ms, i, lat12_2d, lon12_2d, t.hour, shp,
                                       distortion12, weight12)]
                if fa is not None:
                    fshp = dict(sx=300.0, sy=250.0, ang=0.0)
                    inc.append(blob_increments(hazard, fa, i, lat12_2d, lon12_2d, t.hour, fshp,
                                               None, weight12))
            f, sig = compose(b12, inc, hazard, LAT12, LON12, orog12, dx12, 0.12 * KM)
            for v in variables:
                n = fine[v].step()
                if v == "tp":
                    f[v] = f[v] * np.exp(0.6 * FINE[v] * n - 0.18 * FINE[v] ** 2)
                else:
                    f[v] = f[v] + 0.6 * FINE[v] * n
                if v == "r850":
                    f[v] = cap_rh(f[v])
                fv[v][p["number"], i] = quantize(v, f[v], PACK_MEMBER)
            fmask[p["number"], i] = mask_from_signal(hazard, sig).astype(np.uint8)
            mtrack.append({"lead_h": int(LEADS[i]), "active": bool(ms["active"][i]),
                           "center_lat": _r(ms["lat"][i]) if ms["active"][i] else None,
                           "center_lon": _r(ms["lon"][i]) if ms["active"][i] else None,
                           "intensity": _r((ms["vmax"] if cyclone else ms["amp"])[i])
                           if ms["active"][i] else None})
        lab = {"number": p["number"], "role": p["role"], "track": mtrack}
        if fa is not None:
            lab["false_alarm_track"] = [
                {"lead_h": int(LEADS[i]), "center_lat": _r(fa["lat"][i]),
                 "center_lon": _r(fa["lon"][i])} for i in range(len(LEADS)) if fa["active"][i]]
        if p["role"] == "miss":
            lab["miss_lead_h"] = round(p["miss_lead"], 1)
        members_lab.append(lab)
    role_code = {"normal": 0, "miss": 1, "false_alarm": 2}
    rv = fds.createVariable("member_role", "i1", ("number",))
    rv[:] = [role_code[p["role"]] for p in plans]
    rv.flag_values = np.array([0, 1, 2], np.int8)
    rv.flag_meanings = "normal miss false_alarm"
    fds.close()

    labels = make_labels(case_id, cfg, hazard, init, valid, st, meta, truth_track,
                         members_lab, bg.source, variables)
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"{case_id}: truth {t_truth:.0f}s, total {time.time() - t_start:.0f}s")
    return labels


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def cell_area_km2(lat):
    return (0.04 * KM) ** 2 * np.cos(np.deg2rad(lat))


def track_entry(i, t, st, mask, sig, hazard, f):
    e = {"lead_h": int(LEADS[i]), "valid_time": t.isoformat(), "active": bool(mask.any())}
    e["center_lat"] = _r(st["lat"][i]) if st["active"][i] else None
    e["center_lon"] = _r(st["lon"][i]) if st["active"][i] else None
    if mask.any():
        ii, jj = np.nonzero(mask)
        w = np.abs(sig[ii, jj])
        e["mask_centroid_lat"] = _r(np.average(LAT5[ii], weights=w))
        e["mask_centroid_lon"] = _r(np.average(LON5[jj], weights=w))
        e["mask_area_km2"] = round(float(cell_area_km2(LAT5[ii]).sum()), 1)
        e["bbox"] = [_r(LAT5[ii].min() - 0.02), _r(LAT5[ii].max() + 0.02),
                     _r(LON5[jj].min() - 0.02), _r(LON5[jj].max() + 0.02)]
    if hazard == "tropical_cyclone":
        e["vmax_ms"] = _r(st["vmax"][i]) if st["active"][i] else None
        e["rmw_km"] = _r(st["rmw"][i]) if st["active"][i] else None
        if mask.any():
            k = np.unravel_index(np.argmin(np.where(mask, f["msl"], np.inf)), mask.shape)
            e["min_msl_pa"] = _r(f["msl"][k], 1)
    else:
        e["amp_K"] = _r(st["amp"][i])
        if mask.any():
            e["peak_anom_K"] = _r(sig[mask].max() if hazard == "heat_dome" else sig[mask].min())
    return e


def make_labels(case_id, cfg, hazard, init, valid, st, meta, track, members, bg_src, variables):
    act = [e for e in track if e["active"]]
    box = None
    if act:
        box = {"lat_min": min(e["bbox"][0] for e in act), "lat_max": max(e["bbox"][1] for e in act),
               "lon_min": min(e["bbox"][2] for e in act), "lon_max": max(e["bbox"][3] for e in act),
               "level_hPa_bottom": LEVELS[hazard][0], "level_hPa_top": LEVELS[hazard][1],
               "time_start": act[0]["valid_time"], "time_end": act[-1]["valid_time"],
               "lead_start_h": act[0]["lead_h"], "lead_end_h": act[-1]["lead_h"]}
    if hazard == "tropical_cyclone":
        ks = [e for e in act if e.get("min_msl_pa") is not None]
        pk = min(ks, key=lambda e: e["min_msl_pa"]) if ks else None
        peak = None if pk is None else {"variable": "msl", "value_pa": pk["min_msl_pa"],
                                        "lead_h": pk["lead_h"],
                                        "vmax_ms": max(e["vmax_ms"] or 0 for e in act)}
    else:
        pk = max(act, key=lambda e: abs(e["peak_anom_K"])) if act else None
        peak = None if pk is None else {"variable": "t2m_anomaly", "value_K": pk["peak_anom_K"],
                                        "lead_h": pk["lead_h"]}
    return {"case_id": case_id, "synthetic": "true", "hazard": hazard,
            "init_time": init.isoformat(), "lead_hours": LEADS.tolist(),
            "valid_times": [t.isoformat() for t in valid], "background": bg_src,
            "generator": VERSION, "seed": cfg["seed"], "variables": variables,
            "mask_definition": MASK_DEF[hazard],
            "mask_storage": "truth_5km.nc:event_mask, fcst_12km.nc:event_mask_truth / event_mask",
            "event_params": meta, "levels_hPa": LEVELS[hazard], "bbox_4d": box, "peak": peak,
            "truth_track": track, "members": members,
            "roles_summary": {r: sum(m["role"] == r for m in members)
                              for r in ("normal", "miss", "false_alarm")}}


if __name__ == "__main__":
    ids = sys.argv[1:] or list(CASES)
    for cid in ids:
        build_case(cid)
