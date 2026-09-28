"""BRIEF4 Phase 1: trackers on REAL data.

A. Real ENSEMBLE forecasts of Amphan 2020 (WB2 IFS ENS 1.5 deg, GenCast 1.5 deg, IFS ENS 0.25 deg
   flagship run), regridded bilinearly to G12, tracked member by member with tracker v2, the GNN
   (full: uses the real ensemble's cross-member edges) and TE-style. Matched to IBTrACS:
   track error by lead (24/48/72/120/168/240 h), ensemble detection probability, and the first
   init at which P(Amphan detected) >= 0.5 (hours before landfall).
B. REAL-VAL cyclones in ERA5 reanalysis (single run): track error vs IBTrACS.
REAL-TEST events are NOT evaluated here (reserved for BRIEF4 Phase 8).

    python -m pipeline.real_eval [all|ens|era5|report] [file-glob]
        -> per-forecast results cached in reports/real_cache/ (resumable), then
           reports/REAL_RESULTS.md, reports/real_results.json
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from .anomaly import Climatology
from .jsonutil import dumps as strict_dumps
from .evaluate2 import orog12
from .gnn_eval import decode, gnn_main_track, infer, load_model
from .graphs import build_graph
from .te_style import te_blobs, te_cyclones
from .track import haversine_km
from .tracker2 import Tracker2

ROOT = Path(__file__).resolve().parents[1]
ENS = ROOT / "data" / "real" / "ensembles"
REP = ROOT / "reports"
CACHE = REP / "real_cache"
LEADS_REPORT = [24, 48, 72, 120, 168, 240]
MATCH_KM = 500.0
PEAK = [0.0]


def rss():
    PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss / 1e6)


def ibtracs(path):
    ib = pd.read_csv(path, keep_default_na=False)
    ib["time"] = pd.to_datetime(ib.ISO_TIME)
    ib = ib[ib.time.dt.hour % 6 == 0].reset_index(drop=True)
    ib["LAT"], ib["LON"] = pd.to_numeric(ib.LAT), pd.to_numeric(ib.LON)
    return ib


def landfall_time(ib):
    d = pd.to_numeric(ib.DIST2LAND, errors="coerce")
    k = int(np.argmax(pd.to_numeric(ib.USA_WIND, errors="coerce").fillna(0).values))
    after = ib.iloc[k:]
    hit = after[pd.to_numeric(after.DIST2LAND, errors="coerce") <= 0]
    return hit.time.iloc[0] if len(hit) else None


def to_g12(ds_var, lat, lon):
    from synth.grids import LAT12, LON12
    from synth.background import Regridder
    rg = Regridder(lat, lon, LAT12, LON12)
    return rg


def match_track(tracks, times, ib):
    """The member track closest to IBTrACS (mean distance over common times; needs >= 1 common time
    within MATCH_KM). Returns {valid_time: (lat, lon)} or None."""
    ibt = ib.set_index("time")
    best = None
    for tr in tracks:
        d = []
        for t, o in tr:
            if times[t] in ibt.index:
                r = ibt.loc[times[t]]
                d.append(haversine_km(o["center_lat"], o["center_lon"], r.LAT, r.LON))
        if d and min(d) <= MATCH_KM and (best is None or np.mean(d) < best[0]):
            best = (np.mean(d), tr)
    if best is None:
        return None
    return {times[t]: (o["center_lat"], o["center_lon"]) for t, o in best[1]}


def run_methods(msl, times, clim, oro, v2p, models, dec, lat12, lon12):
    """msl: (M, T, 333, 333) float32 on G12 (NaN where no lead). Returns method -> member tracks."""
    out = {}
    tk = Tracker2("tropical_cyclone", clim, oro, lat12, lon12, **v2p)
    out["v2"] = [tk.run(msl[m].astype(np.float64), times)[0] for m in range(msl.shape[0])]
    rss()
    out["te_style"] = [te_cyclones(msl[m].astype(np.float64), lat12, lon12) for m in range(msl.shape[0])]
    rss()
    g = build_graph("tropical_cyclone", msl, times, lat12, lon12, clim)   # float32 (memory)
    rss()
    for v in ("full", "temporal"):
        model, st, _ = models[v]
        pn, pe = infer(model, st, g)
        out[f"gnn_{v}"] = [decode(g, pn, pe, m, "tropical_cyclone", **dec[v]["tropical_cyclone"])
                           for m in range(msl.shape[0])]
    del g
    gc.collect()
    return out


def score_init(tracks_by_method, times, init, ib):
    res = {}
    for meth, member_tracks in tracks_by_method.items():
        matched = [match_track(trs, times, ib) for trs in member_tracks]
        M = len(matched)
        det = sum(x is not None for x in matched) / M
        per_lead = {}
        for L in LEADS_REPORT:
            vt = init + pd.Timedelta(hours=L)
            row = ib[ib.time == vt]
            if row.empty:
                continue
            la, lo = float(row.LAT.iloc[0]), float(row.LON.iloc[0])
            pos = [x[vt] for x in matched if x is not None and vt in x]
            errs = [haversine_km(a, b, la, lo) for a, b in pos]
            e = {"p_detect": len(pos) / M, "n": len(pos)}
            if pos:
                ma, mo = np.mean([p[0] for p in pos]), np.mean([p[1] for p in pos])
                e.update({"ens_mean_err_km": float(haversine_km(ma, mo, la, lo)),
                          "median_member_err_km": float(np.median(errs))})
            per_lead[L] = e
        res[meth] = {"p_detect_any": det, "per_lead": per_lead}
    return res


def load_ensemble(path, clim_rg_cache):
    ds = xr.open_dataset(path)
    init = pd.Timestamp(ds.attrs["init_time"])
    steps = ds.step.values.astype(int)
    lat, lon = ds.latitude.values, ds.longitude.values
    key = (len(lat), len(lon))
    if key not in clim_rg_cache:
        clim_rg_cache[key] = to_g12(None, lat, lon)
    rg = clim_rg_cache[key]
    grid6 = np.arange(0, steps.max() + 1, 6)
    M = ds.sizes["number"]
    out = np.full((M, len(grid6), 333, 333), np.nan, np.float32)
    msl = ds.msl.values
    for j, s in enumerate(steps):
        k = int(np.where(grid6 == s)[0][0])
        for m in range(M):
            out[m, k] = rg(msl[m, j])
    ds.close()
    times = pd.DatetimeIndex([init + pd.Timedelta(hours=int(h)) for h in grid6])
    return out, times, init


def main():
    clim, oro = Climatology(), orog12()
    from synth.grids import LAT12, LON12
    v2p = json.loads((REP / "tracker2_params.json").read_text())["chosen"]["tropical_cyclone"]["params"]
    gr = json.loads((REP / "gnn_results.json").read_text())
    dec = gr["decode"]
    models = {v: load_model(v) for v in ("full", "temporal")}
    ib = ibtracs(ROOT / "data/real/ibtracs/amphan_2020_ibtracs.csv")
    lf = landfall_time(ib)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*"
    CACHE.mkdir(parents=True, exist_ok=True)
    if mode in ("ens", "all"):
        rgc = {}
        for p in sorted(ENS.glob(f"{pattern}.nc")):
            name = p.stem                                # e.g. ifs_ens_1p5_2020051600
            cp = CACHE / f"{name}.json"
            if cp.exists():                              # resumable / idempotent
                continue
            msl, times, init = load_ensemble(p, rgc)
            rss()
            try:
                for t in times:
                    clim.get("msl", t)
            except KeyError as e:                        # climatology not fetched yet: retry later
                print(f"skip {name}: {e}", flush=True)
                del msl
                continue
            tracks = run_methods(msl, times, clim, oro, v2p, models, dec, LAT12, LON12)
            with xr.open_dataset(p) as d:
                step = int(np.diff(d.step.values)[0])
            r = {"init": str(init), "members": int(msl.shape[0]), "lead_step_h": step,
                 **score_init(tracks, times, init, ib), "peak_rss_mb": PEAK[0]}
            cp.write_text(strict_dumps(r, indent=1))
            print(name, {m: round(v["p_detect_any"], 2) for m, v in r.items() if isinstance(v, dict)},
                  f"RSS {PEAK[0]:.0f} MB", flush=True)
            del msl, tracks
            gc.collect()
    if mode in ("era5", "all"):
        era5 = era5_realval(clim, oro, v2p, models, dec, LAT12, LON12)
        v2all = json.loads((REP / "tracker2_params.json").read_text())["chosen"]
        hc = era5_heatcold_realval(clim, oro, v2all, models, dec)
        rss()
        (CACHE / "era5_real_scores.json").write_text(strict_dumps(
            {"results": era5, "heat_cold": hc, "peak_rss_mb": PEAK[0]}, indent=1))
    if mode in ("report", "all"):
        res_ens = {p.stem: json.loads(p.read_text()) for p in sorted(CACHE.glob("*_20*.json"))}
        ep = CACHE / "era5_real_scores.json"
        e = json.loads(ep.read_text()) if ep.exists() else {"results": {}, "heat_cold": {}, "peak_rss_mb": 0}
        peak = max([r.get("peak_rss_mb", 0) for r in res_ens.values()] + [e["peak_rss_mb"]])
        out = {"landfall": str(lf), "genesis": str(ib.time.iloc[0]), "ensembles": res_ens, "era5_real_val": e["results"],
               "era5_heat_cold": e.get("heat_cold", {}), "peak_rss_mb": peak}
        (REP / "real_results.json").write_text(strict_dumps(out, indent=1))
        write_report(out)


def era5_realval(clim, oro, v2p, models, dec, LAT12, LON12):
    splits = json.loads((ROOT / "data/real/SPLITS.json").read_text())
    out = {}
    for eid in splits["REAL-VAL"]:
        ev = splits["events"][eid]
        if ev["hazard"] != "tropical_cyclone":
            continue
        f = (ROOT / "data/real/era5/amphan_era5_g12.nc" if eid == "amphan_2020"
             else ROOT / f"data/real/events/{eid}_sfc_g12.nc")
        if not f.exists():
            out[eid] = {"status": "missing ERA5 file"}
            continue
        ib = ibtracs(ROOT / "data/real/ibtracs/events" / f"{eid}.csv")
        ds = xr.open_dataset(f)
        times = pd.to_datetime(ds.time.values)
        try:
            for t in times[[0, -1]]:
                clim.get("msl", t)
        except KeyError as e:
            out[eid] = {"status": f"no climatology: {e}"}
            continue
        msl = ds.msl.values.astype(np.float32)[None]
        ds.close()
        tracks = run_methods(msl, times, clim, oro, v2p, models, dec, LAT12, LON12)
        r = {}
        for meth, mt in tracks.items():
            x = match_track(mt[0], times, ib)
            ibt = ib.set_index("time")
            errs = [haversine_km(a, b, ibt.loc[t].LAT, ibt.loc[t].LON) for t, (a, b) in (x or {}).items()
                    if t in ibt.index]
            n_ib = int(((ib.time >= times[0]) & (ib.time <= times[-1])).sum())
            r[meth] = {"matched_fixes": len(errs), "ibtracs_fixes": n_ib,
                       "err_mean_km": float(np.mean(errs)) if errs else None,
                       "err_median_km": float(np.median(errs)) if errs else None,
                       "n_tracks": len(mt[0])}
        out[eid] = r
        print(eid, {m: (v["matched_fixes"], v["err_mean_km"] and round(v["err_mean_km"])) for m, v in r.items()},
              flush=True)
        gc.collect()
    return out


def heatcold_reference(t2m, times, clim, reg, hz, min_steps=8, min_area_km2=50000.0):
    """REFERENCE event mask for a real heat/cold wave: IMD operational criteria applied to ERA5
    daily Tmax/Tmin (departure >= 4.5 C from the ERA5 1990-2019 normal + regional absolute
    threshold), kept where it persists >= 2 days (8 steps) and covers >= min_area_km2 at that step.
    This is NOT an independent truth (it comes from the same reanalysis the trackers read)."""
    from .tracker2 import cell_area_km2, imd_field
    from synth.grids import LAT12, LON12
    f = imd_field(t2m, times, clim, reg, hz)
    m = f >= 4.5
    run = np.zeros_like(m, np.int16)                     # persistence: consecutive steps
    for t in range(len(m)):
        run[t] = np.where(m[t], (run[t - 1] + 1) if t else 1, 0)
    keep = np.zeros_like(m)
    for t in range(len(m)):                              # a cell is in the event if it is inside
        lo, hi = max(0, t - min_steps + 1), min(len(m), t + min_steps)   # a >= 2-day run
        keep[t] = m[t] & (run[lo:hi].max(0) >= min_steps)
    area = cell_area_km2(LAT12, LON12)
    ok_t = (keep * area).sum((1, 2)) >= min_area_km2
    keep[~ok_t] = False
    return keep


def score_heatcold(tracks, ref):
    """Step-level POD / FAR / CSI (event present anywhere) and IoU on steps where both exist."""
    T = len(ref)
    det = np.zeros(ref.shape, bool)
    for tr in tracks:
        for t, o in tr:
            det[t] |= o["mask"]
    has_ref, has_det = ref.any((1, 2)), det.any((1, 2))
    hits = int((has_ref & has_det).sum())
    miss = int((has_ref & ~has_det).sum())
    fa = int((~has_ref & has_det).sum())
    ious = [float((ref[t] & det[t]).sum() / max((ref[t] | det[t]).sum(), 1)) for t in range(T)
            if has_ref[t] and has_det[t]]
    return {"steps": T, "ref_steps": int(has_ref.sum()), "hits": hits, "misses": miss, "false_alarms": fa,
            "pod": hits / max(hits + miss, 1), "far": fa / max(hits + fa, 1), "csi": hits / max(hits + miss + fa, 1),
            "iou_mean": float(np.mean(ious)) if ious else None, "n_tracks": len(tracks)}


def imd_check(eid, t2m, times, ref, hz):
    """Agreement of ERA5 daily Tmax/Tmin with IMD 1 deg gridded Tmax/Tmin inside the reference
    event (bias, MAE), if data/real/imd/imd_{tmax|tmin}_{year}_1deg.nc exists."""
    from .tracker2 import trailing
    kind = "tmax" if hz == "heat_dome" else "tmin"
    yrs = sorted({t.year for t in times})
    fs = [ROOT / f"data/real/imd/imd_{kind}_{y}_1deg.nc" for y in yrs]
    if not all(f.exists() for f in fs):
        return {"status": "no IMD temperature file"}
    imd = xr.concat([xr.open_dataset(f) for f in fs], "time")
    v = list(imd.data_vars)[0]
    ext = trailing(t2m, "max" if hz == "heat_dome" else "min") - 273.15
    from synth.grids import LAT12, LON12
    d, n = [], 0
    for t in range(len(times)):
        if times[t].hour != 12 or not ref[t].any():
            continue
        day = times[t].normalize()
        if day not in pd.to_datetime(imd.time.values):
            continue
        g = imd[v].sel(time=day).interp(latitude=xr.DataArray(LAT12, dims="y"), longitude=xr.DataArray(LON12, dims="x")).values
        ok = ref[t] & np.isfinite(g) & (g > -90) & (g < 90)
        d.append((ext[t][ok] - g[ok]))
        n += int(ok.sum())
    if not n:
        return {"status": "no overlapping IMD cells"}
    d = np.concatenate(d)
    return {"status": "ok", "cells_days": n, "bias_era5_minus_imd_C": float(d.mean()), "mae_C": float(np.abs(d).mean())}


def era5_heatcold_realval(clim, oro, v2p_all, models, dec):
    from .tracker2 import regions
    from synth.grids import LAT12, LON12
    splits = json.loads((ROOT / "data/real/SPLITS.json").read_text())
    reg = regions(oro, LAT12, LON12)
    out = {}
    for eid in splits["REAL-VAL"]:
        ev = splits["events"][eid]
        hz = ev["hazard"]
        if hz == "tropical_cyclone":
            continue
        f = ROOT / f"data/real/events/{eid}_sfc_g12.nc"
        if not f.exists():
            out[eid] = {"status": "missing ERA5 file"}
            continue
        ds = xr.open_dataset(f)
        times = pd.to_datetime(ds.time.values)
        t2m = ds.t2m.values.astype(np.float64)
        ds.close()
        try:
            for t in times[[0, -1]]:
                clim.get("t2m", t)
        except KeyError as e:
            out[eid] = {"status": f"no climatology: {e}"}
            continue
        ref = heatcold_reference(t2m, times, clim, reg, hz)
        tk = Tracker2(hz, clim, oro, LAT12, LON12, **v2p_all[hz]["params"])
        res = {"v2": score_heatcold(tk.run(t2m, times)[0], ref)}
        g = build_graph(hz, t2m[None].astype(np.float32), times, LAT12, LON12, clim)
        model, st, _ = models["temporal"]                 # a single run: no cross-member edges
        pn, pe = infer(model, st, g)
        res["gnn_temporal"] = score_heatcold(decode(g, pn, pe, 0, hz, **dec["temporal"][hz]), ref)
        res["te_style"] = score_heatcold(te_blobs(t2m, times, clim, oro, LAT12, LON12, hz), ref)
        out[eid] = {"hazard": hz, "reference": "IMD criteria on ERA5 daily Tmax/Tmin, >= 2 days, >= 50000 km2",
                    "ref_peak_area_cells": int(ref.sum((1, 2)).max()), "methods": res,
                    "imd_check": imd_check(eid, t2m, times, ref, hz)}
        print(eid, {m: (round(v["pod"], 2), round(v["far"], 2), v["iou_mean"] and round(v["iou_mean"], 2))
                    for m, v in res.items()}, flush=True)
        del g, t2m
        gc.collect()
    return out


def fmt(x, n=0):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def write_report(out):
    L = ["# Real-data evaluation (BRIEF4 Phase 1)", "",
         "Everything in this report is **REAL**: WeatherBench 2 public ensemble forecasts, ERA5 reanalysis and "
         "IBTrACS best tracks. The trackers were trained or tuned on SYNTHETIC data only. REAL-TEST events are "
         "locked in data/real/SPLITS.json and were NOT evaluated here.", "",
         f"Amphan landfall (first IBTrACS fix with DIST2LAND <= 0 after peak): {out['landfall']}.", "",
         "## A. Real ensemble forecasts of Amphan", "",
         "The forecasts are regridded bilinearly to G12 and tracked member by member. A member \"detects\" "
         f"Amphan when one of its tracks comes within {MATCH_KM:.0f} km of IBTrACS at a common time; the "
         "closest such track is used.",
         "- `p_detect_any`: the fraction of members that detect Amphan at any lead.",
         "- Per-lead columns: `p` = fraction of members with a matched position at that lead; errors (km) of "
         "the ensemble-mean position and of the median member.",
         "- GenCast has 12-hourly leads (odd 6-h steps are empty).", ""]
    ens = out["ensembles"]
    methods = ["v2", "gnn_full", "gnn_temporal", "te_style"]
    for model in ("ifs_ens_1p5", "gencast_1p5", "ifs_ens_0p25"):
        keys = [k for k in ens if k.startswith(model)]
        if not keys:
            continue
        L += [f"### {model} ({ens[keys[0]]['members']} members, {ens[keys[0]]['lead_step_h']} h steps)", "",
              "#### Track error by lead time (km)", "",
              "Error of the ensemble-mean position of the matched member tracks, averaged over all inits that have "
              "an IBTrACS fix at that valid time; p = mean fraction of members with a matched position; n = inits:", "",
              "| method | " + " | ".join(f"{l} h" for l in LEADS_REPORT) + " |",
              "|---|" + "---|" * len(LEADS_REPORT)]
        for meth in methods:
            cells = []
            for Ld in LEADS_REPORT:
                e = [ens[k][meth]["per_lead"].get(Ld) or ens[k][meth]["per_lead"].get(str(Ld)) for k in keys]
                e = [x for x in e if x]
                em = [x["ens_mean_err_km"] for x in e if "ens_mean_err_km" in x]
                cells.append(f"{fmt(np.mean(em))} (p {np.mean([x['p_detect'] for x in e]):.2f}, n={len(e)})"
                             if e else "n/a")
            L.append(f"| {meth} | " + " | ".join(cells) + " |")
        L += ["", "#### P ≥ 0.5 first-flag lead time", "",
              "The earliest forecast init at which at least half of the members detect Amphan (a track within "
              f"{MATCH_KM:.0f} km of IBTrACS at a common time), in hours before landfall and before the first IBTrACS "
              "fix (genesis):", "",
              "| method | earliest init flagged | hours before landfall | hours before first IBTrACS fix | P at that init |",
              "|---|---|---|---|---|"]
        lf = pd.Timestamp(out["landfall"]) if out["landfall"] != "None" else None
        for meth in methods:
            fl = [(pd.Timestamp(ens[k]["init"]), ens[k][meth]["p_detect_any"]) for k in sorted(keys)
                  if ens[k][meth]["p_detect_any"] >= 0.5]
            if fl:
                t0, p = fl[0]
                g0 = pd.Timestamp(out["genesis"]) if out.get("genesis") else None
                L.append(f"| {meth} | {t0} | {fmt((lf - t0) / pd.Timedelta('1h')) if lf else 'n/a'} | "
                         f"{fmt((g0 - t0) / pd.Timedelta('1h')) if g0 is not None else 'n/a'} | {p:.2f} |")
            else:
                L.append(f"| {meth} | never | - | - | - |")
        L += ["", "Detection fraction by init:", "", "| init | " + " | ".join(methods) + " |",
              "|---|" + "---|" * len(methods)]
        for k in sorted(keys):
            L.append(f"| {ens[k]['init']} | " + " | ".join(f"{ens[k][m]['p_detect_any']:.2f}" for m in methods) + " |")
        L.append("")
    L += ["## B. REAL-VAL cyclones in ERA5 reanalysis (single run)", "",
          "| event | method | matched fixes / IBTrACS fixes | mean err (km) | median err (km) | tracks |",
          "|---|---|---|---|---|---|"]
    for eid, r in out["era5_real_val"].items():
        if "status" in r:
            L.append(f"| {eid} | - | {r['status']} | | | |")
            continue
        for meth, v in r.items():
            L.append(f"| {eid} | {meth} | {v['matched_fixes']}/{v['ibtracs_fixes']} | {fmt(v['err_mean_km'])} | "
                     f"{fmt(v['err_median_km'])} | {v['n_tracks']} |")
    hc = out.get("era5_heat_cold", {})
    if hc:
        L += ["", "## C. REAL-VAL heat and cold waves in ERA5 reanalysis (single run)", "",
              "Reference = IMD operational criteria (departure >= 4.5 C from the ERA5 1990-2019 normal + regional "
              "absolute Tmax/Tmin threshold) applied to ERA5 daily Tmax/Tmin, persisting >= 2 days and covering "
              ">= 50,000 km2. **This reference comes from the same reanalysis the trackers read, so it is not an "
              "independent truth**; tracker v2 thresholds the same field and is close to the reference by "
              "construction. Step-level scores (event present anywhere in the domain), IoU on hit steps.", "",
              "| event | method | ref steps / steps | POD | FAR | CSI | IoU | tracks |", "|---|---|---|---|---|---|---|---|"]
        for eid, r in hc.items():
            if "status" in r:
                L.append(f"| {eid} | - | {r['status']} | | | | | |")
                continue
            for meth, v in r["methods"].items():
                L.append(f"| {eid} | {meth} | {v['ref_steps']}/{v['steps']} | {v['pod']:.2f} | {v['far']:.2f} | "
                         f"{v['csi']:.2f} | {fmt(v['iou_mean'], 2)} | {v['n_tracks']} |")
        L += ["", "ERA5 vs IMD gridded temperature inside the reference event (independent check of the input):", "",
              "| event | IMD check | cell-days | bias ERA5 - IMD (C) | MAE (C) |", "|---|---|---|---|---|"]
        for eid, r in hc.items():
            c = r.get("imd_check", {"status": r.get("status", "n/a")})
            L.append(f"| {eid} | {c['status']} | {c.get('cells_days', '')} | {fmt(c.get('bias_era5_minus_imd_C'), 2)} | "
                     f"{fmt(c.get('mae_C'), 2)} |")
    L += ["", f"Peak RSS {out['peak_rss_mb']:.0f} MB.", "", "## What did not work", ""]
    L += what_failed(out)
    (REP / "REAL_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/REAL_RESULTS.md")


def what_failed(out):
    w = []
    for k, r in out["ensembles"].items():
        for meth in ("gnn_full", "v2"):
            if r[meth]["p_detect_any"] < 0.5:
                w.append(f"- {k} {meth}: only {r[meth]['p_detect_any']:.2f} of members detect Amphan.")
    for eid, r in out["era5_real_val"].items():
        if "status" in r:
            w.append(f"- {eid}: {r['status']}.")
            continue
        for meth, v in r.items():
            if v["ibtracs_fixes"] and v["matched_fixes"] < 0.5 * v["ibtracs_fixes"]:
                w.append(f"- {eid} {meth}: matched {v['matched_fixes']}/{v['ibtracs_fixes']} IBTrACS fixes.")
    for eid, r in out.get("era5_heat_cold", {}).items():
        if "status" in r:
            w.append(f"- {eid}: {r['status']}.")
            continue
        for meth, v in r["methods"].items():
            if v["pod"] < 0.5 or v["far"] > 0.5:
                w.append(f"- {eid} {meth}: POD {v['pod']:.2f}, FAR {v['far']:.2f} against the ERA5 IMD-criteria reference.")
        if r.get("imd_check", {}).get("status") != "ok":
            w.append(f"- {eid}: no IMD temperature check ({r.get('imd_check', {}).get('status')}).")
    return w or ["- Nothing below the thresholds used here."]


if __name__ == "__main__":
    main()
