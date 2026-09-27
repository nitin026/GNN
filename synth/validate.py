"""Phase 4 realism check: synthetic vs real (ERA5 / IMD / IBTrACS).

    python -m synth.validate   ->  reports/SYNTH_VALIDATION.md + reports/figures/val_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from .noise import radial_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
REAL = ROOT / "data" / "real"
REP = ROOT / "reports"
FIG = REP / "figures"
WINDOW = {"tropical_cyclone": "amphan", "heat_dome": "heatwave", "cold_wave": "coldwave"}
KT = 0.514444


def labels():
    return {p.name: json.loads((p / "labels.json").read_text())
            for p in sorted(SYN.iterdir()) if (p / "labels.json").exists()}


def read_packed(path, var, steps=None):
    ds = netCDF4.Dataset(path)
    v = ds.variables[var]
    v.set_auto_maskandscale(False)
    a = (v[:] if steps is None else v[steps]).astype(np.float64) * float(v.scale_factor) + \
        float(v.add_offset)
    ds.close()
    return a


def sample(a, n=200_000, seed=0):
    a = np.asarray(a).ravel()
    a = a[np.isfinite(a)]
    if a.size > n:
        a = np.random.default_rng(seed).choice(a, n, replace=False)
    return a


# ------------------------------------------------------------------------ 1. histograms + p99
def hist_and_p99(labs):
    rows, fig = [], plt.figure(figsize=(16, 11))
    variables = ["t2m", "msl", "wind10", "tp"]
    for h, hz in enumerate(("tropical_cyclone", "heat_dome", "cold_wave")):
        ids = [k for k, l in labs.items() if l["hazard"] == hz]
        real = xr.open_dataset(REAL / "era5" / f"{WINDOW[hz]}_era5_g5.nc")
        rsel = real.isel(time=slice(0, None, 3))     # every 3rd 6-h step cycles through all hours
        for j, var in enumerate(variables):
            syn = []
            for cid in ids:
                p = SYN / cid / "truth_5km.nc"
                steps = slice(0, 41, 3)
                if var == "wind10":
                    x = np.hypot(read_packed(p, "u10", steps), read_packed(p, "v10", steps))
                else:
                    x = read_packed(p, var, steps)
                syn.append(sample(x, 100_000, seed=len(syn)))
            syn = np.concatenate(syn)
            if var == "wind10":
                rv = np.hypot(rsel.u10.values, rsel.v10.values)
            else:
                rv = rsel[var].values
            rv = sample(rv, 400_000, seed=1)
            ax = fig.add_subplot(3, 4, 4 * h + j + 1)
            lo, hi = np.percentile(np.concatenate([syn, rv]), [0.1, 99.95])
            if var == "tp":
                lo, hi = 0, max(hi, 1)
            bins = np.linspace(lo, hi, 60)
            ax.hist(rv, bins, density=True, alpha=0.5, label="real ERA5 (G5 interp.)")
            ax.hist(syn, bins, density=True, histtype="step", lw=1.5, color="k",
                    label="SYNTHETIC truth 5 km")
            if var == "tp":
                ax.set_yscale("log")
            ax.set_title(f"{hz}: {var}", fontsize=9)
            if h == 0 and j == 0:
                ax.legend(fontsize=7)
            rows.append({"hazard": hz, "var": var, "syn_p99": float(np.percentile(syn, 99)),
                         "real_p99": float(np.percentile(rv, 99)),
                         "syn_p50": float(np.percentile(syn, 50)),
                         "real_p50": float(np.percentile(rv, 50)),
                         "syn_max": float(syn.max()), "real_max": float(rv.max())})
        real.close()
    fig.suptitle("Value distributions: SYNTHETIC truth (all cases of a hazard) vs real ERA5 "
                 "of the background window", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "val_histograms.png", dpi=100)
    plt.close(fig)
    return rows


def imd_rain_check(labs):
    """Daily rain p99 over land: synthetic cyclone cases vs IMD (May 2020) vs ERA5 G12."""
    out = []
    imd = xr.open_dataset(REAL / "imd" / "imd_rain_2020_0p25.nc")
    r = imd.rain.sel(time=slice("2020-05-10", "2020-05-25"))
    imd_vals = sample(r.values[r.values > -1], 400_000)
    out.append({"source": "IMD 0.25 deg (real), 2020-05-10..25", "p99": np.percentile(imd_vals, 99),
                "p999": np.percentile(imd_vals, 99.9), "max": imd_vals.max()})
    land = xr.open_dataset(REAL / "dem" / "dem_g12.nc").orog.values > 1
    for cid, l in labs.items():
        if l["hazard"] != "tropical_cyclone":
            continue
        ds = xr.open_dataset(SYN / cid / "fcst_12km.nc", decode_timedelta=False)
        tp = ds.tp_truth.values                      # 6-h accumulations, 41 steps
        daily = np.stack([tp[i + 1:i + 5].sum(0) for i in range(0, 36, 4)])
        v = sample(daily[:, land], 400_000)
        out.append({"source": f"SYNTHETIC {cid} (12 km, daily, land)", "p99": np.percentile(v, 99),
                    "p999": np.percentile(v, 99.9), "max": v.max()})
        ds.close()
    e = xr.open_dataset(REAL / "era5" / "amphan_era5_g12.nc")
    tp = e.tp.values
    daily = np.stack([tp[i + 1:i + 5].sum(0) for i in range(0, len(tp) - 5, 4)])
    v = sample(daily[:, land], 400_000)
    out.append({"source": "ERA5 G12 (real), 2020-05-10..25, daily, land", "p99": np.percentile(v, 99),
                "p999": np.percentile(v, 99.9), "max": v.max()})
    return out


# ------------------------------------------------------------------------ 2. spectra
def spectra(labs):
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, var in zip(axes, ("t2m", "u10")):
        cid = "heat_01" if var == "t2m" else "amphan_replay"
        win = "heatwave" if var == "t2m" else "amphan"
        l = labs[cid]
        step = 20
        t = pd.Timestamp(l["valid_times"][step])
        syn = read_packed(SYN / cid / "truth_5km.nc", var, step)[250:762, 250:762]
        k, p = radial_spectrum(syn, 0.04 * 111.2)
        ax.loglog(k, p, "k", label="SYNTHETIC truth 5 km")
        r25 = xr.open_dataset(REAL / "era5" / f"{win}_era5_0p25.nc")[var].sel(time=t).values
        k2, p2 = radial_spectrum(r25[40:120, 40:120] if r25.shape[0] > 120 else r25, 0.25 * 111.2)
        ax.loglog(k2, p2, "C0", label="real ERA5 0.25 deg")
        g5 = xr.open_dataset(REAL / "era5" / f"{win}_era5_g5.nc")[var].sel(time=t).values
        k3, p3 = radial_spectrum(g5[250:762, 250:762], 0.04 * 111.2)
        ax.loglog(k3, p3, "C1", label="real ERA5 interp. to 5 km")
        ref = p[np.searchsorted(k, 1 / 500)] * (k / (1 / 500)) ** (-5 / 3 - 1)
        ax.loglog(k, ref, "k:", lw=0.8, label="k^-8/3 (2-D power of a k^-5/3 energy spectrum)")
        sel = (k > 1 / 200) & (k < 1 / 20)
        slope_s = np.polyfit(np.log(k[sel]), np.log(p[sel]), 1)[0]
        sel3 = (k3 > 1 / 200) & (k3 < 1 / 20)
        slope_r = np.polyfit(np.log(k3[sel3]), np.log(p3[sel3]), 1)[0]
        res[var] = {"case": cid, "valid_time": str(t), "slope_syn_20_200km": slope_s,
                    "slope_era5interp_20_200km": slope_r,
                    "power_ratio_syn_over_era5interp_at_25km": float(
                        np.interp(1 / 25, k, p) / np.interp(1 / 25, k3, p3))}
        ax.set_xlabel("wavenumber (1/km)")
        ax.set_ylabel("PSD (units^2 km^2)")
        ax.set_title(f"{var}: {cid} vs ERA5 at {t:%Y-%m-%d %H}Z")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "val_spectra.png", dpi=100)
    plt.close(fig)
    return res


# ------------------------------------------------------------------------ 3. cyclone profile
def azimuthal_profile(speed, lat, lon, la0, lo0, rmax=600, dr=5):
    x = (lon[None, :] - lo0) * 111.2 * np.cos(np.deg2rad(la0))
    y = (lat[:, None] - la0) * 111.2
    r = np.sqrt(x ** 2 + y ** 2)
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins)
    s = np.bincount(idx, weights=speed.ravel(), minlength=len(bins) + 1)
    c = np.bincount(idx, minlength=len(bins) + 1)
    rc = 0.5 * (bins[1:] + bins[:-1])
    return rc, (s[1:len(bins)] / np.maximum(c[1:len(bins)], 1))


def cyclone_profile(labs):
    l = labs["amphan_replay"]
    act = [e for e in l["truth_track"] if e.get("vmax_ms")]
    pk = max(act, key=lambda e: e["vmax_ms"])
    step = pk["lead_h"] // 6
    t = pd.Timestamp(pk["valid_time"])
    ds = netCDF4.Dataset(SYN / "amphan_replay" / "truth_5km.nc")
    lat, lon = ds.variables["latitude"][:], ds.variables["longitude"][:]
    ds.close()
    sp = np.hypot(read_packed(SYN / "amphan_replay" / "truth_5km.nc", "u10", step),
                  read_packed(SYN / "amphan_replay" / "truth_5km.nc", "v10", step))
    rc, prof = azimuthal_profile(sp, lat, lon, pk["center_lat"], pk["center_lon"])
    e = xr.open_dataset(REAL / "era5" / "amphan_era5_0p25.nc").sel(time=t)
    esp = np.hypot(e.u10.values, e.v10.values)
    ib = pd.read_csv(REAL / "ibtracs" / "amphan_2020_ibtracs.csv")
    ib["time"] = pd.to_datetime(ib.ISO_TIME)
    row = ib.iloc[(ib.time - t).abs().argmin()]
    rc2, prof2 = azimuthal_profile(esp, e.latitude.values, e.longitude.values, row.LAT, row.LON,
                                   dr=25)
    ib_v10 = float(row.USA_WIND) * KT * 0.88
    ib_rmw = float(row.USA_RMW) * 1.852
    out = {"valid_time": str(t), "lead_h": pk["lead_h"],
           "syn_vmax_azimean": float(prof.max()), "syn_rmw_km": float(rc[prof.argmax()]),
           "syn_vmax_point": float(sp[np.hypot((lat[:, None] - pk["center_lat"]) * 111,
                                               (lon[None, :] - pk["center_lon"]) * 104) < 300].max()),
           "ib_vmax_10min_ms": ib_v10, "ib_vmax_1min_kt": float(row.USA_WIND), "ib_rmw_km": ib_rmw,
           "era5_vmax_azimean": float(prof2.max()), "era5_rmw_km": float(rc2[prof2.argmax()]),
           "era5_vmax_point": float(esp.max())}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rc, prof, "k", label="SYNTHETIC Amphan replay (5 km, azimuthal mean)")
    ax.plot(rc2, prof2, "C0", label="real ERA5 0.25 deg (azimuthal mean)")
    ax.axvline(ib_rmw, color="C3", ls="--", label=f"IBTrACS USA_RMW = {ib_rmw:.0f} km")
    ax.axhline(ib_v10, color="C3", ls=":", label=f"IBTrACS Vmax (x0.88 10-min) = {ib_v10:.0f} m/s")
    ax.set_xlabel("radius (km)")
    ax.set_ylabel("10 m wind speed (m/s)")
    ax.set_title(f"Amphan radial wind profile at {t:%Y-%m-%d %H}Z")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "val_cyclone_profile.png", dpi=100)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------ 4. IMD criteria
def imd_criteria(labs):
    """Daily Tmax (heat) / Tmin (cold) departures from climatology inside the exact mask,
    compared with IMD criteria (plains): heat wave departure >= 4.5 C (severe >= 6.5) with
    Tmax >= 40 C; cold wave departure <= -4.5 C (severe <= -6.5) with Tmin <= 10 C."""
    clim = xr.open_dataset(REAL / "clim" / "era5_clim_g12.nc")
    rows = []
    for cid, l in labs.items():
        hz = l["hazard"]
        if hz not in ("heat_dome", "cold_wave"):
            continue
        ds = xr.open_dataset(SYN / cid / "fcst_12km.nc", decode_timedelta=False)
        t2 = ds.t2m_truth.values
        mask = ds.event_mask_truth.values.astype(bool)
        times = pd.to_datetime(l["valid_times"])
        best = None
        for d0 in range(0, 40, 4):                      # days = 4 consecutive 6-hourly steps
            sl = slice(d0, d0 + 4)
            m = mask[sl].any(0)
            if m.sum() < 10:
                continue
            ext = t2[sl].max(0) if hz == "heat_dome" else t2[sl].min(0)
            cm = []
            for t in times[sl]:
                doy = t.dayofyear if t.dayofyear in clim.dayofyear.values else \
                    int(clim.dayofyear.values[np.argmin(np.abs(clim.dayofyear.values - t.dayofyear))])
                cm.append(clim.t2m_mean.sel(dayofyear=doy, hour=t.hour).values)
            cm = np.array(cm)
            cext = cm.max(0) if hz == "heat_dome" else cm.min(0)
            dep = (ext - cext)[m]
            absT = (ext - 273.15)[m]
            score = dep.mean() if hz == "heat_dome" else -dep.mean()
            if best is None or score > best["score"]:
                best = {"score": score, "day_start": str(times[d0].date()), "dep": dep, "absT": absT}
        ds.close()
        if best is None:
            continue
        dep, absT = best["dep"], best["absT"]
        if hz == "heat_dome":
            r = {"case": cid, "hazard": hz, "peak_day": best["day_start"],
                 "mean_departure_C": float(dep.mean()), "max_departure_C": float(dep.max()),
                 "frac_ge_4p5": float((dep >= 4.5).mean()), "frac_ge_6p5": float((dep >= 6.5).mean()),
                 "frac_abs_threshold": float((absT >= 40).mean()),
                 "frac_imd_heatwave": float(((dep >= 4.5) & (absT >= 40)).mean()),
                 "injected_peak_K": l["event_params"]["amp_peak_K"]}
        else:
            r = {"case": cid, "hazard": hz, "peak_day": best["day_start"],
                 "mean_departure_C": float(dep.mean()), "max_departure_C": float(dep.min()),
                 "frac_ge_4p5": float((dep <= -4.5).mean()), "frac_ge_6p5": float((dep <= -6.5).mean()),
                 "frac_abs_threshold": float((absT <= 10).mean()),
                 "frac_imd_heatwave": float(((dep <= -4.5) & (absT <= 10)).mean()),
                 "injected_peak_K": l["event_params"]["amp_peak_K"]}
        rows.append(r)
    # real reference: ERA5 2024 heatwave, NW India box, peak-day Tmax departure
    e = xr.open_dataset(REAL / "era5" / "heatwave_era5_g12.nc")
    box = dict(latitude=slice(24, 31), longitude=slice(70, 80))
    daily_max = e.t2m.sel(**box).resample(time="1D").max()
    deps = []
    for t in daily_max.time.values:
        t = pd.Timestamp(t)
        cm = np.array([clim.t2m_mean.sel(dayofyear=t.dayofyear, hour=h).sel(**box).values
                       for h in (0, 6, 12, 18)]).max(0)
        deps.append(float((daily_max.sel(time=t).values - cm).mean()))
    i = int(np.argmax(deps))
    real_ref = {"era5_2024_nw_india_peak_day": str(pd.Timestamp(daily_max.time.values[i]).date()),
                "era5_box_mean_tmax_departure_C": deps[i],
                "era5_box_mean_tmax_C": float(daily_max.isel(time=i).mean() - 273.15)}
    return rows, real_ref


# ------------------------------------------------------------------------ report
def f(x, n=2):
    return f"{x:.{n}f}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    labs = labels()
    h = hist_and_p99(labs)
    rain = imd_rain_check(labs)
    sp = spectra(labs)
    cp = cyclone_profile(labs)
    crit, real_ref = imd_criteria(labs)
    from .rain_calibration import imd_amphan_sample, quantiles, synth_sample
    swath = {"IMD Amphan 2020 (real)": quantiles(imd_amphan_sample())}
    for cid, l in labs.items():
        if l["hazard"] == "tropical_cyclone":
            swath[cid] = quantiles(synth_sample(cid))
    (REP / "synth_validation.json").write_text(json.dumps(
        {"hist": h, "rain": rain, "swath": swath, "spectra": sp, "cyclone": cp, "imd": crit,
         "real_ref": real_ref},
        indent=1, default=float))
    units = {"t2m": "K", "msl": "Pa", "wind10": "m/s", "tp": "mm/6h"}
    L = ["# Synthetic-data realism check (Phase 4)", "",
         "Everything labelled SYNTHETIC below was produced by `synth/` and is **not** an observation "
         "or a real forecast. Real references: ERA5 (WeatherBench 2 / ARCO-ERA5), IMD 0.25 deg gridded "
         "rainfall (imdlib), IBTrACS v04r01.", "",
         "## 1. Value distributions and 99th percentiles", "",
         "Synthetic: 5 km truth of all cases of a hazard (every 3rd 6-h step, so all synoptic hours are "
         "sampled). Real: ERA5 of the same background window interpolated to the same 5 km grid (every 3rd "
         "step), whole India box.", "",
         "| hazard | variable | unit | synthetic p50 | real p50 | synthetic p99 | real p99 | synthetic max | real max |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in h:
        L.append(f"| {r['hazard']} | {r['var']} | {units[r['var']]} | {f(r['syn_p50'])} | {f(r['real_p50'])} | "
                 f"{f(r['syn_p99'])} | {f(r['real_p99'])} | {f(r['syn_max'])} | {f(r['real_max'])} |")
    L += ["", "![histograms](figures/val_histograms.png)", "",
          "### Daily rainfall over land vs IMD", "",
          "| source | p99 (mm/day) | p99.9 (mm/day) | max (mm/day) |", "|---|---|---|---|"]
    for r in rain:
        L.append(f"| {r['source']} | {f(r['p99'], 1)} | {f(r['p999'], 1)} | {f(r['max'], 1)} |")
    ref = swath["IMD Amphan 2020 (real)"]
    L += ["", "### Cyclone rain swath vs IMD Amphan (calibration target)", "",
          "Daily land rain within 500 km of the storm centre, ~0.25 deg, wet cells (>= 1 mm/day). "
          "Cyclone rain is quantile-mapped to IMD Amphan (`synth/rain_qm.json`, fitted on the TRAIN "
          "cases cyc_01 and cyc_02 only). Target: p99 within 20 % of IMD.", "",
          "| sample | n | p50 | p90 | p99 | p99 vs IMD | p99.9 | max |", "|---|---|---|---|---|---|---|---|"]
    for k, q in swath.items():
        if not q.get("n"):
            continue
        L.append(f"| {k} | {q['n']} | {f(q['p50'], 1)} | {f(q['p90'], 1)} | {f(q['p99'], 1)} | "
                 f"{(q['p99'] / ref['p99'] - 1) * 100:+.0f} % | {f(q['p999'], 1)} | {f(q['max'], 1)} |")
    L += ["", "## 2. Radially averaged power spectra", "",
          "| variable | case / time | slope 20-200 km, SYNTHETIC | slope 20-200 km, ERA5 interp. 5 km | "
          "power ratio syn/ERA5-interp at 25 km |", "|---|---|---|---|---|"]
    for v, r in sp.items():
        L.append(f"| {v} | {r['case']} {r['valid_time']} | {f(r['slope_syn_20_200km'])} | "
                 f"{f(r['slope_era5interp_20_200km'])} | {f(r['power_ratio_syn_over_era5interp_at_25km'], 1)} |")
    L += ["", "A k^-5/3 energy spectrum corresponds to a 2-D power slope of about -8/3 = -2.67.", "",
          "![spectra](figures/val_spectra.png)", "",
          "## 3. Cyclone radial wind profile vs Amphan observations", "",
          f"At the synthetic replay's peak ({cp['valid_time']}, lead {cp['lead_h']} h):", "",
          "| quantity | SYNTHETIC replay | IBTrACS (real) | ERA5 0.25 deg (real) |", "|---|---|---|---|",
          f"| Vmax, azimuthal mean (m/s) | {f(cp['syn_vmax_azimean'], 1)} | "
          f"{f(cp['ib_vmax_10min_ms'], 1)} (USA_WIND {cp['ib_vmax_1min_kt']:.0f} kt 1-min x 0.88) | "
          f"{f(cp['era5_vmax_azimean'], 1)} |",
          f"| Vmax, grid point (m/s) | {f(cp['syn_vmax_point'], 1)} | - | {f(cp['era5_vmax_point'], 1)} |",
          f"| radius of max wind (km) | {f(cp['syn_rmw_km'], 0)} | {f(cp['ib_rmw_km'], 0)} (USA_RMW) | "
          f"{f(cp['era5_rmw_km'], 0)} |", "",
          "![profile](figures/val_cyclone_profile.png)", "",
          "## 4. Heat-wave / cold-wave magnitude vs IMD criteria", "",
          "IMD plains criteria: heat wave if Tmax departure >= 4.5 C (severe >= 6.5 C) and Tmax >= 40 C; "
          "cold wave if Tmin departure <= -4.5 C (severe <= -6.5 C) and Tmin <= 10 C. Here Tmax/Tmin are "
          "taken from the four 6-hourly 12 km truth values of the most extreme day, and departures are "
          "relative to the ERA5 1990-2019 climatology (not IMD station normals). Fractions are over the "
          "cells inside the exact event mask.", "",
          "| case | peak day | injected peak (K) | mean departure (C) | extreme departure (C) | "
          "frac >= 4.5 C | frac >= 6.5 C | frac meeting absolute threshold | frac meeting IMD (both) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in crit:
        L.append(f"| {r['case']} | {r['peak_day']} | {f(r['injected_peak_K'])} | {f(r['mean_departure_C'])} | "
                 f"{f(r['max_departure_C'])} | {f(r['frac_ge_4p5'])} | {f(r['frac_ge_6p5'])} | "
                 f"{f(r['frac_abs_threshold'])} | {f(r['frac_imd_heatwave'])} |")
    L += ["", f"Real reference: ERA5 2024 heatwave, NW-India box (24-31N, 70-80E), peak day "
          f"{real_ref['era5_2024_nw_india_peak_day']}: box-mean Tmax departure "
          f"{f(real_ref['era5_box_mean_tmax_departure_C'])} C, box-mean Tmax "
          f"{f(real_ref['era5_box_mean_tmax_C'])} C (6-hourly sampling).", "",
          "## 5. Where the synthetic data is NOT realistic", ""]
    L += limitations(h, sp, cp, crit, rain)
    (REP / "SYNTH_VALIDATION.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote reports/SYNTH_VALIDATION.md")


def limitations(h, sp, cp, crit, rain):
    out = []
    ratio = sp.get("t2m", {}).get("power_ratio_syn_over_era5interp_at_25km", np.nan)
    out.append(f"- **Small scales are statistical, not dynamical.** The fine structure below ~50 km is "
               f"spectral noise (k^-5/3) plus the DEM rain factor. It has {ratio:.0f}x the 25 km power of "
               "interpolated ERA5, because ERA5 has almost none at that scale. Nothing ties it to the "
               "weather: no fronts, no sea-breeze, no convection organisation. Validate against a real "
               "km-scale product (e.g. IMD 0.0625 deg or a convection-permitting model) before trusting it.")
    out.append("- **Backgrounds are tamed on purpose.** Real ERA5 anomalies are soft-clipped to 1.5 sigma "
               "and the real Amphan vortex is smoothed away. As a result the tails of t2m/msl/tp in the "
               "background are lighter than in reality, and in the tables above most of the heavy tail "
               "comes from the injected event.")
    out.append(f"- **The cyclone is an analytic, symmetric Holland vortex** plus a motion asymmetry and "
               f"log-spiral bands. There are no eyewall replacement cycles, no shear-induced asymmetry and "
               f"no wind-pressure-rain coupling beyond the parametrisation. At peak the replay's "
               f"azimuthal-mean Vmax is {cp['syn_vmax_azimean']:.0f} m/s (IBTrACS 10-min equivalent "
               f"{cp['ib_vmax_10min_ms']:.0f} m/s) and its RMW is {cp['syn_rmw_km']:.0f} km (IBTrACS "
               f"{cp['ib_rmw_km']:.0f} km). ERA5 itself only reaches {cp['era5_vmax_azimean']:.0f} m/s, "
               "so the synthetic winds are far stronger than anything a 0.25 deg reanalysis or a 12 km "
               "global EPS will show.")
    tc = next((r for r in h if r["hazard"] == "tropical_cyclone" and r["var"] == "tp"), None)
    imd = next((r for r in rain if r["source"].startswith("IMD")), None)
    syn_d = [r for r in rain if r["source"].startswith("SYNTHETIC")]
    ratio = max(r["max"] for r in syn_d) / imd["max"]
    head = ("the local maxima are still too extreme" if ratio > 1.3 else
            "local maxima are now close to IMD")
    out.append(f"- **Rain amounts are parametric; {head}.** The eyewall peak rate "
               "is 3 + 0.3*Vmax mm/h, gated by 850 hPa moisture-flux convergence, scaled by the upslope "
               "factor (<= 2x) and quantile-mapped to IMD Amphan (fitted on 2 train cases). The "
               f"6-h maximum is {tc['syn_max']:.0f} mm against {tc['real_max']:.0f} mm in ERA5, and the daily "
               f"land maxima reach up to {max(r['max'] for r in syn_d):.0f} mm/day against {imd['max']:.0f} "
               "mm/day in IMD (May 2020); the swath table above has the calibrated comparison. "
               "Calibrating every storm to one storm (Amphan) is itself an "
               "assumption. The gating uses ERA5 850 hPa moisture and winds (Amphan window only). "
               "Heat and cold cases inject no rain; they only damp the background.")
    out.append("- **Heat domes and cold waves are 2-D surface blobs.** They have no vertical structure (the "
               "4-D box level range is surface-only), and their wind response is a geostrophic "
               "increment times 0.6. There is no soil-moisture feedback. The diurnal cycle is a fixed "
               "+/-20 % modulation, not the observed Tmax/Tmin asymmetry.")
    if crit:
        hw = [r for r in crit if r["hazard"] == "heat_dome"]
        cw = [r for r in crit if r["hazard"] == "cold_wave"]
        if hw:
            out.append(f"- **IMD criteria are only partly met.** Across the heat cases, on average "
                       f"{np.mean([r['frac_imd_heatwave'] for r in hw]) * 100:.0f} % of mask cells meet the "
                       "full IMD heat-wave definition (departure plus the 40 C absolute threshold) on the peak "
                       "day. 6-hourly sampling misses the true Tmax, and ERA5 climatology is not the IMD normal.")
        if cw:
            out.append(f"- Across the cold-wave cases, {np.mean([r['frac_imd_heatwave'] for r in cw]) * 100:.0f} % "
                       "of mask cells meet the IMD cold-wave definition (departure plus Tmin <= 10 C).")
    if crit:
        hw = [r for r in crit if r["hazard"] == "heat_dome"]
        if hw:
            out.append("- **Departures are larger than the injected amplitude.** In the heat cases the "
                       "departure from climatology exceeds the injected peak (mean departure vs. injected "
                       "peak in the table above). The real 2024 heatwave background, even soft-clipped to "
                       "1.5 sigma (~+4 K), adds to the injected dome. The exact label is the injected part "
                       "only, so a detector that sees the full anomaly will find a larger event than the label.")
    out.append("- **Ensembles are not a model.** Members are the truth event with prescribed, "
               "lead-dependent position and intensity errors, 2 'miss' and 1-2 'false alarm' members, and "
               "large-scale noise. Their spread-skill relation is built in, not emergent.")
    out.append("- **Grid.** G5 is 0.04 deg (~4.4 km) rather than 0.045 deg, so that 3x3 block "
               "averaging onto the 0.12 deg grid is exact.")
    return out


if __name__ == "__main__":
    main()
