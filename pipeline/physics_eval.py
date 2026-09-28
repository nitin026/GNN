"""BRIEF4 Phase 4: physics-informed downscaling -> reports/PHYSICS_DOWNSCALE.md, physics_results.json

    python -m pipeline.physics_eval

1. Physics-violation rates (pipeline.physics.ViolationCounter) for bicubic+lapse, U-Net, diffusion
   (sample) and diffusion+physics, on the SYNTHETIC TEST patches (same sampling as
   pipeline.downscale_eval) and on the REAL IMD perfect-model check (rain only).
   diffusion+physics = the diffusion residual sampler on top of the physics-fine-tuned U-Net mean
   (models/downscale/unet_phys, weights chosen on VAL), followed by the same exact projection.
2. Data metrics of the physics model next to the others (RMSE, p99 ratio, 10 km power) so the cost
   of the constraints is visible.
3. Crop-aware inference: downscale only the tracker's 4-D box + 100 km margin instead of the full
   333x333 domain; wall time and peak memory for both (U-Net, one lead, member 0).
"""
import gc
import json
import time
import tracemalloc
from pathlib import Path

import netCDF4
import numpy as np
import torch
import xarray as xr

from . import downscale_eval as de
from .downscale import VARS, UNetDownscaler, baseline, cond_input, project
from .jsonutil import dumps
from .physics import ViolationCounter, div
from .splits import TEST

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "models/downscale"
REP = ROOT / "reports"
SYN = ROOT / "data/synthetic"
MODELS = ["bicubic+lapse", "unet", "diffusion_sample", "unet_phys", "diffusion_phys_sample"]


def load_all():
    norm, unet, diff, _, _ = de.load_models()
    cp = json.loads((MD / "unet_phys" / "config.json").read_text())
    up_ = UNetDownscaler(cp["base"])
    up_.load_state_dict(torch.load(MD / "unet_phys" / "model.pt", map_location="cpu"))
    up_.eval()
    return norm, unet, diff, up_, cp


@torch.no_grad()
def run_all(X, D12, D5, norm, unet, diff, uphys, gen):
    x, a12, a5 = (torch.tensor(v, dtype=torch.float32) for v in (X, D12, D5))
    out = {"bicubic+lapse": baseline(x, a12, a5)}
    inp, _ = cond_input(x, a12, a5, norm)
    for name, mean_model in (("", unet), ("_phys", uphys)):
        m = mean_model(x, a12, a5, norm)
        out["unet" + name] = m
        c = diff.cond(norm.n(m), inp)
        out[f"diffusion{name}_sample"] = project(norm.d(norm.n(m) + diff.sample(c, de.STEPS, gen)), x)
    return out, x, a5


def truth_div_threshold(n_leads=4):
    """99.9th percentile of |div V10| of the 5 km TRUTH over a few TEST leads."""
    vals = []
    for case in TEST:
        with netCDF4.Dataset(SYN / case / "truth_5km.nc") as t5:
            for s in de.LEAD_STEPS[:n_leads]:
                u = de.decode(t5.variables["u10"], s)
                v = de.decode(t5.variables["v10"], s)
                d = div(torch.tensor(u[None, None]), torch.tensor(v[None, None])).abs().numpy()
                vals.append(np.quantile(d, 0.999))
    return float(np.mean(vals))


def synthetic(norm, unet, diff, uphys, thr):
    viol = {m: ViolationCounter(thr) for m in MODELS + ["truth"]}
    acc = {m: {v: de.Acc(v) for v in ("tp", "wind", "t2m")} for m in MODELS}
    dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)
    dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)
    rng = np.random.default_rng(2026)
    gen = torch.Generator().manual_seed(7)
    for case in TEST:
        t5 = netCDF4.Dataset(SYN / case / "truth_5km.nc")
        f12 = netCDF4.Dataset(SYN / case / "fcst_12km.nc")
        masks = f12.variables["event_mask_truth"]
        for s in de.LEAD_STEPS:
            y5 = np.stack([(de.decode(t5.variables[v], s) - de.OFFSET[v]) * de.SCALE[v] for v in VARS])
            x12 = np.stack([(f12.variables[f"{v}_truth"][s].astype(np.float64) - de.OFFSET[v]) * de.SCALE[v]
                            for v in VARS])
            ev = np.argwhere(masks[s][:].astype(bool))
            starts = []
            for k in range(de.PATCHES_PER_LEAD):
                if k % 2 == 0 and len(ev):
                    ci, cj = ev[rng.integers(len(ev))]
                    starts.append((int(np.clip(ci - de.P12 // 2, 0, 333 - de.P12)),
                                   int(np.clip(cj - de.P12 // 2, 0, 333 - de.P12))))
                else:
                    starts.append(tuple(int(v) for v in rng.integers(0, 333 - de.P12, 2)))
            P = de.P12
            X = np.array([x12[:, i:i + P, j:j + P] for i, j in starts], np.float32)
            Y = np.array([y5[:, 3 * i:3 * (i + P), 3 * j:3 * (j + P)] for i, j in starts], np.float32)
            D12 = np.array([dem12[None, i:i + P, j:j + P] for i, j in starts])
            D5 = np.array([dem5[None, 3 * i:3 * (i + P), 3 * j:3 * (j + P)] for i, j in starts])
            preds, x, a5 = run_all(X, D12, D5, norm, unet, diff, uphys, gen)
            viol["truth"].add(torch.tensor(Y), x, a5)
            tv = de.to_eval_vars(Y)
            for m, p in preds.items():
                viol[m].add(p, x, a5)
                pv = de.to_eval_vars(p.numpy())
                for v in acc[m]:
                    acc[m][v].add(pv[v], tv[v])
            del y5, x12, X, Y, preds
            gc.collect()
            de.rss()
        t5.close()
        f12.close()
        print(f"{case} done, RSS {de.rss():.0f} MB", flush=True)
    data = {m: {v: {k: a.result(v)[k] for k in ("rmse", "p99_ratio", "peak_ratio", "psd_ratio_10km")}
                for v, a in acc[m].items()} for m in MODELS}
    return {m: c.rates() for m, c in viol.items()}, data


@torch.no_grad()
def imd(norm, unet, diff, uphys):
    """Rain-only violation rates on the REAL IMD perfect-model check (JJAS 2020, 0.75 -> 0.25 deg)."""
    da = xr.open_dataset(ROOT / "data/real/imd/imd_rain_2020_0p25.nc").rain.sel(time=slice("2020-06-01", "2020-09-30"))
    lat, lon = da.latitude.values, da.longitude.values
    dem = np.nan_to_num(xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.interp(latitude=lat, longitude=lon).values).astype(np.float32)
    ny, nx = (len(lat) // 3) * 3, (len(lon) // 3) * 3
    viol = {m: ViolationCounter(None) for m in MODELS + ["truth"]}
    mean = norm.m.numpy()[0, :, 0, 0]
    gen = torch.Generator().manual_seed(11)
    days = da.time.values
    for k in range(0, len(days), 8):
        R = np.nan_to_num(da.sel(time=days[k:k + 8]).values[:, :ny, :nx].astype(np.float64))
        x = R.reshape(len(R), ny // 3, 3, nx // 3, 3).mean((2, 4))
        py, px = (-x.shape[1]) % 4, (-x.shape[2]) % 4
        xp = np.pad(x, ((0, 0), (0, py), (0, px)), mode="edge")
        X = np.repeat(mean[None, :, None, None], len(R), 0) * np.ones((1, 1) + xp.shape[1:])
        X[:, 4] = xp
        d5 = np.pad(dem[:ny, :nx], ((0, 3 * py), (0, 3 * px)), mode="edge")
        d12 = d5.reshape(d5.shape[0] // 3, 3, d5.shape[1] // 3, 3).mean((1, 3))
        preds, xt, a5 = run_all(X.astype(np.float32), np.repeat(d12[None, None], len(R), 0),
                                np.repeat(d5[None, None], len(R), 0), norm, unet, diff, uphys, gen)
        Yt = torch.tensor(np.repeat(np.pad(R, ((0, 0), (0, 3 * py), (0, 3 * px)), mode="edge")[:, None], 5, 1),
                          dtype=torch.float32)
        viol["truth"].add(Yt, xt, a5)
        for m, p in preds.items():
            viol[m].add(p, xt, a5)
        gc.collect()
    out = {}
    for m, c in viol.items():
        r = c.rates()
        out[m] = {"rain_noconv": r["rain_noconv"], "neg_rain": r["neg_rain"],
                  "note": "rain only; winds are the training mean, so the convergence proxy is ~0 everywhere"}
    return out


CROP_CHILD = r"""
import sys, time, json, numpy as np, torch, psutil
sys.path.insert(0, r"{root}")
from pipeline import downscale_eval as de
norm, unet, _, _, _ = de.load_models()
a = np.load(r"{npz}")
x, d12, d5 = (torch.tensor(a[k]) for k in ("x", "d12", "d5"))
base = psutil.Process().memory_info()
with torch.no_grad():
    t0 = time.perf_counter()
    y = unet(x, d12, d5, norm)
    dt = time.perf_counter() - t0
mi = psutil.Process().memory_info()
print(json.dumps({{"seconds": dt, "peak_mb": getattr(mi, "peak_wset", mi.rss) / 1e6, "base_mb": base.rss / 1e6}}))
"""


def crop_benchmark():
    """Full-domain vs crop-aware (4-D box + 100 km) U-Net inference for cyc_04, member 0, one lead.
    Each variant runs in a FRESH process so its peak working set (psutil) is its own."""
    import subprocess
    import sys
    import tempfile
    from .run_operational import crop_of
    meta = json.loads((ROOT / "backend/products/cyc_04/meta.json").read_text())
    box = meta["bbox4d"][0]
    f = xr.open_dataset(SYN / "cyc_04/fcst_12km.nc", decode_timedelta=False)
    s = box["lead_start_h"] // 6 + 4
    x = np.stack([(f[v].isel(number=0, step=s).values - de.OFFSET[v]) * de.SCALE[v] for v in VARS])[None].astype(np.float32)
    f.close()
    dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)
    dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)
    i0, i1, j0, j1 = crop_of(box)
    pad = lambda a, n: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, n), (0, n)], mode="edge")
    cases = {"full": (pad(x, 3), pad(dem12, 3)[None, None], pad(dem5, 9)[None, None]),
             "crop": (x[..., i0:i1, j0:j1], dem12[None, None, i0:i1, j0:j1], dem5[None, None, 3 * i0:3 * i1, 3 * j0:3 * j1])}
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for k, (xx, a12, a5) in cases.items():
            npz = Path(td) / f"{k}.npz"
            np.savez(npz, x=xx, d12=a12, d5=a5)
            runs = [json.loads(subprocess.run([sys.executable, "-c", CROP_CHILD.format(root=ROOT, npz=npz)],
                                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
                    for _ in range(3)]
            res[k] = {"seconds": min(r["seconds"] for r in runs), "peak_mb": min(r["peak_mb"] for r in runs),
                      "base_mb": min(r["base_mb"] for r in runs)}
    return {"box": box, "crop_cells_12km": [int(i1 - i0), int(j1 - j0)], "full_cells_12km": [336, 336],
            "full_seconds": round(res["full"]["seconds"], 3), "crop_seconds": round(res["crop"]["seconds"], 3),
            "full_peak_mb": round(res["full"]["peak_mb"]), "crop_peak_mb": round(res["crop"]["peak_mb"]),
            "process_base_mb": round(res["crop"]["base_mb"]),
            "time_saved_pct": round(100 * (1 - res["crop"]["seconds"] / res["full"]["seconds"]), 1),
            "area_fraction": round((i1 - i0) * (j1 - j0) / 336 ** 2, 3),
            "note": "U-Net, cyc_04 member 0, one lead; best of 3 fresh processes; peak = process peak working set "
                    "(includes the ~base_mb of Python + torch + model)"}


def main():
    import sys
    if "--crop-only" in sys.argv:                  # re-measure the crop benchmark, keep the rest
        out = json.loads((REP / "physics_results.json").read_text())
        out["crop"] = crop_benchmark()
        (REP / "physics_results.json").write_text(dumps(out, indent=1))
        write_report(out)
        return
    norm, unet, diff, uphys, cp = load_all()
    thr = truth_div_threshold()
    viol, data = synthetic(norm, unet, diff, uphys, thr)
    out = {"physics_config": cp, "div_threshold_truth_p999": thr, "violations_test": viol, "data_metrics_test": data,
           "violations_imd": imd(norm, unet, diff, uphys), "crop": crop_benchmark(),
           "peak_rss_mb": de.PEAK["rss"]}
    (REP / "physics_results.json").write_text(dumps(out, indent=1))
    write_report(out)


def write_report(out):
    f2 = lambda x: "n/a" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    v, d = out["violations_test"], out["data_metrics_test"]
    L = ["# Physics-informed downscaling (BRIEF4 Phase 4)", "",
         "SYNTHETIC TEST patches (same sampling as reports/DOWNSCALE_RESULTS.md) and the REAL IMD perfect-model check. "
         f"Physics weights of `unet_phys` (chosen on VAL, see reports/TUNING_LOG.md): {out['physics_config'].get('physics_weights')}. "
         "The hard conservation projection is unchanged for every model.", "",
         "## Physics-violation rate (% of the relevant cells), SYNTHETIC TEST", "",
         "| model | rain where convergence <= 0 | wind divergence > truth p99.9 | lapse-rate wrong sign | rain < 0 | q <= q_sat |",
         "|---|---|---|---|---|---|"]
    for m in ["truth"] + MODELS:
        r = v[m]
        L.append(f"| {m} | {f2(r['rain_noconv'])} | {f2(r['div'])} | {f2(r['lapse'])} | {f2(r['neg_rain'])} | n/a |")
    L += ["", "Definitions: " + "; ".join(f"{k}: {s}" for k, s in v["unet"]["denominators"].items()) +
          ". The convergence is a proxy (-div of the 12 km 10 m wind; the downscaler has no humidity or 850 hPa wind), so "
          "even the synthetic TRUTH 'violates' it where rain was placed by 850 hPa moisture-flux convergence. q <= q_sat "
          "cannot be violated because humidity is not a downscaled variable (pipeline/physics.py has qsat for when it is).", "",
          "## Cost of the constraints (SYNTHETIC TEST)", "",
          "| model | rain RMSE | rain p99 ratio | rain patch-peak ratio | rain 10 km power | wind 10 km power | t2m RMSE |",
          "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        L.append(f"| {m} | {d[m]['tp']['rmse']:.3f} | {d[m]['tp']['p99_ratio']:.3f} | {d[m]['tp']['peak_ratio']:.3f} | "
                 f"{d[m]['tp']['psd_ratio_10km']:.3f} | {d[m]['wind']['psd_ratio_10km']:.3f} | {d[m]['t2m']['rmse']:.3f} |")
    L += ["", "## REAL IMD perfect-model check (rain only)", "", "| model | rain where convergence <= 0 | rain < 0 |", "|---|---|---|"]
    for m, r in out["violations_imd"].items():
        L.append(f"| {m} | {f2(r['rain_noconv'])} | {f2(r['neg_rain'])} |")
    L += ["", "Only rain is given in this test (winds = training mean), so the convergence proxy carries no information "
          "here and the rain-without-convergence rate is not meaningful; it is shown for completeness.", "",
          "## Crop-aware inference", ""]
    c = out["crop"]
    L += [f"Downscaling only the tracker's 4-D box + 100 km ({c['crop_cells_12km'][0]} x {c['crop_cells_12km'][1]} cells at "
          f"12 km, {100 * c['area_fraction']:.0f} % of the domain) instead of the full 336 x 336: "
          f"**{c['crop_seconds']} s vs {c['full_seconds']} s ({c['time_saved_pct']} % less time)**, peak process memory "
          f"{c['crop_peak_mb']} MB vs {c['full_peak_mb']} MB (of which ~{c['process_base_mb']} MB is Python + torch + the "
          "model; U-Net, cyc_04 member 0, one lead, best of 3 fresh processes).", "",
          ]
    nr = [json.loads((REP / f).read_text()) for f in ("noise_retune_heat04.json", "noise_retune.json") if (REP / f).exists()]
    if nr:
        base = nr[-1]
        L += ["## Retuning the synthetic fine-scale noise (scripts/retune_noise.py)", "",
              "Reference: ERA5's native 0.25 deg spectrum fitted over 100-400 km (the scales ERA5 resolves) and "
              "extrapolated to 25 km (ERA5 interpolated to 5 km has almost no 25 km power, so it cannot be the target; "
              "there is no real km-scale product here). New amplitude A solves P_background + A^2 P_unit-noise = P_ref at 25 km.", "",
              "| variable | reference case | ERA5 slope 100-400 km | synthetic / reference power at 25 km (before) | amplitude before -> after |",
              "|---|---|---|---|---|"]
        for var_ in ("t2m", "u10"):
            r = base[var_]
            L.append(f"| {var_} | {r['case']} | {r['era5_native_slope_100_400km']:.2f} | {r['old_ratio_syn_over_ref_25km']:.2f} | "
                     f"{r['old_amplitude']:.3g} -> {r['new_amplitude']:.3g} (x{r['scale']:.2f}) |")
        L += ["", "Regenerated TEST cases (retuned copies in data/synthetic_retuned; the originals, and every model trained on "
              "them, are unchanged):", "", "| case | variable | power at 25 km before -> after | power at 50 km before -> after | reference at 25 km |",
              "|---|---|---|---|---|"]
        for r in nr:
            g = r.get("regenerated")
            if g:
                L.append(f"| {g['case']} | {g['var']} | {g['p25_before']:.2f} -> {g['p25_after']:.2f} | {g['p50_before']:.2f} -> "
                         f"{g['p50_after']:.2f} | {base[g['var']]['p_ref_25km']:.2f} |")
        L += ["", "The earlier statement that synthetic fine scales carry ~19x ERA5's power at 25 km compared against ERA5 "
              "*interpolated* to 5 km. Against ERA5's own extrapolated slope, synthetic t2m was too WEAK (x0.5) and u10 too "
              "strong (x7). The one-shot retune overshoots for t2m (the fine noise is not the only small-scale source in the "
              "generator: event shapes and the DEM term add power), so it is a direction, not a finished calibration.",
              "", "![noise retune](figures/noise_retune.png)", ""]
    L += ["## What did not work", ""]
    w = []
    for key, name in (("rain_noconv", "rain without convergence"), ("div", "divergence"), ("lapse", "lapse rate")):
        a, b = v["unet"][key], v["unet_phys"][key]
        if a is not None and b is not None and b >= a:
            w.append(f"- {name}: the physics U-Net does not reduce violations ({b:.2f} % vs {a:.2f} %).")
    if d["unet_phys"]["tp"]["rmse"] > 1.05 * d["unet"]["tp"]["rmse"]:
        w.append(f"- The physics terms cost rain RMSE: {d['unet_phys']['tp']['rmse']:.3f} vs {d['unet']['tp']['rmse']:.3f}.")
    w.append("- q <= q_sat is not applicable (no humidity output); the moisture-convergence term uses a 10 m-wind proxy.")
    w.append("- The diffusion residual model was NOT retrained with the physics loss (CPU budget); diffusion+physics = the "
             "same sampler on the physics-fine-tuned mean.")
    w.append("- Synthetic data regenerated with the retuned noise for 2 TEST cases only; the 13-case dataset and the models "
             "were not regenerated / retrained (several CPU-hours), so BRIEF Phase 4 and BRIEF4 Phase 0 were not re-run on "
             "retuned data. The retune itself overshoots t2m and leaves u10 ~2.4x above the reference.")
    L += w
    (REP / "PHYSICS_DOWNSCALE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/PHYSICS_DOWNSCALE.md")


if __name__ == "__main__":
    main()
