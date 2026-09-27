"""Phase 3 evaluation: 12 km -> 5 km downscaling on held-out TEST patches (SYNTHETIC, exact 5 km
truth) and a sanity check on REAL ERA5 Amphan rain vs IMD.

Metrics per variable: RMSE; conservation error max|avgpool(out) - x12|; amplitude ratios
p99/p99.9 (pred / truth, all test pixels); peak-rain error (patch maximum tp, pred - truth);
power-spectrum ratio pred/truth at 9-20 km scales (patch Nyquist 8.9 km); CRPS (diffusion ensemble; for deterministic
models CRPS = MAE).
    python -m pipeline.downscale_eval   -> reports/DOWNSCALING.md, reports/downscaling_results.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

from .downscale import (VARS, Normalizer, ResidualDiffusion, UNetDownscaler, avgpool, baseline,  # noqa: E402
                        bicubic, cond_input, crps_ensemble, project, radial_psd)

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "models" / "downscale"
REP = ROOT / "reports"
FIG = REP / "figures"
N_TEST = 300          # test patches used (75 per TEST case); diffusion sampling is the cost
N_SAMPLES = 8
STEPS = 25
UNITS = {"t2m": "degC", "u10": "m/s", "v10": "m/s", "msl": "hPa-1000", "tp": "mm/6h"}


def load_models():
    st = np.load(MD / "unet" / "stats.npz")
    norm = Normalizer(st["mean"], st["std"])
    cu = json.loads((MD / "unet" / "config.json").read_text())
    unet = UNetDownscaler(cu["base"])
    unet.load_state_dict(torch.load(MD / "unet" / "model.pt", map_location="cpu"))
    unet.eval()
    diff = None
    if (MD / "diffusion" / "model.pt").exists():
        cd = json.loads((MD / "diffusion" / "config.json").read_text())
        diff = ResidualDiffusion(cd["base"])
        diff.load_state_dict(torch.load(MD / "diffusion" / "model.pt", map_location="cpu"))
        diff.eval()
    return norm, unet, diff


@torch.no_grad()
def predict_all(x12, dem12, dem5, norm, unet, diff, bs=25, seed=0):
    out = {"bicubic+lapse": [], "unet_unconstrained": [], "unet": [], "diffusion_mean": [],
           "diffusion_sample": []}
    samples = []
    g = torch.Generator().manual_seed(seed)
    for i in range(0, len(x12), bs):
        x, d12, d5 = x12[i:i + bs], dem12[i:i + bs], dem5[i:i + bs]
        out["bicubic+lapse"].append(baseline(x, d12, d5))
        raw = unet(x, d12, d5, norm, do_project=False)
        out["unet_unconstrained"].append(raw)
        m = project(raw, x)
        out["unet"].append(m)
        if diff is not None:
            inp, _ = cond_input(x, d12, d5, norm)
            c = diff.cond(norm.n(m), inp)
            ss = [project(norm.d(norm.n(m) + diff.sample(c, STEPS, g)), x) for _ in range(N_SAMPLES)]
            ss = torch.stack(ss)
            samples.append(ss)
            out["diffusion_mean"].append(project(ss.mean(0), x))
            out["diffusion_sample"].append(ss[0])
    out = {k: torch.cat(v).numpy() for k, v in out.items() if v}
    return out, (torch.cat(samples, 1).numpy() if samples else None)


def metrics(pred, y, x12, name, samples=None):
    r = {}
    for v, var in enumerate(VARS):
        p, t = pred[:, v], y[:, v]
        k, Pp = radial_psd(p[:, :, :])
        _, Pt = radial_psd(t[:, :, :])
        n = p.shape[-1]
        # wavenumber k (cycles per patch) = wavelength n*4.44/k km; 20 km .. Nyquist (8.9 km)
        hi = (k >= n * 4.44 / 20) & (k <= n / 2)
        cons = float(np.abs(avgpool(torch.tensor(pred[:, v:v + 1])).numpy()[:, 0] - x12[:, v]).max())
        e = {"rmse": float(np.sqrt(((p - t) ** 2).mean())),
             "p99_ratio": float(np.percentile(p, 99) / np.percentile(t, 99)) if var == "tp" else
             float((np.percentile(p, 99) - np.median(t)) / (np.percentile(t, 99) - np.median(t))),
             "p999_ratio": float(np.percentile(p, 99.9) / np.percentile(t, 99.9)) if var == "tp" else
             float((np.percentile(p, 99.9) - np.median(t)) / (np.percentile(t, 99.9) - np.median(t))),
             "spectral_ratio_9_20km": float(np.mean(Pp[hi] / Pt[hi])),
             "conservation_err": cons}
        if var == "tp":
            pk_p, pk_t = p.reshape(len(p), -1).max(1), t.reshape(len(t), -1).max(1)
            e["peak_rain_bias"] = float(np.mean(pk_p - pk_t))
            e["peak_rain_mae"] = float(np.mean(np.abs(pk_p - pk_t)))
            e["peak_rain_ratio"] = float(np.mean(pk_p) / np.mean(pk_t))
        e["crps"] = crps_ensemble(samples[:, :, v], t) if samples is not None else float(np.abs(p - t).mean())
        r[var] = e
    return r


@torch.no_grad()
def real_amphan(norm, unet, diff):
    """Downscale REAL ERA5 G12 Amphan fields; compare daily 5 km rain with IMD 0.25 (sanity check).
    Conservation means the 12 km (and hence ~0.25 deg) totals equal ERA5's by construction."""
    e = xr.open_dataset(ROOT / "data/real/era5/amphan_era5_g12.nc")
    dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)
    dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)
    times = pd.to_datetime(e.time.values)
    days = pd.date_range("2020-05-19", "2020-05-21")        # Amphan landfall / inland days
    # window over coastal WB / Odisha / Bangladesh: 12 km cells i0..i0+48, j0..j0+48
    i0, j0 = int((18.5 - 0) / 0.12), int((84.5 - 60) / 0.12)
    sl12 = (slice(i0, i0 + 48), slice(j0, j0 + 48))
    sl5 = (slice(3 * i0, 3 * i0 + 144), slice(3 * j0, 3 * j0 + 144))
    res = {k: [] for k in ("bicubic+lapse", "unet", "diffusion_sample", "era5_12km")}
    for day in days:
        idx = [i for i, t in enumerate(times) if day - pd.Timedelta("21h") < t <= day + pd.Timedelta("3h")]
        tot = {k: 0 for k in res}
        for i in idx:
            f = np.stack([e.t2m.values[i] - 273.15, e.u10.values[i], e.v10.values[i],
                          e.msl.values[i] / 100 - 1000, e.tp.values[i]])[:, sl12[0], sl12[1]]
            x = torch.tensor(f[None], dtype=torch.float32)
            d12 = torch.tensor(dem12[sl12][None, None])
            d5 = torch.tensor(dem5[sl5][None, None])
            b = baseline(x, d12, d5)
            u = unet(x, d12, d5, norm)
            tot["bicubic+lapse"] = tot["bicubic+lapse"] + b[0, 4].numpy()
            tot["unet"] = tot["unet"] + u[0, 4].numpy()
            if diff is not None:
                inp, _ = cond_input(x, d12, d5, norm)
                s = project(norm.d(norm.n(u) + diff.sample(diff.cond(norm.n(u), inp), STEPS,
                                                            torch.Generator().manual_seed(1))), x)
                tot["diffusion_sample"] = tot["diffusion_sample"] + s[0, 4].numpy()
            tot["era5_12km"] = tot["era5_12km"] + f[4]
        for k in res:
            if not isinstance(tot[k], int):
                res[k].append(tot[k])
    imd = xr.open_dataset(ROOT / "data/real/imd/imd_rain_2020_0p25.nc").rain.sel(
        time=days, latitude=slice(18.5, 18.5 + 48 * 0.12), longitude=slice(84.5, 84.5 + 48 * 0.12)).values
    out = {"window": "18.5-24.3N, 84.5-90.3E; IMD days 2020-05-19..21 (real data)"}
    q = lambda a: {"p99": float(np.nanpercentile(a, 99)), "max": float(np.nanmax(a)),
                   "mean": float(np.nanmean(a))}
    out["IMD_0p25"] = q(imd[np.isfinite(imd)])
    for k, v in res.items():
        if v:
            out[k] = q(np.array(v))
    e.close()
    return out


def figures(y, preds, x12):
    # example patch: the TEST patch with the largest truth rain
    i = int(np.argmax(y[:, 4].reshape(len(y), -1).max(1)))
    cols = ["truth_5km", "input_12km", "bicubic+lapse", "unet", "diffusion_sample"]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.4))
    for r, (v, cmap) in enumerate(((4, "Blues"), (0, "RdYlBu_r"))):
        vmax = y[i, v].max()
        vmin = 0 if v == 4 else y[i, v].min()
        for c, name in enumerate(cols):
            a = {"truth_5km": y[i, v], "input_12km": np.kron(x12[i, v], np.ones((3, 3)))}.get(name)
            if a is None:
                a = preds[name][i, v] if name in preds else np.zeros_like(y[i, v])
            axes[r, c].imshow(a, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r, c].set_title(f"{name}\n{VARS[v]} max {a.max():.1f}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle("SYNTHETIC TEST patch (heaviest rain): 12 km -> 5 km", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "ds_example.png", dpi=100)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, v in zip(axes, (4, 0, 1)):
        k, Pt = radial_psd(y[:, v])
        ax.loglog(144 * 4.44 / np.maximum(k[1:], 1), Pt[1:], "k", lw=2, label="truth 5 km")
        for name, p in preds.items():
            _, Pp = radial_psd(p[:, v])
            ax.loglog(144 * 4.44 / np.maximum(k[1:], 1), Pp[1:], label=name, lw=1)
        ax.invert_xaxis()
        ax.set_xlabel("wavelength (km)")
        ax.set_title(f"{VARS[v]} power spectrum (TEST patches)")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ds_spectra.png", dpi=100)
    plt.close(fig)


def fmt(x, n=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "bundle" / "downscale_test.npz")
    cases = [str(c) for c in z["cases"]]
    meta = z["meta"]
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(meta[:, 0] == ci)[0], N_TEST // len(cases), replace=False)
                          for ci in range(len(cases))])
    t = lambda k: torch.tensor(z[k][idx].astype(np.float32))
    x12, y5, dem12, dem5 = t("x12"), t("y5"), t("dem12"), t("dem5")
    norm, unet, diff = load_models()
    preds, samples = predict_all(x12, dem12, dem5, norm, unet, diff)
    y, xx = y5.numpy(), x12.numpy()
    res = {name: metrics(p, y, xx, name, samples if name == "diffusion_mean" else None)
           for name, p in preds.items()}
    real = real_amphan(norm, unet, diff)
    figures(y, preds, xx)
    cfg = {m: json.loads((MD / m / "config.json").read_text()) for m in ("unet", "diffusion")
           if (MD / m / "config.json").exists()}
    (REP / "downscaling_results.json").write_text(json.dumps(
        {"test_cases": cases, "n_patches": int(len(idx)), "metrics": res, "real_amphan": real,
         "configs": cfg}, indent=1))
    write_report(res, real, cases, len(idx), cfg)


def write_report(res, real, cases, n, cfg):
    L = ["# 12 km -> 5 km downscaling (BRIEF2 Phase 3)", "",
         f"The test set is {n} **SYNTHETIC** patches (48x48 cells at 12 km -> 144x144 at 5 km) from the held-out "
         f"TEST cases {', '.join(cases)}, with exact 5 km truth. The models were trained on TRAIN patches only "
         "(`scripts/train_downscaler.py`) and selected on VAL patches.", "",
         "Models (`pipeline/downscale.py`); every one ends in the same exact conservation projection:",
         "- (a) **bicubic+lapse**: bicubic interpolation plus a -6.5 K/km DEM lapse-rate correction for t2m.",
         "- (b) **unet**: a U-Net predicting the residual over bicubic, conditioned on the DEM. Loss = MSE "
         "+ 0.1 spectral + 0.5 pinball(q = 0.99). **unet_unconstrained** is the same network without "
         "the projection.",
         "- (c) **diffusion**: CorrDiff-style, i.e. the U-Net mean plus a DDPM on the residual (T = 500), "
         f"sampled with DDIM ({STEPS} steps). **diffusion_mean** is the mean of {N_SAMPLES} samples; "
         "**diffusion_sample** is a single sample, the realistic-texture product.", "",
         "The projection is additive for t2m/u10/v10/msl and multiplicative for tp (so rain stays >= 0). "
         "avgpool(5 km output) equals the 12 km input to float32 rounding.", "",
         "Amplitude ratio columns:",
         "- For tp: p99(pred) / p99(truth).",
         "- For the other variables: (p99 - median) / (truth p99 - median), a tail-amplitude ratio.",
         "- 1.0 is perfect; below 1 means extremes are smoothed away.", ""]
    for var in VARS:
        L += [f"## {var} ({UNITS[var]})", "",
              "| model | RMSE | CRPS | p99 ratio | p99.9 ratio | spectrum ratio 9-20 km (Nyquist 8.9 km) | conservation err |"
              + (" peak-rain bias | peak-rain ratio |" if var == "tp" else ""),
              "|---|---|---|---|---|---|---|" + ("---|---|" if var == "tp" else "")]
        for name, r in res.items():
            e = r[var]
            row = (f"| {name} | {fmt(e['rmse'])} | {fmt(e['crps'])} | {fmt(e['p99_ratio'])} | "
                   f"{fmt(e['p999_ratio'])} | {fmt(e['spectral_ratio_9_20km'])} | {e['conservation_err']:.1e} |")
            if var == "tp":
                row += f" {fmt(e['peak_rain_bias'], 2)} | {fmt(e['peak_rain_ratio'])} |"
            L.append(row)
        L.append("")
    L += ["CRPS for deterministic models equals their MAE. The diffusion row uses the "
          f"{N_SAMPLES}-member sample ensemble.", "",
          "![example](figures/ds_example.png)", "", "![spectra](figures/ds_spectra.png)", "",
          "## Real-data sanity check: ERA5 Amphan -> 5 km vs IMD 0.25 deg", "",
          f"Window: {real['window']}. Daily rain (mm/day). The input is ERA5, so the 12 km (and therefore "
          "~0.25 deg) totals are ERA5's by construction; this checks only that the downscaled extremes are "
          "plausible next to IMD. It is not a skill test.", "",
          "| source | mean | p99 | max |", "|---|---|---|---|"]
    for k in ("IMD_0p25", "era5_12km", "bicubic+lapse", "unet", "diffusion_sample"):
        if k in real:
            L.append(f"| {k} | {fmt(real[k]['mean'], 1)} | {fmt(real[k]['p99'], 1)} | {fmt(real[k]['max'], 1)} |")
    L += ["", "## Training", "", "| model | epochs | best epoch | val loss | train time (s) | compute |",
          "|---|---|---|---|---|---|"]
    for m, c in cfg.items():
        L.append(f"| {m} | {c['epochs']} | {c['best_epoch']} | {fmt(c['best_val_loss'], 4)} | "
                 f"{c['train_seconds']:.0f} | {c['compute']} |")
    L.append("")
    (REP / "DOWNSCALING.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/DOWNSCALING.md")


if __name__ == "__main__":
    main()
