"""BRIEF4 Phase 0: streaming evaluation of the 12 km -> 5 km downscalers.

Memory-safe: data are read from the full-precision NetCDF files one case and one lead at a time,
and scored in patch batches (8 patches of 48x48 -> 144x144). Every metric is accumulated online:
running sums (RMSE, MAE, CRPS, spread/skill), fixed-bin histograms (p99/p99.9), averaged
per-patch radial spectra, rank-histogram counts, and running max/min (conservation, rain >= 0).

Test set: the synthetic TEST cases (pipeline/splits.py). Leads every 24 h (0..240 h). Per lead,
PATCHES_PER_LEAD patches: half centred on the exact event mask (if present), half random
(fixed seed). All three models are scored on the same pixels.
Real-data check: IMD 0.25 deg rain 2020 (JJAS) coarsened 3x to 0.75 deg and downscaled back
(a PERFECT-MODEL test on real data, out of distribution for the models: other input channels are
set to their training mean, DEM from G12).

    python -m pipeline.downscale_eval  -> reports/DOWNSCALE_RESULTS.md, reports/downscaling_results.json
"""
import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

from .downscale import (Normalizer, ResidualDiffusion, UNetDownscaler, baseline,  # noqa: E402
                        cond_input, project)
from .jsonutil import dumps as strict_dumps  # noqa: E402
from .splits import TEST  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "models" / "downscale"
REP = ROOT / "reports"
FIG = REP / "figures"
SYN = ROOT / "data" / "synthetic"
N_SAMPLES, STEPS = 8, 25
LEAD_STEPS = list(range(0, 41, 4))            # every 24 h
PATCHES_PER_LEAD = 8
P12 = 48
DX5 = 4.44                                     # km
EVAL_VARS = ["t2m", "wind", "msl", "tp"]       # wind = |(u10, v10)|
MODELS = ["bicubic+lapse", "unet", "diffusion_mean", "diffusion_sample"]
HIST = {"t2m": (-40, 60, 0.01), "wind": (0, 120, 0.01), "msl": (-80, 60, 0.01), "tp": (0, 600, 0.02)}
UNITS = {"t2m": "degC", "wind": "m/s", "msl": "hPa-1000", "tp": "mm/6h"}
OFFSET = {"t2m": 273.15, "u10": 0.0, "v10": 0.0, "msl": 100000.0, "tp": 0.0}
SCALE = {"t2m": 1.0, "u10": 1.0, "v10": 1.0, "msl": 0.01, "tp": 1.0}
PEAK = {"rss": 0.0}


def rss():
    r = psutil.Process().memory_info().rss / 1e6
    PEAK["rss"] = max(PEAK["rss"], r)
    return r


# ----------------------------------------------------------------------------- accumulators
class Acc:
    """Online metrics for one (model, variable)."""

    def __init__(self, var):
        lo, hi, w = HIST[var]
        self.edges = np.arange(lo, hi + w, w)
        self.hp = np.zeros(len(self.edges) - 1)
        self.ht = np.zeros(len(self.edges) - 1)
        self.n = 0
        self.se = self.ae = 0.0
        self.psd_p = self.psd_t = None
        self.npsd = 0
        self.peak_err = []                       # per-patch (pred_max - truth_max)
        self.peak_truth = []
        self.crps = 0.0
        self.var_sum = 0.0
        self.rank = np.zeros(N_SAMPLES + 1)
        self.cons = 0.0
        self.minval = np.inf

    def add(self, p, t, x12=None, samples=None, spectra=True):
        """p, t: (B, H, W) float64 patches; samples: (S, B, H, W) for diffusion."""
        d = p - t
        self.se += float((d ** 2).sum())
        self.ae += float(np.abs(d).sum())
        self.n += d.size
        self.hp += np.histogram(np.clip(p, self.edges[0], self.edges[-1] - 1e-9), self.edges)[0]
        self.ht += np.histogram(np.clip(t, self.edges[0], self.edges[-1] - 1e-9), self.edges)[0]
        if spectra:
            k, Pp = radial_psd(p)
            _, Pt = radial_psd(t)
            self.psd_p = Pp * len(p) if self.psd_p is None else self.psd_p + Pp * len(p)
            self.psd_t = Pt * len(t) if self.psd_t is None else self.psd_t + Pt * len(t)
            self.npsd += len(p)
            self.k = k
        self.peak_err += list(p.reshape(len(p), -1).max(1) - t.reshape(len(t), -1).max(1))
        self.peak_truth += list(t.reshape(len(t), -1).max(1))
        self.minval = min(self.minval, float(p.min()))
        if x12 is not None:
            self.cons = max(self.cons, float(np.abs(pool(p) - x12).max()))
        if samples is not None:
            s = samples
            t1 = np.abs(s - t[None]).mean(0)
            srt = np.sort(s, axis=0)
            # fair CRPS with the sorted-sample identity: E|X-X'| = 2/(S^2) sum (2i-S-1) x_(i)
            S = s.shape[0]
            w = (2 * np.arange(1, S + 1) - S - 1)[:, None, None, None]
            t2 = 2.0 * (w * srt).sum(0) / (S * (S - 1))
            self.crps += float((t1 - 0.5 * t2).sum())
            self.var_sum += float(s.var(0, ddof=1).sum())
            r = (s < t[None]).sum(0).ravel()
            self.rank += np.bincount(r, minlength=S + 1)[:S + 1]

    def quantile(self, h, q):
        c = np.cumsum(h) / h.sum()
        i = int(np.searchsorted(c, q))
        return float(self.edges[min(i + 1, len(self.edges) - 1)])

    def result(self, var):
        out = {"rmse": np.sqrt(self.se / self.n), "mae": self.ae / self.n, "n_pixels": self.n,
               "p99_pred": self.quantile(self.hp, 0.99), "p99_truth": self.quantile(self.ht, 0.99),
               "p999_pred": self.quantile(self.hp, 0.999), "p999_truth": self.quantile(self.ht, 0.999),
               "conservation_err": self.cons, "min_value": self.minval}
        med = self.quantile(self.ht, 0.5)
        for q in ("p99", "p999"):
            if var in ("tp", "wind"):
                out[f"{q}_ratio"] = out[f"{q}_pred"] / max(out[f"{q}_truth"], 1e-9)
            else:                               # tail amplitude above the median
                out[f"{q}_ratio"] = (out[f"{q}_pred"] - med) / max(out[f"{q}_truth"] - med, 1e-9)
        pe = np.array(self.peak_err)
        out["peak_bias"] = float(pe.mean())
        out["peak_mae"] = float(np.abs(pe).mean())
        out["peak_ratio"] = float((np.array(self.peak_truth) + pe).mean() / np.mean(self.peak_truth))
        if self.npsd:
            Pp, Pt = self.psd_p / self.npsd, self.psd_t / self.npsd
            wl = 144 * DX5 / np.maximum(self.k, 1e-9)
            for L in (10, 25, 50):
                j = int(np.argmin(np.abs(wl - L)))
                out[f"psd_ratio_{L}km"] = float(Pp[j] / Pt[j])
            out["spectrum"] = {"wavelength_km": wl[1:].tolist(), "pred": Pp[1:].tolist(),
                               "truth": Pt[1:].tolist()}
        if self.rank.sum():
            out["crps"] = self.crps / self.n
            out["rank_hist"] = (self.rank / self.rank.sum()).tolist()
            out["spread_skill"] = float(np.sqrt(self.var_sum / self.n) / out["rmse"])
        return out


def pool(a, k=3):
    B, H, W = a.shape
    return a.reshape(B, H // k, k, W // k, k).mean((2, 4))


def radial_psd(a):
    a = a - a.mean((-2, -1), keepdims=True)
    n = a.shape[-1]
    win = np.outer(np.hanning(n), np.hanning(n))
    F = np.abs(np.fft.fftshift(np.fft.fft2(a * win), axes=(-2, -1))) ** 2
    ky, kx = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")
    k = np.hypot(ky, kx).astype(int).ravel()
    cnt = np.bincount(k)
    P = np.array([np.bincount(k, f.ravel()) / np.maximum(cnt, 1) for f in F]).mean(0)
    return np.arange(len(P)), P


# ----------------------------------------------------------------------------- models
def load_models():
    st = np.load(MD / "unet" / "stats.npz")
    norm = Normalizer(st["mean"], st["std"])
    cu = json.loads((MD / "unet" / "config.json").read_text())
    unet = UNetDownscaler(cu["base"])
    unet.load_state_dict(torch.load(MD / "unet" / "model.pt", map_location="cpu"))
    unet.eval()
    cd = json.loads((MD / "diffusion" / "config.json").read_text())
    diff = ResidualDiffusion(cd["base"])
    diff.load_state_dict(torch.load(MD / "diffusion" / "model.pt", map_location="cpu"))
    diff.eval()
    return norm, unet, diff, cu, cd


def to_eval_vars(y):
    """(B, 5, H, W) bundle units -> dict of eval vars (B, H, W) float64."""
    y = y.astype(np.float64)
    return {"t2m": y[:, 0], "wind": np.hypot(y[:, 1], y[:, 2]), "msl": y[:, 3], "tp": y[:, 4]}


@torch.no_grad()
def run_models(x12, d12, d5, norm, unet, diff, gen):
    x, a12, a5 = (torch.tensor(v, dtype=torch.float32) for v in (x12, d12, d5))
    out = {"bicubic+lapse": baseline(x, a12, a5).numpy()}
    m = unet(x, a12, a5, norm)
    out["unet"] = m.numpy()
    inp, _ = cond_input(x, a12, a5, norm)
    c = diff.cond(norm.n(m), inp)
    s = torch.stack([project(norm.d(norm.n(m) + diff.sample(c, STEPS, gen)), x) for _ in range(N_SAMPLES)])
    out["diffusion_mean"] = project(s.mean(0), x).numpy()
    out["diffusion_sample"] = s[0].numpy()
    return out, s.numpy()


# ----------------------------------------------------------------------------- synthetic TEST
def decode(v, idx):
    v.set_auto_maskandscale(False)
    return v[idx].astype(np.float64) * float(v.scale_factor) + float(v.add_offset)


def synthetic_eval(norm, unet, diff):
    acc = {m: {v: Acc(v) for v in EVAL_VARS} for m in MODELS}
    dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)
    dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)
    rng = np.random.default_rng(2026)
    gen = torch.Generator().manual_seed(7)
    example = None
    per_case = {}
    for case in TEST:
        t0 = time.time()
        t5 = netCDF4.Dataset(SYN / case / "truth_5km.nc")
        f12 = netCDF4.Dataset(SYN / case / "fcst_12km.nc")
        masks = f12.variables["event_mask_truth"]
        n_p = 0
        for s in LEAD_STEPS:
            y5 = np.stack([(decode(t5.variables[v], s) - OFFSET[v]) * SCALE[v] for v in
                           ("t2m", "u10", "v10", "msl", "tp")])
            x12 = np.stack([(f12.variables[f"{v}_truth"][s].astype(np.float64) - OFFSET[v]) * SCALE[v]
                            for v in ("t2m", "u10", "v10", "msl", "tp")])
            ev = np.argwhere(masks[s][:].astype(bool))
            starts = []
            for k in range(PATCHES_PER_LEAD):
                if k % 2 == 0 and len(ev):
                    ci, cj = ev[rng.integers(len(ev))]
                    i0 = int(np.clip(ci - P12 // 2, 0, 333 - P12))
                    j0 = int(np.clip(cj - P12 // 2, 0, 333 - P12))
                else:
                    i0, j0 = (int(v) for v in rng.integers(0, 333 - P12, 2))
                starts.append((i0, j0))
            X = np.array([x12[:, i:i + P12, j:j + P12] for i, j in starts], np.float32)
            Y = np.array([y5[:, 3 * i:3 * (i + P12), 3 * j:3 * (j + P12)] for i, j in starts])
            D12 = np.array([dem12[None, i:i + P12, j:j + P12] for i, j in starts])
            D5 = np.array([dem5[None, 3 * i:3 * (i + P12), 3 * j:3 * (j + P12)] for i, j in starts])
            preds, samples = run_models(X, D12, D5, norm, unet, diff, gen)
            tv = to_eval_vars(Y)
            xv = to_eval_vars(X)
            sv = {v: np.stack([to_eval_vars(smp)[v] for smp in samples]) for v in EVAL_VARS}
            for mname, p in preds.items():
                pv = to_eval_vars(p)
                for v in EVAL_VARS:
                    acc[mname][v].add(pv[v], tv[v], x12=xv[v] if v != "wind" else None,
                                      samples=sv[v] if mname == "diffusion_mean" else None)
            # example panel: heaviest-rain patch seen so far
            j = int(np.argmax(Y[:, 4].reshape(len(Y), -1).max(1)))
            if example is None or Y[j, 4].max() > example["truth"][4].max():
                example = {"case": case, "lead_h": 6 * s, "x12": X[j], "truth": Y[j],
                           **{m: preds[m][j] for m in preds}}
            n_p += len(starts)
            del y5, x12, X, Y, preds, samples, sv
            gc.collect()
            rss()
        t5.close()
        f12.close()
        per_case[case] = {"patches": n_p, "seconds": round(time.time() - t0, 1), "rss_mb": round(rss(), 0)}
        print(f"{case}: {n_p} patches, {time.time() - t0:.0f}s, RSS {rss():.0f} MB", flush=True)
    res = {m: {v: acc[m][v].result(v) for v in EVAL_VARS} for m in MODELS}
    return res, example, per_case


# ----------------------------------------------------------------------------- real IMD perfect-model test
@torch.no_grad()
def imd_eval(norm, unet, diff):
    """IMD 0.25 deg JJAS 2020 rain -> 3x3 block mean (0.75 deg) -> downscale -> score vs 0.25 deg.
    Inputs other than rain are set to the TRAIN mean (normalised 0); DEM is G12 interpolated to
    the IMD grid. Rain in mm/day is fed as-is (the models were trained on mm/6h): OUT OF
    DISTRIBUTION in scale and units, reported as a stress test."""
    imd = xr.open_dataset(ROOT / "data/real/imd/imd_rain_2020_0p25.nc").rain
    imd = imd.sel(time=slice("2020-06-01", "2020-09-30"))
    lat, lon = imd.latitude.values, imd.longitude.values
    dem = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.interp(latitude=lat, longitude=lon).values
    dem = np.nan_to_num(dem).astype(np.float32)
    ny, nx = (len(lat) // 3) * 3, (len(lon) // 3) * 3                  # 129 x 135 -> exact 3x
    acc = {m: Acc("tp") for m in MODELS}
    gen = torch.Generator().manual_seed(11)
    mean = norm.m.numpy()[0, :, 0, 0]
    days = imd.time.values
    for k in range(0, len(days), 8):
        R = imd.sel(time=days[k:k + 8]).values[:, :ny, :nx].astype(np.float64)
        valid = np.isfinite(R).all(0)
        R = np.nan_to_num(R)
        x = R.reshape(len(R), ny // 3, 3, nx // 3, 3).mean((2, 4))              # 0.75 deg
        # pad coarse grid to a multiple of 4 (fine = multiple of 4 as the U-Net needs)
        py, px = (-x.shape[1]) % 4, (-x.shape[2]) % 4
        xp = np.pad(x, ((0, 0), (0, py), (0, px)), mode="edge")
        X = np.repeat(mean[None, :, None, None], len(R), 0) * np.ones((1, 1) + xp.shape[1:])
        X[:, 4] = xp
        d5 = np.pad(dem[:ny, :nx], ((0, 3 * py), (0, 3 * px)), mode="edge")
        d12 = d5.reshape(d5.shape[0] // 3, 3, d5.shape[1] // 3, 3).mean((1, 3))
        D5 = np.repeat(d5[None, None], len(R), 0)
        D12 = np.repeat(d12[None, None], len(R), 0)
        preds, samples = run_models(X.astype(np.float32), D12, D5, norm, unet, diff, gen)
        m = np.repeat(valid[None], len(R), 0)
        for mname, p in preds.items():
            pr = p[:, 4, :ny, :nx]
            s = samples[:, :, 4, :ny, :nx] if mname == "diffusion_mean" else None
            # score only valid (land) IMD cells: set sea cells to truth (zero contribution)
            pr = np.where(m, pr, R)
            if s is not None:
                s = np.where(m[None], s, R[None])
            acc[mname].add(pr, R, x12=None, samples=s, spectra=False)
        del R, X, preds, samples
        gc.collect()
        rss()
    out = {}
    for mname in MODELS:
        r = acc[mname].result("tp")
        out[mname] = r
    out["_note"] = ("PERFECT-MODEL test on REAL IMD 0.25 deg JJAS 2020 rain, coarsened 3x to 0.75 deg. "
                    "Out of distribution: the models were trained at 12->5 km on synthetic mm/6h with all "
                    "5 variables; here only rain is given (other channels = training mean), in mm/day. "
                    "Sea cells are excluded (set equal to the truth).")
    return out


# ----------------------------------------------------------------------------- figures / report
def figures(res, example):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, v in zip(axes, ("tp", "wind", "t2m")):
        sp = res["unet"][v]["spectrum"]
        ax.loglog(sp["wavelength_km"], sp["truth"], "k", lw=2.2, label="truth 5 km")
        for m, sty in (("bicubic+lapse", "C0"), ("unet", "C1"), ("diffusion_sample", "C2"),
                       ("diffusion_mean", "C3")):
            s = res[m][v]["spectrum"]
            ax.loglog(s["wavelength_km"], s["pred"], color=sty, lw=1.2, label=m)
        for L in (10, 25, 50):
            ax.axvline(L, color="grey", ls=":", lw=0.8)
        ax.invert_xaxis()
        ax.set_xlabel("wavelength (km)")
        ax.set_ylabel("power")
        ax.set_title(f"{v}: radially averaged spectrum (SYNTHETIC TEST)")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ds4_spectra.png", dpi=100)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, q in zip(axes, ("p99_ratio", "p999_ratio")):
        xs = np.arange(len(EVAL_VARS))
        for i, m in enumerate(MODELS):
            ax.bar(xs + 0.2 * i - 0.3, [res[m][v][q] for v in EVAL_VARS], 0.2, label=m)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axhline(0.9, color="k", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(EVAL_VARS)
        ax.set_title(f"{q} (pred / truth; dotted = 0.9 target)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ds4_p99.png", dpi=100)
    plt.close(fig)
    cols = [("x12", "12 km input"), ("bicubic+lapse", "bicubic+lapse"), ("unet", "U-Net"),
            ("diffusion_sample", "diffusion sample"), ("diffusion_mean", "diffusion mean"),
            ("truth", "truth 5 km")]
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.6))
    for r, (vi, name, cmap) in enumerate(((4, "tp mm/6h", "viridis"), (None, "wind m/s", "magma"))):
        t = example["truth"]
        tv = t[4] if vi == 4 else np.hypot(t[1], t[2])
        for c, (k, title) in enumerate(cols):
            a = example[k]
            a = a[4] if vi == 4 else np.hypot(a[1], a[2])
            if k == "x12":
                a = np.kron(a, np.ones((3, 3)))
            axes[r, c].imshow(a, origin="lower", cmap=cmap, vmin=0, vmax=tv.max())
            axes[r, c].set_title(f"{title}\n{name} max {a.max():.1f}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle(f"SYNTHETIC TEST {example['case']} lead {example['lead_h']} h (patch with the heaviest rain)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "ds4_example.png", dpi=100)
    plt.close(fig)


def f(x, n=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def write_report(res, imd, per_case, cfg, seconds, ex_max=None):
    L = ["# Downscaling results (BRIEF4 Phase 0; completes BRIEF2 Phase 3)", "",
         f"The test set is the **SYNTHETIC** TEST cases {', '.join(TEST)}, with exact 5 km truth. For each case it "
         f"takes leads every 24 h and {PATCHES_PER_LEAD} patches per lead (48x48 cells at 12 km -> 144x144 at 5 km; "
         "half centred on the event, half random; fixed seed). Data are read from the full-precision NetCDF, and "
         "every model is scored on the same pixels. The models were trained on TRAIN patches and selected on "
         "VAL patches.", "",
         f"Diffusion: {N_SAMPLES} DDIM samples ({STEPS} steps) per patch. `diffusion_mean` is their mean, "
         "re-projected; `diffusion_sample` is sample 1. The evaluation streams one case and one lead at a time, "
         f"and its peak RSS was **{PEAK['rss']:.0f} MB** (the previous version was killed for lack of memory).", "",
         "Ratio columns:",
         "- For tp and wind: p99(pred) / p99(truth).",
         "- For t2m and msl: (p99 - median) / (truth p99 - median).",
         "- 1 is perfect; values below 1 mean the extremes were smoothed.", "",
         "PSD ratio: power(pred) / power(truth) at that wavelength (below 1 means too smooth).", ""]
    for v in EVAL_VARS:
        extra = " peak bias | peak ratio |" if v in ("tp", "wind") else ""
        L += [f"## {v} ({UNITS[v]})", "",
              "| model | RMSE | MAE | CRPS | p99 ratio | p99.9 ratio | PSD 10 km | PSD 25 km | PSD 50 km |"
              " conservation err | min |" + extra,
              "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if extra else "")]
        for m in MODELS:
            r = res[m][v]
            crps = r.get("crps", r["mae"])
            row = (f"| {m} | {f(r['rmse'])} | {f(r['mae'])} | {f(crps)} | {f(r['p99_ratio'])} | "
                   f"{f(r['p999_ratio'])} | {f(r['psd_ratio_10km'])} | {f(r['psd_ratio_25km'])} | "
                   f"{f(r['psd_ratio_50km'])} | {r['conservation_err']:.1e} | {f(r['min_value'], 2)} |")
            if extra:
                row += f" {f(r['peak_bias'], 2)} | {f(r['peak_ratio'])} |"
            L.append(row)
        dm = res["diffusion_mean"][v]
        L += ["", f"Diffusion ensemble: spread/skill = {f(dm['spread_skill'])} (1 = well dispersed); rank "
                  f"histogram (9 bins) = {[round(x, 3) for x in dm['rank_hist']]}."
              + (" For tp, ties at zero rain (truth = all samples = 0) count as rank 0, which inflates "
                 "the first bin." if v == "tp" else ""), ""]
    L += ["CRPS for the deterministic models (and for the single diffusion sample) is their MAE. The "
          "`diffusion_mean` row reports the CRPS of the 8-sample ensemble. Conservation error is "
          "max|avgpool(pred) - x12| (wind is not conserved, only its components are).", "",
          "![spectra](figures/ds4_spectra.png)", "", "![p99](figures/ds4_p99.png)", "",
          "![example](figures/ds4_example.png)", "",
          "## Real data: IMD perfect-model test", "", imd["_note"], "",
          "| model | RMSE (mm/day) | MAE | CRPS | p99 ratio | p99.9 ratio | min | peak ratio |",
          "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        r = imd[m]
        L.append(f"| {m} | {f(r['rmse'], 2)} | {f(r['mae'], 2)} | {f(r.get('crps', r['mae']), 2)} | "
                 f"{f(r['p99_ratio'])} | {f(r['p999_ratio'])} | {f(r['min_value'], 2)} | {f(r['peak_ratio'])} |")
    hl_u = res["unet"]["tp"]["p99_ratio"]
    hl_d = res["diffusion_sample"]["tp"]["p99_ratio"]
    margin = 0.05                                 # a contrast smaller than this is not called a difference
    meets = hl_d >= 0.9 and hl_d > hl_u
    contrast = hl_d - hl_u >= margin
    pk_d, pk_u = res["diffusion_sample"]["tp"]["peak_ratio"], res["unet"]["tp"]["peak_ratio"]
    ps_d, ps_u = res["diffusion_sample"]["tp"]["psd_ratio_10km"], res["unet"]["tp"]["psd_ratio_10km"]
    verdict = ("HOLDS" if meets and contrast else
               "technically meets the threshold, but the claimed CONTRAST is NOT supported at p99" if meets
               else "DOES NOT HOLD")
    L += ["", "## Headline claim: diffusion keeps extremes where the U-Net smooths them", "",
          "- Target: the diffusion p99 ratio (rain) is >= 0.9 and above the U-Net's. A contrast only counts "
          f"if it exceeds {margin}.",
          f"- Result: diffusion sample {f(hl_d)}, diffusion mean {f(res['diffusion_mean']['tp']['p99_ratio'])}, "
          f"U-Net {f(hl_u)}, bicubic {f(res['bicubic+lapse']['tp']['p99_ratio'])}.",
          f"- **Verdict on SYNTHETIC TEST: the claim {verdict}.** The U-Net does not smooth p99 either. "
          "Pixel p99 over all patches is dominated by the 12 km amplitude, which the exact conservation "
          "projection preserves for every model, bicubic included.",
          f"- The contrast does show at small scales and in local peaks. Patch-peak rain ratio: diffusion "
          f"sample {f(pk_d)} vs U-Net {f(pk_u)}. 10 km rain power: {f(ps_d)} vs {f(ps_u)} of the truth. So "
          "diffusion keeps fine-scale texture and peaks, while the U-Net smooths them; p99 does not show it.", "",
          "## What did not work", ""]
    L += what_did_not_work(res, imd)
    if ex_max:
        L.append(f"- The heaviest-rain TEST patch ({ex_max['case']}, lead {ex_max['lead_h']} h): the truth peak is "
                 f"{ex_max['truth']:.1f} mm/6h, while every model stays near the 12 km-limited level (bicubic "
                 f"{ex_max['bicubic+lapse']:.1f}, U-Net {ex_max['unet']:.1f}, diffusion sample "
                 f"{ex_max['diffusion_sample']:.1f}). Sub-grid peaks well above the 12 km value are not recovered.")
    L.append("- The diffusion sample shows noisy patch borders (the wind panel of ds4_example.png), a boundary "
             "artefact of padding-free patch sampling. Tiling with overlap would be needed at inference.")
    L += ["", "## Run", "", "| case | patches | seconds | RSS (MB) |", "|---|---|---|---|"]
    for c, r in per_case.items():
        L.append(f"| {c} | {r['patches']} | {r['seconds']} | {r['rss_mb']} |")
    L += ["", f"Total {seconds:.0f} s on {cfg['compute']}; peak RSS {PEAK['rss']:.0f} MB.", ""]
    (REP / "DOWNSCALE_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def what_did_not_work(res, imd):
    out = []
    d, u = res["diffusion_sample"]["tp"]["p99_ratio"], res["unet"]["tp"]["p99_ratio"]
    if d - u < 0.05:
        out.append(f"- Headline contrast at p99 not shown: the diffusion rain p99 ratio is {d:.3f} vs U-Net {u:.3f} "
                   "(difference < 0.05). The advantage appears only in 10 km power and patch peaks.")
    for v in EVAL_VARS:
        for m in ("unet", "diffusion_sample"):
            r = res[m][v]
            if r["p99_ratio"] < 0.9:
                out.append(f"- {m} {v}: p99 ratio {r['p99_ratio']:.3f} < 0.9 (extremes smoothed).")
            if r["psd_ratio_10km"] < 0.5:
                out.append(f"- {m} {v}: 10 km power is {r['psd_ratio_10km']:.2f} of the truth (too smooth at small scales).")
    d = res["diffusion_mean"]
    for v in EVAL_VARS:
        ss = d[v]["spread_skill"]
        if not 0.8 <= ss <= 1.2:
            out.append(f"- Diffusion ensemble {v}: spread/skill {ss:.2f} (outside 0.8-1.2, "
                       f"{'under' if ss < 0.8 else 'over'}-dispersed).")
    b, u = imd["bicubic+lapse"]["rmse"], imd["unet"]["rmse"]
    if u > b:
        out.append(f"- IMD perfect-model test: U-Net RMSE {u:.2f} > bicubic {b:.2f} mm/day (out of distribution).")
    return out or ["- All targets in this phase were met."]


def rewrite_report():
    """Rebuild DOWNSCALE_RESULTS.md from reports/downscaling_results.json (no re-run)."""
    js = json.loads((REP / "downscaling_results.json").read_text())
    PEAK["rss"] = js["peak_rss_mb"]
    write_report(js["metrics"], js["imd_perfect_model"], js["per_case"], js["configs"]["unet"], js["seconds"],
                 js.get("example_tp_max"))


def main():
    t0 = time.time()
    torch.manual_seed(0)
    norm, unet, diff, cu, cd = load_models()
    res, example, per_case = synthetic_eval(norm, unet, diff)
    imd = imd_eval(norm, unet, diff)
    figures(res, example)
    ex_max = {"case": example["case"], "lead_h": example["lead_h"],
              **{k: float(example[k][4].max()) for k in ("truth", "bicubic+lapse", "unet",
                                                        "diffusion_sample", "diffusion_mean")}}
    seconds = time.time() - t0
    slim = {m: {v: {k: x for k, x in r.items() if k != "spectrum"} for v, r in d.items()} for m, d in res.items()}
    (REP / "downscaling_results.json").write_text(strict_dumps(
        {"test_cases": TEST, "lead_steps": LEAD_STEPS, "patches_per_lead": PATCHES_PER_LEAD,
         "n_samples": N_SAMPLES, "metrics": slim,
         "spectra": {m: {v: res[m][v]["spectrum"] for v in EVAL_VARS} for m in MODELS},
         "imd_perfect_model": imd, "per_case": per_case, "peak_rss_mb": PEAK["rss"],
         "seconds": seconds, "configs": {"unet": cu, "diffusion": cd}, "example_tp_max": ex_max},
        indent=1))
    write_report(res, imd, per_case, cu, seconds, ex_max)
    print(f"done in {seconds:.0f}s, peak RSS {PEAK['rss']:.0f} MB")


if __name__ == "__main__":
    import sys
    rewrite_report() if "--report-only" in sys.argv else main()
