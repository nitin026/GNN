"""Phase 5 baseline: z-score anomalies + EFI + threshold/label/Hungarian tracker.

Evaluates on every synthetic case (IoU vs exact masks, centroid error in km, ensemble
detection, EFI discrimination) and on real Amphan (ERA5 G12 vs IBTrACS track error).
Writes reports/BASELINE_RESULTS.md, reports/baseline_results.json and figures.

    python -m pipeline.evaluate
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from .anomaly import Climatology  # noqa: E402
from .efi import efi_gaussian  # noqa: E402
from .track import detect, haversine_km, iou, link, main_track, track_masks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
REP = ROOT / "reports"
FIG = REP / "figures"

# hazard -> (variable, sign, z threshold, min cells, link gate km)
CONFIG = {"tropical_cyclone": ("msl", -1, 2.0, 9, 400.0),
          "heat_dome": ("t2m", +1, 1.75, 100, 500.0),
          "cold_wave": ("t2m", -1, 1.75, 100, 500.0)}
SMOOTH_CELLS = 2.0   # Gaussian smoothing (sigma, G12 cells ~ 25 km) of z before thresholding


def decode(ds, v):
    return ds[v].values.astype(np.float64)


def roc_auc(score, label):
    """Mann-Whitney AUC (probability a random event pixel outscores a non-event pixel)."""
    s, y = np.asarray(score, float).ravel(), np.asarray(label, bool).ravel()
    if y.all() or not y.any():
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (len(s) - npos)))


def run_detection(fields, times, lat, lon, clim, hazard):
    var, sign, thr, minc, gate = CONFIG[hazard]
    frames = []
    for i, t in enumerate(times):
        z = gaussian_filter(clim.z(var, fields[i], t), SMOOTH_CELLS)
        cf = -fields[i] if hazard == "tropical_cyclone" else sign * z
        frames.append(detect(z, lat, lon, thr, sign=sign, min_cells=minc, center_field=cf))
    tracks = link(frames, max_km=gate, key="center" if hazard == "tropical_cyclone" else "centroid")
    return frames, tracks


def ref_position(e, hazard):
    if hazard == "tropical_cyclone":
        return e.get("center_lat"), e.get("center_lon")
    return e.get("mask_centroid_lat"), e.get("mask_centroid_lon")


def evaluate_case(case_dir, clim):
    lab = json.loads((case_dir / "labels.json").read_text())
    hazard = lab["hazard"]
    var, sign, thr, *_ = CONFIG[hazard]
    ds = xr.open_dataset(case_dir / "fcst_12km.nc", decode_timedelta=False)
    lat, lon = ds.latitude.values, ds.longitude.values
    times = pd.to_datetime(lab["valid_times"])
    truth = decode(ds, f"{var}_truth")
    tmask = ds["event_mask_truth"].values.astype(bool)
    frames, tracks = run_detection(truth, times, lat, lon, clim, hazard)
    mt = main_track(tracks)
    pmask = track_masks(mt, len(times), tmask.shape[1:])
    both = [i for i in range(len(times)) if tmask[i].any() or pmask[i].any()]
    ious = [iou(pmask[i], tmask[i]) for i in both]
    inter = (pmask & tmask).sum()
    union = (pmask | tmask).sum()
    errs = []
    key = "center" if hazard == "tropical_cyclone" else "centroid"
    if mt:
        for t, o in mt:
            la, lo = ref_position(lab["truth_track"][t], hazard)
            if la is not None and tmask[t].any():
                errs.append(haversine_km(o[f"{key}_lat"], o[f"{key}_lon"], la, lo))
    # ---- ensemble
    ens = decode(ds, var)                                          # (M, T, y, x)
    roles = ds["member_role"].values
    M = ens.shape[0]
    peak_lead = lab["peak"]["lead_h"] if lab.get("peak") else None
    pi = lab["lead_hours"].index(peak_lead) if peak_lead is not None else None
    hits, fa_tracks, lead_err = [], 0, {}
    for m in range(M):
        _, tr_m = run_detection(ens[m], times, lat, lon, clim, hazard)
        mm = main_track(tr_m)
        mmask = track_masks(mm, len(times), tmask.shape[1:])
        if pi is not None:
            hits.append(bool((mmask[pi] & tmask[pi]).any()))
        for tr in tr_m:                   # tracks never touching the truth mask
            if len(tr) >= 2 and not any((o["mask"] & tmask[t]).any() for t, o in tr):
                fa_tracks += 1
        if mm and roles[m] == 0:
            for t, o in mm:
                la, lo = ref_position(lab["truth_track"][t], hazard)
                if la is not None and tmask[t].any():
                    lead_err.setdefault(lab["lead_hours"][t], []).append(
                        haversine_km(o[f"{key}_lat"], o[f"{key}_lon"], la, lo))
    # ---- EFI at the peak lead
    efi_auc = efi_in = efi_out = np.nan
    efi_map = None
    if pi is not None:
        mu, sd = clim.get(var, times[pi])
        e = efi_gaussian(ens[:, pi], mu, sd)
        efi_map = e
        score = sign * e
        efi_auc = roc_auc(score, tmask[pi])
        efi_in = float(np.mean(score[tmask[pi]])) if tmask[pi].any() else np.nan
        efi_out = float(np.mean(score[~tmask[pi]]))
    res = {"case_id": lab["case_id"], "hazard": hazard, "variable": var,
           "z_threshold": thr * sign, "n_steps_with_event": int(tmask.any(axis=(1, 2)).sum()),
           "iou_mean": float(np.nanmean(ious)) if ious else np.nan,
           "iou_aggregate": float(inter / union) if union else np.nan,
           "centroid_err_km_mean": float(np.mean(errs)) if errs else np.nan,
           "centroid_err_km_median": float(np.median(errs)) if errs else np.nan,
           "n_tracks_truth": len(tracks),
           "ens_hit_rate_at_peak": float(np.mean(hits)) if hits else np.nan,
           "ens_false_alarm_tracks": fa_tracks,
           "ens_roles": {"miss": int((roles == 1).sum()), "false_alarm": int((roles == 2).sum())},
           "efi_auc_at_peak": efi_auc, "efi_mean_in_mask": efi_in, "efi_mean_outside": efi_out,
           "ens_track_err_km_by_lead": {int(k): float(np.mean(v)) for k, v in sorted(lead_err.items())}}
    viz = {"lat": lat, "lon": lon, "pi": pi, "tmask": tmask, "pmask": pmask, "efi": efi_map,
           "z": clim.z(var, truth[pi], times[pi]) if pi is not None else None,
           "track": lab["truth_track"], "mt": mt}
    ds.close()
    return res, viz


def evaluate_amphan_real(clim):
    ds = xr.open_dataset(ROOT / "data/real/era5/amphan_era5_g12.nc")
    ib = pd.read_csv(ROOT / "data/real/ibtracs/amphan_2020_ibtracs.csv")
    ib["time"] = pd.to_datetime(ib.ISO_TIME)
    times = pd.to_datetime(ds.time.values)
    msl = ds["msl"].values.astype(float)
    lat, lon = ds.latitude.values, ds.longitude.values
    frames, tracks = run_detection(msl, times, lat, lon, clim, "tropical_cyclone")
    t0, t1 = ib.time.min(), ib.time.max()
    ib6 = ib[ib.time.dt.hour % 6 == 0].set_index("time")
    # the track that best overlaps the IBTrACS period
    def overlap(tr):
        return sum(t0 <= times[t] <= t1 for t, _ in tr)
    best = max(tracks, key=lambda tr: (overlap(tr), len(tr))) if tracks else None
    rows = []
    if best:
        for t, o in best:
            if times[t] in ib6.index:
                r = ib6.loc[times[t]]
                rows.append({"time": str(times[t]), "era5_lat": o["center_lat"],
                             "era5_lon": o["center_lon"], "ib_lat": float(r.LAT),
                             "ib_lon": float(r.LON),
                             "era5_min_msl_hpa": float(msl[t][o["mask"]].min() / 100),
                             "ib_usa_pres_hpa": pd.to_numeric(r.USA_PRES, errors="coerce"),
                             "err_km": float(haversine_km(o["center_lat"], o["center_lon"],
                                                          r.LAT, r.LON))})
    ib_times = ib6.index[(ib6.index >= times[0]) & (ib6.index <= times[-1])]
    ds.close()
    return {"n_ibtracs_6h_fixes": len(ib_times), "n_matched": len(rows),
            "track_err_km_mean": float(np.mean([r["err_km"] for r in rows])) if rows else np.nan,
            "track_err_km_median": float(np.median([r["err_km"] for r in rows])) if rows else np.nan,
            "track_err_km_max": float(np.max([r["err_km"] for r in rows])) if rows else np.nan,
            "rows": rows}


# ------------------------------------------------------------------------------ figures
def fig_examples(vizs, results):
    picks = {}
    for (res, v) in zip(results, vizs):
        if res["hazard"] not in picks and v["pi"] is not None:
            picks[res["hazard"]] = (res, v)
    fig, axes = plt.subplots(2, len(picks), figsize=(5 * len(picks), 9), squeeze=False)
    for k, (hz, (res, v)) in enumerate(picks.items()):
        ax = axes[0, k]
        lim = 4
        im = ax.pcolormesh(v["lon"], v["lat"], v["z"], cmap="RdBu_r", vmin=-lim, vmax=lim,
                           shading="auto")
        ax.contour(v["lon"], v["lat"], v["tmask"][v["pi"]], [0.5], colors="k", linewidths=1.2)
        ax.contour(v["lon"], v["lat"], v["pmask"][v["pi"]], [0.5], colors="lime", linewidths=1.0,
                   linestyles="--")
        tr = [(e["center_lon"], e["center_lat"]) for e in v["track"] if e["center_lat"]]
        if tr:
            ax.plot(*zip(*tr), "k.-", ms=3, lw=0.8)
        ax.set_title(f"{res['case_id']} lead {v['pi'] * 6} h: z({res['variable']})\n"
                     "black = exact mask, green dashed = tracker")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[1, k]
        im = ax.pcolormesh(v["lon"], v["lat"], v["efi"], cmap="RdBu_r", vmin=-1, vmax=1,
                           shading="auto")
        ax.contour(v["lon"], v["lat"], v["tmask"][v["pi"]], [0.5], colors="k", linewidths=1.2)
        ax.set_title(f"EFI ({res['variable']}), AUC={res['efi_auc_at_peak']:.2f}")
        plt.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.ravel():
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.suptitle("SYNTHETIC cases - baseline detection and EFI at peak lead", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_examples.png", dpi=110)
    plt.close(fig)


def fig_lead_error(results):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hz, c in (("tropical_cyclone", "C0"), ("heat_dome", "C3"), ("cold_wave", "C2")):
        acc = {}
        for r in results:
            if r["hazard"] == hz:
                for k, v in r["ens_track_err_km_by_lead"].items():
                    acc.setdefault(k, []).append(v)
        if acc:
            ks = sorted(acc)
            ax.plot(ks, [np.mean(acc[k]) for k in ks], "o-", color=c, ms=3, label=hz)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("member position error vs truth (km)")
    ax.set_title("SYNTHETIC ensembles: tracker position error vs lead (normal members)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "baseline_error_vs_lead.png", dpi=110)
    plt.close(fig)


def fig_amphan(am):
    fig, ax = plt.subplots(figsize=(6, 6))
    ib = pd.read_csv(ROOT / "data/real/ibtracs/amphan_2020_ibtracs.csv")
    ax.plot(ib.LON, ib.LAT, "k.-", label="IBTrACS best track")
    if am["rows"]:
        ax.plot([r["era5_lon"] for r in am["rows"]], [r["era5_lat"] for r in am["rows"]], "ro-",
                ms=4, label="tracker on ERA5 G12 (MSLP z-score)")
    ax.set_xlim(80, 95)
    ax.set_ylim(5, 28)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"REAL Amphan 2020: mean track error {am['track_err_km_mean']:.0f} km")
    fig.tight_layout()
    fig.savefig(FIG / "baseline_amphan_real.png", dpi=110)
    plt.close(fig)


def fmt(x, n=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def write_report(results, am):
    L = ["# Baseline results (Phase 5 smoke test)", "",
         "Baseline = z-score anomaly vs ERA5 1990-2019 climatology -> threshold -> "
         "`scipy.ndimage.label` -> Hungarian frame-to-frame matching (`pipeline/track.py`), plus "
         "the ECMWF EFI (`pipeline/efi.py`). No learning is involved. All 12 km. "
         "z is smoothed with a 25 km Gaussian before thresholding. Thresholds: cyclone MSLP z <= -2.0, "
         "heat dome T2m z >= +1.75, cold wave T2m z <= -1.75; min object size 9 cells (cyclone) / 100 cells "
         "(heat, cold). These were set by hand once for all cases after a first look at heat_01, "
         "not tuned per case.", "",
         "**Synthetic cases are SYNTHETIC** (exact labels). The Amphan row at the end is real ERA5 vs "
         "real IBTrACS.", "",
         "## Synthetic cases: tracker on the 12 km truth (block-averaged 5 km truth)", "",
         "| case | hazard | steps w/ event | IoU mean | IoU aggregate | centroid err mean (km) | "
         "median (km) | ens hit rate @peak | ens false-alarm tracks | EFI AUC @peak | EFI in mask | EFI outside |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['case_id']} | {r['hazard']} | {r['n_steps_with_event']} | {fmt(r['iou_mean'])} | "
                 f"{fmt(r['iou_aggregate'])} | {fmt(r['centroid_err_km_mean'], 0)} | "
                 f"{fmt(r['centroid_err_km_median'], 0)} | {fmt(r['ens_hit_rate_at_peak'])} | "
                 f"{r['ens_false_alarm_tracks']} | {fmt(r['efi_auc_at_peak'])} | "
                 f"{fmt(r['efi_mean_in_mask'])} | {fmt(r['efi_mean_outside'])} |")
    L += ["", "EFI columns use sign-adjusted EFI (for cyclone MSLP and cold-wave T2m the EFI is "
          "negated so that 'more extreme' is positive).", "",
          "## Summary by hazard", "", "| hazard | n | IoU mean | centroid err (km) | EFI AUC |",
          "|---|---|---|---|---|"]
    for hz in ("tropical_cyclone", "heat_dome", "cold_wave"):
        rs = [r for r in results if r["hazard"] == hz]
        if rs:
            L.append(f"| {hz} | {len(rs)} | {fmt(np.nanmean([r['iou_mean'] for r in rs]))} | "
                     f"{fmt(np.nanmean([r['centroid_err_km_mean'] for r in rs]), 0)} | "
                     f"{fmt(np.nanmean([r['efi_auc_at_peak'] for r in rs]))} |")
    L += ["", "## Ensemble position error vs lead (normal members, mean over cases)", "",
          "| lead (h) | " + " | ".join(("tropical_cyclone", "heat_dome", "cold_wave")) + " |",
          "|---|---|---|---|"]
    for ld in (24, 72, 120, 168, 240):
        row = []
        for hz in ("tropical_cyclone", "heat_dome", "cold_wave"):
            v = [r["ens_track_err_km_by_lead"].get(ld) for r in results if r["hazard"] == hz]
            v = [x for x in v if x is not None]
            row.append(fmt(np.mean(v), 0) if v else "n/a")
        L.append(f"| {ld} | " + " | ".join(row) + " |")
    L += ["", "![examples](figures/baseline_examples.png)", "",
          "![error vs lead](figures/baseline_error_vs_lead.png)", "",
          "## Real data: Amphan 2020, tracker on ERA5 G12 vs IBTrACS", "",
          f"- IBTrACS 6-hourly fixes inside the ERA5 window: {am['n_ibtracs_6h_fixes']}; matched by tracker: "
          f"{am['n_matched']}",
          f"- Track error (km): mean {fmt(am['track_err_km_mean'], 0)}, median "
          f"{fmt(am['track_err_km_median'], 0)}, max {fmt(am['track_err_km_max'], 0)}", "",
          "| time | ERA5 lat | ERA5 lon | IBTrACS lat | IBTrACS lon | error km | ERA5 min MSLP hPa | IBTrACS USA_PRES hPa |",
          "|---|---|---|---|---|---|---|---|"]
    for r in am["rows"]:
        L.append(f"| {r['time']} | {r['era5_lat']:.2f} | {r['era5_lon']:.2f} | {r['ib_lat']:.1f} | "
                 f"{r['ib_lon']:.1f} | {r['err_km']:.0f} | {r['era5_min_msl_hpa']:.1f} | "
                 f"{fmt(r['ib_usa_pres_hpa'], 0)} |")
    L += ["", "![amphan](figures/baseline_amphan_real.png)", "",
          "## Reading these numbers", "",
          "- The synthetic IoU/centroid numbers measure how well a *fixed-threshold* detector recovers an "
          "exactly known mask whose definition (physical-unit injected anomaly) differs from the detector's "
          "(z-score of the full field). They are a floor for the planned GNN tracker, not a skill claim.",
          "- The synthetic backgrounds had real anomalies soft-clipped to 1.5 sigma, which makes detection "
          "easier than on raw ERA5; the Amphan-on-ERA5 row is the honest real-data check.",
          "- Ensemble spread, 'miss' and 'false alarm' members were prescribed by the generator, so ensemble "
          "scores here test the pipeline plumbing, not forecast skill.",
          "- **The per-member tracker is weak.** The generator adds large-scale background noise to each "
          "member (T2m up to ~1.5 K at 240 h), which creates many threshold exceedances. That produces "
          "hundreds of 'false-alarm tracks' per case, a low hit rate, and heat/cold member position errors "
          "of several hundred km even at short leads, because the 'main track' is sometimes a noise object "
          "and not the event. Fixed thresholds with largest-area track selection cannot separate them. "
          "This is exactly the gap the planned GNN tracker (learned, ensemble-aware) has to close. The "
          "EFI, which uses the whole ensemble at once, discriminates the events far better (AUC column).", ""]
    (REP / "BASELINE_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    clim = Climatology()
    results, vizs = [], []
    for d in sorted(p for p in SYN.iterdir() if (p / "labels.json").exists()):
        r, v = evaluate_case(d, clim)
        print(f"{r['case_id']}: IoU {r['iou_mean']:.2f} err {r['centroid_err_km_mean']:.0f} km "
              f"EFI AUC {r['efi_auc_at_peak']:.2f}")
        results.append(r)
        vizs.append(v)
    am = evaluate_amphan_real(clim)
    print(f"Amphan real: mean track error {am['track_err_km_mean']:.0f} km over {am['n_matched']} fixes")
    fig_examples(vizs, results)
    fig_lead_error(results)
    fig_amphan(am)
    (REP / "baseline_results.json").write_text(json.dumps({"synthetic": results, "amphan_real": am},
                                                          indent=1, default=float))
    write_report(results, am)


if __name__ == "__main__":
    main()
