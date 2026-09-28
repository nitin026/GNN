"""BRIEF4 Phase 3: the object GNN, fixed -> reports/OBJECT_GNN_V3.md, reports/object_gnn_v3.json

    python scripts/train_tracker.py --graphs data/graphs_v2 --out models/tracker_v3 --variant full --readout
    (and temporal / cross / none), then  python -m pipeline.phase3_eval

What changed against BRIEF2 (reports/GNN_RESULTS.md):
  * edges carry real information (pipeline.graphs EDGE_FEATURES_V2: Kalman-residual, change of
    mean intensity, steering-wind residual and cosine, member agreement) + a per-lead consensus
    readout in the GNN (pipeline.gnn, readout=True);
  * heat/cold targets and verification use IMD-consistent DAILY labels (scripts/imd_labels.py);
  * the India (Natural Earth) + 250 km domain everywhere (pipeline.tracker2.regions);
  * the TempestExtremes-style comparator has a wind criterion and a moist-core proxy for the
    warm-core criterion (pipeline.te_style.te_cyclones);
  * lead-band calibration fitted on VAL (pipeline.calibration);
  * an early-warning mode: alert at lead t when the calibrated probability that an event exists
    in the domain within the next 72 h exceeds a threshold chosen on VAL.
TEST is only reported. Every choice made on VAL is listed in the report and in TUNING_LOG.md.
"""
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import calibration as cal
from .anomaly import Climatology
from .efi import efi_gaussian
from .evaluate2 import load_case, orog12
from .gnn_eval import case_metrics, decode, infer, load_model
from .jsonutil import dumps
from .splits import TEST, TRAIN, VAL
from .te_style import te_blobs, te_cyclones
from .tracker2 import Tracker2, regions

ROOT = Path(__file__).resolve().parents[1]
GDIR = ROOT / "data/graphs_v2"
MDIR = ROOT / "models/tracker_v3"
SYN = ROOT / "data/synthetic"
REP = ROOT / "reports"
VARIANTS = ["full", "temporal", "cross", "none"]
EW_WINDOW = 12                       # 72 h / 6 h


def load_case_imd(case):
    """evaluate2.load_case with the IMD daily labels for heat/cold (cyclones unchanged)."""
    d = load_case(case)
    if d["hazard"] != "tropical_cyclone":
        z = np.load(SYN / case / "labels_imd.npz")
        X = int(z["shape"][-1])
        d["emask"] = np.unpackbits(z["event_mask_imd"], axis=-1)[..., :X].astype(bool)
        d["tmask"] = np.unpackbits(z["event_mask_imd_truth"], axis=-1)[..., :X].astype(bool)
    return d


def aux_fields(case):
    with xr.open_dataset(SYN / case / "fcst_12km.nc", decode_timedelta=False) as ds:
        wind = np.hypot(ds.u10.values.astype(np.float32), ds.v10.values.astype(np.float32))
        wt = np.hypot(ds.u10_truth.values, ds.v10_truth.values).astype(np.float32)
        rh = ds.r850.values.astype(np.float32) if "r850" in ds else None
        rht = ds.r850_truth.values.astype(np.float32) if "r850_truth" in ds else None
    return wind, wt, rh, rht


def run_baselines(d, case, clim, oro, v2p):
    hz = d["hazard"]
    tk = Tracker2(hz, clim, oro, d["lat"], d["lon"], **v2p)
    out = {"v2": case_metrics(d, [tk.run(d["ens"][m], d["times"])[0] for m in range(20)],
                              tk.run(d["truth"], d["times"])[0])}
    if hz == "tropical_cyclone":
        wind, wt, rh, rht = aux_fields(case)
        te = [te_cyclones(d["ens"][m], d["lat"], d["lon"], wind=wind[m], rh850=None if rh is None else rh[m])
              for m in range(20)]
        out["te_style"] = case_metrics(d, te, te_cyclones(d["truth"], d["lat"], d["lon"], wind=wt, rh850=rht))
        del wind, rh
    else:
        fn = lambda s: te_blobs(s, d["times"], clim, oro, d["lat"], d["lon"], hz)
        out["te_style"] = case_metrics(d, [fn(d["ens"][m]) for m in range(20)], fn(d["truth"]))
    return out


def load_graph(case):
    z = np.load(GDIR / f"{case}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def choose_decode(models, graphs):
    grid = [(tn, ml) for tn in (0.3, 0.5, 0.7) for ml in (2, 4, 8)]
    dec = {v: {} for v in models}
    for c in VAL:                                # one case in memory at a time
        d, g = load_case_imd(c), graphs[c]
        for v, (model, st, _) in models.items():
            pn, pe = infer(model, st, g)
            best = None
            for tn, ml in grid:
                r = case_metrics(d, [decode(g, pn, pe, m, d["hazard"], tau_n=tn, min_len=ml) for m in range(20)],
                                 decode(g, pn, pe, -1, d["hazard"], tau_n=tn, min_len=ml))
                key = (r["csi"], -r["spurious_tracks"])
                if best is None or key > best[0]:
                    best = (key, {"tau_n": tn, "tau_e": 0.5, "min_len": ml})
            dec[v][d["hazard"]] = best[1]
        del d
    return dec


# ----------------------------------------------------------------------------- calibration
def node_data(model, st, g):
    pn, _ = infer(model, st, g)
    ens = g["member"] >= 0
    return pn[ens], g["y_truth"][ens], 6 * g["t"][ens].astype(int), cal.ens_frequency(g)[ens]


def calibrate(models, graphs, key="full", fname="calibration.json"):
    model, st, _ = models[key]
    val = {c: node_data(model, st, graphs[c]) for c in VAL}
    cals, choice = {}, {}
    for name, a, b in cal.BANDS:
        pb = {c: v[0][(v[2] >= a) & (v[2] <= b)] for c, v in val.items()}
        yb = {c: v[1][(v[2] >= a) & (v[2] <= b)] for c, v in val.items()}
        cals[name], scores, base = cal.fit_band(pb, yb)
        choice[name] = {"chosen": cals[name].kind, "loo_brier": scores, "val_base_rate": base,
                        **({"T": cals[name].T} if cals[name].kind == "temperature" else {})}
    cal.save(MDIR / fname, cals, {"fitted_on": VAL, "choice": choice, "model": key})
    test = {c: node_data(model, st, graphs[c]) for c in TEST}
    P = np.concatenate([v[0] for v in test.values()])
    Y = np.concatenate([v[1] for v in test.values()])
    Ld = np.concatenate([v[2] for v in test.values()])
    EF = np.concatenate([v[3] for v in test.values()])
    Pc = cal.apply(cals, P, Ld)
    out = {}
    for name, a, b in cal.BANDS:
        k = (Ld >= a) & (Ld <= b)
        if not k.any():
            continue
        base = choice[name]["val_base_rate"]
        bs_raw, bs_cal = cal.brier(P[k], Y[k]), cal.brier(Pc[k], Y[k])
        bs_clim, bs_ens = cal.brier(np.full(k.sum(), base), Y[k]), cal.brier(EF[k], Y[k])
        rr, rc = cal.reliability(P[k], Y[k]), cal.reliability(Pc[k], Y[k])
        out[name] = {"n": int(k.sum()), "test_base_rate": float(Y[k].mean()),
                     "brier_raw": bs_raw, "brier_cal": bs_cal, "brier_clim": bs_clim, "brier_ens_freq": bs_ens,
                     "bss_cal_vs_clim": 1 - bs_cal / bs_clim, "bss_cal_vs_ens_freq": 1 - bs_cal / bs_ens,
                     "bss_raw_vs_clim": 1 - bs_raw / bs_clim,
                     "slope_raw": cal.slope(rr), "slope_cal": cal.slope(rc), "rel_raw": rr, "rel_cal": rc}
    return cals, choice, out


# ----------------------------------------------------------------------------- early warning
def ew_prob(g, pn, cals, T):
    """P_t = mean over members of the max calibrated node probability over leads [t, t+72 h]."""
    pc = cal.apply(cals, pn, 6 * g["t"].astype(int))
    q = np.zeros((20, T))
    for i in np.nonzero(g["member"] >= 0)[0]:
        q[g["member"][i], g["t"][i]] = max(q[g["member"][i], g["t"][i]], pc[i])
    return np.array([q[:, t:t + EW_WINDOW + 1].max(1).mean() for t in range(T)])


def efi_prob(d, clim, reg):
    """EFI-only: fraction of the domain-max EFI >= 0.8 over leads [t, t+72 h] (0/1 per lead)."""
    var = "msl" if d["hazard"] == "tropical_cyclone" else "t2m"
    sign = -1.0 if d["hazard"] in ("tropical_cyclone", "cold_wave") else 1.0
    dom = reg > 0 if d["hazard"] != "tropical_cyclone" else np.ones_like(reg, bool)
    e = np.array([np.nanmax(np.where(dom, sign * efi_gaussian(d["ens"][:, t], *clim.get(var, d["times"][t])), -1))
                  for t in range(len(d["times"]))])
    return np.array([float(e[t:t + EW_WINDOW + 1].max() >= 0.8) for t in range(len(e))]), e


def v2_prob(d, clim, oro, v2p):
    """Tracker v2: fraction of members with a tracked object within [t, t+72 h]."""
    tk = Tracker2(d["hazard"], clim, oro, d["lat"], d["lon"], **v2p)
    T = len(d["times"])
    q = np.zeros((20, T))
    for m in range(20):
        for tr in tk.run(d["ens"][m], d["times"])[0]:
            for t, _ in tr:
                q[m, t] = 1
    return np.array([q[:, t:t + EW_WINDOW + 1].max(1).mean() for t in range(T)])


def first_flag(p, thr):
    k = np.nonzero(p >= thr)[0]
    return int(k[0]) if len(k) else None


def early_warning(models, graphs, cals, clim, oro, v2p_all):
    from synth.grids import LAT12, LON12
    model, st, _ = models["full"]
    reg = regions(oro, LAT12, LON12)
    series = {}
    for c in VAL + TEST:
        d, g = load_case_imd(c), graphs[c]
        pn, _ = infer(model, st, g)
        T = len(d["times"])
        onset = next((t for t in range(T) if d["tmask"][t].any()), None)
        series[c] = {"gnn": ew_prob(g, pn, cals, T), "v2": v2_prob(d, clim, oro, v2p_all[d["hazard"]]["params"]),
                     "efi": efi_prob(d, clim, reg)[0], "onset": onset}
        del d
    # threshold on VAL: the largest tau in {0.3..0.8} that flags every VAL case no later than onset
    taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ok = [tau for tau in taus if all(first_flag(series[c]["gnn"], tau) is not None and
                                     first_flag(series[c]["gnn"], tau) <= series[c]["onset"] for c in VAL)]
    tau = max(ok) if ok else 0.5
    res = {}
    for c in VAL + TEST:
        s = series[c]
        fg, fv, fe = first_flag(s["gnn"], tau), first_flag(s["v2"], 0.5), first_flag(s["efi"], 0.5)
        warn = lambda f: None if f is None or s["onset"] is None else 6 * (s["onset"] - f)
        res[c] = {"split": "val" if c in VAL else "test", "onset_lead_h": 6 * s["onset"] if s["onset"] is not None else None,
                  "warning_h": {"gnn_ew": warn(fg), "v2": warn(fv), "efi_only": warn(fe)},
                  "hours_gained_vs_v2": (warn(fg) - warn(fv)) if None not in (warn(fg), warn(fv)) else None,
                  "hours_gained_vs_efi": (warn(fg) - warn(fe)) if None not in (warn(fg), warn(fe)) else None,
                  "p_gnn": s["gnn"].round(3).tolist()}
    return tau, res


def simplify(variant="none_small"):
    """GNN-only evaluation of a simplified linker (no baselines re-run), appended to object_gnn_v3.json."""
    import time as _t
    out = json.loads((REP / "object_gnn_v3.json").read_text())
    model, st, cfg = load_model(variant, MDIR)
    full = load_model("full", MDIR)
    graphs = {c: load_graph(c) for c in VAL + TEST}
    dec = choose_decode({variant: (model, st, cfg)}, graphs)[variant]
    res, t_inf = {}, {"full": 0.0, variant: 0.0}
    for c in VAL + TEST:
        d, g = load_case_imd(c), graphs[c]
        for name, (m, s_, _) in (("full", full), (variant, (model, st, cfg))):
            t0 = _t.perf_counter()
            pn, pe = infer(m, s_, g)
            t_inf[name] += _t.perf_counter() - t0
            if name == variant:
                res[c] = case_metrics(d, [decode(g, pn, pe, k, d["hazard"], **dec[d["hazard"]]) for k in range(20)],
                                      decode(g, pn, pe, -1, d["hazard"], **dec[d["hazard"]]))
        del d
    decs = json.loads((MDIR / "decode.json").read_text())
    decs[variant] = dec
    (MDIR / "decode.json").write_text(json.dumps(decs, indent=1))
    _, choice_s, cal_s = calibrate({variant: (model, st, cfg)}, graphs, key=variant, fname=f"calibration_{variant}.json")
    out["simplified_calibration_choice"], out["simplified_calibration_test"] = choice_s, cal_s
    out["simplified"] = {"variant": variant, "n_params": int(sum(p.numel() for p in model.parameters())),
                         "n_params_full": int(sum(p.numel() for p in full[0].parameters())),
                         "inference_seconds": t_inf, "decode": dec, "cases": res,
                         "config": cfg}
    (REP / "object_gnn_v3.json").write_text(dumps(out, indent=1))
    write_report(out)


def main():
    import sys
    if "--simplify" in sys.argv:
        return simplify()
    clim, oro = Climatology(), orog12()
    v2p = json.loads((REP / "tracker2_params.json").read_text())["chosen"]
    models = {v: load_model(v, MDIR) for v in VARIANTS if (MDIR / v / "model.pt").exists()}
    cases = TRAIN + VAL + TEST
    graphs = {c: load_graph(c) for c in cases}
    dec = choose_decode(models, graphs)
    (MDIR / "decode.json").write_text(json.dumps(dec, indent=1))
    print("decode", dec, flush=True)
    res = {}
    for c in cases:
        d, g = load_case_imd(c), graphs[c]
        hz = d["hazard"]
        r = run_baselines(d, c, clim, oro, v2p[hz]["params"])
        for v, (model, st, _) in models.items():
            pn, pe = infer(model, st, g)
            r[f"gnn_{v}"] = case_metrics(d, [decode(g, pn, pe, m, hz, **dec[v][hz]) for m in range(20)],
                                         decode(g, pn, pe, -1, hz, **dec[v][hz]))
        res[c] = {"hazard": hz, "split": "train" if c in TRAIN else "val" if c in VAL else "test", **r}
        print(c, {k: (round(x["iou"], 2), round(x["csi"], 2), x["spurious_tracks"]) for k, x in r.items()
                  if isinstance(x, dict)}, flush=True)
        del d
        import gc
        gc.collect()
    cals, choice, calres = calibrate(models, graphs)
    tau, ew = early_warning(models, graphs, cals, clim, oro, v2p)
    out = {"cases": res, "decode": dec, "calibration_choice": choice, "calibration_test": calres,
           "early_warning_tau": tau, "early_warning": ew,
           "train_configs": {v: {**models[v][2], "n_params": int(sum(p.numel() for p in models[v][0].parameters()))}
                             for v in models},
           "imd_labels": json.loads((REP / "imd_labels.json").read_text())}
    (REP / "object_gnn_v3.json").write_text(dumps(out, indent=1))
    write_report(out)


def write_report(out):
    old = json.loads((REP / "gnn_results.json").read_text())
    R = out["cases"]
    f2 = lambda x: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
    methods = ["v2", "te_style"] + [f"gnn_{v}" for v in VARIANTS if f"gnn_{v}" in R[TEST[0]]]
    mean = lambda split, m, k: float(np.nanmean([R[c][m][k] if R[c][m][k] is not None else np.nan
                                                  for c in R if R[c]["split"] == split]))
    L = ["# Object GNN v3 (BRIEF4 Phase 3)", "",
         "All cases are **SYNTHETIC**. Heat/cold cases are now labelled and verified with IMD-consistent **daily** "
         "labels (scripts/imd_labels.py; reports/imd_labels.json), so the numbers below are not directly comparable "
         "with reports/GNN_RESULTS.md for heat and cold (cyclones use the same labels as before). TEST is only "
         "reported; decoding thresholds, calibrators and the early-warning threshold were chosen on VAL.", "",
         "## Mean over TEST cases", "", "| method | IoU | CSI | POD | FAR | spurious / case | cases with < 10 spurious |",
         "|---|---|---|---|---|---|---|"]
    for m in methods:
        n_ok = sum(R[c][m]["spurious_tracks"] < 10 for c in R)
        L.append(f"| {m} | {f2(mean('test', m, 'iou'))} | {f2(mean('test', m, 'csi'))} | {f2(mean('test', m, 'pod'))} | "
                 f"{f2(mean('test', m, 'far'))} | {mean('test', m, 'spurious_tracks'):.1f} | {n_ok}/{len(R)} (all splits) |")
    L += ["", "BRIEF2 object GNN (old edges, old labels) on TEST for reference: " + ", ".join(
        f"{k} {np.mean([old['cases'][c]['gnn_full'][k] for c in TEST]):.2f}" for k in ("iou", "csi", "far")), "",
          "## Do the links help now? (edge ablations, TEST and VAL means)", "",
          "| variant | TEST CSI | TEST FAR | TEST spurious | VAL CSI | VAL spurious | val AP node+edge | params |", "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        m = f"gnn_{v}"
        if m not in R[TEST[0]]:
            continue
        cfg = out["train_configs"][v]
        L.append(f"| {v} | {f2(mean('test', m, 'csi'))} | {f2(mean('test', m, 'far'))} | {mean('test', m, 'spurious_tracks'):.1f} | "
                 f"{f2(mean('val', m, 'csi'))} | {mean('val', m, 'spurious_tracks'):.1f} | {cfg['best_val_ap_sum']:.3f} | "
                 f"{cfg.get('n_params', 'n/a')} |")
    L += ["", "## Per case (TEST)", "", "| case | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for c in TEST:
        L.append(f"| {c} | " + " | ".join(f"CSI {R[c][m]['csi']:.2f} / IoU {R[c][m]['iou']:.2f} / spur {R[c][m]['spurious_tracks']}"
                                           for m in methods) + " |")
    L += ["", "## Calibration (GNN full, node event probability vs the truth)", "",
          "Calibrator per lead band chosen on VAL by leave-one-case-out Brier: " + ", ".join(
              f"{b}: {v['chosen']}" + (f" (T = {v['T']:.2f})" if 'T' in v else "") for b, v in out["calibration_choice"].items()), "",
          "| lead band | n (TEST nodes) | reliability slope raw | slope calibrated | Brier raw | Brier cal | BSS cal vs climatology | BSS cal vs raw ensemble frequency | target slope 0.8-1.2 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b, v in out["calibration_test"].items():
        okb = v["slope_cal"] is not None and 0.8 <= v["slope_cal"] <= 1.2
        L.append(f"| {b} | {v['n']} | {f2(v['slope_raw'])} | {f2(v['slope_cal'])} | {v['brier_raw']:.4f} | {v['brier_cal']:.4f} | "
                 f"{v['bss_cal_vs_clim']:.2f} | {v['bss_cal_vs_ens_freq']:.2f} | {'yes' if okb else 'NO'} |")
    L += ["", f"## Early-warning mode (threshold {out['early_warning_tau']} chosen on VAL)", "",
          "Alert at lead t when the calibrated probability that an event exists within [t, t+72 h] exceeds the "
          "threshold. Warning = hours between the first alert and the truth onset (valid time); v2 and EFI-only use "
          "the same 72 h look-ahead (v2: >= 50 % of members track an object; EFI-only: domain-max EFI >= 0.8).", "",
          "| case | split | onset lead (h) | warning GNN-EW (h) | v2 (h) | EFI-only (h) | gained vs v2 | gained vs EFI |",
          "|---|---|---|---|---|---|---|---|"]
    for c, v in out["early_warning"].items():
        w = v["warning_h"]
        L.append(f"| {c} | {v['split']} | {v['onset_lead_h']} | {w['gnn_ew']} | {w['v2']} | {w['efi_only']} | "
                 f"{v['hours_gained_vs_v2']} | {v['hours_gained_vs_efi']} |")
    L += ["", "Every synthetic case contains an event, so false-alarm behaviour of the early-warning mode cannot be "
          "measured here; the onset in several cases is at lead 0-6 h, where no warning time is possible.", "",
          "## Heat/cold labels and heat_04", "", "| case | old label cells | IMD label cells | truth peak (C) | peak departure (C) | plains cells >= 40 C |",
          "|---|---|---|---|---|---|"]
    for c, v in out["imd_labels"].items():
        L.append(f"| {c} | {v['old_label_cells']} | {v['imd_label_cells']} | {v['truth_peak_abs_C']:.1f} | "
                 f"{v['truth_peak_departure_C']:.1f} | {v.get('plains_cells_ge_40C', '-')} |")
    L += ["", "heat_04 meets the IMD heat-wave criteria only marginally (peak Tmax 40.5 C, 10 plains cells >= 40 C), which "
          "is why tracker v2 (IMD 40 C rule) never fires on it; the case is kept as generated rather than edited after "
          "seeing TEST results.", ""]
    w = []
    for b, v in out["calibration_test"].items():
        if v["slope_cal"] is None or not 0.8 <= v["slope_cal"] <= 1.2:
            w.append(f"- Calibration slope target missed in {b}: {f2(v['slope_cal'])}.")
        if v["bss_cal_vs_ens_freq"] < 0:
            w.append(f"- {b}: calibrated GNN is worse than the raw ensemble frequency (BSS {v['bss_cal_vs_ens_freq']:.2f}).")
    bad = [c for c in R if R[c]["gnn_full"]["spurious_tracks"] >= 10]
    if bad:
        w.append(f"- Spurious-track target (< 10 per case in 13/13) missed in: {', '.join(bad)}.")
    g0 = [c for c, v in out["early_warning"].items() if v["split"] == "test" and not (v["hours_gained_vs_v2"] or 0) > 0]
    if g0:
        w.append(f"- No lead-time gain over v2 on TEST cases {', '.join(g0)}.")
    full_csi, none_csi = mean("test", "gnn_full", "csi"), mean("test", "gnn_none", "csi") if "gnn_none" in R[TEST[0]] else None
    if none_csi is not None and full_csi - none_csi < 0.02:
        w.append(f"- Links still add little on TEST (full CSI {full_csi:.2f} vs node-only {none_csi:.2f}).")
    w.append("- The TempestExtremes warm-core criterion needs 300-500 hPa temperature, which the synthetic cases do not "
             "have; a moist-core proxy (RH850 >= 80 %) is used for cyclones.")
    sm = out.get("simplified")
    if not sm:
        L += ["## What did not work", ""]
    if sm:
        cs = [c for c in TEST]
        m_ = lambda k: float(np.mean([sm["cases"][c][k] for c in cs]))
        L += ["", "## Simplification (links add little)", "",
              f"Because the edge ablations show no benefit from message passing, a node-only model with hidden 32 "
              f"(`{sm['variant']}`, {sm['n_params']} parameters vs {sm['n_params_full']} for full) was trained on the same "
              f"graphs: TEST CSI {m_('csi'):.2f}, FAR {m_('far'):.2f}, spurious {m_('spurious_tracks'):.1f} per case; "
              f"inference over the VAL+TEST graphs {sm['inference_seconds'][sm['variant']]:.2f} s vs "
              f"{sm['inference_seconds']['full']:.2f} s for full. Its own lead-band calibration (fitted on VAL, "
              f"calibration_{sm['variant']}.json) gives TEST reliability slopes "
              + ", ".join(f"{b} {v['slope_cal']:.2f}" for b, v in out.get('simplified_calibration_test', {}).items()) +
              f". **The operational pipeline therefore uses the simplified model** "
              "(pipeline/run_operational.py, scripts/export_extras.py); whether cross-member links help on real NEPS-G "
              "ensembles is untested.", "", "## What did not work", ""]
    L += w or ["- Nothing below the targets."]
    (REP / "OBJECT_GNN_V3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/OBJECT_GNN_V3.md")


if __name__ == "__main__":
    main()
