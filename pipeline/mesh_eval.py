"""Evaluate the icosahedral mesh GNN (BRIEF4 Phase 2) -> reports/MESH_GNN_RESULTS.md, mesh_results.json

    python -m pipeline.mesh_eval [model ...]          (default: every trained models/mesh/*)

Two-level pipeline: mesh GNN per-cell probability -> hysteresis connected components (high/low,
min area) -> candidate objects -> the existing object GNN (models/tracker/full) links them across
leads and members -> tracks + 4-D boxes. Scored with pipeline.gnn_eval.case_metrics, i.e. exactly
the IoU / CSI / POD / FAR / spurious-track definitions of reports/GNN_RESULTS.md.
Also 'segmentation only': IoU of the thresholded probability of the truth run vs the truth mask.

Hysteresis thresholds are chosen on VAL (segmentation IoU of the truth run); TEST is only reported.
Probabilities are cached per case as uint8 in data/graphs/mesh_probs/ (gitignored).
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from .anomaly import Climatology
from .evaluate2 import load_case
from .gnn_eval import case_metrics, decode, infer, load_model
from .graphs import CAND, build_graph
from .jsonutil import dumps
from .mesh_gnn import HAZ, N_IN, Graph, GridCNN, MeshGNN, Static, features
from .splits import TEST, VAL
from .tracker2 import cell_area_km2, detect_hysteresis

ROOT = Path(__file__).resolve().parents[1]
MDIR = ROOT / "models/mesh"
PCACHE = ROOT / "data/graphs/mesh_probs"
REP = ROOT / "reports"
GRID = [(h, l) for h in (0.5, 0.6, 0.7, 0.8) for l in (0.3, 0.4, 0.5) if l < h]
PEAK = [0.0]


def rss():
    PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss / 1e6)


def load_mesh_model(name):
    from synth.grids import LAT12, LON12
    cfg = json.loads((MDIR / name / "config.json").read_text())
    m = MeshGNN(Graph(cfg["level"], LAT12, LON12), hidden=cfg["hidden"]) if cfg["model"] == "mesh" else GridCNN()
    m.load_state_dict(torch.load(MDIR / name / "model.pt", map_location="cpu"))
    m.eval()
    return m, cfg


@torch.no_grad()
def case_probs(name, model, case, clim, st):
    """(21, 41, 333, 333) uint8 probability x 255 of the case's hazard channel: members 0..19,
    then the truth run (index 20)."""
    PCACHE.mkdir(parents=True, exist_ok=True)
    p = PCACHE / f"{name}_{case}.npz"
    if p.exists():
        return np.load(p)["p"]
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_mesh_gnn import case_efi, member_fields
    lab = json.loads((ROOT / "data/synthetic" / case / "labels.json").read_text())
    times = pd.to_datetime(lab["valid_times"])
    hi = HAZ.index(lab["hazard"])
    efi = case_efi(case, clim)
    out = np.zeros((21, len(times), 333, 333), np.uint8)
    for k, m in enumerate(list(range(20)) + [-1]):
        f, _ = member_fields(case, m)
        for t0 in range(0, len(times), 4):
            ts = range(t0, min(t0 + 4, len(times)))
            X = torch.tensor(np.stack([features(f, t, times[t], 6 * t, efi[t], clim, st).reshape(N_IN, -1).T for t in ts]))
            P = torch.sigmoid(model(X))[..., hi].numpy()
            out[k, list(ts)] = np.round(P * 255).reshape(len(ts), 333, 333).astype(np.uint8)
        del f
        gc.collect()
        rss()
    np.savez_compressed(p, p=out)
    return out


def seg_mask(prob, hz, high, low, area):
    """Hysteresis connected components of one probability map (union of objects)."""
    from synth.grids import LAT12, LON12
    objs = detect_hysteresis(prob, LAT12, LON12, high, low, CAND[hz]["min_area_km2"], area)
    m = np.zeros(prob.shape, bool)
    for o in objs:
        if o["core"]:                      # a component with a cell >= high and the minimum area
            m |= o["mask"]
    return m


def seg_iou(P, tmask, hz, high, low, area):
    """Mean IoU over leads where truth or prediction is present (truth run)."""
    vals = []
    for t in range(len(tmask)):
        m = seg_mask(P[t], hz, high, low, area)
        if m.any() or tmask[t].any():
            vals.append((m & tmask[t]).sum() / max((m | tmask[t]).sum(), 1))
    return float(np.mean(vals)) if vals else 0.0


def pipeline_metrics(d, P, clim, high, low, obj_model, dec):
    """mesh probabilities -> objects -> object GNN -> case_metrics."""
    from synth.grids import LAT12, LON12
    hz = d["hazard"]
    probs = {m: P[m].astype(np.float32) / 255.0 for m in range(20)}
    probs[-1] = P[20].astype(np.float32) / 255.0
    g = build_graph(hz, d["ens"], d["times"], LAT12, LON12, clim, truth=d["truth"], emask=d["emask"],
                    tmask=d["tmask"], det=(probs, high, low, CAND[hz]["min_area_km2"]))
    model, st, _ = obj_model
    pn, pe = infer(model, st, g)
    members = [decode(g, pn, pe, m, hz, **dec) for m in range(20)]
    truth = decode(g, pn, pe, -1, hz, **dec)
    r = case_metrics(d, members, truth)
    r["nodes"] = int(len(g["X"]))
    return r


def evaluate_model(name, clim, st, obj_model, dec_all):
    model, cfg = load_mesh_model(name)
    from synth.grids import LAT12, LON12
    area = cell_area_km2(LAT12, LON12)
    t0 = time.time()
    probs = {}
    for c in VAL + TEST:
        probs[c] = case_probs(name, model, c, clim, st)
        print(f"{name} {c}: probs {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    # thresholds on VAL (per hazard), segmentation IoU of the truth run
    thr = {}
    for c in VAL:
        d = load_case(c)
        hz = d["hazard"]
        scores = {(h, l): seg_iou(probs[c][20].astype(np.float32) / 255, d["tmask"], hz, h, l, area) for h, l in GRID}
        best = max(scores, key=scores.get)
        thr[hz] = {"high": best[0], "low": best[1], "val_seg_iou": scores[best], "val_case": c}
        del d
    res = {}
    for c in VAL + TEST:
        d = load_case(c)
        hz = d["hazard"]
        h, l = thr[hz]["high"], thr[hz]["low"]
        P = probs[c]
        seg = seg_iou(P[20].astype(np.float32) / 255, d["tmask"], hz, h, l, area)
        pm = pipeline_metrics(d, P, clim, h, l, obj_model, dec_all[hz])
        res[c] = {"split": "val" if c in VAL else "test", "hazard": hz, "seg_iou": seg, "pipeline": pm}
        print(f"{name} {c}: seg IoU {seg:.2f}, pipeline IoU {pm['iou']:.2f} CSI {pm['csi']:.2f} FAR {pm['far']:.2f} "
              f"spurious {pm['spurious_tracks']}", flush=True)
        del d
        gc.collect()
        rss()
    return {"config": cfg, "thresholds": thr, "cases": res, "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def real_val(name, clim, st, obj_temporal, dec_temporal, thr):
    """Mesh pipeline on the REAL-VAL cyclones (ERA5 single run) -> track error vs IBTrACS."""
    import xarray as xr
    from .mesh_gnn import efi_fields
    from .real_eval import ibtracs, match_track
    from .track import haversine_km
    from synth.grids import LAT12, LON12
    model, _ = load_mesh_model(name)
    splits = json.loads((ROOT / "data/real/SPLITS.json").read_text())
    out = {}
    for eid in splits["REAL-VAL"]:
        ev = splits["events"][eid]
        if ev["hazard"] != "tropical_cyclone":
            continue
        f = ROOT / ("data/real/era5/amphan_era5_g12.nc" if eid == "amphan_2020" else f"data/real/events/{eid}_sfc_g12.nc")
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        times = pd.to_datetime(ds.time.values)
        fld = {v: ds[v].values.astype(np.float32) for v in ("t2m", "u10", "v10", "msl", "tp") if v in ds}
        ds.close()
        for v in ("u10", "v10", "tp", "t2m"):
            fld.setdefault(v, np.zeros_like(fld["msl"]))
        efi = efi_fields(fld["msl"][None], fld["t2m"][None], times, clim)
        P = np.zeros((len(times), 333, 333), np.float32)
        for t0 in range(0, len(times), 4):
            ts = list(range(t0, min(t0 + 4, len(times))))
            X = torch.tensor(np.stack([features(fld, t, times[t], 0, efi[t], clim, st).reshape(N_IN, -1).T for t in ts]))
            P[ts] = torch.sigmoid(model(X))[..., 0].numpy().reshape(len(ts), 333, 333)
        try:
            g = build_graph("tropical_cyclone", fld["msl"][None].astype(np.float64), times, LAT12, LON12, clim,
                            det=({0: P}, thr["high"], thr["low"], CAND["tropical_cyclone"]["min_area_km2"]))
        except ValueError:
            out[eid] = {"status": "no objects"}
            continue
        m, s_, _ = obj_temporal
        pn, pe = infer(m, s_, g)
        tr = decode(g, pn, pe, 0, "tropical_cyclone", **dec_temporal)
        ib = ibtracs(ROOT / "data/real/ibtracs/events" / f"{eid}.csv")
        x = match_track(tr, times, ib)
        ibt = ib.set_index("time")
        errs = [haversine_km(a, b, ibt.loc[t].LAT, ibt.loc[t].LON) for t, (a, b) in (x or {}).items() if t in ibt.index]
        n_ib = int(((ib.time >= times[0]) & (ib.time <= times[-1])).sum())
        out[eid] = {"matched_fixes": len(errs), "ibtracs_fixes": n_ib,
                    "err_mean_km": float(np.mean(errs)) if errs else None, "n_tracks": len(tr)}
        print(f"REAL-VAL {eid}: {out[eid]}", flush=True)
        gc.collect()
    return out


def main():
    from synth.grids import LAT12, LON12
    names = sys.argv[1:] or sorted(p.name for p in MDIR.iterdir() if (p / "model.pt").exists())
    clim, st = Climatology(), Static(LAT12, LON12)
    gr = json.loads((REP / "gnn_results.json").read_text())
    obj = load_model("full")
    out_p = REP / "mesh_results.json"
    allres = json.loads(out_p.read_text()) if out_p.exists() else {}
    obj_t = load_model("temporal")
    for n in names:
        allres[n] = evaluate_model(n, clim, st, obj, gr["decode"]["full"])
        if n == "mesh_l6":
            allres[n]["real_val"] = real_val(n, clim, st, obj_t, gr["decode"]["temporal"]["tropical_cyclone"],
                                             allres[n]["thresholds"]["tropical_cyclone"])
        allres["_peak_rss_mb"] = PEAK[0]
        out_p.write_text(dumps(allres, indent=1))
    write_report(allres, gr)


def write_report(allres, gr):
    models = [k for k in allres if not k.startswith("_")]
    main_m = "mesh_l6" if "mesh_l6" in allres else models[0]
    R = allres[main_m]
    L = ["# Icosahedral mesh GNN (BRIEF4 Phase 2)", "",
         "All cases are **SYNTHETIC** (exact labels). Train: 6 TRAIN cases; model epoch and hysteresis thresholds "
         "chosen on VAL; TEST (cyc_04, heat_04, cold_04, amphan_replay) only reported. Built by "
         "`python -m pipeline.mesh_eval`.", "",
         "Pipeline: mesh GNN per-cell probability (`pipeline/mesh_gnn.py`, icosahedral multi-mesh levels 0..L over "
         "the India box + 12 deg halo, `pipeline/icomesh.py`) -> hysteresis connected components -> candidate objects "
         "-> the object GNN (models/tracker/full) links them across leads and members. Metrics use the same "
         "`case_metrics` as reports/GNN_RESULTS.md.", "",
         f"Main model `{main_m}`: {R['config']['params']} parameters, trained {R['config']['train_seconds']:.0f} s on "
         f"{R['config']['compute']}, best epoch {R['config']['best_epoch']} (VAL IoU {R['config']['best_val_iou']:.3f}).",
         "", "Hysteresis thresholds (chosen on VAL): " + ", ".join(
             f"{hz}: high {v['high']}, low {v['low']}" for hz, v in R["thresholds"].items()), "",
         "## TEST: mesh pipeline vs the Phase-2 object GNN (BRIEF2) and tracker v2", "",
         "| case | hazard | method | IoU | CSI | POD | FAR | spurious tracks | centroid err (km) |",
         "|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda x: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
    f0 = lambda x: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.0f}"
    for c in TEST:
        r = R["cases"][c]
        p = r["pipeline"]
        L.append(f"| {c} | {r['hazard']} | **mesh GNN + object GNN** | **{f2(p['iou'])}** | {f2(p['csi'])} | {f2(p['pod'])} | "
                 f"{f2(p['far'])} | {p['spurious_tracks']} | {f0(p['centroid_err_km'])} |")
        L.append(f"| {c} | | mesh GNN segmentation only (truth run) | {f2(r['seg_iou'])} | | | | | |")
        for mth in ("gnn_full", "v2"):
            o = gr["cases"][c][mth]
            L.append(f"| {c} | | {mth} (BRIEF2) | {f2(o['iou'])} | {f2(o['csi'])} | {f2(o['pod'])} | {f2(o['far'])} | "
                     f"{o['spurious_tracks']} | {f0(o['centroid_err_km'])} |")
    mean = lambda key, src: float(np.mean([src(c)[key] for c in TEST]))
    mp = lambda c: R["cases"][c]["pipeline"]
    og = lambda c: gr["cases"][c]["gnn_full"]
    L += ["", "## Targets on TEST", "", "| target | mesh pipeline | object GNN (BRIEF2) | met? |", "|---|---|---|---|"]
    per_hz = {R["cases"][c]["hazard"]: [] for c in TEST}
    for c in TEST:
        per_hz[R["cases"][c]["hazard"]].append(mp(c)["iou"])
    iou_hz = {hz: float(np.mean(v)) for hz, v in per_hz.items()}
    L.append("| IoU >= 0.40 on every hazard | " + ", ".join(f"{hz} {v:.2f}" for hz, v in iou_hz.items()) +
             f" | mean {mean('iou', og):.2f} | {'yes' if min(iou_hz.values()) >= 0.40 else 'NO'} |")
    L.append(f"| CSI >= 0.65 | {mean('csi', mp):.2f} | {mean('csi', og):.2f} | {'yes' if mean('csi', mp) >= 0.65 else 'NO'} |")
    L.append(f"| FAR <= 0.20 | {mean('far', mp):.2f} | {mean('far', og):.2f} | {'yes' if mean('far', mp) <= 0.20 else 'NO'} |")
    L += ["", "## Ablations (TEST means)", "",
          "| model | params | mesh level | seg IoU (truth run) | pipeline IoU | CSI | FAR | spurious / case | train s |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in models:
        A = allres[n]
        cm = lambda k: float(np.mean([A["cases"][c]["pipeline"][k] for c in TEST]))
        L.append(f"| {n} | {A['config']['params']} | {A['config']['level'] or '- (lat-lon grid CNN)'} | "
                 f"{np.mean([A['cases'][c]['seg_iou'] for c in TEST]):.2f} | {cm('iou'):.2f} | {cm('csi'):.2f} | "
                 f"{cm('far'):.2f} | {cm('spurious_tracks'):.1f} | {A['config']['train_seconds']:.0f} |")
    L += ["", "The grid CNN has about the same parameter count and sees the same inputs on the lat-lon grid; the mesh "
          "levels change the processor resolution (level 5 ~ 240 km, 6 ~ 120 km, 7 ~ 60 km edges). Over the India box "
          "(0-40 N) a lat-lon grid cell shrinks by only cos(40 deg) = 0.77, so the 'no polar distortion' advantage of "
          "the sphere mesh is small here; the comparison tests whether the mesh's multi-scale long edges help.", "",
          ]
    if R.get("real_val"):
        rr = json.loads((REP / "real_results.json").read_text())["era5_real_val"] if (REP / "real_results.json").exists() else {}
        L += ["## REAL-VAL cyclones (ERA5 reanalysis vs IBTrACS)", "",
              "The mesh GNN was trained on SYNTHETIC data only; here it segments real ERA5 fields (single run, so the "
              "object GNN is the temporal-only variant), compared with the BRIEF4 Phase 1 numbers of the object pipeline.", "",
              "| event | mesh pipeline matched / IBTrACS | mesh mean err (km) | object-GNN temporal (Phase 1) err (km) | v2 err (km) |",
              "|---|---|---|---|---|"]
        for eid, v in R["real_val"].items():
            o = rr.get(eid, {})
            L.append(f"| {eid} | {v.get('matched_fixes', '-')}/{v.get('ibtracs_fixes', '-')} | {f0(v.get('err_mean_km'))} | "
                     f"{f0(o.get('gnn_temporal', {}).get('err_mean_km'))} | {f0(o.get('v2', {}).get('err_mean_km'))} |")
        L.append("")
    L += ["## What did not work", ""]
    w = []
    for hz, v in iou_hz.items():
        if v < 0.40:
            w.append(f"- IoU target missed for {hz}: {v:.2f} < 0.40.")
    if mean("csi", mp) < 0.65:
        w.append(f"- CSI {mean('csi', mp):.2f} < 0.65.")
    if mean("far", mp) > 0.20:
        w.append(f"- FAR {mean('far', mp):.2f} > 0.20.")
    w.append("- The 850 hPa moisture-flux-convergence input is a proxy (RH850 x 10 m wind); heat/cold cases have no "
             "RH850 at all, so that channel is 0 there.")
    w.append(f"- Trained on CPU with a small budget ({R['config']['epochs']} epochs, {R['config']['members_per_case']} "
             "members per case per epoch, every 3rd lead); a GPU run with all members is in notebooks/train_colab.ipynb.")
    L += w
    (REP / "MESH_GNN_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/MESH_GNN_RESULTS.md")


if __name__ == "__main__":
    main()
