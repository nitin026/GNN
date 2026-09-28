"""End-to-end operational pipeline (BRIEF4 Phase 5).

    python -m pipeline.run_operational FILE.nc [--hazards tropical_cyclone,heat_dome,cold_wave]
                                       [--samples 1] [--lead-stride 1] [--members all] [--out backend/products]
    python -m pipeline.run_operational --watch inbox/          # process every new file dropped there

Input: a 12 km ensemble NetCDF in the fcst_12km.nc / NEPS-G layout (dims number, step or time,
latitude, longitude; msl [Pa] required, u10, v10, t2m, tp optional; other grids are regridded to
G12). GRIB2 input goes through cfgrib when it is installed (Linux / Docker image); on this Windows
machine eccodes is not available, so GRIB2 raises a clear error and NetCDF is the supported input.

Stages (each timed; peak RSS recorded):
  1 load            12 km ensemble
  2 anomaly + EFI   z-scores and EFI of low MSLP / T2m from the whole ensemble
  3 mesh GNN        per-cell probability of each hazard (icosahedral mesh GNN, models/mesh/mesh_l6)
  4 objects+tracks  hysteresis components of the mesh probability -> object GNN links them across
                    leads and members (models/tracker_v3 if trained, else models/tracker) -> tracks,
                    consensus, 4-D boxes, strike probability
  5 calibration     lead-band calibration of the node probabilities (models/tracker_v3/calibration.json)
  6 crop            the 4-D box + 100 km margin (only this region is downscaled)
  7 downscaling     U-Net mean + diffusion residual, `--samples` samples per member, box leads
  8 alerts          P(exceed IMD threshold) from the 5 km samples (fraction of member x sample
                    realisations, 50 km neighbourhood) -> low/moderate/severe (docs/ALERT_RULES.md),
                    each with a pinpoint core, a 5 km impact radius, valid time, probability, reason
Outputs: backend/products/{run_id}/ with the files the API and dashboard read (meta.json,
tracks.geojson, strike_prob/, fields_12km/, fields_5km/, alerts.json, alert_grid.npz) + run.json
(stage times). Idempotent: a run id is the SHA-1 of the input file; a finished run is skipped, and
stage results are checkpointed in {run_id}/_work/ so an interrupted run resumes.
"""
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.jsonutil import dumps  # noqa: E402

HAZ = ["tropical_cyclone", "heat_dome", "cold_wave"]
KIND = {"tropical_cyclone": ["wind", "rain"], "heat_dome": ["heat"], "cold_wave": ["cold"]}
PEAK = [0.0]


def rss():
    PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss / 1e6)
    return PEAK[0]


def export_module():
    spec = importlib.util.spec_from_file_location("export_products", ROOT / "scripts/export_products.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_id(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "op_" + h.hexdigest()[:12]


class Stages:
    def __init__(self, work):
        self.work, self.rec = work, []

    def run(self, name, fn, cache=True):
        p = self.work / f"{name.split()[0]}.pkl"
        t0 = time.perf_counter()
        if cache and p.exists():
            out = pickle.loads(p.read_bytes())
            self.rec.append({"stage": name, "seconds": round(time.perf_counter() - t0, 2), "resumed": True})
        else:
            out = fn()
            if cache:
                p.write_bytes(pickle.dumps(out, protocol=4))
            self.rec.append({"stage": name, "seconds": round(time.perf_counter() - t0, 2), "resumed": False,
                             "rss_mb": round(rss())})
        print(f"  {name:<34s} {self.rec[-1]['seconds']:8.1f} s{' (resumed)' if self.rec[-1]['resumed'] else ''}", flush=True)
        return out


def load_input(path, ep):
    path = Path(path)
    if path.suffix.lower() in (".grib", ".grib2", ".grb", ".grb2"):
        try:
            import cfgrib  # noqa: F401
        except ImportError as e:
            raise RuntimeError("GRIB2 input needs cfgrib + eccodes (not installable on this Windows machine; "
                               "available in the Docker image). Convert to NetCDF, e.g. `grib_to_netcdf`.") from e
        import xarray as xr
        ds = xr.open_dataset(path, engine="cfgrib")
        nc = path.with_suffix(".nc")
        ds.rename({k: v for k, v in {"t2m": "t2m", "u10": "u10", "v10": "v10", "msl": "msl", "tp": "tp"}.items()
                   if k in ds}).to_netcdf(nc)
        path = nc
    return ep.load_file(path, "tropical_cyclone", path.stem)


def mesh_probs(c, clim, st, name="mesh_l6", members=None):
    """(M, T, 3, 333, 333) uint8 probabilities of the 3 hazards."""
    from pipeline.mesh_eval import load_mesh_model
    from pipeline.mesh_gnn import N_IN, efi_fields, features
    model, _ = load_mesh_model(name)
    e = c["ens"]
    M = e["msl"].shape[0] if members is None else len(members)
    mem = list(range(e["msl"].shape[0])) if members is None else members
    efi = efi_fields(e["msl"], e["t2m"], c["times"], clim)
    T = len(c["times"])
    out = np.zeros((M, T, 3, 333, 333), np.uint8)
    with torch.no_grad():
        for k, m in enumerate(mem):
            f = {v: e[v][m] for v in ("t2m", "u10", "v10", "msl", "tp")}
            for t0 in range(0, T, 4):
                ts = list(range(t0, min(t0 + 4, T)))
                X = torch.tensor(np.stack([features(f, t, c["times"][t], 6 * t, efi[t], clim, st).reshape(N_IN, -1).T
                                           for t in ts]))
                P = torch.sigmoid(model(X)).numpy()                       # (B, N, 3)
                out[k, ts] = np.round(255 * P.transpose(0, 2, 1).reshape(len(ts), 3, 333, 333)).astype(np.uint8)
            rss()
    return out, efi


def track_hazard(c, hz, probs, clim, thr):
    """mesh probabilities -> objects -> object GNN tracks for one hazard."""
    from pipeline.gnn_eval import decode, infer, load_model
    from pipeline.graphs import CAND, build_graph
    from synth.grids import LAT12, LON12
    mdir = ROOT / "models/tracker_v3" if (ROOT / "models/tracker_v3/full/model.pt").exists() else ROOT / "models/tracker"
    M = probs.shape[0]
    # BRIEF4 Phase 3: the links add nothing on TEST, so the simplified node-only linker is used
    variant = "none_small" if (mdir / "none_small/model.pt").exists() else ("full" if M > 1 else "temporal")
    model, st, cfg = load_model(variant, mdir)
    hi = HAZ.index(hz)
    var = "msl" if hz == "tropical_cyclone" else "t2m"
    det = ({m: probs[m, :, hi].astype(np.float32) / 255 for m in range(M)}, thr["high"], thr["low"],
           CAND[hz]["min_area_km2"])
    ev = int(cfg["n_edge_features"])
    wind = {m: (c["ens"]["u10"][m], c["ens"]["v10"][m]) for m in range(M)} if ev > 6 else None
    try:
        g = build_graph(hz, c["ens"][var].astype(np.float64), c["times"], LAT12, LON12, clim, det=det,
                        edge_version=2 if ev > 6 else 1, wind=wind)
    except ValueError:
        return None
    pn, pe = infer(model, st, g)
    dec_file = mdir / "decode.json"
    dec = json.loads(dec_file.read_text())[variant][hz] if dec_file.exists() else \
        json.loads((ROOT / "reports/gnn_results.json").read_text())["decode"][variant][hz]
    tracks = [decode(g, pn, pe, m, hz, **dec) for m in range(M)]
    return {"graph_nodes": int(len(g["X"])), "tracks": tracks, "pn": pn, "t": g["t"], "member": g["member"],
            "model": str(mdir.relative_to(ROOT)) + "/" + variant, "decode": dec}


def calibrate_tracks(tr):
    from pipeline import calibration as cal
    v = tr["model"].split("/")[-1] if tr else ""
    f = ROOT / f"models/tracker_v3/calibration_{v}.json"
    f = f if f.exists() else ROOT / "models/tracker_v3/calibration.json"
    if not f.exists() or tr is None:
        return None
    cals = cal.load(f)
    for trs in tr["tracks"]:
        for track in trs:
            for t, o in track:
                o["p_cal"] = float(cal.apply(cals, np.array([o["p"]]), np.array([6 * t]))[0])
    return str(f.relative_to(ROOT))


def crop_of(box, margin_km=100.0):
    from synth.grids import LAT12, LON12
    m = margin_km / 111.2
    i0 = int(np.searchsorted(LAT12, box["lat_min"] - m))
    i1 = int(np.searchsorted(LAT12, box["lat_max"] + m))
    j0 = int(np.searchsorted(LON12, box["lon_min"] - m))
    j1 = int(np.searchsorted(LON12, box["lon_max"] + m))
    h = max(8, ((i1 - i0 + 3) // 4) * 4)
    w = max(8, ((j1 - j0 + 3) // 4) * 4)
    i0, j0 = max(0, min(i0, 333 - h)), max(0, min(j0, 333 - w))
    return i0, i0 + h, j0, j0 + w


@torch.no_grad()
def downscale_crop(c, crop, leads, samples, steps=25, seed=0):
    """(M * samples, L, 5, 3h, 3w) float32 physical 5 km realisations in the crop (bundle units)."""
    from pipeline import downscale_eval as de
    from pipeline.downscale import cond_input, project
    import xarray as xr
    norm, unet, diff, _, _ = de.load_models()
    phys = ROOT / "models/downscale/unet_phys/model.pt"
    if phys.exists():                                   # physics-informed mean if trained (Phase 4)
        unet.load_state_dict(torch.load(phys, map_location="cpu"))
    i0, i1, j0, j1 = crop
    dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)[i0:i1, j0:j1]
    dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)[3 * i0:3 * i1, 3 * j0:3 * j1]
    e = c["ens"]
    M = e["msl"].shape[0]
    gen = torch.Generator().manual_seed(seed)
    out = np.zeros((M * samples, len(leads), 5, 3 * (i1 - i0), 3 * (j1 - j0)), np.float32)
    off = {"t2m": 273.15, "u10": 0, "v10": 0, "msl": 100000.0, "tp": 0}
    sc = {"t2m": 1, "u10": 1, "v10": 1, "msl": 0.01, "tp": 1}
    d12 = torch.tensor(np.repeat(dem12[None, None], M, 0))
    d5 = torch.tensor(np.repeat(dem5[None, None], M, 0))
    for k, t in enumerate(leads):
        x = torch.tensor(np.stack([(e[v][:, t, i0:i1, j0:j1] - off[v]) * sc[v] for v in ("t2m", "u10", "v10", "msl", "tp")], 1),
                         dtype=torch.float32)
        m = unet(x, d12, d5, norm)
        inp, _ = cond_input(x, d12, d5, norm)
        cnd = diff.cond(norm.n(m), inp)
        for s in range(samples):
            y = project(norm.d(norm.n(m) + diff.sample(cnd, steps, gen)), x)
            out[s * M:(s + 1) * M, k] = y.numpy()
        rss()
    return out


def alerts_5km(c, hz, real5, leads, crop, clim, run_case):
    """P(exceed IMD threshold) from the 5 km realisations, 50 km neighbourhood, per lead."""
    from backend.alert_rules import CATEGORIES, COLOURS, categorise, probabilities, reason
    from pipeline.tracker2 import cached_normals, regions
    from pipeline.evaluate2 import orog12
    from scipy import ndimage
    from synth.grids import LAT5, LAT12, LON5, LON12
    i0, i1, j0, j1 = crop
    lat5, lon5 = LAT5[3 * i0:3 * i1], LON5[3 * j0:3 * j1]
    reg12 = regions(orog12(), LAT12, LON12)[i0:i1, j0:j1]
    ep = export_module()
    reg5 = np.kron(reg12, np.ones((3, 3), int))
    alerts, grids = [], {}
    for kind in KIND[hz]:
        cat_all = np.zeros((len(leads),) + reg5.shape, np.int8)
        for k, t in enumerate(leads):
            if kind == "wind":
                f, v = np.hypot(real5[:, k, 1], real5[:, k, 2]) * 3.6, None
            elif kind == "rain":
                # 24 h rain needs 4 consecutive downscaled leads ending at t
                ks = [leads.index(tt) for tt in range(t - 3, t + 1) if tt in leads]
                if len(ks) < 4:
                    continue
                f, v = real5[:, ks, 4].sum(1), None
            else:
                ks = [leads.index(tt) for tt in range(t - 1, t + 3) if tt in leads]      # centred 24 h window
                ext = real5[:, ks, 0].max(1) if kind == "heat" else real5[:, ks, 0].min(1)
                nk = "max" if kind == "heat" else "min"
                normal = cached_normals(clim, c["times"][[t]], nk)[0][i0:i1, j0:j1] - 273.15
                dep = ext - np.kron(normal, np.ones((3, 3)))
                absT = ext
                if kind == "heat":
                    ok = ((reg5 == 1) & (absT >= 40)) | ((reg5 == 2) & (absT >= 37)) | ((reg5 == 3) & (absT >= 30)) | ((reg5 > 0) & (absT >= 45))
                else:
                    ok = ((reg5 == 1) & (absT <= 10)) | ((reg5 == 2) & (absT <= 15)) | ((reg5 == 3) & (absT <= 0)) | ((reg5 == 1) & (absT <= 4))
                f, v = dep, ok & (reg5 > 0)
            p1, p2, p3 = probabilities(kind, f, v, neighbourhood=True, dx_km=4.45)
            cat = categorise(p1, p2, p3)
            cat_all[k] = cat
            lab, n = ndimage.label(cat >= 1, structure=np.ones((3, 3)))
            for r in range(1, n + 1):
                rm = lab == r
                if rm.sum() < 9:
                    continue
                lvl = CATEGORIES[int(cat[rm].max())]
                if lvl == "severe" or (lvl == "moderate" and p3[rm].max() >= 0.2):
                    pc, which = p3, "severe"
                elif lvl == "moderate" or (lvl == "low" and p2[rm].max() >= 0.2):
                    pc, which = p2, "moderate"
                else:
                    pc, which = p1, "low"
                pr = float(pc[rm].max())
                ii, jj = np.unravel_index(np.argmax(np.where(rm, pc + 1e-3 * (f.mean(0) if kind != "cold" else -f.mean(0)), -1)), rm.shape)
                plat, plon = float(lat5[ii]), float(lon5[jj])
                alerts.append({"id": f"{run_case}-{hz}-{kind}-{6 * t:03d}-{r}", "case": run_case, "hazard": hz, "kind": kind,
                               "lead_h": 6 * t, "valid_time": str(c["times"][t]), "category": lvl, "colour": COLOURS[lvl],
                               "probability": round(pr, 3), "probability_type": "5 km diffusion realisations, neighbourhood (50 km)",
                               "probability_cell": round(float(pc[ii, jj]), 3),
                               "reason": reason(kind, which, pr) + " within 50 km (5 km ensemble)",
                               "pinpoint": {"lat": round(plat, 3), "lon": round(plon, 3), "grid": "G5 0.04 deg"},
                               "impact_polygon": {"type": "Polygon", "coordinates": [ep.circle(plat, plon)]},
                               "impact_radius_km": 5.0,
                               "region_bbox": [float(lon5[np.nonzero(rm)[1]].min()), float(lat5[np.nonzero(rm)[0]].min()),
                                               float(lon5[np.nonzero(rm)[1]].max()), float(lat5[np.nonzero(rm)[0]].max())],
                               "region_cells_5km": int(rm.sum()), "in_india": bool((reg5[rm] > 0).any()),
                               "n_members": int(f.shape[0]), "synthetic": c["synthetic"]})
        grids[kind] = cat_all
    return alerts, grids


def process(path, out_root, hazards, samples, lead_stride, steps, force=False):
    ep = export_module()
    from pipeline.anomaly import Climatology
    from pipeline.gnn_eval import cone
    from pipeline.mesh_gnn import Static
    from synth.grids import LAT12, LON12
    rid = run_id(path)
    out = Path(out_root) / rid
    if (out / "run.json").exists() and not force:
        print(f"{path}: already processed as {rid} (idempotent skip)")
        return json.loads((out / "run.json").read_text())
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    S = Stages(work)
    t_all = time.perf_counter()
    print(f"operational run {rid} <- {path}", flush=True)
    clim, st = Climatology(), Static(LAT12, LON12)
    c = S.run("1 load 12 km ensemble", lambda: load_input(path, ep))
    c["case"] = rid
    probs, efi = S.run("2+3 anomaly/EFI + mesh GNN", lambda: mesh_probs(c, clim, st))
    mres = json.loads((ROOT / "reports/mesh_results.json").read_text())["mesh_l6"]["thresholds"] \
        if (ROOT / "reports/mesh_results.json").exists() else {}
    thr = lambda hz: mres.get(hz, {"high": 0.6, "low": 0.4})
    tracks = S.run("4 objects + object-GNN tracks", lambda: {hz: track_hazard(c, hz, probs, clim, thr(hz)) for hz in hazards})
    calf = S.run("5 calibration", lambda: {hz: calibrate_tracks(tracks[hz]) for hz in hazards}, cache=False)
    # choose the hazard with the most tracked evidence as the headline one; products per hazard
    ev = {hz: sum(sum(o.get("p", 0) for _, o in tr) for trs in (tracks[hz] or {"tracks": []})["tracks"] for tr in trs)
          for hz in hazards}
    main = max(ev, key=ev.get)
    alerts_all, grids_all, boxes_all, feats = [], {}, [], []
    run_crops = {}
    for hz in [h for h in hazards if tracks[h] and any(tracks[h]["tracks"])]:
        cz = {**c, "hazard": hz}
        mt = tracks[hz]["tracks"]
        ep.core_masks(cz, mt, clim)
        cn = cone({"lat": LAT12, "lon": LON12}, mt)
        gj, boxes = ep.tracks_geojson(cz, mt, cn)
        feats += gj["features"]
        boxes_all += boxes
        if not boxes:
            continue
        crop = crop_of(boxes[0])
        run_crops[hz] = crop
        b = boxes[0]
        leads = list(range(max(0, b["lead_start_h"] // 6 - 3), min(len(c["times"]) - 1, b["lead_end_h"] // 6 + 2) + 1, lead_stride))
        real5 = S.run(f"7 downscaling {hz}", lambda: downscale_crop(c, crop, leads, samples, steps))
        al, gr = S.run(f"8 alerts {hz}", lambda: alerts_5km(c, hz, real5, leads, crop, clim, rid))
        alerts_all += al
        grids_all[hz] = {"crop": crop, "leads": leads, **gr}
        real5 = None                       # free the 5 km realisations before the next hazard
    S.rec.append({"stage": "6 crop (box + 100 km)", "seconds": 0.0, "crops": {k: list(map(int, v)) for k, v in run_crops.items()}})
    # products in the API layout
    (out / "tracks.geojson").write_text(dumps({"type": "FeatureCollection", "features": feats}))
    (out / "alerts.json").write_text(dumps({"case": rid, "synthetic": c["synthetic"], "rules": "docs/ALERT_RULES.md",
                                            "alerts": alerts_all}))
    LA, LO = np.meshgrid(LAT12, LON12, indexing="ij")
    cz = {**c, "hazard": main}
    mt = tracks[main]["tracks"] if tracks[main] else [[] for _ in range(c["ens"]["msl"].shape[0])]
    for t in range(len(c["times"])):
        ep.write_png(out / "strike_prob" / f"{6 * t:03d}.png", ep.strike(cz, mt, t, LA, LO), "strike")
        e0 = {v: c["ens"][v][0, t].astype(np.float64) for v in ("t2m", "u10", "v10", "msl", "tp")}
        ep.write_png(out / "fields_12km/t2m" / f"{6 * t:03d}.png", e0["t2m"] - 273.15, "t2m")
        ep.write_png(out / "fields_12km/wind" / f"{6 * t:03d}.png", np.hypot(e0["u10"], e0["v10"]), "wind")
        ep.write_png(out / "fields_12km/msl" / f"{6 * t:03d}.png", e0["msl"] / 100, "msl")
        ep.write_png(out / "fields_12km/tp" / f"{6 * t:03d}.png", e0["tp"], "tp")
        var = "msl" if main == "tropical_cyclone" else "t2m"
        ep.write_png(out / "fields_12km/anom" / f"{6 * t:03d}.png", clim.z(var, e0[var], c["times"][t]), "anom")
    total = time.perf_counter() - t_all
    meta = ep.build_meta({**c, "hazard": main, "labels": None}, boxes_all, "operational", {}, total)
    meta.update({"tracker": f"mesh GNN (models/mesh/mesh_l6) + object GNN ({tracks[main]['model'] if tracks[main] else 'n/a'})",
                 "downscaler": "U-Net mean (physics-informed if trained) + diffusion residual, crop-aware",
                 "leads_5km_h": [], "operational": True, "hazards_tracked": {h: bool(tracks[h] and any(tracks[h]["tracks"])) for h in hazards}})
    (out / "meta.json").write_text(dumps(meta, indent=1))
    np.savez_compressed(out / "alert_grid_5km.npz", **{f"{hz}_{k}": v for hz, g in grids_all.items() for k, v in g.items()
                                                      if k not in ("crop", "leads")})
    rec = {"run_id": rid, "input": str(path).replace("\\", "/"), "members": int(c["ens"]["msl"].shape[0]),
           "leads": int(len(c["times"])), "stages": S.rec, "total_seconds": round(total, 1), "peak_rss_mb": round(PEAK[0]),
           "hazards": hazards, "main_hazard": main, "alerts": len(alerts_all), "samples_per_member": samples,
           "lead_stride": lead_stride, "ddim_steps": steps, "calibration": calf,
           "device": "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}
    (out / "run.json").write_text(dumps(rec, indent=1))
    ep.write_index(Path(out_root))
    print(f"done {rid}: {len(alerts_all)} alerts, {total:.0f} s, peak RSS {PEAK[0]:.0f} MB", flush=True)
    return rec


def watch(folder, out_root, **kw):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"watching {folder} for *.nc / *.grib2 (Ctrl+C to stop)", flush=True)
    seen = set()
    while True:
        for p in sorted(list(folder.glob("*.nc")) + list(folder.glob("*.grib2"))):
            if p in seen:
                continue
            size = p.stat().st_size
            time.sleep(2)
            if p.stat().st_size != size:                     # still being copied
                continue
            try:
                process(p, out_root, **kw)
            except Exception as e:                           # one bad file must not stop the worker
                print(f"FAILED {p}: {type(e).__name__}: {e}", flush=True)
            seen.add(p)
        time.sleep(5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("--watch", default=None)
    ap.add_argument("--out", default=str(ROOT / "backend/products"))
    ap.add_argument("--hazards", default=",".join(HAZ))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--lead-stride", type=int, default=1)
    ap.add_argument("--steps", type=int, default=25, help="DDIM steps")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    kw = dict(hazards=a.hazards.split(","), samples=a.samples, lead_stride=a.lead_stride, steps=a.steps)
    if a.watch:
        watch(a.watch, a.out, **kw)
    elif a.input:
        process(a.input, a.out, force=a.force, **kw)
    else:
        ap.error("give an input file or --watch DIR")


if __name__ == "__main__":
    main()
