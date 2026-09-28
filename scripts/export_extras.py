"""BRIEF4 Phase 6: extra web products on top of scripts/export_products.py.

    python scripts/export_extras.py [case ...]           (default: the committed demo cases)
    python scripts/export_extras.py --real               REAL-VAL storms (ERA5 single runs) + the real
                                                          IFS ENS 0.25 deg Amphan ensemble as cases

Per case, into backend/products/{case}/:
  mesh_prob/{lead}.png   ensemble-mean mesh-GNN probability of the case hazard (models/mesh/mesh_l6)
  strike_cal/{lead}.png  CALIBRATED strike probability from the v3 object GNN (models/tracker_v3):
                         mean over members of the max calibrated node probability of the member's
                         tracked objects whose centre is within 120 km (cyclones) / that cover the cell
  efi/{lead}.png         EFI of the hazard variable (low MSLP for cyclones, T2m +/- for heat/cold)
  truth/{lead}.png       SYNTHETIC cases only: the exact truth event mask (IMD daily labels for heat/cold)
  scen/{var}/{s}/{lead}.png   5 km diffusion scenarios inside the 4-D box + 100 km: samples 1..N, mean,
                         p90 of tp and wind, every 12 h inside the box (same colour scales as fields_5km)
  physviol/{lead}.png    cells of the scenario mean where rain > 1 mm/6h falls without low-level
                         convergence (pipeline/physics.py proxy)
  alerts.json            each alert gets an "explain" block: members exceeding each threshold in the
                         50 km neighbourhood of the pinpoint, their values, and the calibration curve
                         of the lead band (reports/object_gnn_v3.json)
  meta.json              "layers" (which extras exist) and "scenario" (bounds, leads, samples); real
                         storms also get "ibtracs" (best-track fixes) and "real_event"
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.jsonutil import dumps  # noqa: E402

OUT = ROOT / "backend/products"
DEMO = ["amphan_replay", "cyc_04", "heat_04", "cold_04", "amphan_era5_real"]
N_SCEN = 4


def ep():
    spec = importlib.util.spec_from_file_location("export_products", ROOT / "scripts/export_products.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.SCALES.update({"efi": ("RdBu_r", -1.0, 1.0, "EFI", None), "mesh": ("inferno", 0.0, 1.0, "probability", 0.05),
                     "truth": ("Greens", 0.0, 1.0, "truth mask", 0.5), "physviol": ("Reds", 0.0, 1.0, "violation", 0.5)})
    return m


def load_any(E, case):
    if (ROOT / "data/synthetic" / case).exists() or case == "amphan_era5_real":
        return E.load(case)
    meta = json.loads((OUT / case / "meta.json").read_text())
    return E.load_file(ROOT / meta["input_file"], meta["hazard"], case)


def mesh_layer(E, case, c, clim, st, out):
    from pipeline.mesh_eval import PCACHE, load_mesh_model
    from pipeline.mesh_gnn import HAZ, N_IN, efi_fields, features
    hi = HAZ.index(c["hazard"])
    p = PCACHE / f"mesh_l6_{case}.npz"
    if p.exists():
        P = np.load(p)["p"][:20].astype(np.float32).mean(0) / 255.0
    else:
        model, _ = load_mesh_model("mesh_l6")
        e = c["ens"]
        efi = efi_fields(e["msl"], e["t2m"], c["times"], clim)
        M = e["msl"].shape[0]
        P = np.zeros((len(c["times"]), 333, 333), np.float32)
        with torch.no_grad():
            for m in range(M):
                f = {v: e[v][m] for v in ("t2m", "u10", "v10", "msl", "tp")}
                for t0 in range(0, len(c["times"]), 4):
                    ts = list(range(t0, min(t0 + 4, len(c["times"]))))
                    X = torch.tensor(np.stack([features(f, t, c["times"][t], 6 * t, efi[t], clim, st).reshape(N_IN, -1).T
                                               for t in ts]))
                    P[ts] += torch.sigmoid(model(X))[..., hi].numpy().reshape(len(ts), 333, 333) / M
    for t in range(len(c["times"])):
        E.write_png(out / "mesh_prob" / f"{6 * t:03d}.png", P[t], "mesh")
    return True


def strike_cal_layer(E, case, c, clim, out):
    from pipeline import calibration as cal
    from pipeline.gnn_eval import decode, infer, load_model
    from pipeline.graphs import build_graph
    from synth.grids import LAT12, LON12
    mdir = ROOT / "models/tracker_v3"
    if not (mdir / "calibration.json").exists():
        return False
    M = c["ens"]["msl"].shape[0]
    variant = "none_small" if (mdir / "none_small/model.pt").exists() else ("full" if M > 1 else "temporal")
    model, st, _ = load_model(variant, mdir)
    gp = ROOT / "data/graphs_v2" / f"{case}.npz"
    if gp.exists():
        z = np.load(gp)
        g = {k: z[k] for k in z.files}
    else:
        var = "msl" if c["hazard"] == "tropical_cyclone" else "t2m"
        g = build_graph(c["hazard"], c["ens"][var].astype(np.float64), c["times"], LAT12, LON12, clim, edge_version=2,
                        wind={m: (c["ens"]["u10"][m], c["ens"]["v10"][m]) for m in range(M)})
    pn, pe = infer(model, st, g)
    dec = json.loads((mdir / "decode.json").read_text())[variant][c["hazard"]]
    cf = mdir / f"calibration_{variant}.json"
    cals = cal.load(cf if cf.exists() else mdir / "calibration.json")
    LA, LO = np.meshgrid(LAT12, LON12, indexing="ij")
    tracks = [decode(g, pn, pe, m, c["hazard"], **dec) for m in range(M)]
    T = len(c["times"])
    S = np.zeros((T, 333, 333), np.float32)
    for trs in tracks:
        hit = np.zeros((T, 333, 333), np.float32)
        for tr in trs:
            for t, o in tr:
                pc = float(cal.apply(cals, np.array([o["p"]]), np.array([6 * t]))[0])
                if c["hazard"] == "tropical_cyclone":
                    la, lo = o["center_lat"], o["center_lon"]
                    m = np.hypot((LA - la) * 111.2, (LO - lo) * 111.2 * np.cos(np.deg2rad(la))) <= 120
                else:
                    m = o["mask"]
                hit[t] = np.where(m, np.maximum(hit[t], pc), hit[t])
        S += hit / M
    for t in range(T):
        E.write_png(out / "strike_cal" / f"{6 * t:03d}.png", S[t], "strike")
    return True


def efi_truth_layers(E, case, c, clim, out):
    from pipeline.efi import efi_gaussian
    var = "msl" if c["hazard"] == "tropical_cyclone" else "t2m"
    sign = -1.0 if c["hazard"] in ("tropical_cyclone", "cold_wave") else 1.0
    for t, vt in enumerate(c["times"]):
        e = sign * efi_gaussian(c["ens"][var][:, t].astype(np.float64), *clim.get(var, vt)) if c["ens"][var].shape[0] > 1 \
            else np.full((333, 333), np.nan)
        E.write_png(out / "efi" / f"{6 * t:03d}.png", e, "efi")
    if not c["synthetic"]:
        return False
    from pipeline.phase3_eval import load_case_imd
    d = load_case_imd(case)
    for t in range(len(c["times"])):
        E.write_png(out / "truth" / f"{6 * t:03d}.png", d["tmask"][t].astype(float), "truth")
    return True


@torch.no_grad()
def scenario_layers(E, case, c, meta, out):
    from pipeline import downscale_eval as de
    from pipeline.downscale import cond_input, project
    from pipeline.physics import RAIN_THR, conv_proxy
    from pipeline.run_operational import crop_of
    from synth.grids import LAT5, LON5
    if not meta.get("bbox4d"):
        return None
    box = meta["bbox4d"][0]
    i0, i1, j0, j1 = crop_of(box)
    norm, unet, diff, _, _ = de.load_models()
    phys = ROOT / "models/downscale/unet_phys/model.pt"
    if phys.exists():
        unet.load_state_dict(torch.load(phys, map_location="cpu"))
    dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)[i0:i1, j0:j1]
    dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)[3 * i0:3 * i1, 3 * j0:3 * j1]
    rep = int(meta.get("fields_member_index", 0))
    leads = [t for t in range(len(c["times"])) if (6 * t) % 12 == 0 and box["lead_start_h"] <= 6 * t <= box["lead_end_h"]]
    gen = torch.Generator().manual_seed(0)
    off = {"t2m": 273.15, "u10": 0, "v10": 0, "msl": 100000.0, "tp": 0}
    sc = {"t2m": 1, "u10": 1, "v10": 1, "msl": 0.01, "tp": 1}
    d12, d5 = torch.tensor(dem12[None, None]), torch.tensor(dem5[None, None])

    def full5(a):                                  # place the crop into a transparent 999x999 frame
        z = np.full((999, 999), np.nan, np.float32)
        z[3 * i0:3 * i1, 3 * j0:3 * j1] = a
        return z
    for t in leads:
        x = torch.tensor(np.stack([(c["ens"][v][rep, t, i0:i1, j0:j1] - off[v]) * sc[v]
                                   for v in ("t2m", "u10", "v10", "msl", "tp")])[None], dtype=torch.float32)
        m = unet(x, d12, d5, norm)
        inp, _ = cond_input(x, d12, d5, norm)
        cnd = diff.cond(norm.n(m), inp)
        S = torch.cat([project(norm.d(norm.n(m) + diff.sample(cnd, 25, gen)), x) for _ in range(N_SCEN)]).numpy()
        for var, fn in (("tp", lambda y: y[:, 4]), ("wind", lambda y: np.hypot(y[:, 1], y[:, 2]))):
            v = fn(S)
            for s in range(N_SCEN):
                E.write_png(out / "scen" / var / f"s{s + 1}" / f"{6 * t:03d}.png", full5(v[s]), var)
            E.write_png(out / "scen" / var / "mean" / f"{6 * t:03d}.png", full5(v.mean(0)), var)
            E.write_png(out / "scen" / var / "p90" / f"{6 * t:03d}.png", full5(np.quantile(v, 0.9, axis=0)), var)
        mean_tp = torch.tensor(S[:, 4:5].mean(0, keepdims=True))
        viol = ((mean_tp > RAIN_THR) & (conv_proxy(x) <= 0)).float().numpy()[0, 0]
        E.write_png(out / "physviol" / f"{6 * t:03d}.png", full5(viol), "physviol")
    return {"bounds": [float(LON5[3 * j0] - 0.02), float(LAT5[3 * i0] - 0.02), float(LON5[3 * j1 - 1] + 0.02),
                       float(LAT5[3 * i1 - 1] + 0.02)], "frame_bounds": meta["bounds"], "leads_h": [6 * t for t in leads],
            "samples": N_SCEN, "vars": ["tp", "wind"], "member": rep,
            "model": "diffusion residual on the " + ("physics-informed " if phys.exists() else "") + "U-Net mean, 25 DDIM steps",
            "physics_overlay": "rain > 1 mm/6h of the scenario mean where -div(V10_12km) <= 0 (proxy)"}


def explain_alerts(E, case, c, out):
    from scipy.ndimage import maximum_filter, minimum_filter
    from backend.alert_rules import THRESHOLDS, footprint
    from pipeline.tracker2 import cached_normals, trailing
    from synth.grids import LAT12, LON12
    aj = json.loads((out / "alerts.json").read_text())
    v3 = json.loads((ROOT / "reports/object_gnn_v3.json").read_text()) if (ROOT / "reports/object_gnn_v3.json").exists() else {}
    calc = v3.get("calibration_test", {})
    e = c["ens"]
    M = e["msl"].shape[0]
    fp = footprint()
    cache = {}
    for a in aj["alerts"]:
        t, kind = a["lead_h"] // 6, a["kind"]
        i = int(np.abs(LAT12 - a["pinpoint"]["lat"]).argmin())
        j = int(np.abs(LON12 - a["pinpoint"]["lon"]).argmin())
        key = (kind, t)
        if key not in cache:
            if kind == "wind":
                f = np.hypot(e["u10"][:, t], e["v10"][:, t]) * 3.6
            elif kind == "rain":
                f = e["tp"][:, max(0, t - 3):t + 1].sum(1)
            else:
                k = "max" if kind == "heat" else "min"
                if ("norm", k) not in cache:
                    cache[("norm", k)] = cached_normals(__import__("pipeline.anomaly", fromlist=["Climatology"]).Climatology(),
                                                        c["times"], k)
                f = np.array([trailing(e["t2m"][m].astype(np.float64), k)[t] for m in range(M)]) - cache[("norm", k)][t]
            filt = minimum_filter if THRESHOLDS[kind][3] == "le" else maximum_filter
            cache[key] = np.array([filt(x, footprint=fp, mode="nearest") for x in f])
        vals = cache[key][:, i, j]
        label, unit, thr, how = THRESHOLDS[kind]
        ex = {lvl: [int(m) for m in range(M) if (vals[m] >= th if how == "ge" else vals[m] <= th)]
              for lvl, th in zip(("low", "moderate", "severe"), thr)}
        band = "0-72h" if a["lead_h"] <= 72 else "78-168h" if a["lead_h"] <= 168 else "174-240h"
        a["explain"] = {"variable": f"{label} ({unit}), max within 50 km of the pinpoint" if how == "ge" else
                        f"{label} ({unit}), min within 50 km of the pinpoint",
                        "member_values": [round(float(v), 1) for v in vals], "thresholds": list(thr),
                        "members_exceeding": ex, "n_members": M,
                        "drivers": ["10 m wind"] if kind == "wind" else ["6 h rain (24 h sum)"] if kind == "rain"
                        else ["T2m daily " + ("max" if kind == "heat" else "min") + " departure from the ERA5 normal"],
                        "calibration_band": band,
                        "calibration_curve": calc.get(band, {}).get("rel_cal"),
                        "calibration_note": "reliability (forecast p, observed frequency, n) of the calibrated object-GNN "
                                            "event probability on SYNTHETIC TEST for this lead band (reports/object_gnn_v3.json)"}
    (out / "alerts.json").write_text(dumps(aj))


def extras(case, clim, st):
    E = ep()
    t0 = time.time()
    out = OUT / case
    meta = json.loads((out / "meta.json").read_text())
    c = load_any(E, case)
    layers = {"mesh_prob": mesh_layer(E, case, c, clim, st, out),
              "strike_cal": strike_cal_layer(E, case, c, clim, out)}
    layers["truth"] = efi_truth_layers(E, case, c, clim, out)
    layers["efi"] = c["ens"]["msl"].shape[0] > 1
    scen = scenario_layers(E, case, c, meta, out)
    layers["scen"] = layers["physviol"] = scen is not None
    explain_alerts(E, case, c, out)
    meta["layers"] = layers
    meta["scenario"] = scen
    if case == "amphan_era5_real" and "ibtracs" not in meta:
        from pipeline.real_eval import ibtracs
        ib = ibtracs(ROOT / "data/real/ibtracs/events/amphan_2020.csv")
        meta["ibtracs"] = [{"time": str(r.time), "lat": float(r.LAT), "lon": float(r.LON)} for _, r in ib.iterrows()]
        meta["real_event"] = "amphan_2020"
    meta["legends"].update({k: E.legend(k) for k in ("efi", "mesh", "truth", "physviol")})
    (out / "meta.json").write_text(dumps(meta, indent=1))
    print(f"{case}: extras {layers} in {time.time() - t0:.0f}s", flush=True)


def export_real(clim, st):
    """REAL-VAL storms (ERA5) and the real IFS ENS 0.25 deg Amphan ensemble as dashboard cases."""
    from pipeline.real_eval import ibtracs
    E = ep()
    import xarray as xr_
    from pipeline.tracker2 import regions
    from pipeline.evaluate2 import orog12
    from synth.grids import LAT12, LON12
    ds = E.Downscaler()
    reg = regions(orog12(), LAT12, LON12)
    india = xr_.open_dataset(ROOT / "data/real/boundary/india_mask_g12.nc").india.values.astype(bool)
    dec = json.loads((ROOT / "reports/gnn_results.json").read_text())["decode"]
    splits = json.loads((ROOT / "data/real/SPLITS.json").read_text())
    jobs = [(f"real_{eid}", ROOT / f"data/real/events/{eid}_sfc_g12.nc", eid) for eid in splits["REAL-VAL"]
            if splits["events"][eid]["hazard"] == "tropical_cyclone" and eid != "amphan_2020"]
    jobs.append(("real_amphan_ifs_ens_20200516", ROOT / "data/real/ensembles/ifs_ens_0p25_2020051600.nc", "amphan_2020"))
    for case, f, eid in jobs:
        if not f.exists():
            continue
        c = E.load_file(f, "tropical_cyclone", case, source=("ERA5 reanalysis" if "events" in str(f) else
                                                              "ECMWF IFS ENS 0.25 deg via WeatherBench 2 (REAL ensemble forecast)"))
        c["synthetic"] = False
        c["badge"] = "REAL (ERA5 reanalysis)" if "events" in str(f) else "REAL (IFS ENS forecast)"
        E.export_case(case, clim, ds, india, reg, dec, c=c)
        ib = ibtracs(ROOT / "data/real/ibtracs/events" / f"{eid}.csv")
        m = json.loads((OUT / case / "meta.json").read_text())
        m["ibtracs"] = [{"time": str(r.time), "lat": float(r.LAT), "lon": float(r.LON),
                         "wind_kt": None if str(r.get("USA_WIND", "")).strip() in ("", " ") else float(r.USA_WIND)}
                        for _, r in ib.iterrows()]
        m["real_event"] = eid
        m["input_file"] = str(f.relative_to(ROOT)).replace("\\", "/")
        (OUT / case / "meta.json").write_text(dumps(m, indent=1))
        extras(case, clim, st)
    E.write_index()


def main():
    from pipeline.anomaly import Climatology
    from pipeline.mesh_gnn import Static
    from synth.grids import LAT12, LON12
    clim, st = Climatology(), Static(LAT12, LON12)
    if "--real" in sys.argv:
        export_real(clim, st)
        return
    for case in [a for a in sys.argv[1:] if not a.startswith("--")] or DEMO:
        extras(case, clim, st)
    ep().write_index()


if __name__ == "__main__":
    main()
