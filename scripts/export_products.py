"""BRIEF3 Phase A: precompute light, web-ready products for every case -> backend/products/{case}/

    python scripts/export_products.py [case ...]      (default: all 13 synthetic + amphan_era5_real)

Per case:
  meta.json            synthetic flag + badge text, source, init/valid times, grid bounds, colour
                       scales (with legend stops and units), metrics copied from reports/*.json
  tracks.geojson       GNN member tracks (LineStrings with per-vertex leads), the consensus track
                       (ensemble-mean position + spread per lead) and 4-D bounding boxes
  strike_prob/{lead}.png      probability overlay (cyclone: event centre within 120 km;
                              heat/cold: cell inside the member's tracked event object)
  fields_12km/{var}/{lead}.png, fields_5km/{var}/{lead}.png   member 0 (control-like member; the
                       ERA5 run for the real case) at 12 km and U-Net-downscaled to 5 km on the SAME
                       colour scale; var in t2m, wind, msl, tp (+ anom at 12 km: hazard z-score)
  alerts.json          alerts per lead from backend/alert_rules.py
All numbers come from the pipeline; nothing is typed in by hand.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from backend.alert_rules import CATEGORIES, COLOURS, NBHD_KM, categorise, probabilities, reason  # noqa: E402
from pipeline.anomaly import Climatology  # noqa: E402
from pipeline.downscale import Normalizer, UNetDownscaler  # noqa: E402
from pipeline.gnn_eval import cone, decode, gnn_main_track, infer, load_graph, load_model  # noqa: E402
from pipeline.splits import ALL  # noqa: E402
from pipeline.track import haversine_km  # noqa: E402
from pipeline.tracker2 import cached_normals, regions, trailing  # noqa: E402
from synth.grids import LAT5, LAT12, LON5, LON12  # noqa: E402

OUT = ROOT / "backend" / "products"
SYN = ROOT / "data" / "synthetic"
BOUNDS = [60.0, 0.0, 99.96, 39.96]              # lon_min, lat_min, lon_max, lat_max (cell edges)
SCALES = {  # var -> (cmap, vmin, vmax, units, transparent_below)
    "t2m": ("RdBu_r", -10.0, 48.0, "degC", None),
    "wind": ("magma", 0.0, 60.0, "m/s", None),
    "msl": ("cividis", 960.0, 1030.0, "hPa", None),
    "tp": ("viridis", 0.0, 60.0, "mm/6h", 0.2),
    "anom": ("RdBu_r", -4.0, 4.0, "z-score (sigma)", None),
    "strike": ("inferno", 0.0, 1.0, "probability", 0.02),
}
HAZ_KIND = {"tropical_cyclone": ["wind", "rain"], "heat_dome": ["heat"], "cold_wave": ["cold"]}
LEVELS = {"tropical_cyclone": [1013, 850], "heat_dome": [1013, 1013], "cold_wave": [1013, 1013]}


# ----------------------------------------------------------------------------- PNG helpers
def lut(cmap):
    c = (matplotlib.colormaps[cmap](np.linspace(0, 1, 255))[:, :3] * 255).astype(np.uint8)
    return np.vstack([[0, 0, 0], c])                   # index 0 = transparent


def write_png(path, a, var):
    cmap, vmin, vmax, _, below = SCALES[var]
    x = np.clip((np.asarray(a, float) - vmin) / (vmax - vmin), 0, 1)
    idx = (1 + np.round(x * 254)).astype(np.uint8)
    bad = ~np.isfinite(a)
    if below is not None:
        bad |= np.asarray(a) < below
    idx[bad] = 0
    im = Image.fromarray(np.flipud(idx), mode="P")    # PNG rows run north -> south
    im.putpalette(lut(cmap).ravel().tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, optimize=True, transparency=0)


def legend(var):
    cmap, vmin, vmax, units, below = SCALES[var]
    stops = np.linspace(0, 1, 9)
    cols = matplotlib.colormaps[cmap](stops)
    return {"cmap": cmap, "vmin": vmin, "vmax": vmax, "units": units, "transparent_below": below,
            "stops": [{"value": float(vmin + s * (vmax - vmin)),
                       "color": matplotlib.colors.to_hex(c)} for s, c in zip(stops, cols)]}


# ----------------------------------------------------------------------------- data loading
def load(case):
    if case == "amphan_era5_real":
        e = xr.open_dataset(ROOT / "data/real/era5/amphan_era5_g12.nc")
        times = pd.to_datetime(e.time.values)
        ds = {v: e[v].values.astype(np.float32)[None] for v in ("t2m", "u10", "v10", "msl", "tp")}
        e.close()
        return {"case": case, "hazard": "tropical_cyclone", "synthetic": False, "times": times,
                "init": times[0], "ens": ds, "labels": None,
                "source": "ERA5 reanalysis (WeatherBench 2), India box, 0.25 deg -> G12 bilinear"}
    lab = json.loads((SYN / case / "labels.json").read_text())
    f = xr.open_dataset(SYN / case / "fcst_12km.nc", decode_timedelta=False)
    ds = {v: f[v].values.astype(np.float32) for v in ("t2m", "u10", "v10", "msl", "tp")}
    f.close()
    return {"case": case, "hazard": lab["hazard"], "synthetic": True,
            "times": pd.to_datetime(lab["valid_times"]), "init": pd.Timestamp(lab["init_time"]),
            "ens": ds, "labels": lab, "source": f"SYNTHETIC ({lab['background']})"}


class Downscaler:
    def __init__(self):
        d = ROOT / "models" / "downscale" / "unet"
        st = np.load(d / "stats.npz")
        cfg = json.loads((d / "config.json").read_text())
        self.norm = Normalizer(st["mean"], st["std"])
        self.net = UNetDownscaler(cfg["base"])
        self.net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        self.net.eval()
        self.dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)
        self.dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)

    @torch.no_grad()
    def __call__(self, t2m, u10, v10, msl, tp):
        """Physical 12 km fields (333x333) -> 5 km (999x999), bundle units internally."""
        x = np.stack([t2m - 273.15, u10, v10, msl / 100.0 - 1000.0, tp])
        pad = lambda a, n: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, n), (0, n)], mode="edge")
        x12 = torch.tensor(pad(x, 3)[None], dtype=torch.float32)   # 336 -> 1008 (divisible by 4)
        d12 = torch.tensor(pad(self.dem12[None], 3)[None])
        d5 = torch.tensor(pad(self.dem5[None], 9)[None])
        y = self.net(x12, d12, d5, self.norm)[0].numpy()[:, :999, :999]
        return {"t2m": y[0] + 273.15, "u10": y[1], "v10": y[2], "msl": (y[3] + 1000.0) * 100.0,
                "tp": y[4]}


# ----------------------------------------------------------------------------- tracks
CORE_THR = {"tropical_cyclone": 2.0, "heat_dome": 3.0, "cold_wave": 3.0}


def core_masks(c, member_tracks, clim):
    """Attach o['core_mask'], the event extent used for the 4-D box (not the permissive candidate
    object): cyclone = closed area >= 4 hPa below the 5.5-deg ring-median MSLP around the centre;
    heat/cold = the departure >= 3 C component containing (or nearest) the object centre."""
    from pipeline.graphs import hazard_field, static
    _, _, domain = static(LAT12, LON12)
    var = "msl" if c["hazard"] == "tropical_cyclone" else "t2m"
    for m, trs in enumerate(member_tracks):
        if not trs:
            continue
        f, _ = hazard_field(c["hazard"], c["ens"][var][m].astype(np.float64), c["times"], clim, domain)
        for tr in trs:
            for t, o in tr:
                if c["hazard"] == "tropical_cyclone":
                    # physical storm extent: closed area with MSLP >= 4 hPa below the 5.5-deg ring
                    # median around the centre (same definition as pipeline.te_style.node_region)
                    from pipeline.te_style import node_region
                    ci = int(np.abs(LAT12 - o["center_lat"]).argmin())
                    cj = int(np.abs(LON12 - o["center_lon"]).argmin())
                    o["core_mask"] = node_region(c["ens"]["msl"][m, t].astype(np.float64), ci, cj,
                                                 LAT12, LON12)
                    continue
                core = o["mask"] & (f[t] >= CORE_THR[c["hazard"]])
                lab, n = ndimage.label(core, structure=np.ones((3, 3)))
                if n == 0:
                    o["core_mask"] = core
                    continue
                # keep the connected core component that contains (or is nearest to) the centre
                ci = int(np.abs(LAT12 - o["center_lat"]).argmin())
                cj = int(np.abs(LON12 - o["center_lon"]).argmin())
                k = lab[ci, cj]
                if k == 0:
                    ii, jj = np.nonzero(lab)
                    q = int(np.argmin((ii - ci) ** 2 + (jj - cj) ** 2))
                    k = lab[ii[q], jj[q]]
                o["core_mask"] = lab == k

def tracks_geojson(c, member_tracks, cn):
    feats = []
    times = c["times"]
    ens = c["ens"]
    for m, trs in enumerate(member_tracks):
        for k, tr in enumerate(trs):
            coords = [[o["center_lon"], o["center_lat"]] for _, o in tr]
            leads = [int(6 * t) for t, _ in tr]
            if c["hazard"] == "tropical_cyclone":
                pk = min(float(ens["msl"][m, t][np.abs(LAT12 - o["center_lat"]).argmin(),
                                                np.abs(LON12 - o["center_lon"]).argmin()])
                         for t, o in tr) / 100.0
                peak = {"min_msl_hpa": round(pk, 1)}
            else:
                peak = {"max_event_prob": round(max(o.get("p", 0) for _, o in tr), 3)}
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                          "properties": {"kind": "member_track", "member": m, "track": k,
                                         "hazard": c["hazard"], "leads_h": leads, **peak,
                                         "main": tr is gnn_main_track(trs)}})
    if cn:
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString", "coordinates": [[x[2], x[1]] for x in cn]},
                      "properties": {"kind": "consensus", "hazard": c["hazard"],
                                     "leads_h": [int(6 * x[0]) for x in cn],
                                     "spread_km": [round(x[3], 1) for x in cn],
                                     "member_count": [int(x[4]) for x in cn],
                                     "n_members": len(member_tracks)}})
    # 4-D box of the consensus event: union of member main-track objects at leads where at
    # least half the members have one
    need = max(1, len(member_tracks) // 2)
    per_lead = {}
    for trs in member_tracks:
        mm = gnn_main_track(trs)
        for t, o in mm or []:
            per_lead.setdefault(t, []).append(o)
    leads = sorted(t for t, v in per_lead.items() if len(v) >= need)
    boxes = []
    if leads:
        m = np.zeros((333, 333), bool)
        for t in leads:
            for o in per_lead[t]:
                m |= o.get("core_mask", o["mask"])
        ii, jj = np.nonzero(m)
        box = {"lat_min": float(LAT12[ii].min() - 0.06), "lat_max": float(LAT12[ii].max() + 0.06),
               "lon_min": float(LON12[jj].min() - 0.06), "lon_max": float(LON12[jj].max() + 0.06),
               "level_hPa_bottom": LEVELS[c["hazard"]][0], "level_hPa_top": LEVELS[c["hazard"]][1],
               "t_start": str(times[leads[0]]), "t_end": str(times[leads[-1]]),
               "lead_start_h": 6 * leads[0], "lead_end_h": 6 * leads[-1],
               "member_count_min": int(min(len(per_lead[t]) for t in leads)),
               "n_members": len(member_tracks), "hazard": c["hazard"], "source": "GNN consensus"}
        boxes.append(box)
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
            [box["lon_min"], box["lat_min"]], [box["lon_max"], box["lat_min"]],
            [box["lon_max"], box["lat_max"]], [box["lon_min"], box["lat_max"]],
            [box["lon_min"], box["lat_min"]]]]}, "properties": {"kind": "bbox4d", **box}})
    return {"type": "FeatureCollection", "features": feats}, boxes


def strike(c, member_tracks, t, LA, LO):
    P = np.zeros((333, 333))
    for trs in member_tracks:
        hit = np.zeros((333, 333), bool)
        for tr in trs:
            for tt, o in tr:
                if tt != t:
                    continue
                if c["hazard"] == "tropical_cyclone":
                    la, lo = o["center_lat"], o["center_lon"]
                    hit |= np.hypot((LA - la) * 111.2, (LO - lo) * 111.2 * np.cos(np.deg2rad(la))) <= 120
                else:
                    hit |= o["mask"]
        P += hit
    return P / len(member_tracks)


# ----------------------------------------------------------------------------- alerts
def circle(lat, lon, r_km=5.0, n=32):
    a = np.linspace(0, 2 * np.pi, n + 1)
    return [[round(lon + r_km / (111.2 * np.cos(np.deg2rad(lat))) * np.cos(x), 5),
             round(lat + r_km / 111.2 * np.sin(x), 5)] for x in a]


def compute_alerts(c, clim, reg, india, ds5_intensity):
    """Alerts for every lead; ds5_intensity[kind][t] is the member-0 5 km field used to put the
    pinpoint (max intensity) inside each 12 km alert region."""
    e, times = c["ens"], c["times"]
    alerts = []
    kinds = HAZ_KIND[c["hazard"]]
    member_fields = {}
    if c["hazard"] in ("heat_dome", "cold_wave"):
        kind = kinds[0]
        k = "max" if kind == "heat" else "min"
        normals = cached_normals(clim, times, k)
        dep, valid = [], []
        for m in range(e["t2m"].shape[0]):
            ext = trailing(e["t2m"][m].astype(np.float64), k)
            d = ext - normals
            absT = ext - 273.15
            if kind == "heat":
                ok = ((reg == 1) & (absT >= 40)) | ((reg == 2) & (absT >= 37)) | ((reg == 3) & (absT >= 30)) \
                    | ((reg > 0) & (absT >= 45))
            else:
                ok = ((reg == 1) & (absT <= 10)) | ((reg == 2) & (absT <= 15)) | ((reg == 3) & (absT <= 0)) \
                    | ((reg == 1) & (absT <= 4))
            dep.append(d.astype(np.float32))
            valid.append(ok & (reg > 0))
        member_fields[kind] = (np.array(dep), np.array(valid))
    for t in range(len(times)):
        for kind in kinds:
            if kind == "wind":
                f, v = np.hypot(e["u10"][:, t], e["v10"][:, t]) * 3.6, None
            elif kind == "rain":
                if t < 3:                        # need a complete 24 h window (4 x 6 h)
                    continue
                f, v = e["tp"][:, t - 3:t + 1].sum(1), None
            else:
                f, v = member_fields[kind][0][:, t], member_fields[kind][1][:, t]
            p1, p2, p3 = probabilities(kind, f, v)
            q1, q2, q3 = probabilities(kind, f, v, neighbourhood=False)
            cat = categorise(p1, p2, p3)
            lab, n = ndimage.label(cat >= 1, structure=np.ones((3, 3)))
            for r in range(1, n + 1):
                reg_m = lab == r
                if reg_m.sum() < 3:
                    continue
                level_i = int(cat[reg_m].max())
                level = CATEGORIES[level_i]
                # cited probability: the one that triggered the level
                if level == "severe" or (level == "moderate" and p3[reg_m].max() >= 0.2):
                    pc, qc, which = p3, q3, "severe"
                elif level in ("moderate",) or (level == "low" and p2[reg_m].max() >= 0.2):
                    pc, qc, which = p2, q2, "moderate"
                else:
                    pc, qc, which = p1, q1, "low"
                pmax = float(pc[reg_m].max())
                pcell = float(qc[reg_m].max())
                ii, jj = np.nonzero(reg_m)
                # pinpoint: 5 km cell of max member-0 downscaled intensity inside the region
                f5 = ds5_intensity[kind][t]
                m5 = np.kron(reg_m, np.ones((3, 3), bool))
                if kind == "cold":
                    k5 = np.argmin(np.where(m5, f5, np.inf))
                else:
                    k5 = np.argmax(np.where(m5, f5, -np.inf))
                pi5, pj5 = np.unravel_index(k5, f5.shape)
                plat, plon = float(LAT5[pi5]), float(LON5[pj5])
                alerts.append({
                    "id": f"{c['case']}-{kind}-{6 * t:03d}-{r}", "case": c["case"], "hazard": c["hazard"],
                    "kind": kind, "lead_h": 6 * t, "valid_time": str(times[t]), "category": level,
                    "colour": COLOURS[level], "probability": round(pmax, 3),
                    "probability_type": f"neighbourhood ({NBHD_KM:g} km)",
                    "probability_cell": round(pcell, 3),
                    "reason": reason(kind, which, pmax) + f" within {NBHD_KM:g} km",
                    "pinpoint": {"lat": round(plat, 3), "lon": round(plon, 3), "grid": "G5 0.04 deg"},
                    "impact_polygon": {"type": "Polygon", "coordinates": [circle(plat, plon)]},
                    "impact_radius_km": 5.0,
                    "region_bbox": [float(LON12[jj].min()), float(LAT12[ii].min()),
                                    float(LON12[jj].max()), float(LAT12[ii].max())],
                    "region_cells_12km": int(reg_m.sum()),
                    "in_india": bool((reg_m & india).any()),
                    "n_members": int(f.shape[0]), "synthetic": c["synthetic"]})
    return alerts


# ----------------------------------------------------------------------------- main
def export_case(case, clim, ds, india, reg, dec):
    t0 = time.time()
    c = load(case)
    out = OUT / case
    out.mkdir(parents=True, exist_ok=True)
    real = not c["synthetic"]
    variant = "temporal" if real else "full"
    model, st, _ = load_model(variant)
    g = load_graph("amphan_era5_real" if real else case)
    pn, pe = infer(model, st, g)
    dp = dec[variant][c["hazard"]]
    M = c["ens"]["msl"].shape[0]
    member_tracks = [decode(g, pn, pe, m, c["hazard"], **dp) for m in range(M)]
    core_masks(c, member_tracks, clim)
    cn = cone({"lat": LAT12, "lon": LON12}, member_tracks)
    gj, boxes = tracks_geojson(c, member_tracks, cn)
    (out / "tracks.geojson").write_text(json.dumps(gj))
    LA, LO = np.meshgrid(LAT12, LON12, indexing="ij")
    T = len(c["times"])
    var_hz = "msl" if c["hazard"] == "tropical_cyclone" else "t2m"
    ds5 = {"wind": [], "rain": [], "heat": [], "cold": []}
    tp5_hist = []
    for t in range(T):
        write_png(out / "strike_prob" / f"{6 * t:03d}.png", strike(c, member_tracks, t, LA, LO), "strike")
        e0 = {v: c["ens"][v][0, t].astype(np.float64) for v in ("t2m", "u10", "v10", "msl", "tp")}
        f5 = ds(**e0)
        # 5 km PNGs every 12 h to stay under the 500 MB budget (12 km: every 6 h)
        for res, f in (("12km", e0), ("5km", f5)) if (6 * t) % 12 == 0 else (("12km", e0),):
            write_png(out / f"fields_{res}" / "t2m" / f"{6 * t:03d}.png", f["t2m"] - 273.15, "t2m")
            write_png(out / f"fields_{res}" / "wind" / f"{6 * t:03d}.png", np.hypot(f["u10"], f["v10"]), "wind")
            write_png(out / f"fields_{res}" / "msl" / f"{6 * t:03d}.png", f["msl"] / 100.0, "msl")
            write_png(out / f"fields_{res}" / "tp" / f"{6 * t:03d}.png", f["tp"], "tp")
        write_png(out / "fields_12km" / "anom" / f"{6 * t:03d}.png",
                  clim.z(var_hz, e0[var_hz], c["times"][t]), "anom")
        ds5["wind"].append((np.hypot(f5["u10"], f5["v10"]) * 3.6).astype(np.float32))
        tp5_hist.append(f5["tp"].astype(np.float32))
        ds5["rain"].append(np.sum(tp5_hist[t - 3:t + 1], 0) if t >= 3 else None)
        ds5["heat"].append(f5["t2m"].astype(np.float32))
        ds5["cold"].append(f5["t2m"].astype(np.float32))
        if t >= 4:
            tp5_hist[t - 4] = None
    alerts = compute_alerts(c, clim, reg, india, ds5)
    (out / "alerts.json").write_text(json.dumps({"case": case, "synthetic": c["synthetic"],
                                                 "rules": "docs/ALERT_RULES.md", "alerts": alerts}))
    meta = build_meta(c, boxes, variant, dp, time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"{case}: {len(alerts)} alerts, {sum(len(x) for x in member_tracks)} tracks, "
          f"{size:.1f} MB, {time.time() - t0:.0f}s", flush=True)
    return size


def build_meta(c, boxes, variant, dp, seconds):
    gr = json.loads((ROOT / "reports/gnn_results.json").read_text())
    ds_path = ROOT / "reports/downscaling_results.json"
    case = c["case"]
    metrics = {}
    if case in gr["cases"]:
        metrics["tracking"] = {k: {kk: v[kk] for kk in ("iou", "csi", "pod", "far", "spurious_tracks",
                                                         "centroid_err_km", "detect_lag_h")}
                               for k, v in gr["cases"][case].items() if isinstance(v, dict)}
        metrics["split"] = gr["cases"][case]["split"]
    if case == "amphan_era5_real":
        metrics["amphan_track_error_vs_ibtracs"] = {k: {kk: v[kk] for kk in ("n_matched", "err_mean_km",
                                                                             "err_median_km")}
                                                    for k, v in gr["amphan_real"].items()}
    metrics["downscaling"] = (json.loads(ds_path.read_text())["metrics"] if ds_path.exists()
                              else "pending: reports/downscaling_results.json not produced yet")
    return {"case": case, "hazard": c["hazard"], "synthetic": c["synthetic"],
            "badge": "SYNTHETIC" if c["synthetic"] else "REAL (ERA5 reanalysis)",
            "source": c["source"], "init_time": str(c["init"]),
            "valid_times": [str(t) for t in c["times"]], "leads_h": [6 * i for i in range(len(c["times"]))],
            "n_members": int(c["ens"]["msl"].shape[0]),
            "bounds": BOUNDS, "grid_12km": "G12 0.12 deg 333x333", "grid_5km": "G5 0.04 deg 999x999",
            "leads_5km_h": [6 * i for i in range(len(c["times"])) if (6 * i) % 12 == 0],
            "fields_member": "member 0" if c["synthetic"] else "ERA5 (single run)",
            "downscaler": "U-Net + exact conservation projection (models/downscale/unet)",
            "tracker": f"GNN {variant} (models/tracker/{variant}), decode {dp}",
            "legends": {v: legend(v) for v in SCALES}, "bbox4d": boxes,
            "label_bbox4d": c["labels"]["bbox_4d"] if c["labels"] else None,
            "metrics": metrics, "export_seconds": round(seconds, 1)}


def main():
    cases = sys.argv[1:] or ALL + ["amphan_era5_real"]
    clim = Climatology()
    ds = Downscaler()
    orog = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float64)
    reg = regions(orog, LAT12, LON12)
    india = xr.open_dataset(ROOT / "data/real/boundary/india_mask_g12.nc").india.values.astype(bool)
    dec = json.loads((ROOT / "reports/gnn_results.json").read_text())["decode"]
    total = 0.0
    for c in cases:
        total += export_case(c, clim, ds, india, reg, dec)
    idx = [json.loads((OUT / c / "meta.json").read_text()) for c in sorted(p.name for p in OUT.iterdir()
                                                                             if (p / "meta.json").exists())]
    (OUT / "index.json").write_text(json.dumps([{k: m[k] for k in ("case", "hazard", "synthetic", "badge",
                                                                  "init_time")}
                                               | {"start": m["valid_times"][0], "end": m["valid_times"][-1]}
                                               for m in idx], indent=1))
    print(f"total exported this run: {total:.1f} MB")


if __name__ == "__main__":
    main()
