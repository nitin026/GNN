"""Candidate-object graphs for the GNN tracker (BRIEF2 Phase 2).

For each case:
  nodes  = candidate anomaly objects per member per lead (permissive hysteresis detection)
  edges  = temporal (same member, t -> t+1 and t -> t+2) and cross-member (same lead)
           pairs within a distance gate, stored once (i < j); the model treats them as undirected
  labels = node: object belongs to the member's own injected event (exact mask);
           edge: both ends are event nodes and their masks overlap / are close
Also stored: y_truth (overlap with the TRUTH mask; for verification only), per-node cell
indices (for IoU), and an "analysis" graph built on the 12 km truth itself (member -1, no
cross-member edges) for truth-run scoring.

    python -m pipeline.graphs [case ...]   -> data/graphs/{case}.npz

Candidate fields (larger = more extreme):
  cyclone    : -z(MSLP), 25 km smoothed (same as tracker v2); high 1.5 / low 1.2 / >= 2000 km2
  heat, cold : daily T departure from the ERA5 normal (centred 24 h Tmax / Tmin), inside
               India + 250 km (Natural Earth mask), excluding > 2500 m; high 3.0 / low 2.0 C,
               >= 10000 km2. Whether IMD criteria hold is a node FEATURE, not a filter.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from .anomaly import Climatology
from .efi import efi_gaussian
from .tracker2 import (cached_normals, cell_area_km2, closed_contour, detect_hysteresis, regions,
                       trailing)

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
OUT = ROOT / "data" / "graphs"
KM = 111.195
HAZARDS = ["tropical_cyclone", "heat_dome", "cold_wave"]
CAND = {"tropical_cyclone": dict(high=1.5, low=1.2, min_area_km2=2000.0, t_gate=600.0, m_gate=400.0),
        "heat_dome": dict(high=3.0, low=2.0, min_area_km2=10000.0, t_gate=500.0, m_gate=500.0),
        "cold_wave": dict(high=3.0, low=2.0, min_area_km2=10000.0, t_gate=500.0, m_gate=500.0)}
NODE_FEATURES = ["lead", "log_area", "peak", "mean", "lat", "lon", "sig_major", "sig_minor",
                 "cos2th", "sin2th", "frac_land", "frac_crit", "aux_max", "efi", "core",
                 "haz_tc", "haz_heat", "haz_cold"]
EDGE_FEATURES = ["dist", "dpeak", "dlogarea", "dt", "overlap", "is_temporal"]


def static(lat, lon):
    orog = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float64)
    india = xr.open_dataset(ROOT / "data/real/boundary/india_mask_g12.nc").india.values.astype(bool)
    india = binary_dilation(india, iterations=1)                      # coastal cells
    dkm = distance_transform_edt(~india) * (lat[1] - lat[0]) * KM
    domain = (dkm <= 250.0) & (orog <= 2500.0) & (orog > 1.0)
    return orog, india, domain


def hazard_field(hz, series, times, clim, domain, smooth=2.0):
    """Candidate field and an auxiliary field used for node features."""
    if hz == "tropical_cyclone":
        f = np.array([-gaussian_filter(clim.z("msl", series[i], t), smooth)
                      for i, t in enumerate(times)])
        return f, series                                             # aux = MSLP (Pa)
    kind = "max" if hz == "heat_dome" else "min"
    ext = trailing(series, kind)
    dep = ext - cached_normals(clim, times, kind)
    f = dep if hz == "heat_dome" else -dep
    f = np.where(domain, f, 0.0)
    f = np.array([gaussian_filter(x, 1.0) for x in f])
    return f, ext - 273.15                                            # aux = Tmax/Tmin (C)


def crit_mask(hz, dep, absT, reg):
    """IMD criteria per cell (see tracker2.imd_field): dep is 'larger = more extreme'."""
    if hz == "heat_dome":
        ok = ((reg == 1) & (absT >= 40)) | ((reg == 2) & (absT >= 37)) | ((reg == 3) & (absT >= 30)) \
            | ((reg > 0) & (absT >= 45))
    else:
        ok = ((reg == 1) & (absT <= 10)) | ((reg == 2) & (absT <= 15)) | ((reg == 3) & (absT <= 0)) \
            | ((reg == 1) & (absT <= 4))
    return ok & (dep >= 4.5)


def shape_moments(ii, jj, lat, lon, w):
    y = lat[ii] * KM
    x = lon[jj] * KM * np.cos(np.deg2rad(lat[ii]))
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cxx = (w * (x - mx) ** 2).sum()
    cyy = (w * (y - my) ** 2).sum()
    cxy = (w * (x - mx) * (y - my)).sum()
    ev, evec = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]) + 1e-6)
    th = np.arctan2(evec[1, 1], evec[0, 1])
    return np.sqrt(max(ev[1], 0)) / 100.0, np.sqrt(max(ev[0], 0)) / 100.0, np.cos(2 * th), np.sin(2 * th)


def objects_for_run(hz, f, aux, lat, lon, area, orog, reg, efi, emask=None, tmask=None):
    p = CAND[hz]
    rows, cells = [], []
    land = orog > 1.0
    for t in range(len(f)):
        objs = detect_hysteresis(f[t], lat, lon, p["high"], p["low"], p["min_area_km2"], area)
        for o in objs:
            m = o["mask"]
            ii, jj = np.nonzero(m)
            vals = f[t][ii, jj]
            s1, s2, c2, s2t = shape_moments(ii, jj, lat, lon, np.maximum(vals - p["low"], 1e-3))
            if hz == "tropical_cyclone":
                k = int(np.argmin(aux[t][ii, jj]))
                ci, cj = ii[k], jj[k]
                frac_crit = float(closed_contour(aux[t], ci, cj, lat, lon, 200.0, 5.5))
                aux_max = float((np.median(aux[t]) - aux[t][ci, cj]) / 100.0)      # depth, hPa
            else:
                ci = int(np.round(np.average(ii, weights=vals)))
                cj = int(np.round(np.average(jj, weights=vals)))
                frac_crit = float(crit_mask(hz, vals, aux[t][ii, jj], reg[ii, jj]).mean())
                aux_max = float(aux[t][ii, jj].max() if hz == "heat_dome" else aux[t][ii, jj].min())
            n = len(ii)
            ev = int((m & emask[t]).sum()) if emask is not None else 0
            y = float(emask is not None and (ev >= 0.25 * n or ev >= 0.5 * max(emask[t].sum(), 1)))
            tv = int((m & tmask[t]).sum()) if tmask is not None else 0
            yt = float(tmask is not None and (tv >= 0.25 * n or tv >= 0.5 * max(tmask[t].sum(), 1)))
            rows.append({"t": t, "lat": float(lat[ci]), "lon": float(lon[cj]),
                         "centroid_lat": o["centroid_lat"], "centroid_lon": o["centroid_lon"],
                         "area": o["area_km2"], "peak": float(vals.max()), "mean": float(vals.mean()),
                         "s1": s1, "s2": s2, "c2": c2, "s2t": s2t, "frac_land": float(land[ii, jj].mean()),
                         "frac_crit": frac_crit, "aux_max": aux_max,
                         "efi": float(efi[t][ci, cj]) if efi is not None else 0.0,
                         "core": float(o["core"]), "y": y, "y_truth": yt})
            cells.append((ii * len(lon) + jj).astype(np.int32))
    return rows, cells


def feat_matrix(rows, hz):
    oh = [float(hz == h) for h in HAZARDS]
    X = [[r["t"] / 40.0, np.log10(r["area"]), r["peak"], r["mean"], (r["lat"] - 20) / 10,
          (r["lon"] - 80) / 10, r["s1"], r["s2"], r["c2"], r["s2t"], r["frac_land"], r["frac_crit"],
          r["aux_max"] / 10.0, r["efi"], r["core"], *oh] for r in rows]
    return np.array(X, np.float32).reshape(len(rows), len(NODE_FEATURES))


def hav(la1, lo1, la2, lo2):
    p1, p2 = np.deg2rad(la1), np.deg2rad(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_edges(meta, cells, hz, cross_member=True):
    p = CAND[hz]
    member, t = meta["member"], meta["t"]
    la, lo = meta["centroid_lat"], meta["centroid_lon"]
    cellset = [set(c.tolist()) for c in cells]
    E, F = [], []
    order = np.lexsort((t, member))
    by_mt = {}
    for i in order:
        by_mt.setdefault((member[i], t[i]), []).append(i)
    lead_nodes = {}
    for i in range(len(t)):
        lead_nodes.setdefault(t[i], []).append(i)

    def add(i, j, temporal):
        d = hav(la[i], lo[i], la[j], lo[j])
        gate = p["t_gate"] * (t[j] - t[i]) if temporal else p["m_gate"]
        if d > gate:
            return
        a, b = cellset[i], cellset[j]
        ov = len(a & b) / max(min(len(a), len(b)), 1) if (a and b) else 0.0
        E.append((i, j))
        F.append([d / 500.0, meta["peak"][j] - meta["peak"][i],
                  np.log10(meta["area"][j]) - np.log10(meta["area"][i]),
                  float(t[j] - t[i]), ov, float(temporal)])

    for (m, tt), nodes in by_mt.items():
        for dt in (1, 2):
            for i in nodes:
                for j in by_mt.get((m, tt + dt), []):
                    add(i, j, True)
    if cross_member:
        for tt, nodes in lead_nodes.items():
            for a_ in range(len(nodes)):
                for b_ in range(a_ + 1, len(nodes)):
                    i, j = nodes[a_], nodes[b_]
                    if member[i] != member[j] and member[i] >= 0 and member[j] >= 0:
                        add(i, j, False)
    E = np.array(E, np.int64).reshape(-1, 2)
    F = np.array(F, np.float32).reshape(-1, len(EDGE_FEATURES))
    return E, F


def edge_labels(E, meta):
    y = meta["y"]
    return (y[E[:, 0]] * y[E[:, 1]]).astype(np.float32)


def build_graph(hz, ens, times, lat, lon, clim, truth=None, emask=None, tmask=None):
    """Graph arrays for an ensemble ens (M, T, y, x) [+ optional truth run as member -1]."""
    orog, india, domain = static(lat, lon)
    reg = regions(orog, lat, lon)
    area = cell_area_km2(lat, lon)
    var = "msl" if hz == "tropical_cyclone" else "t2m"
    sign = -1.0 if hz in ("tropical_cyclone", "cold_wave") else 1.0
    efi = np.array([sign * efi_gaussian(ens[:, t], *clim.get(var, times[t])) for t in range(len(times))])
    all_rows, all_cells, member = [], [], []
    runs = list(range(ens.shape[0])) + ([-1] if truth is not None else [])
    for m in runs:
        series = ens[m] if m >= 0 else truth
        f, aux = hazard_field(hz, series, times, clim, domain)
        own = None if emask is None else (emask[m] if m >= 0 else tmask)
        rows, cells = objects_for_run(hz, f, aux, lat, lon, area, orog, reg, efi,
                                      emask=own, tmask=tmask)
        all_rows += rows
        all_cells += cells
        member += [m] * len(rows)
    meta = {k: np.array([r[k] for r in all_rows]) for k in all_rows[0]}
    meta["member"] = np.array(member)
    X = feat_matrix(all_rows, hz)
    E, EF = build_edges(meta, all_cells, hz)
    ye = edge_labels(E, meta)
    offs = np.cumsum([0] + [len(c) for c in all_cells]).astype(np.int64)
    return dict(X=X, E=E, EF=EF, y_node=meta["y"].astype(np.float32),
                y_truth=meta["y_truth"].astype(np.float32), y_edge=ye,
                member=meta["member"].astype(np.int16), t=meta["t"].astype(np.int16),
                lat=meta["lat"], lon=meta["lon"], clat=meta["centroid_lat"],
                clon=meta["centroid_lon"], area=meta["area"], cells=np.concatenate(all_cells),
                cell_offsets=offs, hazard=hz, node_features=np.array(NODE_FEATURES),
                edge_features=np.array(EDGE_FEATURES))


def build_case(case, clim=None):
    clim = clim or Climatology()
    lab = json.loads((SYN / case / "labels.json").read_text())
    hz = lab["hazard"]
    ds = xr.open_dataset(SYN / case / "fcst_12km.nc", decode_timedelta=False)
    lat, lon = ds.latitude.values, ds.longitude.values
    times = pd.to_datetime(lab["valid_times"])
    var = "msl" if hz == "tropical_cyclone" else "t2m"
    ens = ds[var].values.astype(np.float64)
    truth = ds[f"{var}_truth"].values.astype(np.float64)
    emask = ds["event_mask"].values.astype(bool)
    tmask = ds["event_mask_truth"].values.astype(bool)
    ds.close()
    g = build_graph(hz, ens, times, lat, lon, clim, truth=truth, emask=emask, tmask=tmask)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{case}.npz", **g, case=case, synthetic="true")
    return {"case": case, "nodes": len(g["X"]), "edges": len(g["E"]),
            "pos_nodes": int(g["y_node"].sum()), "pos_edges": int(g["y_edge"].sum()),
            "temporal_edges": int(g["EF"][:, -1].sum()), "cross_edges": int((g["EF"][:, -1] == 0).sum())}


def build_amphan_real(clim=None):
    """Single-member graph of REAL ERA5 Amphan MSLP on G12 (no labels)."""
    clim = clim or Climatology()
    ds = xr.open_dataset(ROOT / "data/real/era5/amphan_era5_g12.nc")
    msl = ds.msl.values.astype(np.float64)[None]
    times = pd.to_datetime(ds.time.values)
    g = build_graph("tropical_cyclone", msl, times, ds.latitude.values, ds.longitude.values, clim)
    ds.close()
    np.savez_compressed(OUT / "amphan_era5_real.npz", **g, case="amphan_era5_real", synthetic="false",
                        times=np.array([str(t) for t in times]))
    return {"case": "amphan_era5_real", "nodes": len(g["X"]), "edges": len(g["E"])}


if __name__ == "__main__":
    from .splits import ALL
    clim = Climatology()
    for c in sys.argv[1:] or ALL + ["amphan_era5_real"]:
        print(build_amphan_real(clim) if c == "amphan_era5_real" else build_case(c, clim), flush=True)
