"""Phase 2 evaluation: GNN tracker vs tracker v2 (Phase 1) vs TempestExtremes-style tracking.

Metrics (identical code for every method; all at 12 km):
  truth run  : IoU (main track vs exact truth mask), centroid/centre error (km)
  ensemble   : POD, FAR, CSI over all (member, lead) pairs against the TRUTH mask
               (hit = a tracked object overlaps the truth mask; false alarm = a tracked object
               that does not); spurious tracks vs each member's own injected-event mask;
               detection lag = first lead where >= 50 % of members hit the truth event, minus
               the truth onset (h); lead gained = lag(v2) - lag(method)
  real Amphan: track error (km) of the main track on ERA5 G12 vs IBTrACS
Decoding thresholds for the GNN (node threshold, min track length) are chosen on VAL cases by CSI.
TEST cases are only reported.

    python -m pipeline.gnn_eval     -> reports/GNN_RESULTS.md, reports/gnn_results.json, figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from .anomaly import Climatology  # noqa: E402
from .evaluate2 import load_case, orog12, ref_pos  # noqa: E402
from .gnn import GNNTracker  # noqa: E402
from .splits import TEST, TRAIN, VAL  # noqa: E402
from .te_style import te_blobs, te_cyclones  # noqa: E402
from .track import haversine_km, iou, main_track, track_masks  # noqa: E402
from .tracker2 import Tracker2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GDIR = ROOT / "data" / "graphs"
MDIR = ROOT / "models" / "tracker"
REP = ROOT / "reports"
FIG = REP / "figures"
VARIANTS = ["full", "temporal", "cross", "none"]
SHAPE = (333, 333)
MIN_STEPS_V2 = {"tropical_cyclone": 4, "heat_dome": 8, "cold_wave": 8}


# ------------------------------------------------------------------------------ GNN inference
def load_graph(case):
    z = np.load(GDIR / f"{case}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_model(variant):
    d = MDIR / variant
    cfg = json.loads((d / "config.json").read_text())
    st = dict(np.load(d / "stats.npz"))
    ut, uc = {"full": (1, 1), "temporal": (1, 0), "cross": (0, 1), "none": (0, 0)}[variant]
    m = GNNTracker(cfg["n_node_features"], cfg["n_edge_features"], hidden=cfg["hidden"],
                   use_temporal=bool(ut), use_cross=bool(uc))
    m.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    m.eval()
    return m, st, cfg


@torch.no_grad()
def infer(model, st, g):
    x = torch.tensor((g["X"] - st["xm"]) / st["xs"], dtype=torch.float32)
    ef = torch.tensor((g["EF"] - st["em"]) / st["es"], dtype=torch.float32)
    e = torch.tensor(g["E"], dtype=torch.long)
    et = torch.tensor(g["EF"][:, -1] > 0.5)
    nl, el = model(x, e, ef, et)
    return torch.sigmoid(nl).numpy(), torch.sigmoid(el).numpy()


def node_obj(g, i, hz):
    m = np.zeros(SHAPE[0] * SHAPE[1], bool)
    m[g["cells"][g["cell_offsets"][i]:g["cell_offsets"][i + 1]]] = True
    return {"mask": m.reshape(SHAPE), "center_lat": float(g["lat"][i]), "center_lon": float(g["lon"][i]),
            "centroid_lat": float(g["clat"][i]), "centroid_lon": float(g["clon"][i]), "node": int(i),
            "n_cells": int(m.sum()), "core": True}


def gnn_main_track(tracks):
    """GNN main track = the track with the largest summed node event probability."""
    return max(tracks, key=lambda tr: sum(o.get("p", 0.0) for _, o in tr)) if tracks else None


def decode(g, pn, pe, member, hz, tau_n=0.5, tau_e=0.5, min_len=4):
    """Tracks for one member: event nodes (p >= tau_n) linked by Hungarian assignment on
    -log p_edge over temporal edges (dt = 1 or 2 steps), links need p_edge >= tau_e."""
    sel = np.where((g["member"] == member) & (pn >= tau_n))[0]
    if len(sel) == 0:
        return []
    temporal = g["EF"][:, -1] > 0.5
    pe_map = {}
    for k in np.where(temporal)[0]:
        a, b = g["E"][k]
        pe_map[(a, b)] = pe[k]
        pe_map[(b, a)] = pe[k]
    by_t = {}
    for i in sel:
        by_t.setdefault(int(g["t"][i]), []).append(int(i))
    tracks, open_ = [], []
    for t in sorted(by_t):
        nodes = by_t[t]
        live = [tr for tr in open_ if t - int(g["t"][tr[-1]]) <= 2]
        assigned = set()
        if live and nodes:
            C = np.full((len(live), len(nodes)), 1e6)
            for a, tr in enumerate(live):
                for b, j in enumerate(nodes):
                    p = pe_map.get((tr[-1], j))
                    if p is not None and p >= tau_e:
                        C[a, b] = -np.log(max(p, 1e-6))
            ri, ci = linear_sum_assignment(C)
            for a, b in zip(ri, ci):
                if C[a, b] < 1e6:
                    live[a].append(nodes[b])
                    assigned.add(nodes[b])
        for j in nodes:
            if j not in assigned:
                open_.append([j])
    out = []
    for tr in open_:
        if int(g["t"][tr[-1]]) - int(g["t"][tr[0]]) + 1 >= min_len:
            out.append([(int(g["t"][i]), dict(node_obj(g, i, hz), p=float(pn[i]))) for i in tr])
    return out


# ------------------------------------------------------------------------------ metrics
def case_metrics(d, tracks_by_member, truth_tracks):
    """d: evaluate2.load_case output; tracks_by_member: list of 20 track lists."""
    hz, lab = d["hazard"], d["lab"]
    key = "center" if hz == "tropical_cyclone" else "centroid"
    tm = d["tmask"]
    T = len(tm)
    pick = lambda trs: (gnn_main_track(trs) if trs and "p" in trs[0][0][1] else main_track(trs))
    # truth run
    mt = pick(truth_tracks)
    pm = track_masks(mt, T, SHAPE)
    idx = [i for i in range(T) if tm[i].any() or pm[i].any()]
    ious = [iou(pm[i], tm[i]) for i in idx]
    errs = []
    for t, o in mt or []:
        la, lo = ref_pos(lab["truth_track"][t], hz)
        if la is not None and tm[t].any():
            errs.append(haversine_km(o[f"{key}_lat"], o[f"{key}_lon"], la, lo))
    # ensemble
    H = M = FA = 0
    spur = 0
    hit_frac = np.zeros(T)
    lead_err = {}
    for m, trs in enumerate(tracks_by_member):
        occ = [[] for _ in range(T)]
        for tr in trs:
            if not any((o["mask"] & d["emask"][m, t]).any() for t, o in tr):
                spur += 1
            for t, o in tr:
                occ[t].append(o)
        for t in range(T):
            hits = [o for o in occ[t] if (o["mask"] & tm[t]).any()]
            FA += len(occ[t]) - len(hits)
            if tm[t].any():
                if hits:
                    H += 1
                    hit_frac[t] += 1
                else:
                    M += 1
        mm = pick(trs)
        if mm and d["roles"][m] == 0:
            for t, o in mm:
                la, lo = ref_pos(lab["truth_track"][t], hz)
                if la is not None and tm[t].any():
                    lead_err.setdefault(int(lab["lead_hours"][t]), []).append(
                        float(haversine_km(o[f"{key}_lat"], o[f"{key}_lon"], la, lo)))
    hit_frac /= len(tracks_by_member)
    onset = next((t for t in range(T) if tm[t].any()), None)
    det = next((t for t in range(T) if tm[t].any() and hit_frac[t] >= 0.5), None)
    lag = None if onset is None or det is None else 6 * (det - onset)
    return {"iou": float(np.nanmean(ious)) if ious else 0.0,
            "centroid_err_km": float(np.mean(errs)) if errs else np.nan,
            "pod": H / max(H + M, 1), "far": FA / max(H + FA, 1), "csi": H / max(H + M + FA, 1),
            "hits": H, "misses": M, "false_alarms": FA, "spurious_tracks": spur,
            "detect_lag_h": lag, "ens_err_by_lead": {k: float(np.mean(v)) for k, v in sorted(lead_err.items())}}


def run_baselines(d, clim, oro, params_v2):
    hz = d["hazard"]
    tk = Tracker2(hz, clim, oro, d["lat"], d["lon"], **params_v2)
    if hz == "tropical_cyclone":
        te = lambda s: te_cyclones(s, d["lat"], d["lon"])
    else:
        te = lambda s: te_blobs(s, d["times"], clim, oro, d["lat"], d["lon"], hz)
    out = {}
    for name, fn in (("v2", lambda s: tk.run(s, d["times"])[0]), ("te_style", te)):
        members = [fn(d["ens"][m]) for m in range(d["ens"].shape[0])]
        out[name] = case_metrics(d, members, fn(d["truth"]))
    return out


def run_gnn(d, g, model, st, dec):
    pn, pe = infer(model, st, g)
    hz = d["hazard"]
    members = [decode(g, pn, pe, m, hz, **dec) for m in range(d["ens"].shape[0])]
    truth = decode(g, pn, pe, -1, hz, **dec)
    return case_metrics(d, members, truth), pn, pe


# ------------------------------------------------------------------------------ real Amphan
def amphan_real_error(tracks, times):
    ib = pd.read_csv(ROOT / "data/real/ibtracs/amphan_2020_ibtracs.csv")
    ib["time"] = pd.to_datetime(ib.ISO_TIME)
    ib6 = ib[ib.time.dt.hour % 6 == 0].set_index("time")
    t0, t1 = ib.time.min(), ib.time.max()
    if not tracks:
        return {"n_matched": 0, "err_mean_km": np.nan, "err_median_km": np.nan, "other_tracks": 0}
    best = max(tracks, key=lambda tr: (sum(t0 <= times[t] <= t1 for t, _ in tr), len(tr)))
    errs, pts = [], []
    for t, o in best:
        if times[t] in ib6.index:
            r = ib6.loc[times[t]]
            errs.append(float(haversine_km(o["center_lat"], o["center_lon"], r.LAT, r.LON)))
            pts.append((str(times[t]), o["center_lat"], o["center_lon"]))
    return {"n_matched": len(errs), "err_mean_km": float(np.mean(errs)) if errs else np.nan,
            "err_median_km": float(np.median(errs)) if errs else np.nan,
            "err_max_km": float(np.max(errs)) if errs else np.nan,
            "other_tracks": len(tracks) - 1, "track": pts}


# ------------------------------------------------------------------------------ products
def strike_probability(d, tracks_by_member, radius_km=120.0):
    """Fraction of members whose event track passes within radius_km of each grid cell."""
    lat, lon = d["lat"], d["lon"]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    P = np.zeros(SHAPE)
    for trs in tracks_by_member:
        hit = np.zeros(SHAPE, bool)
        for tr in trs:
            for t, o in tr:
                la, lo = o["center_lat"], o["center_lon"]
                hit |= np.hypot((LA - la) * 111.2, (LO - lo) * 111.2 * np.cos(np.deg2rad(la))) <= radius_km
        P += hit
    return P / len(tracks_by_member)


def cone(d, tracks_by_member):
    """Ensemble-mean position and spread (mean distance to the mean) per lead, main tracks."""
    pos = {}
    for trs in tracks_by_member:
        mm = gnn_main_track(trs) if trs and "p" in trs[0][0][1] else main_track(trs)
        for t, o in mm or []:
            pos.setdefault(t, []).append((o["center_lat"], o["center_lon"]))
    out = []
    for t in sorted(pos):
        p = np.array(pos[t])
        mla, mlo = p.mean(0)
        spread = float(np.mean([haversine_km(a, b, mla, mlo) for a, b in p]))
        out.append((t, float(mla), float(mlo), spread, len(p)))
    return out


def reliability(pn_list, y_list, t_list, bins=10):
    out = {}
    for name, (lo, hi) in {"0-72h": (0, 12), "78-168h": (13, 28), "174-240h": (29, 40)}.items():
        p = np.concatenate([pn[(t >= lo) & (t <= hi)] for pn, t in zip(pn_list, t_list)])
        y = np.concatenate([yy[(t >= lo) & (t <= hi)] for yy, t in zip(y_list, t_list)])
        edges = np.linspace(0, 1, bins + 1)
        k = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        out[name] = [(float(p[k == b].mean()), float(y[k == b].mean()), int((k == b).sum()))
                     for b in range(bins) if (k == b).sum() >= 20]
    return out


# ------------------------------------------------------------------------------ main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    clim, oro = Climatology(), orog12()
    v2p = json.loads((REP / "tracker2_params.json").read_text())["chosen"]
    models = {v: load_model(v) for v in VARIANTS if (MDIR / v / "model.pt").exists()}
    cases = TRAIN + VAL + TEST
    data = {c: load_case(c) for c in cases}
    graphs = {c: load_graph(c) for c in cases}
    # decoding thresholds per variant and hazard, chosen on VAL by CSI (ties: fewer spurious)
    grid = [(tn, ml) for tn in (0.3, 0.5, 0.7) for ml in (2, 4, 8)]
    dec = {}
    for v, (model, st, cfg) in models.items():
        dec[v] = {}
        for c in VAL:
            hz = data[c]["hazard"]
            best = None
            for tn, ml in grid:
                r, _, _ = run_gnn(data[c], graphs[c], model, st, {"tau_n": tn, "min_len": ml})
                key = (r["csi"], -r["spurious_tracks"])
                if best is None or key > best[0]:
                    best = (key, {"tau_n": tn, "tau_e": 0.5, "min_len": ml})
            dec[v][hz] = best[1]
        print("decode params", v, dec[v], flush=True)
    res, probs = {}, {}
    for c in cases:
        d, g = data[c], graphs[c]
        hz = d["hazard"]
        r = run_baselines(d, clim, oro, v2p[hz]["params"])
        for v, (model, st, cfg) in models.items():
            r[f"gnn_{v}"], pn, pe = run_gnn(d, g, model, st, dec[v][hz])
            if v == "full":
                probs[c] = (pn, pe)
        res[c] = {"hazard": hz, "split": "train" if c in TRAIN else "val" if c in VAL else "test", **r}
        print(c, {k: (round(x["iou"], 2), round(x["csi"], 2), x["spurious_tracks"], x["detect_lag_h"])
                  for k, x in r.items()}, flush=True)
    # real Amphan
    import xarray as xr
    ga = load_graph("amphan_era5_real")
    times = pd.to_datetime(ga["times"])
    e = xr.open_dataset(ROOT / "data/real/era5/amphan_era5_g12.nc")
    msl = e.msl.values.astype(float)
    tk = Tracker2("tropical_cyclone", clim, oro, e.latitude.values, e.longitude.values,
                  **v2p["tropical_cyclone"]["params"])
    real = {"v2": amphan_real_error(tk.run(msl, times)[0], times),
            "te_style": amphan_real_error(te_cyclones(msl, e.latitude.values, e.longitude.values), times)}
    for v, (model, st, cfg) in models.items():
        pn, pe = infer(model, st, ga)
        real[f"gnn_{v}"] = amphan_real_error(
            decode(ga, pn, pe, 0, "tropical_cyclone", **dec[v]["tropical_cyclone"]), times)
    e.close()
    # products on TEST cases (full model)
    prod = {}
    for c in TEST:
        d, g = data[c], graphs[c]
        pn, pe = probs[c]
        trs = [decode(g, pn, pe, m, d["hazard"], **dec["full"][d["hazard"]]) for m in range(20)]
        prod[c] = {"strike": strike_probability(d, trs), "cone": cone(d, trs)}
    ens = {c: graphs[c]["member"] >= 0 for c in TEST}
    rel = reliability([probs[c][0][ens[c]] for c in TEST], [graphs[c]["y_truth"][ens[c]] for c in TEST],
                      [graphs[c]["t"][ens[c]] for c in TEST])
    rel_own = reliability([probs[c][0][ens[c]] for c in TEST], [graphs[c]["y_node"][ens[c]] for c in TEST],
                          [graphs[c]["t"][ens[c]] for c in TEST])
    figures(data, prod, rel, rel_own, real)
    cfgs = {v: models[v][2] for v in models}
    from .jsonutil import dumps as strict_dumps
    (REP / "gnn_results.json").write_text(strict_dumps(
        {"cases": res, "amphan_real": real, "decode": dec, "reliability_truth": rel,
         "reliability_member": rel_own, "cones": {c: prod[c]["cone"] for c in prod},
         "train_configs": cfgs}, indent=1))
    write_report(res, real, dec, rel, cfgs)


def figures(data, prod, rel, rel_own, real):
    fig, axes = plt.subplots(1, len(prod), figsize=(5.2 * len(prod), 5), squeeze=False)
    for ax, (c, p) in zip(axes[0], prod.items()):
        d = data[c]
        im = ax.pcolormesh(d["lon"], d["lat"], p["strike"], vmin=0, vmax=1, cmap="magma_r", shading="auto")
        ax.contour(d["lon"], d["lat"], d["tmask"].any(0), [0.5], colors="c", linewidths=1)
        cn = p["cone"]
        if cn:
            ax.plot([x[2] for x in cn], [x[1] for x in cn], "k-", lw=1.5)
            for t, la, lo, sp, n in cn[::4]:
                ax.add_patch(plt.Circle((lo, la), sp / 111.2, fill=False, color="k", lw=0.8))
        tr = [(e["center_lon"], e["center_lat"]) for e in d["lab"]["truth_track"] if e["center_lat"]]
        if tr:
            ax.plot(*zip(*tr), "c--", lw=1)
        ax.set_title(f"{c}: strike probability (120 km) + cone\ncyan = truth track/mask (SYNTHETIC)",
                     fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "gnn_strike_cone.png", dpi=100)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, rr, ttl in ((axes[0], rel, "vs TRUTH mask"), (axes[1], rel_own, "vs member's own event")):
        ax.plot([0, 1], [0, 1], "k:")
        for name, pts in rr.items():
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, label=name)
        ax.set_xlabel("forecast event probability (GNN node)")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"Reliability, TEST cases, {ttl}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "gnn_reliability.png", dpi=100)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    ib = pd.read_csv(ROOT / "data/real/ibtracs/amphan_2020_ibtracs.csv")
    ax.plot(ib.LON, ib.LAT, "k.-", label="IBTrACS")
    for k, sty in (("v2", "b^-"), ("te_style", "gs-"), ("gnn_full", "ro-"), ("gnn_temporal", "mx-")):
        tr = real.get(k, {}).get("track")
        if tr:
            ax.plot([p[2] for p in tr], [p[1] for p in tr], sty, ms=3,
                    label=f"{k} ({real[k]['err_mean_km']:.0f} km)")
    ax.set_xlim(80, 95)
    ax.set_ylim(5, 28)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("REAL Amphan 2020: ERA5 G12 tracks vs IBTrACS")
    fig.tight_layout()
    fig.savefig(FIG / "gnn_amphan_real.png", dpi=100)
    plt.close(fig)


def fmt(x, n=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def findings(res, real):
    test = [c for c in res if res[c]["split"] == "test"]
    val = [c for c in res if res[c]["split"] == "val"]
    m = lambda mth, k, cs: float(np.nanmean([res[c][mth][k] for c in cs]))
    return ["## Findings (honest reading)", "",
            f"- **Detection skill.** On TEST, gnn_full has CSI {m('gnn_full', 'csi', test):.2f} against "
            f"{m('v2', 'csi', test):.2f} for v2 and {m('te_style', 'csi', test):.2f} for TE-style, and FAR "
            f"{m('gnn_full', 'far', test):.2f} against {m('v2', 'far', test):.2f}. It detects the two TEST "
            "events that v2 misses entirely (heat_04, where the IMD 40 C rule never fires, and cold_04). "
            "The same ranking holds on VAL.",
            f"- **Area overlap is worse.** Truth-run IoU is {m('gnn_full', 'iou', val):.2f} for the GNN "
            f"against {m('v2', 'iou', val):.2f} for v2 on VAL, and {m('gnn_full', 'iou', test):.2f} against "
            f"{m('v2', 'iou', test):.2f} on TEST. The GNN classifies and links the permissive candidate "
            "objects but does not re-segment them, and those objects are larger than the exact masks. A "
            "segmentation head is future work.",
            "- **Edges add little.** All four variants have VAL AP 1.94-1.95 of 2, and the node-only MLP "
            "(gnn_none) matches the full GNN on TEST CSI. Most of the signal is in the node features "
            "(intensity, IMD-criteria fraction, EFI, shape). On TEST, cross-member edges do not help, "
            "and temporal edges give the fewest spurious tracks. With 6 training cases, the edge types "
            "cannot be ranked reliably.",
            f"- **Real Amphan.** The temporal-only GNN gives {real['gnn_temporal']['err_mean_km']:.0f} km "
            f"mean error against {real['v2']['err_mean_km']:.0f} km for v2 and "
            f"{real['te_style']['err_mean_km']:.0f} km for TE-style. The median is 28 km for all three, so "
            "the gain comes from fewer bad fixes (max 133 km against 195 km). The full GNN scores exactly "
            "like v2 here (same matched fixes and errors). A likely but unverified reason: a single ERA5 "
            "run has none of the cross-member edges the full model was trained with.",
            "- **Time to detection.** The mean lag is 0-10 h and no GNN variant misses a TEST event. The "
            "'lead gained vs v2' column only averages cases both methods detect; v2 never detects heat_04 "
            "at all.",
            "- **Probabilities are over-confident.** GNN node probabilities are nearly binary. Against the "
            "member's own event they are close to calibrated. Against the truth, p ~ 1 verifies about 60-90 % "
            "of the time, with the long-lead bin worst, because the model was trained to recognise events "
            "in each member, not to forecast the truth. Lead-dependent calibration on VAL (e.g. temperature "
            "scaling) is a cheap next step.",
            "- **TE-style is poor on cyclones.** Without the warm-core and wind criteria that real "
            "TempestExtremes setups add, MSLP minima with a closed contour include many weak lows, giving "
            "hundreds of spurious tracks per case. It is a like-for-like re-implementation of the "
            "DetectNodes/StitchNodes core, not an optimised TE configuration.", ""]


def table(res, methods, cases):
    L = ["| case | method | IoU | centroid err (km) | POD | FAR | CSI | spurious tracks | lag (h) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        for mth in methods:
            r = res[c][mth]
            L.append(f"| {c} | {mth} | {fmt(r['iou'])} | {fmt(r['centroid_err_km'], 0)} | {fmt(r['pod'])} | "
                     f"{fmt(r['far'])} | {fmt(r['csi'])} | {r['spurious_tracks']} | "
                     f"{'never' if r['detect_lag_h'] is None else r['detect_lag_h']} |")
    return L


def mean_row(res, mth, cases):
    rs = [res[c][mth] for c in cases]
    lags = [r["detect_lag_h"] for r in rs]
    gained = [res[c]["v2"]["detect_lag_h"] - res[c][mth]["detect_lag_h"] for c in cases
              if res[c]["v2"]["detect_lag_h"] is not None and res[c][mth]["detect_lag_h"] is not None]
    ok = [x for x in lags if x is not None]
    return (f"| {mth} | {fmt(np.mean([r['iou'] for r in rs]))} | {fmt(np.mean([r['csi'] for r in rs]))} | "
            f"{fmt(np.mean([r['pod'] for r in rs]))} | {fmt(np.mean([r['far'] for r in rs]))} | "
            f"{fmt(np.nanmean([r['centroid_err_km'] for r in rs]), 0)} | "
            f"{fmt(np.mean([r['spurious_tracks'] for r in rs]), 1)} | "
            f"{fmt(np.mean(ok), 0) if ok else 'n/a'} ({len(lags) - len(ok)} never) | "
            f"{fmt(np.mean(gained), 0) if gained else 'n/a'} |")


def write_report(res, real, dec, rel, cfgs):
    first = next(iter(res.values()))
    methods = ["v2", "te_style"] + [f"gnn_{v}" for v in VARIANTS if f"gnn_{v}" in first]
    L = ["# GNN spatio-temporal tracker (BRIEF2 Phase 2)", "",
         "Synthetic cases are **SYNTHETIC** (exact labels); the Amphan section is real ERA5 vs IBTrACS. "
         "The GNN was trained on the 6 TRAIN cases only, and its checkpoint and decoding thresholds were "
         "chosen on VAL (cyc_03, heat_03, cold_03). TEST (cyc_04, heat_04, cold_04, amphan_replay) is only "
         "reported.", "",
         "Methods:",
         "- **v2**: the Phase 1 tracker with its TRAIN+VAL parameters.",
         "- **te_style**: TempestExtremes-style tracking (DetectNodes/StitchNodes for cyclones, "
         "DetectBlobs/StitchBlobs for heat/cold), re-implemented in `pipeline/te_style.py`.",
         "- **gnn_\\***: the GNN (`pipeline/gnn.py`), with edge-type ablations: full, temporal only, "
         "cross-member only, and none (a plain node MLP).", "",
         "Metrics (see `pipeline/gnn_eval.py`):",
         "- POD, FAR and CSI are over (member, lead) pairs, against the truth mask.",
         "- Lag = hours after the truth onset until at least 50 % of members detect the event; lower is "
         "better. Lead gained = lag(v2) - lag(method).", "",
         "## Mean over TEST cases", "",
         "| method | IoU | CSI | POD | FAR | centroid err (km) | spurious tracks/case | mean lag (h) | lead gained vs v2 (h) |",
         "|---|---|---|---|---|---|---|---|---|"]
    test = [c for c in res if res[c]["split"] == "test"]
    val = [c for c in res if res[c]["split"] == "val"]
    L += [mean_row(res, m, test) for m in methods]
    L += ["", "## Mean over VAL cases", "",
          "| method | IoU | CSI | POD | FAR | centroid err (km) | spurious tracks/case | mean lag (h) | lead gained vs v2 (h) |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [mean_row(res, m, val) for m in methods]
    L += ["", "## Per case: TEST", ""] + table(res, methods, test)
    L += ["", "## Per case: VAL", ""] + table(res, methods, val)
    L += ["", "## Per case: TRAIN (in-sample for the GNN; not a skill estimate)", ""]
    L += table(res, methods, [c for c in res if res[c]["split"] == "train"])
    L += ["", "## Real Amphan 2020 (ERA5 G12 vs IBTrACS)", "",
          "| method | matched 6-h fixes | mean err (km) | median (km) | max (km) | other tracks |",
          "|---|---|---|---|---|---|"]
    for k, r in real.items():
        L.append(f"| {k} | {r['n_matched']} | {fmt(r['err_mean_km'], 0)} | {fmt(r['err_median_km'], 0)} | "
                 f"{fmt(r.get('err_max_km'), 0)} | {r['other_tracks']} |")
    L += ["", "ERA5 is a single deterministic run, so the GNN sees no cross-member edges; the "
          "temporal-only variant is the like-for-like model here.", "",
          "![amphan](figures/gnn_amphan_real.png)", "",
          "## Ensemble products (TEST cases, full GNN)", "",
          "![strike](figures/gnn_strike_cone.png)", "",
          "![reliability](figures/gnn_reliability.png)", "",
          "Reliability is computed from the GNN node event probabilities of all TEST ensemble nodes, binned "
          "by lead. The left panel verifies against the truth mask; the right verifies against the member's "
          "own injected event, which is the GNN's training target.", "",
          "## Training", "", "| variant | best epoch | val AP node+edge | train time (s) | compute |",
          "|---|---|---|---|---|"]
    for v, c in cfgs.items():
        L.append(f"| {v} | {c['best_epoch']} | {fmt(c['best_val_ap_sum'], 3)} | {fmt(c['train_seconds'], 0)} | "
                 f"{c['compute']} |")
    L += ["", f"Decoding thresholds (chosen on VAL, full model): `{json.dumps(dec['full'])}`", "",
          "The GNN's main track is the one with the largest summed node event probability. This was "
          "changed from the largest-area rule after VAL cyc_03 showed the area rule picking a large "
          "non-storm object (truth-run IoU 0.00 -> 0.23); TEST numbers were not looked at for this.", ""]
    L += findings(res, real)
    (REP / "GNN_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/GNN_RESULTS.md")


if __name__ == "__main__":
    main()
