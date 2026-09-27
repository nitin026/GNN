"""Before/after evaluation of the Phase 5 baseline tracker vs tracker v2 (BRIEF2 Phase 1).

Both trackers run on identical inputs and are scored with identical definitions:
  iou_mean        mean over steps where truth or tracker mask is non-empty (main track vs
                  exact 12 km truth mask)
  centroid_err    km, main track vs label centre (cyclone) / mask centroid (heat, cold)
  truth_spurious  tracks on the 12 km truth that never overlap the truth mask
  ens_spurious    sum over the 20 members of tracks that never overlap THAT member's own
                  injected-event mask (tracker false alarms; the generator's deliberate
                  false-alarm events are part of the member mask and are not counted)
  ens_spurious_vs_truth  same, but against the truth mask (definition used in Phase 5)
Old tracker tracks are counted when they last >= 2 steps (the Phase 5 convention); v2 tracks
have already passed the v2 duration filter.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .anomaly import Climatology
from .evaluate import CONFIG, run_detection
from .track import haversine_km, iou, main_track, track_masks
from .tracker2 import Tracker2

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"


def load_case(case):
    lab = json.loads((SYN / case / "labels.json").read_text())
    ds = xr.open_dataset(SYN / case / "fcst_12km.nc", decode_timedelta=False)
    var = "msl" if lab["hazard"] == "tropical_cyclone" else "t2m"
    d = {"lab": lab, "hazard": lab["hazard"], "var": var,
         "lat": ds.latitude.values, "lon": ds.longitude.values,
         "times": pd.to_datetime(lab["valid_times"]),
         "truth": ds[f"{var}_truth"].values.astype(np.float64),
         "tmask": ds["event_mask_truth"].values.astype(bool),
         "ens": ds[var].values.astype(np.float64),
         "emask": ds["event_mask"].values.astype(bool),
         "roles": ds["member_role"].values}
    ds.close()
    return d


def ref_pos(e, hazard):
    if hazard == "tropical_cyclone":
        return e.get("center_lat"), e.get("center_lon")
    return e.get("mask_centroid_lat"), e.get("mask_centroid_lon")


def score_tracks(tracks, tmask, lab, hazard, key, min_len=1):
    tracks = [t for t in tracks if len(t) >= min_len]
    mt = main_track(tracks)
    pm = track_masks(mt, len(tmask), tmask.shape[1:])
    idx = [i for i in range(len(tmask)) if tmask[i].any() or pm[i].any()]
    ious = [iou(pm[i], tmask[i]) for i in idx]
    errs = []
    if mt:
        for t, o in mt:
            la, lo = ref_pos(lab["truth_track"][t], hazard)
            if la is not None and tmask[t].any():
                errs.append(haversine_km(o[f"{key}_lat"], o[f"{key}_lon"], la, lo))
    spur = sum(1 for tr in tracks if not any((o["mask"] & tmask[t]).any() for t, o in tr))
    dious = []                              # secondary: daily masks (union of 4 six-hourly steps)
    for d0 in range(0, len(tmask) - 3, 4):
        a, b = pm[d0:d0 + 4].any(0), tmask[d0:d0 + 4].any(0)
        if a.any() or b.any():
            dious.append(iou(a, b))
    return {"iou_mean": float(np.nanmean(ious)) if ious else 0.0,
            "iou_daily": float(np.nanmean(dious)) if dious else 0.0,
            "centroid_err_km": float(np.mean(errs)) if errs else np.nan,
            "n_tracks": len(tracks), "spurious": spur, "main": mt, "pmask": pm}


def run_old(d, clim, series):
    _, tracks = run_detection(series, d["times"], d["lat"], d["lon"], clim, d["hazard"])
    return tracks


def evaluate(d, clim, orog, trackers=("old", "v2"), members=True, params=None):
    hz, lab = d["hazard"], d["lab"]
    key = "center" if hz == "tropical_cyclone" else "centroid"
    peak = lab["peak"]["lead_h"] // 6 if lab.get("peak") else None
    res = {}
    for name in trackers:
        if name == "old":
            run = lambda s: run_old(d, clim, s)
            min_len = 2
        else:
            tk = Tracker2(hz, clim, orog, d["lat"], d["lon"], **(params or {}))
            run = lambda s: tk.run(s, d["times"])[0]
            min_len = 1
        tr = run(d["truth"])
        st = score_tracks(tr, d["tmask"], lab, hz, key, min_len)
        r = {"iou_mean": st["iou_mean"], "iou_daily": st["iou_daily"],
             "centroid_err_km": st["centroid_err_km"],
             "truth_tracks": st["n_tracks"], "truth_spurious": st["spurious"]}
        if members:
            sp_own = sp_truth = 0
            hits, lead_err = [], {}
            for m in range(d["ens"].shape[0]):
                trm = [t for t in run(d["ens"][m]) if len(t) >= min_len]
                sp_own += sum(1 for t_ in trm if not any((o["mask"] & d["emask"][m, t]).any()
                                                          for t, o in t_))
                sp_truth += sum(1 for t_ in trm if not any((o["mask"] & d["tmask"][t]).any()
                                                            for t, o in t_))
                mm = main_track(trm)
                pm = track_masks(mm, len(d["tmask"]), d["tmask"].shape[1:])
                if peak is not None:
                    hits.append(bool((pm[peak] & d["tmask"][peak]).any()))
                if mm and d["roles"][m] == 0:
                    for t, o in mm:
                        la, lo = ref_pos(lab["truth_track"][t], hz)
                        if la is not None and d["tmask"][t].any():
                            lead_err.setdefault(int(lab["lead_hours"][t]), []).append(
                                float(haversine_km(o[f"{key}_lat"], o[f"{key}_lon"], la, lo)))
            r.update({"ens_spurious": sp_own, "ens_spurious_vs_truth": sp_truth,
                      "ens_hit_rate_at_peak": float(np.mean(hits)) if hits else np.nan,
                      "ens_err_by_lead": {k: float(np.mean(v)) for k, v in sorted(lead_err.items())}})
        res[name] = r
    return res


def orog12():
    return xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float64)


if __name__ == "__main__":
    import sys
    clim, oro = Climatology(), orog12()
    for c in sys.argv[1:]:
        d = load_case(c)
        r = evaluate(d, clim, oro)
        print(c, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                 for kk, vv in v.items() if kk != "ens_err_by_lead"}
                             for k, v in r.items()}))
