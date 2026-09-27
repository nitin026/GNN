"""TempestExtremes-style comparison trackers (re-implemented; not the TempestExtremes code).

Cyclones (DetectNodes + StitchNodes, settings commonly used with TempestExtremes, e.g.
Zarzycki & Ullrich 2017): MSLP local minima; closed contour: MSLP rises by 200 Pa within
5.5 deg (closedcontourcmd "PSL,200.0,5.5,0"); minima within 6 deg merged (keep the lowest);
stitching within 8 deg great-circle distance per step, max gap 1 step, min length 4 steps.
Each node's region, needed for IoU, is the connected area within 5.5 deg where
MSLP <= (median MSLP on the 5.5 deg ring) - 400 Pa.
Heat / cold (DetectBlobs + StitchBlobs): blobs where the IMD departure field (tracker2.imd_field)
is >= 4.5 C, minimum area 10000 km2; blobs linked when they overlap the previous step's blob;
minimum duration 8 steps (2 days).
Output format matches pipeline.tracker2 (list of tracks, each a list of (t, object)).
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import minimum_filter

from .track import haversine_km
from .tracker2 import cell_area_km2, closed_contour, imd_field, regions


def detect_nodes(msl, lat, lon, dp=200.0, r_deg=5.5, merge_deg=6.0):
    mins = (msl == minimum_filter(msl, size=9, mode="nearest"))
    ii, jj = np.nonzero(mins)
    order = np.argsort(msl[ii, jj])
    kept = []
    for k in order:
        i, j = ii[k], jj[k]
        if any(haversine_km(lat[i], lon[j], lat[a], lon[b]) < merge_deg * 111.2 for a, b in kept):
            continue
        if closed_contour(msl, i, j, lat, lon, dp, r_deg):
            kept.append((i, j))
    return kept


def node_region(msl, i, j, lat, lon, r_deg=5.5):
    dlat = lat[1] - lat[0]
    n = int(r_deg / dlat)
    i0, i1 = max(0, i - n), min(msl.shape[0], i + n + 1)
    j0, j1 = max(0, j - n), min(msl.shape[1], j + n + 1)
    sub = msl[i0:i1, j0:j1]
    yy, xx = np.mgrid[i0:i1, j0:j1]
    rr = np.hypot((yy - i) * dlat, (xx - j) * dlat * np.cos(np.deg2rad(lat[i])))
    ring = sub[(rr > r_deg - 0.3) & (rr <= r_deg)]
    thr = np.median(ring) - 400.0 if ring.size else msl[i, j] + 400.0
    lab, _ = ndimage.label((sub <= thr) & (rr <= r_deg))
    m = np.zeros(msl.shape, bool)
    k = lab[i - i0, j - j0]
    if k > 0:
        m[i0:i1, j0:j1] = lab == k
    else:
        m[i, j] = True
    return m


def te_cyclones(msl_series, lat, lon, range_deg=8.0, max_gap=1, min_len=4):
    frames = []
    for t, msl in enumerate(msl_series):
        objs = []
        for i, j in detect_nodes(msl, lat, lon):
            m = node_region(msl, i, j, lat, lon)
            objs.append({"mask": m, "center_lat": float(lat[i]), "center_lon": float(lon[j]),
                         "centroid_lat": float(lat[i]), "centroid_lon": float(lon[j]),
                         "peak": float(-msl[i, j]), "core": True, "n_cells": int(m.sum()),
                         "area_km2": 0.0, "bbox": [0, 0, 0, 0]})
        frames.append(objs)
    return stitch_points(frames, range_deg * 111.2, max_gap, min_len)


def stitch_points(frames, range_km, max_gap, min_len):
    """Greedy nearest-neighbour stitching (as StitchNodes): each open track takes the closest
    unclaimed node within range at the next step(s)."""
    tracks, open_ = [], []
    for t, objs in enumerate(frames):
        claimed = set()
        cand = []
        for k, tr in enumerate(open_):
            lt, lo = tr[-1]
            for j, o in enumerate(objs):
                d = haversine_km(lo["center_lat"], lo["center_lon"], o["center_lat"], o["center_lon"])
                if d <= range_km * (t - lt):
                    cand.append((d, k, j))
        used = set()
        for d, k, j in sorted(cand):
            if k in used or j in claimed:
                continue
            open_[k].append((t, objs[j]))
            used.add(k)
            claimed.add(j)
        for j, o in enumerate(objs):
            if j not in claimed:
                open_.append([(t, o)])
        still = []
        for tr in open_:
            if t - tr[-1][0] < max_gap + 1:
                still.append(tr)
            else:
                tracks.append(tr)
        open_ = still
    tracks += open_
    return [tr for tr in tracks if tr[-1][0] - tr[0][0] + 1 >= min_len]


def te_blobs(series, times, clim, orog, lat, lon, hazard, thr=4.5, min_area_km2=10000.0, min_len=8):
    reg = regions(orog, lat, lon)
    f = imd_field(series, times, clim, reg, hazard)
    area = cell_area_km2(lat, lon)
    frames = []
    for t in range(len(f)):
        lab, n = ndimage.label(f[t] >= thr, structure=np.ones((3, 3)))
        objs = []
        for k in range(1, n + 1):
            m = lab == k
            ar = float(area[m].sum())
            if ar < min_area_km2:
                continue
            ii, jj = np.nonzero(m)
            w = f[t][ii, jj]
            objs.append({"mask": m, "centroid_lat": float(np.average(lat[ii], weights=w)),
                         "centroid_lon": float(np.average(lon[jj], weights=w)),
                         "center_lat": float(lat[ii[np.argmax(w)]]), "center_lon": float(lon[jj[np.argmax(w)]]),
                         "peak": float(w.max()), "core": True, "n_cells": int(m.sum()), "area_km2": ar,
                         "bbox": [0, 0, 0, 0]})
        frames.append(objs)
    tracks, open_ = [], []
    for t, objs in enumerate(frames):
        nxt, used = [], set()
        for tr in open_:
            last = tr[-1][1]["mask"]
            best = None
            for j, o in enumerate(objs):
                ov = int((o["mask"] & last).sum())
                if ov > 0 and j not in used and (best is None or ov > best[0]):
                    best = (ov, j)
            if best:
                tr.append((t, objs[best[1]]))
                used.add(best[1])
                nxt.append(tr)
            else:
                tracks.append(tr)
        for j, o in enumerate(objs):
            if j not in used:
                nxt.append([(t, o)])
        open_ = nxt
    tracks += open_
    return [tr for tr in tracks if tr[-1][0] - tr[0][0] + 1 >= min_len]
