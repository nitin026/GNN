"""Baseline object tracker: threshold -> scipy.ndimage.label -> Hungarian matching.

detect()   : connected components of (sign * field >= threshold), size-filtered
link()     : frame-to-frame assignment (scipy.optimize.linear_sum_assignment) on centroid
             distance with a gate; unmatched objects start new tracks
boxes_4d() : (lat, lon, level, time) bounding box of each track
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

KM_PER_DEG = 111.195


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def detect(field, lat, lon, threshold, sign=1, min_cells=9, center_field=None):
    """Objects where sign*field >= threshold. Returns list of dicts with a boolean mask.

    center = location of the extreme of `center_field` (default: `field`) inside the object
    (e.g. MSLP minimum for a cyclone); centroid = intensity-weighted mean position.
    """
    f = sign * np.asarray(field, float)
    binary = f >= threshold
    lab, n = ndimage.label(binary, structure=np.ones((3, 3)))
    if n == 0:
        return []
    sizes = ndimage.sum(binary, lab, index=np.arange(1, n + 1))
    cf = f if center_field is None else np.asarray(center_field, float)
    out = []
    for k in np.where(sizes >= min_cells)[0] + 1:
        m = lab == k
        ii, jj = np.nonzero(m)
        w = f[ii, jj] - threshold + 1e-6
        c = np.argmax(cf[ii, jj])
        out.append({"mask": m, "n_cells": int(m.sum()),
                    "centroid_lat": float(np.average(lat[ii], weights=w)),
                    "centroid_lon": float(np.average(lon[jj], weights=w)),
                    "center_lat": float(lat[ii[c]]), "center_lon": float(lon[jj[c]]),
                    "peak": float(f[ii, jj].max() * sign),
                    "bbox": [float(lat[ii].min()), float(lat[ii].max()),
                             float(lon[jj].min()), float(lon[jj].max())]})
    return out


def link(frames, max_km=500.0, key="centroid"):
    """frames: list (per time) of detect() outputs. Returns list of tracks; each track is a
    list of (time_index, object)."""
    tracks, open_ = [], {}
    for t, objs in enumerate(frames):
        prev = [(tid, tracks[tid][-1][1]) for tid in open_ if tracks[tid][-1][0] == t - 1]
        assigned = set()
        if prev and objs:
            C = np.array([[haversine_km(p[f"{key}_lat"], p[f"{key}_lon"],
                                        o[f"{key}_lat"], o[f"{key}_lon"]) for o in objs]
                          for _, p in prev])
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] <= max_km:
                    tracks[prev[i][0]].append((t, objs[j]))
                    assigned.add(j)
        for j, o in enumerate(objs):
            if j not in assigned:
                tracks.append([(t, o)])
                open_[len(tracks) - 1] = True
    return tracks


def boxes_4d(tracks, times, levels=(1013, 1013)):
    out = []
    for tr in tracks:
        bb = np.array([o["bbox"] for _, o in tr])
        out.append({"lat_min": bb[:, 0].min(), "lat_max": bb[:, 1].max(),
                    "lon_min": bb[:, 2].min(), "lon_max": bb[:, 3].max(),
                    "level_bottom": levels[0], "level_top": levels[1],
                    "time_start": str(times[tr[0][0]]), "time_end": str(times[tr[-1][0]]),
                    "n_steps": len(tr)})
    return out


def main_track(tracks, min_len=2):
    """Pick the dominant track: largest area summed over its lifetime (then longest)."""
    cand = [t for t in tracks if len(t) >= min_len] or tracks
    if not cand:
        return None
    return max(cand, key=lambda t: (sum(o["n_cells"] for _, o in t), len(t)))


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def track_masks(track, n_times, shape):
    m = np.zeros((n_times,) + tuple(shape), bool)
    if track:
        for t, o in track:
            m[t] |= o["mask"]
    return m
