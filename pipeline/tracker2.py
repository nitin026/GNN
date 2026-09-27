"""Tracker v2 (BRIEF2 Phase 1): hysteresis objects, hazard-specific fields, closed-contour
cyclone test, Kalman motion prior in the Hungarian cost, area/duration/persistence filters.

Hazard-specific detection fields (all "larger = more extreme"):
  tropical_cyclone : -z(MSLP), 25-km smoothed; the object must also pass a TempestExtremes-style
                     closed-contour test (MSLP rises by >= dp_closed within r_closed of the
                     centre in every one of 16 directions), and a track must start over sea
                     (tropical cyclones form over the ocean; removes land heat lows).
  heat_dome        : IMD heat-wave criteria on daily Tmax (max of the 4 six-hourly values in a
                     centred 24 h window), inside a rough IMD domain (imd_domain): departure from the ERA5 1990-2019 normal and an absolute
                     threshold by region (plains 40 C, coastal 37 C, hills 30 C).
  cold_wave        : IMD cold-wave criteria on daily Tmin: departure <= -4.5 C with Tmin <= 10 C
                     (plains), <= 15 C (coastal), <= 0 C (hills) - departure made positive.
Hysteresis: objects are connected components of (field >= low) that contain at least one cell
(field >= high); only such "cored" objects can start a track, weaker components (>= low, no
core) can only continue an existing track.
"""
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import linear_sum_assignment

from .track import haversine_km

KM = 111.195

DEFAULTS = {
    "tropical_cyclone": dict(high=2.0, low=1.5, min_area_km2=5000.0, min_steps=4, smooth=2.0,
                             dp_closed=200.0, r_closed_deg=5.5, gate_sigma=3.0, max_gap=1),
    "heat_dome": dict(high=4.5, low=3.0, min_area_km2=30000.0, min_steps=8, smooth=1.0,
                      gate_sigma=3.0, max_gap=1),
    "cold_wave": dict(high=4.5, low=3.0, min_area_km2=30000.0, min_steps=8, smooth=1.0,
                      gate_sigma=3.0, max_gap=1),
}


# ----------------------------------------------------------------------------- regions / normals
def regions(orog, lat, lon, coast_km=50.0):
    """IMD region classes: 0 sea or excluded (> 2500 m), 1 plains, 2 coastal (< 50 km from sea),
    3 hills (1000-2500 m)."""
    land = orog > 1.0
    dy = (lat[1] - lat[0]) * KM
    dist_sea = distance_transform_edt(land) * dy          # km to nearest sea cell (approx)
    reg = np.where(land, 1, 0)
    reg = np.where(land & (dist_sea <= coast_km), 2, reg)
    reg = np.where(land & (orog > 1000.0), 3, reg)
    reg = np.where(orog > 2500.0, 0, reg)      # Tibetan plateau / high Karakoram: not IMD domain
    reg = np.where(imd_domain(orog, lat), reg, 0)
    return reg


def imd_domain(orog, lat, lat_max=37.0, crest_m=4500.0, crest_from=26.0):
    """Rough IMD warning domain without a country boundary: latitude <= 37 N and south of the
    Himalayan / Hindu Kush crest (for each longitude, everything north of the first cell above
    4500 m (main Himalayan ridge) met going north from 26 N is excluded: Tibet, Tarim Basin).
    Cells above 2500 m (e.g. Ladakh) are excluded separately in regions()."""
    ok = (lat <= lat_max)[:, None] * np.ones_like(orog, bool)
    i0 = int(np.searchsorted(lat, crest_from))
    high = orog[i0:] > crest_m
    for j in range(orog.shape[1]):
        k = np.argmax(high[:, j]) if high[:, j].any() else None
        if k is not None:
            ok[i0 + k:, j] = False
    return ok


def daily_normals(clim, times, kind):
    """Climatological daily Tmax (kind='max') or Tmin ('min') valid at each step [K]."""
    out = []
    for t in pd.to_datetime(times):
        vals = [clim.get("t2m", t.normalize() + pd.Timedelta(hours=h))[0] for h in (0, 6, 12, 18)]
        out.append(np.max(vals, 0) if kind == "max" else np.min(vals, 0))
    return np.array(out)


def trailing(x, kind, n=4, back=1):
    """24-h max/min of 6-hourly data over the window [t - back, t - back + n - 1] (default:
    centred, t-6h .. t+12h). A forecast contains its own future leads, so the centred window
    is available at issue time and removes the one-day lag of a trailing window."""
    out = np.empty_like(x)
    for t in range(len(x)):
        lo = min(max(0, t - back), max(0, len(x) - n))
        w = x[lo:lo + n]
        out[t] = w.max(0) if kind == "max" else w.min(0)
    return out


_NORMALS = {}


def cached_normals(clim, times, kind):
    key = (id(clim), tuple(pd.to_datetime(times)), kind)
    if key not in _NORMALS:
        _NORMALS.clear() if len(_NORMALS) > 8 else None
        _NORMALS[key] = daily_normals(clim, times, kind)
    return _NORMALS[key]


def imd_field(t2m, times, clim, reg, hazard):
    """Signed departure field [C] that is set to 0 where the IMD absolute criterion fails."""
    if hazard == "heat_dome":
        tx = trailing(t2m, "max")
        dep = tx - cached_normals(clim, times, "max")
        absT = tx - 273.15
        ok = ((reg == 1) & (absT >= 40)) | ((reg == 2) & (absT >= 37)) | ((reg == 3) & (absT >= 30))
        ok |= (reg > 0) & (absT >= 45)                        # IMD: Tmax >= 45 C regardless
        return np.where(ok & (reg > 0), dep, 0.0)
    tn = trailing(t2m, "min")
    dep = -(tn - cached_normals(clim, times, "min"))
    absT = tn - 273.15
    ok = ((reg == 1) & (absT <= 10)) | ((reg == 2) & (absT <= 15)) | ((reg == 3) & (absT <= 0))
    ok |= (reg == 1) & (absT <= 4)                            # IMD: Tmin <= 4 C (plains)
    return np.where(ok & (reg > 0), dep, 0.0)


def cyclone_field(msl, times, clim, smooth):
    return np.array([-gaussian_filter(clim.z("msl", msl[i], t), smooth)
                     for i, t in enumerate(pd.to_datetime(times))])


# ----------------------------------------------------------------------------- objects
def cell_area_km2(lat, lon):
    return ((lat[1] - lat[0]) * KM) * ((lon[1] - lon[0]) * KM) * np.cos(np.deg2rad(lat))[:, None] \
        * np.ones((1, len(lon)))


def closed_contour(msl, i, j, lat, lon, dp, r_deg, n_dir=16):
    """True if MSLP rises by >= dp within r_deg of (i, j) along every one of n_dir rays."""
    p0 = msl[i, j]
    dlat = lat[1] - lat[0]
    nstep = int(r_deg / dlat)
    for a in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
        ok = False
        for s in range(1, nstep + 1):
            ii = int(round(i + s * np.sin(a)))
            jj = int(round(j + s * np.cos(a) / max(np.cos(np.deg2rad(lat[i])), 0.3)))
            if not (0 <= ii < msl.shape[0] and 0 <= jj < msl.shape[1]):
                break
            if msl[ii, jj] >= p0 + dp:
                ok = True
                break
        if not ok:
            return False
    return True


def detect_hysteresis(f, lat, lon, high, low, min_area_km2, area, msl=None, dp_closed=None,
                      r_closed_deg=None):
    weak = f >= low
    lab, n = ndimage.label(weak, structure=np.ones((3, 3)))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    has_core = ndimage.maximum(f, lab, idx) >= high
    areas = ndimage.sum(area, lab, idx)
    out = []
    for k, core, ar in zip(idx, has_core, areas):
        if ar < 0.5 * min_area_km2:          # too small even to continue a track
            continue
        core = bool(core) and ar >= min_area_km2   # only large cored objects may start tracks
        m = lab == k
        ii, jj = np.nonzero(m)
        vals = f[ii, jj]
        if msl is not None:
            c = int(np.argmin(msl[ii, jj]))
        else:
            c = int(np.argmax(vals))
        ci, cj = ii[c], jj[c]
        if msl is not None and dp_closed is not None and core:
            core = closed_contour(msl, ci, cj, lat, lon, dp_closed, r_closed_deg)
        w = vals - low + 1e-6
        out.append({"mask": m, "n_cells": int(m.sum()), "area_km2": float(ar), "core": bool(core),
                    "centroid_lat": float(np.average(lat[ii], weights=w)),
                    "centroid_lon": float(np.average(lon[jj], weights=w)),
                    "center_lat": float(lat[ci]), "center_lon": float(lon[cj]),
                    "peak": float(vals.max()),
                    "bbox": [float(lat[ii].min()), float(lat[ii].max()),
                             float(lon[jj].min()), float(lon[jj].max())]})
    return out


# ----------------------------------------------------------------------------- Kalman linking
class KalmanCV:
    """Constant-velocity Kalman filter in a local km frame (state x, y, vx, vy; dt in hours)."""

    def __init__(self, lat, lon, dt=6.0, q=20.0, r=40.0, v0=300.0):
        self.lat0, self.lon0, self.dt = lat, lon, dt
        self.x = np.zeros(4)
        self.P = np.diag([r ** 2, r ** 2, v0 ** 2, v0 ** 2])  # v in km per step
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.Q = np.diag([q ** 2, q ** 2, (0.5 * q) ** 2, (0.5 * q) ** 2]) * 4
        self.H = np.eye(2, 4)
        self.R = np.eye(2) * r ** 2

    def to_xy(self, lat, lon):
        return np.array([(lon - self.lon0) * KM * np.cos(np.deg2rad(self.lat0)),
                         (lat - self.lat0) * KM])

    def to_ll(self, xy):
        return (self.lat0 + xy[1] / KM,
                self.lon0 + xy[0] / (KM * np.cos(np.deg2rad(self.lat0))))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, lat, lon):
        z = self.to_xy(lat, lon)
        S = self.H @ self.P @ self.H.T + self.R
        d = z - self.H @ self.x
        return float(d @ np.linalg.solve(S, d))            # squared Mahalanobis distance

    def update(self, lat, lon):
        z = self.to_xy(lat, lon)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P


def link_kalman(frames, key="center", gate_sigma=3.0, max_gap=1, q=20.0, r=40.0):
    """Hungarian assignment on squared Mahalanobis distance to each track's Kalman prediction.
    Only cored objects may start tracks; tracks survive up to max_gap missed steps."""
    tracks, active = [], []            # active: list of dict(track_idx, kf, missed)
    gate = gate_sigma ** 2
    for t, objs in enumerate(frames):
        for a in active:
            a["kf"].predict()
        assigned = set()
        if active and objs:
            C = np.array([[a["kf"].innovation(o[f"{key}_lat"], o[f"{key}_lon"]) for o in objs]
                          for a in active])
            Cg = np.where(C <= gate, C, 1e6)
            ri, ci = linear_sum_assignment(Cg)
            for i, j in zip(ri, ci):
                if Cg[i, j] < 1e6:
                    a = active[i]
                    a["kf"].update(objs[j][f"{key}_lat"], objs[j][f"{key}_lon"])
                    tracks[a["idx"]].append((t, objs[j]))
                    a["missed"] = -1
                    assigned.add(j)
        for a in active:
            a["missed"] += 1
        active = [a for a in active if a["missed"] <= max_gap]
        for j, o in enumerate(objs):
            if j not in assigned and o["core"]:
                kf = KalmanCV(o[f"{key}_lat"], o[f"{key}_lon"], q=q, r=r)
                kf.x[:2] = kf.to_xy(o[f"{key}_lat"], o[f"{key}_lon"])
                tracks.append([(t, o)])
                active.append({"idx": len(tracks) - 1, "kf": kf, "missed": 0})
    return tracks


# ----------------------------------------------------------------------------- driver
class Tracker2:
    def __init__(self, hazard, clim, orog, lat, lon, **overrides):
        self.hazard = hazard
        self.p = dict(DEFAULTS[hazard], **overrides)
        self.clim, self.lat, self.lon = clim, lat, lon
        self.area = cell_area_km2(lat, lon)
        self.reg = regions(orog, lat, lon)
        self.land = orog > 1.0

    def field(self, series, times):
        if self.hazard == "tropical_cyclone":
            return cyclone_field(series, times, self.clim, self.p["smooth"])
        f = imd_field(series, times, self.clim, self.reg, self.hazard)
        return np.array([gaussian_filter(x, self.p["smooth"]) for x in f]) if self.p["smooth"] else f

    def run(self, series, times, field=None):
        """series: (T, y, x) MSLP [Pa] for cyclones, T2m [K] otherwise."""
        p = self.p
        f = self.field(series, times) if field is None else field
        cyc = self.hazard == "tropical_cyclone"
        frames = [detect_hysteresis(f[t], self.lat, self.lon, p["high"], p["low"], p["min_area_km2"],
                                    self.area, msl=series[t] if cyc else None,
                                    dp_closed=p.get("dp_closed"), r_closed_deg=p.get("r_closed_deg"))
                  for t in range(len(f))]
        tracks = link_kalman(frames, key="center" if cyc else "centroid",
                             gate_sigma=p["gate_sigma"], max_gap=p["max_gap"],
                             q=20.0 if cyc else 30.0, r=40.0 if cyc else 120.0)
        # duration / persistence filter (heat & cold: IMD-style >= 2 days)
        tracks = [tr for tr in tracks if (tr[-1][0] - tr[0][0] + 1) >= p["min_steps"]]
        if cyc and p.get("genesis_over_sea", True):
            tracks = [tr for tr in tracks if not self._over_land(tr[0][1])]
        return tracks, f

    def _over_land(self, o):
        i = min(int(np.searchsorted(self.lat, o["center_lat"])), len(self.lat) - 1)
        j = min(int(np.searchsorted(self.lon, o["center_lon"])), len(self.lon) - 1)
        return self.land[i, j]


def track_distance_km(o, lat, lon, key):
    return haversine_km(o[f"{key}_lat"], o[f"{key}_lon"], lat, lon)
