"""Read-only access to the precomputed products in backend/products/ (BRIEF3 Phase B).

Everything the API returns comes from files written by scripts/export_products.py (or by a POST /run
job, which runs the same export code). Nothing is computed from typed-in numbers.
"""
import json
import math
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.alert_rules import CATEGORIES, COLOURS, reason
from pipeline.jsonutil import clean

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = ROOT / "backend" / "products"
STATIC = ROOT / "backend" / "static"
LAT0, DLAT, NLAT = 0.06, 0.12, 333            # G12 cell centres (synth.grids.LAT12 / LON12)
LON0, DLON, NLON = 60.06, 0.12, 333
VARS_12 = ("t2m", "wind", "msl", "tp", "anom")
VARS_5 = ("t2m", "wind", "msl", "tp")


def cell_of(lat, lon):
    """(i, j) of the G12 cell containing (lat, lon), or None outside the India box."""
    i = int(round((lat - LAT0) / DLAT))
    j = int(round((lon - LON0) / DLON))
    return (i, j) if 0 <= i < NLAT and 0 <= j < NLON else None


def haversine_km(la1, lo1, la2, lo2):
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


class Store:
    def __init__(self, root=PRODUCTS):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._cache = {}

    # ----------------------------------------------------------------- basic files
    def _json(self, path):
        path = Path(path)
        key = (str(path), path.stat().st_mtime_ns)
        with self._lock:
            if key not in self._cache:
                self._cache = {k: v for k, v in self._cache.items() if k[0] != str(path)}
                self._cache[key] = clean(json.loads(path.read_text(encoding="utf-8")))   # NaN -> null
            return self._cache[key]

    def index(self):
        p = self.root / "index.json"
        return self._json(p) if p.exists() else []

    def has(self, case):
        return (self.root / case / "meta.json").exists() and "/" not in case and "\\" not in case \
            and not case.startswith(".")

    def meta(self, case):
        return self._json(self.root / case / "meta.json")

    def tracks(self, case):
        return self._json(self.root / case / "tracks.geojson")

    def alerts(self, case):
        return self._json(self.root / case / "alerts.json")["alerts"]

    def grids(self, case):
        p = self.root / case / "alert_grid.npz"
        if not p.exists():
            return None
        key = (str(p), p.stat().st_mtime_ns)
        with self._lock:
            if key not in self._cache:
                z = np.load(p)
                self._cache[key] = {k: z[k] for k in z.files}
            return self._cache[key]

    def field_path(self, case, var, res, lead):
        if var == "strike":
            return self.root / case / "strike_prob" / f"{lead:03d}.png"
        return self.root / case / f"fields_{res}" / var / f"{lead:03d}.png"

    # ----------------------------------------------------------------- tracks
    def tracks_at(self, case, lead=None):
        fc = self.tracks(case)
        if lead is None:
            return fc
        out = []
        for f in fc["features"]:
            p = f["properties"]
            if p["kind"] in ("member_track", "consensus"):
                keep = [i for i, h in enumerate(p["leads_h"]) if h <= lead]
                if not keep:
                    continue
                coords = [f["geometry"]["coordinates"][i] for i in keep]
                props = {k: ([v[i] for i in keep] if isinstance(v, list) and len(v) == len(p["leads_h"]) else v)
                         for k, v in p.items()}
                props["active"] = p["leads_h"][keep[-1]] == lead
                out.append({"type": "Feature", "properties": props,
                            "geometry": {"type": "LineString", "coordinates": coords}})
                if props["active"]:
                    pos = {"kind": f"{p['kind']}_position", "lead_h": lead,
                           **{k: p[k] for k in ("member", "track", "hazard", "main") if k in p}}
                    if p["kind"] == "consensus":
                        pos["spread_km"] = props["spread_km"][-1]
                        pos["member_count"] = props["member_count"][-1]
                    out.append({"type": "Feature", "properties": pos,
                                "geometry": {"type": "Point", "coordinates": coords[-1]}})
            else:                                                       # bbox4d
                out.append({**f, "properties": {**p, "active": p["lead_start_h"] <= lead <= p["lead_end_h"]}})
        return {"type": "FeatureCollection", "features": out, "lead_h": lead}

    # ----------------------------------------------------------------- alerts
    def alerts_filtered(self, case=None, lead=None, lat=None, lon=None, min_category="low"):
        cases = [case] if case else [c["case"] for c in self.index()]
        rank = CATEGORIES.index(min_category)
        cell = cell_of(lat, lon) if lat is not None and lon is not None else None
        out = []
        for c in cases:
            if not self.has(c):
                continue
            g = self.grids(c) if cell else None
            for a in self.alerts(c):
                if lead is not None and a["lead_h"] != lead:
                    continue
                if CATEGORIES.index(a["category"]) < rank:
                    continue
                if lat is not None and lon is not None:
                    if cell is None:
                        continue
                    x0, y0, x1, y1 = a["region_bbox"]
                    if not (x0 - 0.06 <= lon <= x1 + 0.06 and y0 - 0.06 <= lat <= y1 + 0.06):
                        continue
                    if g is not None and f"{a['kind']}_cat" in g:
                        if g[f"{a['kind']}_cat"][a["lead_h"] // 6][cell] < 1:
                            continue                       # inside the bbox but not in the region
                    d = haversine_km(lat, lon, a["pinpoint"]["lat"], a["pinpoint"]["lon"])
                    a = {**a, "distance_to_pinpoint_km": round(d, 1),
                         "within_impact_radius": d <= a["impact_radius_km"]}
                out.append(a)
        out.sort(key=lambda a: (-CATEGORIES.index(a["category"]), -a["probability"], a["lead_h"]))
        return out


@lru_cache(maxsize=1)
def districts():
    """District index raster on G12 + names (scripts/build_districts.py; GADM 4.1 level 2)."""
    d = json.loads((STATIC / "districts.json").read_text(encoding="utf-8"))
    idx = np.load(ROOT / "data/real/boundary/districts_g12.npz")["index"] \
        if (ROOT / "data/real/boundary/districts_g12.npz").exists() else None
    return d, idx


def district_rollup(store, case, lead=None):
    """District-level roll-up of the per-cell alert categories. For each district and kind: the
    highest category over its cells, the neighbourhood probability of the rule that set it, and the
    fraction of its cells at >= low. lead=None aggregates over all leads (max) and reports the first
    and last lead at which the district is warned."""
    g = store.grids(case)
    if g is None:
        return None
    dmeta, idx = districts()
    if idx is None:
        return None
    ds = dmeta["districts"]
    flat = idx.ravel()
    n = len(ds)
    counts = np.bincount(flat[flat >= 0], minlength=n)
    kinds = [str(k) for k in g["kinds"]]
    leads = g["leads_h"]
    rows = []
    for kind in kinds:
        cat, pl, pm, ps = (g[f"{kind}_cat"], g[f"{kind}_p_low"], g[f"{kind}_p_moderate"], g[f"{kind}_p_severe"])
        ts = [int(lead // 6)] if lead is not None else range(len(leads))
        best = np.zeros(n, np.int8)
        pbest = {k: np.zeros(n) for k in ("low", "moderate", "severe")}
        cover = np.zeros(n)
        first = np.full(n, -1)
        last = np.full(n, -1)
        for t in ts:
            if t >= len(cat):
                continue
            c = cat[t].ravel()
            per = np.zeros(n, np.int8)
            np.maximum.at(per, flat[flat >= 0], c[flat >= 0])
            # districts owning no cell use their centroid cell
            for d in ds:
                if d["n_cells"] == 0:
                    per[d["id"]] = max(per[d["id"]], cat[t][tuple(d["cell"])])
            hit = per >= 1
            first = np.where(hit & (first < 0), leads[t], first)
            last = np.where(hit, leads[t], last)
            best = np.maximum(best, per)
            for k, arr in (("low", pl), ("moderate", pm), ("severe", ps)):
                v = np.zeros(n)
                np.maximum.at(v, flat[flat >= 0], arr[t].ravel()[flat >= 0] / 100.0)
                pbest[k] = np.maximum(pbest[k], v)
            fr = np.bincount(flat[flat >= 0], weights=(c[flat >= 0] >= 1).astype(float), minlength=n)
            cover = np.maximum(cover, fr / np.maximum(counts, 1))
        for i in np.nonzero(best >= 1)[0]:
            level = CATEGORIES[int(best[i])]
            if level == "severe" or (level == "moderate" and pbest["severe"][i] >= 0.2):
                which = "severe"
            elif level == "moderate" or (level == "low" and pbest["moderate"][i] >= 0.2):
                which = "moderate"
            else:
                which = "low"
            p = float(pbest[which][i])
            d = ds[i]
            rows.append({"district": d["district"], "state": d["state"], "gid": d["gid"], "kind": kind,
                         "category": level, "colour": COLOURS[level], "probability": round(p, 2),
                         "reason": reason(kind, which, p) + " (neighbourhood 50 km)",
                         "fraction_of_district": round(float(cover[i]), 2),
                         "first_lead_h": int(first[i]), "last_lead_h": int(last[i]),
                         "centroid": {"lat": d["lat"], "lon": d["lon"]}})
    rows.sort(key=lambda r: (-CATEGORIES.index(r["category"]), -r["probability"], r["state"], r["district"]))
    return {"case": case, "lead_h": lead, "unit": "district (GADM 4.1 level 2, rasterised to G12)",
            "source": dmeta["source"], "n_districts_warned": len({r["gid"] for r in rows}), "districts": rows}
