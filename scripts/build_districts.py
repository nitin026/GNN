"""District boundaries for the alert roll-up (BRIEF3 Phase B: GET /alerts/districts).

    python scripts/build_districts.py

Source: GADM 4.1 India level 2 (districts), https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip
(GADM licence: free for academic and other non-commercial use). The raw file goes to
data/real/boundary/raw/ (gitignored). Outputs:
  data/real/boundary/districts_g12.npz   int16 district index per G12 cell (-1 = none); small
                                         districts that own no cell centre get their nearest cell
  backend/static/districts.json          index -> {gid, district, state, lat, lon, n_cells}
  backend/static/districts.geojson       simplified outlines for the dashboard (display only)
GADM's simplified JSON writes names without spaces ("NicobarIslands"); spaces are restored
before capitals ("Nicobar Islands"). Boundaries are GADM's and are shown as given; they are not
an official Survey of India product.
"""
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synth.grids import LAT12, LON12  # noqa: E402

URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip"
RAW = ROOT / "data/real/boundary/raw"


def spaced(s):
    if not s or s == "NA":
        return s
    s = re.sub(r"(?<=[a-z])and(?=[A-Z])", " and ", s)             # "AndamanandNicobar"
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).replace("  ", " ").strip()


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return polys


def simplify(ring, tol=0.04):
    """Drop vertices closer than tol degrees to the last kept one (display only)."""
    out = [ring[0]]
    for p in ring[1:]:
        if abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) >= tol:
            out.append(p)
    if out[-1] != ring[-1]:
        out.append(ring[-1])
    return [[round(x, 3), round(y, 3)] for x, y in out] if len(out) >= 4 else None


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    js = RAW / "gadm41_IND_2.json"
    if not js.exists():
        z = RAW / "gadm41_IND_2.json.zip"
        urllib.request.urlretrieve(URL, z)
        zipfile.ZipFile(z).extractall(RAW)
    fc = json.loads(js.read_text(encoding="utf-8"))
    LO, LA = np.meshgrid(LON12, LAT12)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    idx = np.full(LA.size, -1, np.int16)
    meta, feats = [], []
    for k, f in enumerate(fc["features"]):
        pr = f["properties"]
        hit = np.zeros(LA.size, bool)
        xs, ys, simple = [], [], []
        for poly in rings(f["geometry"]):
            outer = np.asarray(poly[0])
            xs.append(outer[:, 0])
            ys.append(outer[:, 1])
            lo0, la0 = outer.min(0)
            lo1, la1 = outer.max(0)
            box = (pts[:, 0] >= lo0) & (pts[:, 0] <= lo1) & (pts[:, 1] >= la0) & (pts[:, 1] <= la1)
            if box.any():
                inside = MPath(outer).contains_points(pts[box])
                for hole in poly[1:]:
                    inside &= ~MPath(np.asarray(hole)).contains_points(pts[box])
                hit[np.nonzero(box)[0][inside]] = True
            s = simplify(poly[0])
            if s:
                simple.append([s])
        x, y = np.concatenate(xs), np.concatenate(ys)
        clat, clon = float(y.mean()), float(x.mean())
        if not hit.any():                                     # small district: nearest cell
            hit[int(np.abs(LAT12 - clat).argmin()) * len(LON12) + int(np.abs(LON12 - clon).argmin())] = True
        idx[hit & (idx < 0)] = k
        meta.append({"id": k, "gid": pr["GID_2"], "district": spaced(pr["NAME_2"]), "state": spaced(pr["NAME_1"]),
                     "lat": round(clat, 3), "lon": round(clon, 3),
                     "cell": [int(np.abs(LAT12 - clat).argmin()), int(np.abs(LON12 - clon).argmin())]})
        if simple:
            feats.append({"type": "Feature", "properties": {"id": k, "district": meta[-1]["district"],
                                                            "state": meta[-1]["state"]},
                          "geometry": {"type": "MultiPolygon", "coordinates": simple}})
    idx = idx.reshape(LA.shape)
    for m in meta:
        m["n_cells"] = int((idx == m["id"]).sum())
    np.savez_compressed(ROOT / "data/real/boundary/districts_g12.npz", index=idx, source=URL,
                        synthetic="false")
    out = ROOT / "backend/static"
    (out / "districts.json").write_text(json.dumps({"source": URL, "licence": "GADM (non-commercial)",
                                                    "grid": "G12 0.12 deg", "districts": meta}))
    (out / "districts.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"{len(meta)} districts, {int((idx >= 0).sum())} G12 cells assigned, "
          f"{sum(m['n_cells'] == 0 for m in meta)} with no cell, "
          f"geojson {(out / 'districts.geojson').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
