"""Package a small (<2 GB) training bundle for Colab/Kaggle (or local) training.

    python scripts/package_training_data.py [--patches-per-case 300] [--out bundle/]

Contents (all SYNTHETIC; the bundle carries synthetic=true in manifest.json):
  downscale_{train,val,test}.npz
      x12  (N, 5, 48, 48)    12 km truth patch  (t2m degC, u10, v10, msl hPa-1000, tp mm/6h)
      y5   (N, 5, 144, 144)  matching 5 km truth patch (exactly 3x3 per 12 km cell)
      dem12 (N, 1, 48, 48), dem5 (N, 1, 144, 144)  Copernicus DEM (m, real)
      mask5 (N, 1, 144, 144) exact event mask; meta (N, 4) = case index, step, i0, j0
  graphs_*.npz  (added by Phase 2: object graphs for the GNN tracker, if present)
  manifest.json  splits, variables, offsets, seeds, sizes, provenance
Fields are float16 after subtracting fixed offsets. Conservation is therefore exact only
to float16 rounding in the bundle; the full-precision NetCDF files remain the reference.
Splits come from pipeline/splits.py (whole cases, no leakage).
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.splits import TEST, TRAIN, VAL  # noqa: E402

SYN = ROOT / "data" / "synthetic"
VARS = ["t2m", "u10", "v10", "msl", "tp"]
OFFSET = {"t2m": 273.15, "u10": 0.0, "v10": 0.0, "msl": 100000.0, "tp": 0.0}
SCALE = {"t2m": 1.0, "u10": 1.0, "v10": 1.0, "msl": 0.01, "tp": 1.0}  # msl -> hPa
P12, K = 48, 3
SEED = 2026


def decode(v, idx):
    v.set_auto_maskandscale(False)
    return v[idx].astype(np.float64) * float(v.scale_factor) + float(v.add_offset)


def case_patches(case, n, rng, dem5, dem12):
    t5 = netCDF4.Dataset(SYN / case / "truth_5km.nc")
    f12 = netCDF4.Dataset(SYN / case / "fcst_12km.nc")
    nsteps = t5.dimensions["step"].size
    m12 = f12.variables["event_mask_truth"][:].astype(bool)
    N12 = f12.dimensions["latitude"].size
    xs, ys, ms, d12, d5, meta = [], [], [], [], [], []
    steps = np.sort(rng.integers(0, nsteps, n))   # sorted -> field cache reuse
    cache = {}
    for k, s in enumerate(steps):
        s = int(s)
        # half of the patches centred on the event (when present), the rest random
        ev = np.argwhere(m12[s])
        if k % 2 == 0 and len(ev):
            ci, cj = ev[rng.integers(len(ev))]
            i0 = int(np.clip(ci - P12 // 2 + rng.integers(-8, 9), 0, N12 - P12))
            j0 = int(np.clip(cj - P12 // 2 + rng.integers(-8, 9), 0, N12 - P12))
        else:
            i0, j0 = (int(v) for v in rng.integers(0, N12 - P12, 2))
        if s not in cache:
            cache.clear()
            cache[s] = ({v: decode(t5.variables[v], s) for v in VARS},
                        {v: f12.variables[f"{v}_truth"][s].astype(np.float64) for v in VARS},
                        t5.variables["event_mask"][s].astype(np.uint8))
        f5, f12v, msk = cache[s]
        sl12 = (slice(i0, i0 + P12), slice(j0, j0 + P12))
        sl5 = (slice(K * i0, K * (i0 + P12)), slice(K * j0, K * (j0 + P12)))
        xs.append(np.stack([(f12v[v][sl12] - OFFSET[v]) * SCALE[v] for v in VARS]))
        ys.append(np.stack([(f5[v][sl5] - OFFSET[v]) * SCALE[v] for v in VARS]))
        ms.append(msk[sl5][None])
        d12.append(dem12[sl12][None])
        d5.append(dem5[sl5][None])
        meta.append((0, s, i0, j0))
    t5.close()
    f12.close()
    return (np.array(xs, np.float16), np.array(ys, np.float16), np.array(ms, np.uint8),
            np.array(d12, np.float16), np.array(d5, np.float16), np.array(meta, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-per-case", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "bundle"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dem5 = xr.open_dataset(ROOT / "data/real/dem/dem_g5.nc").orog.values.astype(np.float32)
    dem12 = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)
    rng = np.random.default_rng(SEED)
    sizes = {}
    for split, cases in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        parts = []
        for ci, c in enumerate(cases):
            p = case_patches(c, a.patches_per_case, rng, dem5, dem12)
            p[5][:, 0] = ci
            parts.append(p)
            print(f"{split}: {c} {len(p[0])} patches", flush=True)
        arrs = [np.concatenate([p[i] for p in parts]) for i in range(6)]
        fn = out / f"downscale_{split}.npz"
        np.savez_compressed(fn, x12=arrs[0], y5=arrs[1], mask5=arrs[2], dem12=arrs[3],
                            dem5=arrs[4], meta=arrs[5], cases=np.array(cases))
        sizes[fn.name] = round(fn.stat().st_size / 1e6, 1)
    for g in sorted((ROOT / "data" / "graphs").glob("*.npz")) if (ROOT / "data/graphs").exists() else []:
        (out / g.name).write_bytes(g.read_bytes())
        sizes[g.name] = round(g.stat().st_size / 1e6, 1)
    manifest = {"synthetic": "true", "splits": {"train": TRAIN, "val": VAL, "test": TEST},
                "variables": VARS, "offsets": OFFSET, "scales": SCALE,
                "patch_12km": P12, "factor": K, "seed": SEED,
                "patches_per_case": a.patches_per_case, "sizes_mb": sizes,
                "note": "SYNTHETIC data from synth.generate (exact labels); DEM is real "
                        "Copernicus GLO-90. float16 after offsets."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # single zip for upload to Colab/Kaggle (npz are already compressed -> store)
    zp = out.parent / "training_bundle.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, f"bundle/{f.name}")
    total = zp.stat().st_size / 1e9
    print(f"bundle: {sizes}  zip {total:.2f} GB -> {zp}")
    if total >= 2.0:
        raise SystemExit(f"bundle too large ({total:.2f} GB >= 2 GB); lower --patches-per-case")


if __name__ == "__main__":
    main()
