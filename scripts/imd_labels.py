"""IMD-consistent heat/cold labels for the SYNTHETIC cases (BRIEF4 Phase 3).

The original labels (event_mask in fcst_12km.nc) are the injected anomaly footprint at each 6-h
step. IMD declares a heat/cold wave from DAILY Tmax/Tmin: departure from normal >= 4.5 C (severe
>= 6.5 C) together with a regional absolute threshold, on at least 2 consecutive days. The new
labels apply exactly that rule (pipeline.tracker2.imd_field: centred 24-h Tmax/Tmin, ERA5
1990-2019 daily normals, plains/coastal/hill thresholds) inside the injected event footprint
(dilated by 2 cells), with the 2-day (8-step) persistence rule.

    python scripts/imd_labels.py [case ...]    -> data/synthetic/{case}/labels_imd.npz
        event_mask_imd (M, T, y, x) and event_mask_imd_truth (T, y, x), packed bits;
        the old labels are kept unchanged in fcst_12km.nc / labels.json.
Also reports, per case, whether the truth event ever meets the IMD criteria (heat_04 check).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.anomaly import Climatology  # noqa: E402
from pipeline.evaluate2 import orog12  # noqa: E402
from pipeline.splits import ALL  # noqa: E402
from pipeline.tracker2 import cached_normals, imd_field, regions, trailing  # noqa: E402
from synth.grids import LAT12, LON12  # noqa: E402

SYN = ROOT / "data/synthetic"


def persist(m, n=8):
    """Keep cells that belong to a run of >= n consecutive steps (IMD 2-day rule)."""
    run = np.zeros(m.shape, np.int16)
    for t in range(len(m)):
        run[t] = np.where(m[t], (run[t - 1] + 1) if t else 1, 0)
    keep = np.zeros_like(m)
    end = run >= n
    for t in range(len(m) - 1, -1, -1):             # propagate run membership backwards
        keep[t] = end[t] | (m[t] & (keep[t + 1] if t + 1 < len(m) else False))
    return keep & m


def imd_mask(t2m, times, clim, reg, hz, footprint):
    f = imd_field(t2m, times, clim, reg, hz)            # 0 where the absolute criterion fails
    fp = np.array([binary_dilation(x, iterations=2) for x in footprint])
    return persist((f >= 4.5) & fp)


def main():
    cases = sys.argv[1:] or [c for c in ALL if not c.startswith("cyc") and c != "amphan_replay"]
    clim = Climatology()
    reg = regions(orog12(), LAT12, LON12)
    report = {}
    for c in cases:
        lab = json.loads((SYN / c / "labels.json").read_text())
        hz = lab["hazard"]
        if hz == "tropical_cyclone":
            continue
        times = pd.to_datetime(lab["valid_times"])
        with xr.open_dataset(SYN / c / "fcst_12km.nc", decode_timedelta=False) as ds:
            tt = ds.t2m_truth.values.astype(np.float64)
            tm = ds.event_mask_truth.values.astype(bool)
            truth = imd_mask(tt, times, clim, reg, hz, tm)
            M = ds.sizes["number"]
            ens = np.zeros((M,) + truth.shape, bool)
            for m in range(M):
                ens[m] = imd_mask(ds.t2m.isel(number=m).values.astype(np.float64), times, clim, reg, hz,
                                  ds.event_mask.isel(number=m).values.astype(bool))
        kind = "max" if hz == "heat_dome" else "min"
        ext = trailing(tt, kind) - 273.15
        dep = trailing(tt, kind) - cached_normals(clim, times, kind)
        inside = tm.any(0)
        rec = {"hazard": hz, "old_label_cells": int(tm.sum()), "imd_label_cells": int(truth.sum()),
               "imd_steps": int(truth.any((1, 2)).sum()), "old_steps": int(tm.any((1, 2)).sum()),
               "truth_peak_abs_C": float(ext[:, inside].max() if hz == "heat_dome" else ext[:, inside].min()),
               "truth_peak_departure_C": float((dep if hz == "heat_dome" else -dep)[:, inside].max()),
               "meets_imd": bool(truth.any())}
        if hz == "heat_dome":
            plains = (reg == 1) & inside
            rec["plains_cells_ge_40C"] = int((ext[:, plains] >= 40).any(0).sum())
        np.savez_compressed(SYN / c / "labels_imd.npz", event_mask_imd=np.packbits(ens, axis=-1),
                            event_mask_imd_truth=np.packbits(truth, axis=-1), shape=np.array(ens.shape),
                            rule="IMD: daily Tmax/Tmin departure >= 4.5 C + regional absolute threshold, "
                                 ">= 2 days, inside the injected footprint (+2 cells)", synthetic="true")
        report[c] = rec
        print(c, rec, flush=True)
    out = ROOT / "reports/imd_labels.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **report}, indent=1))


def load_imd(case):
    z = np.load(SYN / case / "labels_imd.npz")
    M, T, Y, X = z["shape"]
    ens = np.unpackbits(z["event_mask_imd"], axis=-1)[..., :X].astype(bool)
    truth = np.unpackbits(z["event_mask_imd_truth"], axis=-1)[..., :X].astype(bool)
    return ens, truth


if __name__ == "__main__":
    main()
