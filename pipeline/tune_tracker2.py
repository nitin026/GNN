"""Choose tracker-v2 parameters on TRAIN + VAL cases only (never on TEST).

Objective per hazard: maximise mean truth IoU subject to < 10 spurious ensemble tracks per case
(tracks not overlapping the member's own event mask, summed over 20 members); if no setting meets
the constraint, take the one with the fewest spurious tracks. Members are subsampled (8 of 20) for
speed and counts are scaled to 20.

    python -m pipeline.tune_tracker2   -> reports/tracker2_params.json
"""
import itertools
import json
from pathlib import Path

import numpy as np

from .anomaly import Climatology
from .evaluate2 import load_case, orog12, evaluate
from .splits import TRAIN, VAL

ROOT = Path(__file__).resolve().parents[1]
GRID = {
    "tropical_cyclone": {"high": [2.0, 2.5], "low": [1.5, 2.0], "min_area_km2": [3000.0, 8000.0]},
    "heat_dome": {"high": [4.5], "low": [2.0, 3.0, 3.5], "min_area_km2": [20000.0, 50000.0]},
    "cold_wave": {"high": [4.5], "low": [2.0, 3.0, 3.5], "min_area_km2": [20000.0, 50000.0]},
}
HAZ = {"tropical_cyclone": "cyc", "heat_dome": "heat", "cold_wave": "cold"}
N_SUB = 8


def main():
    clim, oro = Climatology(), orog12()
    chosen, table = {}, []
    for hz, grid in GRID.items():
        cases = [c for c in TRAIN + VAL if c.startswith(HAZ[hz])]
        data = []
        for c in cases:
            d = load_case(c)
            sel = np.linspace(0, d["ens"].shape[0] - 1, N_SUB).astype(int)
            d["ens"], d["emask"], d["roles"] = d["ens"][sel], d["emask"][sel], d["roles"][sel]
            data.append(d)
        best = None
        for vals in itertools.product(*grid.values()):
            p = dict(zip(grid, vals))
            if p["low"] > p["high"]:
                continue
            rs = [evaluate(d, clim, oro, trackers=("v2",), params=p)["v2"] for d in data]
            iou = float(np.mean([r["iou_mean"] for r in rs]))
            spur = float(np.mean([r["ens_spurious"] for r in rs])) * 20 / N_SUB
            row = {"hazard": hz, **p, "iou_trainval": iou,
                   "iou_daily_trainval": float(np.mean([r["iou_daily"] for r in rs])),
                   "spurious_per_case_trainval": spur}
            table.append(row)
            print(row, flush=True)
            key = (spur < 10, iou if spur < 10 else -spur)
            if best is None or key > best[0]:
                best = (key, p, row)
        chosen[hz] = {"params": best[1], "trainval": best[2], "cases": cases}
    out = {"chosen": chosen, "grid": table, "note": "selected on TRAIN+VAL only; TEST untouched"}
    (ROOT / "reports" / "tracker2_params.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
