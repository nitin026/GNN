"""Tracker: exact recovery on an idealised moving blob, and IoU on a real synthetic case."""
import json

import numpy as np
import pandas as pd
import xarray as xr

from pipeline.track import detect, haversine_km, iou, link, main_track, track_masks


def test_tracker_recovers_moving_blobs():
    lat = np.linspace(0, 40, 201)
    lon = np.linspace(60, 100, 201)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    frames, truth = [], []
    for t in range(8):
        c1 = (15 + 0.5 * t, 85 - 0.5 * t)
        c2 = (30 - 0.3 * t, 70 + 0.6 * t)
        f = np.exp(-((LA - c1[0]) ** 2 + (LO - c1[1]) ** 2) / 2) + \
            np.exp(-((LA - c2[0]) ** 2 + (LO - c2[1]) ** 2) / 3)
        truth.append(f >= 0.5)
        frames.append(detect(f, lat, lon, 0.5))
    tracks = link(frames, max_km=300)
    assert len(tracks) == 2 and all(len(tr) == 8 for tr in tracks)
    m = np.zeros((8, 201, 201), bool)
    for tr in tracks:
        m |= track_masks(tr, 8, (201, 201))
    assert iou(m, np.array(truth)) > 0.99
    tr0 = min(tracks, key=lambda tr: tr[0][1]["centroid_lat"])
    for t, o in tr0:
        assert haversine_km(o["centroid_lat"], o["centroid_lon"], 15 + 0.5 * t, 85 - 0.5 * t) < 15


def test_tracker_iou_on_synthetic_case(cases):
    """Baseline tracker on the 12 km truth of a synthetic heat-dome case: the detected
    event must overlap the exact label mask (IoU > 0.3) with centroid error < 250 km."""
    from pipeline.anomaly import Climatology
    from pipeline.evaluate import evaluate_case
    case = next(c for c in cases if json.loads((c / "labels.json").read_text())["hazard"]
                == "heat_dome")
    res, _ = evaluate_case(case, Climatology())
    assert res["iou_mean"] > 0.3, res
    assert res["centroid_err_km_mean"] < 250, res
