"""Tracker v2: hysteresis + Kalman linking on an idealised case, IMD domain, rain QM factor."""
import json

import numpy as np

from pipeline.tracker2 import (KalmanCV, cell_area_km2, detect_hysteresis, imd_domain,
                               link_kalman, regions)


def _blob(LA, LO, c, s=1.5):
    return 6 * np.exp(-((LA - c[0]) ** 2 + (LO - c[1]) ** 2) / (2 * s ** 2))


def test_hysteresis_and_kalman_track_one_moving_object():
    lat = np.arange(0, 40, 0.12) + 0.06
    lon = np.arange(60, 100, 0.12) + 0.06
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    area = cell_area_km2(lat, lon)
    rng = np.random.default_rng(0)
    frames = []
    for t in range(10):
        f = _blob(LA, LO, (15 + 0.4 * t, 85 - 0.6 * t)) + rng.normal(0, 0.3, LA.shape)
        if t == 5:
            f = np.minimum(f, 3.9)                    # weak step: below 'high', above 'low'
        if t == 6:
            f = rng.normal(0, 0.3, LA.shape)          # missed step (gap)
        frames.append(detect_hysteresis(f, lat, lon, high=4.5, low=3.0,
                                        min_area_km2=5000, area=area))
    assert all(not o["core"] for o in frames[5])      # weak object cannot start a track
    tracks = link_kalman(frames, key="centroid", max_gap=1, r=60.0)
    tracks = [tr for tr in tracks if len(tr) >= 3]
    assert len(tracks) == 1, [len(t) for t in tracks]  # noise specks never start a track
    assert [t for t, _ in tracks[0]] == [0, 1, 2, 3, 4, 5, 7, 8, 9]   # gap at 6 bridged


def test_kalman_prediction_follows_constant_velocity():
    kf = KalmanCV(15.0, 85.0, r=10.0)
    for t in range(6):
        kf.predict()
        kf.update(15.0 + 0.5 * (t + 1), 85.0)
    kf.predict()
    la, lo = kf.to_ll(kf.x[:2])
    assert abs(la - 18.5) < 0.2 and abs(lo - 85.0) < 0.2


def test_imd_domain_excludes_tibet_and_tarim_keeps_india():
    import xarray as xr
    from pathlib import Path
    o = xr.open_dataset(Path(__file__).resolve().parents[1] / "data/real/dem/dem_g12.nc").orog.values
    lat = np.arange(333) * 0.12 + 0.06
    lon = lat + 60
    reg = regions(o, lat, lon)
    at = lambda a, b: reg[np.searchsorted(lat, a), np.searchsorted(lon, b)]
    for p in [(28.6, 77.2), (34.0, 74.8), (22.6, 88.4), (17.4, 78.5)]:    # Delhi Srinagar Kolkata Hyd.
        assert at(*p) > 0, p
    for p in [(39.0, 81.0), (29.6, 91.1)]:                                  # Tarim, Lhasa
        assert at(*p) == 0, p
    assert imd_domain(o, lat).shape == o.shape


def test_rain_qm_factor_monotone_and_bounded():
    from synth.rain_calibration import QM, apply_factor
    t = json.loads(QM.read_text())
    D = np.linspace(1, 600, 200)
    out = D * apply_factor(D, t)
    assert np.all(np.diff(out) >= -1e-9)             # mapping preserves rank order
    assert np.all((apply_factor(D, t) >= 0.1) & (apply_factor(D, t) <= 10))
