"""Physical constraints and flags on every synthetic file: rain >= 0, RH <= 100 %,
synthetic="true", required shapes."""
import json

import netCDF4
import numpy as np


def test_rain_nonnegative_and_rh_capped(cases):
    for case in cases:
        for fn in ("truth_5km.nc", "fcst_12km.nc"):
            ds = netCDF4.Dataset(case / fn)
            tp = ds.variables["tp"]
            tp.set_auto_maskandscale(False)
            assert int(tp[:].min()) >= 0, f"{case.name}/{fn}: negative rain"
            assert float(ds.variables["tp_truth"][:].min()) >= 0 if "tp_truth" in ds.variables else True
            if "r850" in ds.variables:
                r = ds.variables["r850"]
                r.set_auto_maskandscale(False)
                assert int(r[:].max()) * float(r.scale_factor) <= 100.0 + 1e-9
            ds.close()


def test_synthetic_flag_and_shapes(cases):
    for case in cases:
        f = netCDF4.Dataset(case / "fcst_12km.nc")
        t = netCDF4.Dataset(case / "truth_5km.nc")
        assert f.getncattr("synthetic") == "true" and t.getncattr("synthetic") == "true"
        assert f.dimensions["number"].size == 20
        np.testing.assert_array_equal(f.variables["step"][:], np.arange(0, 241, 6))
        assert t.dimensions["latitude"].size == 3 * f.dimensions["latitude"].size
        f.close()
        t.close()
        lab = json.loads((case / "labels.json").read_text())
        assert lab["synthetic"] == "true"
        for k in ("hazard", "truth_track", "bbox_4d", "peak", "members"):
            assert lab.get(k) is not None, f"{case.name}: labels missing {k}"
        roles = [m["role"] for m in lab["members"]]
        assert "miss" in roles and "false_alarm" in roles


def test_physics_helpers():
    from synth.physics import cap_rh, place_rain
    rr = np.array([[5.0, 5.0], [5.0, -1.0]])
    mfc = np.array([[1.0, -1.0], [1.0, 1.0]])
    acc = place_rain(rr, mfc, np.ones_like(rr))
    assert acc[0, 1] == 0 and acc.min() >= 0 and acc[0, 0] == 30.0
    assert cap_rh(np.array([120.0, -5.0, 50.0])).tolist() == [100.0, 0.0, 50.0]
