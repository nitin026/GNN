"""avgpool(truth_5km) == truth at 12 km (exact block-average conservation)."""
import netCDF4
import numpy as np

from synth.grids import LAT5, LAT12, LON5, LON12, avgpool


def test_grids_nest_exactly():
    assert len(LAT5) == 3 * len(LAT12) and len(LON5) == 3 * len(LON12)
    np.testing.assert_allclose(avgpool(LAT5[None, :, None] * np.ones((1, 1, 3)))[0, :, 0], LAT12,
                               atol=1e-9)
    np.testing.assert_allclose(LON5.reshape(-1, 3).mean(1), LON12, atol=1e-9)


def test_avgpool_conserves_area_mean():
    a = np.random.default_rng(0).gamma(0.5, 3.0, size=(4, 999, 999))
    p = avgpool(a)
    assert p.shape == (4, 333, 333)
    np.testing.assert_allclose(p.mean(axis=(1, 2)), a.mean(axis=(1, 2)), rtol=1e-12)


def _decode(v, step):
    v.set_auto_maskandscale(False)
    return v[step].astype(np.float64) * float(v.scale_factor) + float(v.add_offset)


def test_case_files_conserve(cases):
    """For every case and variable: avgpool(decoded truth_5km) == <var>_truth in fcst_12km."""
    for case in cases:
        t5 = netCDF4.Dataset(case / "truth_5km.nc")
        f12 = netCDF4.Dataset(case / "fcst_12km.nc")
        for name in ("t2m", "msl", "tp", "u10", "v10", "r850"):
            if name not in t5.variables:
                continue
            for step in (0, 20, 40):
                fine = _decode(t5.variables[name], step)
                coarse = f12.variables[f"{name}_truth"][step].astype(np.float64)
                np.testing.assert_allclose(avgpool(fine), coarse, rtol=2e-7, atol=1e-4,
                                           err_msg=f"{case.name}:{name}:step{step}")
        t5.close()
        f12.close()
