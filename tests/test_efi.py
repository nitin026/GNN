"""EFI sanity: climate-like ensemble -> EFI ~ 0; +3 sigma shifted ensemble -> EFI > 0.8."""
import numpy as np

from pipeline.efi import efi_gaussian, efi_sample


def test_efi_climate_ensemble_is_zero():
    rng = np.random.default_rng(42)
    ens = rng.standard_normal((50, 2000))           # 50 members at 2000 independent points
    e = efi_gaussian(ens, np.zeros(2000), np.ones(2000))
    assert abs(e.mean()) < 0.02
    assert np.abs(e).max() < 0.6


def test_efi_shifted_ensemble_is_extreme():
    rng = np.random.default_rng(1)
    ens = rng.standard_normal((50, 500)) + 3.0
    e = efi_gaussian(ens, np.zeros(500), np.ones(500))
    assert e.min() > 0.8
    e_neg = efi_gaussian(-ens, np.zeros(500), np.ones(500))
    assert e_neg.max() < -0.8


def test_efi_bounds_and_sample_climate():
    rng = np.random.default_rng(3)
    clim = rng.standard_normal((5000, 10))
    assert np.all(efi_sample(np.full((20, 10), 100.0), clim) > 0.99)
    assert abs(efi_sample(rng.standard_normal((20, 10)), clim).mean()) < 0.15


def test_efi_gaussian_fast_path_matches_quantile_formula():
    import numpy as np
    from scipy.special import ndtri
    from pipeline.efi import _P, efi_from_quantiles, efi_gaussian
    rng = np.random.default_rng(3)
    m = rng.normal(size=(30, 40))
    s = rng.uniform(0.5, 2.0, (30, 40))
    ens = rng.normal(1.0, 1.5, (20, 30, 40)) * s + m
    q = m[None] + s[None] * ndtri(_P)[:, None, None]
    assert np.abs(efi_from_quantiles(ens, q) - efi_gaussian(ens, m, s)).max() < 1e-12
