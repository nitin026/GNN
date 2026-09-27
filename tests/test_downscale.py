"""Downscaling: the projection layer enforces avgpool(5 km) == 12 km exactly, rain >= 0."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.downscale import (Normalizer, ResidualDiffusion, UNetDownscaler, avgpool, baseline,  # noqa: E402
                                cond_input, project)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x12 = torch.randn(4, 5, 16, 16, generator=g) * 3
    x12[:, 4] = torch.rand(4, 16, 16, generator=g) * 20           # rain >= 0
    x12[:, 4, :4, :4] = 0.0                                        # dry cells
    dem12 = torch.rand(4, 1, 16, 16, generator=g) * 2000
    dem5 = torch.rand(4, 1, 48, 48, generator=g) * 2000
    return x12, dem12, dem5


def test_projection_is_exact_for_arbitrary_input():
    x12, _, _ = _batch()
    y = torch.randn(4, 5, 48, 48) * 10                             # arbitrary, even negative rain
    out = project(y, x12)
    assert torch.allclose(avgpool(out), x12, atol=1e-4)
    assert out[:, 4].min() >= 0
    assert torch.all(out[:, 4, :12, :12] == 0)                     # dry 12 km cells stay dry


def test_baseline_and_models_conserve():
    x12, d12, d5 = _batch(1)
    assert torch.allclose(avgpool(baseline(x12, d12, d5)), x12, atol=1e-4)
    norm = Normalizer(np.zeros(5), np.ones(5))
    torch.manual_seed(0)
    unet = UNetDownscaler(base=8).eval()
    with torch.no_grad():
        y = unet(x12, d12, d5, norm)
        assert y.shape == (4, 5, 48, 48)
        assert torch.allclose(avgpool(y), x12, atol=1e-4) and y[:, 4].min() >= 0
        diff = ResidualDiffusion(base=8, T=20).eval()
        inp, _ = cond_input(x12, d12, d5, norm)
        r = diff.sample(diff.cond(norm.n(y), inp), steps=3)
        s = project(norm.d(norm.n(y) + r), x12)
        assert torch.allclose(avgpool(s), x12, atol=1e-4)
