"""BRIEF4 Phases 3-4: calibration (pipeline/calibration.py) and physics terms (pipeline/physics.py)."""
import numpy as np
import torch

from pipeline import calibration as cal
from pipeline.physics import ViolationCounter, div, physics_loss, qsat


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(float)       # true p = sigmoid(z)
    p_over = 1 / (1 + np.exp(-3 * z))                                     # over-confident by T = 3
    c = cal.Temperature().fit(p_over, y)
    assert 2.5 < c.T < 3.5
    assert cal.brier(c(p_over), y) < cal.brier(p_over, y)
    s = cal.slope(cal.reliability(c(p_over), y))
    assert 0.9 < s < 1.1


def test_isotonic_and_band_selection():
    rng = np.random.default_rng(1)
    p = rng.random(3000)
    y = (rng.random(3000) < p ** 2).astype(float)                         # miscalibrated, monotone
    pc = {f"c{i}": p[i::3] for i in range(3)}
    yc = {f"c{i}": y[i::3] for i in range(3)}
    c, scores, base = cal.fit_band(pc, yc)
    assert set(scores) == {"temperature", "isotonic"} and 0 < base < 1
    assert cal.brier(c(p), y) < cal.brier(p, y)
    d = c.to_json()
    c2 = cal.from_json(d)
    assert np.allclose(c(p[:50]), c2(p[:50]))
    applied = cal.apply({b: c for b, _, _ in cal.BANDS}, p[:10], np.array([0, 60, 72, 78, 120, 168, 174, 200, 240, 6]))
    assert applied.shape == (10,)


def test_divergence_and_physics_loss():
    H = 30
    yy, xx = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")
    u = torch.tensor(xx * 4448.0 * 1e-5, dtype=torch.float32)[None, None]  # du/dx = 1e-5 1/s
    v = torch.zeros_like(u)
    d = div(u, v)[0, 0, 2:-2, 2:-2]
    assert torch.allclose(d, torch.full_like(d, 1e-5), rtol=1e-3)
    # rain only where the (input) wind converges -> no rain penalty; rain in divergence -> penalty
    x12 = torch.zeros(1, 5, 12, 12)
    x12[:, 1] = -torch.arange(12.0)[None, :].repeat(12, 1)                  # u decreasing east: convergence
    y = torch.zeros(1, 5, 36, 36)
    y[:, 4] = 5.0
    dem = torch.zeros(1, 1, 36, 36)
    l_conv, parts_conv = physics_loss(y, x12, dem, {"rain": 1.0})
    x12[:, 1] = -x12[:, 1]                                                   # now divergence everywhere
    l_div, parts_div = physics_loss(y, x12, dem, {"rain": 1.0})
    assert parts_div["rain"] > parts_conv["rain"]


def test_violation_counter_and_qsat():
    vc = ViolationCounter(div_thr=1.0)
    x12 = torch.zeros(1, 5, 12, 12)
    y = torch.zeros(1, 5, 36, 36)
    y[:, 4] = -0.1                                                           # negative rain
    vc.add(y, x12, torch.zeros(1, 1, 36, 36))
    r = vc.rates()
    assert r["neg_rain"] == 100.0 and r["div"] == 0.0 and "n/a" in r["q_le_qsat"]
    assert 0.014 < qsat(20.0, 1000.0) < 0.016                                # ~14.7 g/kg at 20 C, 1000 hPa
