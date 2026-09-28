"""Physics constraints for the 12 -> 5 km downscaler (BRIEF4 Phase 4): loss terms + violation rates.

Fields are the downscaler's physical units at 5 km (t2m degC, u10 / v10 m/s, msl hPa-1000,
tp mm/6h), tensors (B, 5, H, W); dem5 (B, 1, H, W) in m.

1. rain_noconv   rain > RAIN_THR where the moisture-flux convergence <= 0 (the PS example).
                 The downscaler has no humidity or 850 hPa wind, so the convergence is the
                 low-level wind convergence -div(V10) of the 12 km INPUT winds bicubically
                 interpolated (moisture taken as locally uniform): a documented proxy. Penalty:
                 mean of relu(tp - RAIN_THR) over those cells.
2. q <= q_sat    NOT APPLICABLE: humidity (q or RH) is not a downscaled variable here, so the
                 Clausius-Clapeyron bound cannot be violated by the model (reported as n/a).
                 q_sat(T, p) is implemented (qsat) for use once humidity is added.
3. div_smooth    wind-divergence smoothness: mean squared Laplacian of div(V10) at 5 km
                 (penalises grid-scale divergence noise). Violation: |div| above the 99.9th
                 percentile of the TRUTH |div| (per evaluation set).
4. lapse         lapse-rate consistency: the sub-12-km T2m anomaly should follow the terrain,
                 dT' ~ -6.5 K/km * dz'. Penalty: mean squared (dT' - LAPSE dz') where |dz'| >
                 DZ_MIN. Violation: cells with |dz'| > DZ_MIN where dT' has the WRONG sign
                 (warmer where higher, and vice versa) by more than 0.5 K.
The hard conservation projection (pipeline.downscale.project) stays as it is.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .downscale import TP, avgpool, bicubic, up

RAIN_THR = 1.0          # mm / 6 h
DZ_MIN = 200.0          # m
LAPSE = -6.5e-3         # K / m
DX5 = 4448.0            # m, G5 (0.04 deg) meridional spacing; zonal uses cos(lat) ~ 1 near 20 N (proxy)


def div(u, v, dx=DX5):
    du = (u[..., :, 2:] - u[..., :, :-2]) / (2 * dx)
    dv = (v[..., 2:, :] - v[..., :-2, :]) / (2 * dx)
    return F.pad(du, (1, 1, 0, 0)) + F.pad(dv, (0, 0, 1, 1))


def conv_proxy(x12):
    """-div(V10) of the bicubic 12 km input winds (B, 1, H5, W5) [1/s]."""
    b = bicubic(x12)
    return -div(b[:, 1:2], b[:, 2:3])


def laplacian(a):
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=a.dtype, device=a.device)[None, None]
    return F.conv2d(F.pad(a, (1, 1, 1, 1), mode="replicate"), k)


def subgrid(a):
    return a - up(avgpool(a))


def physics_loss(y, x12, dem5, w):
    """y: physical 5 km prediction. w: dict of weights (rain, div, lapse). Returns (loss, parts)."""
    parts = {}
    conv = conv_proxy(x12)
    tp = y[:, TP:TP + 1]
    parts["rain"] = (F.relu(tp - RAIN_THR) * (conv <= 0)).mean()
    d = div(y[:, 1:2], y[:, 2:3]) * 1e4            # ~1e-4 1/s -> O(1)
    parts["div"] = laplacian(d).pow(2).mean()
    dz = subgrid(dem5)
    dT = subgrid(y[:, 0:1])
    m = (dz.abs() > DZ_MIN).float()
    parts["lapse"] = ((dT - LAPSE * dz) ** 2 * m).sum() / m.sum().clamp(min=1)
    loss = sum(w.get(k, 0.0) * v for k, v in parts.items())
    return loss, {k: float(v) for k, v in parts.items()}


def qsat(t_c, p_hpa):
    """Saturation specific humidity [kg/kg] (Magnus / Clausius-Clapeyron, Bolton 1980)."""
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    return 0.622 * es / (p_hpa - 0.378 * es)


class ViolationCounter:
    """Streaming violation rates. Call add(pred, x12, dem5, truth=...) per batch."""

    def __init__(self, div_thr=None):
        self.n = {"rain_noconv": [0, 0], "div": [0, 0], "lapse": [0, 0], "neg_rain": [0, 0]}
        self.div_thr = div_thr

    @torch.no_grad()
    def add(self, y, x12, dem5):
        conv = conv_proxy(x12)
        tp = y[:, TP:TP + 1]
        wet = tp > RAIN_THR
        self.n["rain_noconv"][0] += int((wet & (conv <= 0)).sum())
        self.n["rain_noconv"][1] += int(wet.sum())
        self.n["neg_rain"][0] += int((tp < 0).sum())
        self.n["neg_rain"][1] += int(tp.numel())
        if self.div_thr is not None:
            d = div(y[:, 1:2], y[:, 2:3]).abs()
            self.n["div"][0] += int((d > self.div_thr).sum())
            self.n["div"][1] += int(d.numel())
        dz, dT = subgrid(dem5), subgrid(y[:, 0:1])
        m = dz.abs() > DZ_MIN
        wrong = (torch.sign(dT) == torch.sign(dz)) & (dT.abs() > 0.5)      # warmer where higher
        self.n["lapse"][0] += int((wrong & m).sum())
        self.n["lapse"][1] += int(m.sum())

    def rates(self):
        out = {k: (100.0 * a / b if b else None) for k, (a, b) in self.n.items()}
        out["q_le_qsat"] = "n/a (humidity is not a downscaled variable)"
        out["denominators"] = {"rain_noconv": "% of wet cells (tp > 1 mm/6h) with convergence <= 0",
                               "div": "% of cells with |div V10| > truth p99.9",
                               "lapse": "% of cells with |dz'| > 200 m where T2m's sub-grid anomaly has the wrong sign (> 0.5 K)",
                               "neg_rain": "% of cells with rain < 0"}
        return out
