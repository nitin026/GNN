"""Amplitude-preserving 12 km -> 5 km downscaling (BRIEF2 Phase 3).

Models (all end in the same exact conservation projection):
  (a) bicubic + DEM lapse-rate baseline (t2m corrected by -6.5 K/km x (DEM5 - DEM12 upsampled))
  (b) U-Net predicting the residual over the bicubic field, conditioned on the DEM
  (c) CorrDiff-style: (b) as the mean + a DDPM on the residual y - mean, conditioned on
      [mean, bicubic, DEM]; sampled with DDIM
Conservation projection (exact, per variable, per 12 km cell, factor 3):
  additive       y <- y + up(x12 - avgpool(y))                     t2m, u10, v10, msl
  multiplicative y <- y * up(x12 / avgpool(y))  (y >= 0 first)     tp (keeps rain >= 0)
so avgpool(output) == 12 km input to float rounding.
Losses: MSE + spectral (log power-spectrum difference) + extreme (pinball loss at q = 0.99 on
the pixel residual, which penalises under-predicting peaks).
Variables / units follow the training bundle (t2m degC, u10, v10 m/s, msl hPa-1000, tp mm/6h).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARS = ["t2m", "u10", "v10", "msl", "tp"]
TP = VARS.index("tp")
K = 3
LAPSE = -6.5e-3                     # K per m


# ----------------------------------------------------------------------------- projection
def avgpool(y, k=K):
    return F.avg_pool2d(y, k)


def up(x, k=K):
    return x.repeat_interleave(k, -2).repeat_interleave(k, -1)


def project(y, x12, eps=1e-6):
    """Exact conservation: avgpool(project(y, x12)) == x12 (float rounding)."""
    out = y.clone()
    add = [i for i in range(y.shape[1]) if i != TP]
    out[:, add] = y[:, add] + up(x12[:, add] - avgpool(y[:, add]))
    r = torch.clamp(y[:, TP:TP + 1], min=0.0)
    pr = avgpool(r)
    ratio = torch.where(pr > eps, x12[:, TP:TP + 1] / pr.clamp(min=eps), torch.zeros_like(pr))
    flat = up(x12[:, TP:TP + 1])                   # where the guess is ~0, fall back to flat rain
    out[:, TP:TP + 1] = torch.where(up(pr) > eps, r * up(ratio), flat)
    return out


# ----------------------------------------------------------------------------- baseline (a)
def bicubic(x12, k=K):
    return F.interpolate(x12, scale_factor=k, mode="bicubic", align_corners=False)


def baseline(x12, dem12, dem5):
    y = bicubic(x12)
    y[:, TP] = y[:, TP].clamp(min=0)
    y[:, 0] = y[:, 0] + LAPSE * (dem5[:, 0] - bicubic(dem12)[:, 0])
    return project(y, x12)


# ----------------------------------------------------------------------------- U-Net
class Block(nn.Module):
    def __init__(self, i, o, temb=0):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.n1 = nn.GroupNorm(8, o)
        self.n2 = nn.GroupNorm(8, o)
        self.t = nn.Linear(temb, o) if temb else None
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, te=None):
        h = F.silu(self.n1(self.c1(x)))
        if self.t is not None and te is not None:
            h = h + self.t(te)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """3-level U-Net (144 -> 72 -> 36), optional timestep embedding for diffusion."""

    def __init__(self, cin, cout, base=32, temb=0):
        super().__init__()
        self.temb = temb
        if temb:
            self.tmlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        c = [base, 2 * base, 4 * base]
        self.inp = nn.Conv2d(cin, c[0], 3, padding=1)
        self.d1 = Block(c[0], c[0], temb)
        self.d2 = Block(c[0], c[1], temb)
        self.d3 = Block(c[1], c[2], temb)
        self.mid = Block(c[2], c[2], temb)
        self.u2 = Block(c[2] + c[1], c[1], temb)
        self.u1 = Block(c[1] + c[0], c[0], temb)
        self.out = nn.Conv2d(c[0], cout, 3, padding=1)

    def tembed(self, t):
        half = self.temb // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None]
        return self.tmlp(torch.cat([a.sin(), a.cos()], 1))

    def forward(self, x, t=None):
        te = self.tembed(t) if (self.temb and t is not None) else None
        h0 = self.d1(self.inp(x), te)
        h1 = self.d2(F.avg_pool2d(h0, 2), te)
        h2 = self.d3(F.avg_pool2d(h1, 2), te)
        h = self.mid(h2, te)
        h = self.u2(torch.cat([F.interpolate(h, scale_factor=2), h1], 1), te)
        h = self.u1(torch.cat([F.interpolate(h, scale_factor=2), h0], 1), te)
        return self.out(h)


class Normalizer:
    """Per-variable standardisation (fitted on TRAIN), plus DEM scaling."""

    def __init__(self, mean, std):
        self.m = torch.tensor(mean, dtype=torch.float32)[None, :, None, None]
        self.s = torch.tensor(std, dtype=torch.float32)[None, :, None, None]

    def to(self, dev):
        self.m, self.s = self.m.to(dev), self.s.to(dev)
        return self

    def n(self, x):
        return (x - self.m) / self.s

    def d(self, z):
        return z * self.s + self.m


def cond_input(x12, dem12, dem5, norm):
    """Network input at 5 km: normalised bicubic fields, DEM (km) and DEM anomaly (km)."""
    b = bicubic(x12)
    return torch.cat([norm.n(b), dem5 / 1000.0, (dem5 - bicubic(dem12)) / 1000.0], 1), b


class UNetDownscaler(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = UNet(len(VARS) + 2, len(VARS), base)

    def forward(self, x12, dem12, dem5, norm, do_project=True):
        inp, b = cond_input(x12, dem12, dem5, norm)
        y = norm.d(norm.n(b) + self.net(inp))
        return project(y, x12) if do_project else y


# ----------------------------------------------------------------------------- losses
def spectral_loss(pred, target):
    P = torch.fft.rfft2(pred).abs() ** 2
    T = torch.fft.rfft2(target).abs() ** 2
    return (torch.log1p(P) - torch.log1p(T)).abs().mean()


def pinball(pred, target, q=0.99):
    d = target - pred
    return torch.maximum(q * d, (q - 1) * d).mean()


def downscale_loss(pred_n, target_n, w_spec=0.1, w_ext=0.5):
    """pred/target normalised. MSE + spectral + extreme (pinball q=0.99 on top-1 % target pixels)."""
    mse = F.mse_loss(pred_n, target_n)
    spec = spectral_loss(pred_n, target_n)
    thr = torch.quantile(target_n.flatten(2), 0.99, dim=2)[:, :, None, None]
    m = target_n >= thr
    ext = pinball(pred_n[m], target_n[m]) if m.any() else pred_n.sum() * 0
    return mse + w_spec * spec + w_ext * ext, {"mse": float(mse), "spec": float(spec), "ext": float(ext)}


# ----------------------------------------------------------------------------- diffusion (c)
class ResidualDiffusion(nn.Module):
    """DDPM (Ho et al. 2020) on the normalised residual r = n(y) - n(mean); DDIM sampling
    (Song et al. 2021). Conditioning: [n(mean), n(bicubic), DEM, DEM anomaly]."""

    def __init__(self, base=32, T=500):
        super().__init__()
        self.T = T
        self.net = UNet(len(VARS) + (2 * len(VARS) + 2), len(VARS), base, temb=64)
        beta = torch.linspace(1e-4, 0.02, T)
        self.register_buffer("abar", torch.cumprod(1 - beta, 0))

    def cond(self, mean_n, inp):
        return torch.cat([mean_n, inp], 1)

    def loss(self, r, c):
        t = torch.randint(0, self.T, (r.shape[0],), device=r.device)
        a = self.abar[t][:, None, None, None]
        eps = torch.randn_like(r)
        xt = a.sqrt() * r + (1 - a).sqrt() * eps
        return F.mse_loss(self.net(torch.cat([xt, c], 1), t), eps)

    @torch.no_grad()
    def sample(self, c, steps=25, gen=None):
        ts = torch.linspace(self.T - 1, 0, steps).long().to(c.device)
        x = torch.randn(c.shape[0], len(VARS), c.shape[2], c.shape[3], device=c.device, generator=gen)
        for i, t in enumerate(ts):
            a = self.abar[t]
            eps = self.net(torch.cat([x, c], 1), t.repeat(c.shape[0]))
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=c.device)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps          # DDIM, eta = 0
        return x


# ----------------------------------------------------------------------------- metrics
def radial_psd(a):
    """Radially averaged power of a batch of fields (N, H, W) -> (k, P)."""
    a = a - a.mean((-2, -1), keepdims=True)
    F_ = np.abs(np.fft.fftshift(np.fft.fft2(a), axes=(-2, -1))) ** 2
    n = a.shape[-1]
    ky, kx = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")
    k = np.hypot(ky, kx).astype(int)
    P = np.array([np.bincount(k.ravel(), f.ravel()) / np.bincount(k.ravel()) for f in F_]).mean(0)
    return np.arange(len(P)), P


def crps_ensemble(samples, y):
    """Fair CRPS of an ensemble (S, ...) against y (...), averaged."""
    s = np.asarray(samples)
    t1 = np.abs(s - y[None]).mean(0)
    t2 = np.abs(s[:, None] - s[None]).mean((0, 1))
    return float((t1 - 0.5 * t2).mean())
