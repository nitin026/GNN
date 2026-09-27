"""Spectral noise fields."""
import numpy as np


def spectral_noise(shape, rng, slope=-5.0 / 3.0, kmin=1.0, n=None):
    """Zero-mean, unit-std 2-D random field whose isotropic ENERGY spectrum E(k) ~ k**slope.

    For a 2-D field, E(k) ~ k * |F(k)|**2, so the Fourier amplitude is |F| ~ k**((slope-1)/2).
    ``n`` fields can be drawn at once (leading axis).
    """
    ny, nx = shape
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.rfftfreq(nx) * nx
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    amp = np.where(k >= kmin, np.maximum(k, kmin) ** ((slope - 1.0) / 2.0), 0.0)
    lead = () if n is None else (n,)
    ph = rng.standard_normal(lead + k.shape) + 1j * rng.standard_normal(lead + k.shape)
    f = np.fft.irfft2(ph * amp, s=shape)
    f -= f.mean(axis=(-2, -1), keepdims=True)
    f /= f.std(axis=(-2, -1), keepdims=True)
    return f.astype(np.float32)


class AR1Noise:
    """Time-correlated spectral noise: x_t = rho x_{t-1} + sqrt(1-rho^2) e_t."""

    def __init__(self, shape, rng, slope=-5.0 / 3.0, rho=0.8, kmin=1.0):
        self.shape, self.rng, self.slope, self.rho, self.kmin = shape, rng, slope, rho, kmin
        self.state = spectral_noise(shape, rng, slope, kmin)

    def step(self):
        e = spectral_noise(self.shape, self.rng, self.slope, self.kmin)
        self.state = self.rho * self.state + np.sqrt(1 - self.rho ** 2) * e
        return self.state


def radial_spectrum(field, dx_km):
    """Radially averaged power spectrum of a 2-D field. Returns (wavenumber [1/km], power)."""
    f = np.asarray(field, float)
    f = f - f.mean()
    ny, nx = f.shape
    wy, wx = np.hanning(ny), np.hanning(nx)
    F = np.fft.fftshift(np.fft.fft2(f * wy[:, None] * wx[None, :]))
    P = np.abs(F) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    dk = 1.0 / (max(nx, ny) * dx_km)
    bins = np.arange(dk, k.max(), dk)
    idx = np.digitize(k.ravel(), bins)
    pw = np.bincount(idx, weights=P.ravel(), minlength=len(bins) + 1)
    cnt = np.bincount(idx, minlength=len(bins) + 1)
    good = (cnt > 0)[1:len(bins)]
    kc = 0.5 * (bins[1:] + bins[:-1])
    return kc[good], (pw[1:len(bins)] / np.maximum(cnt[1:len(bins)], 1))[good]
