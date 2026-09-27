"""ECMWF Extreme Forecast Index (Lalaurette 2003; Zsoter 2006).

    EFI = (2/pi) * integral_0^1 (p - F_f(p)) / sqrt(p (1 - p)) dp

where F_f(p) is the fraction of ensemble members below the climate p-quantile.
EFI is in [-1, 1]; 0 when the ensemble is distributed like the climate, -> +1 when
all members exceed the climate maximum.
"""
import numpy as np
from scipy.special import ndtri

# Substituting p = sin^2(theta) gives dp / sqrt(p(1-p)) = 2 dtheta, which removes the endpoint
# singularity: EFI = (2/pi) * int_0^{pi/2} 2 (p - F_f(p)) dtheta  (midpoint rule in theta).
_NP = 400
_TH = (np.arange(_NP) + 0.5) * (np.pi / 2) / _NP
_P = np.sin(_TH) ** 2
_W = np.full(_NP, 2.0 * (np.pi / 2) / _NP)


def efi_from_quantiles(ens, clim_q):
    """ens: (M, ...) members; clim_q: (len(_P), ...) climate quantiles at probabilities _P."""
    ens = np.asarray(ens, float)
    F = (ens[None, ...] < clim_q[:, None, ...]).mean(axis=1)       # (P, ...)
    integrand = (_P.reshape((-1,) + (1,) * (F.ndim - 1)) - F) * \
        _W.reshape((-1,) + (1,) * (F.ndim - 1))
    return (2.0 / np.pi) * integrand.sum(axis=0)


def efi_gaussian(ens, mean, std):
    """EFI against a Gaussian climate N(mean, std) (per grid point)."""
    mean = np.asarray(mean, float)
    std = np.asarray(std, float)
    q = mean[None, ...] + std[None, ...] * ndtri(_P).reshape((-1,) + (1,) * mean.ndim)
    return efi_from_quantiles(ens, q)


def efi_sample(ens, clim_sample):
    """EFI against an empirical climate sample (N, ...)."""
    q = np.quantile(np.asarray(clim_sample, float), _P, axis=0)
    return efi_from_quantiles(ens, q)
