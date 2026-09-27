"""Physics-consistency helpers: moisture-flux convergence gating, orographic enhancement,
humidity cap, non-negative rain."""
import numpy as np
from scipy.ndimage import gaussian_filter

KM_PER_DEG = 111.195


def grid_spacing_km(lat, lon):
    dy = (lat[1] - lat[0]) * KM_PER_DEG
    dx = (lon[1] - lon[0]) * KM_PER_DEG * np.cos(np.deg2rad(lat))[:, None]
    return dx, dy


def moisture_flux_convergence(q, u, v, lat, lon, smooth_km=25.0):
    """MFC = -div(q V) [kg kg-1 s-1] on a regular lat/lon grid (centred differences)."""
    dx, dy = grid_spacing_km(lat, lon)
    qu, qv = q * u, q * v
    dqu_dx = np.gradient(qu, axis=-1) / (dx * 1000.0)
    dqv_dy = np.gradient(qv, axis=-2) / (dy * 1000.0)
    mfc = -(dqu_dx + dqv_dy)
    if smooth_km:
        sig = smooth_km / dy
        mfc = gaussian_filter(mfc, sigma=sig)
    return mfc


def orographic_factor(u, v, orog, lat, lon, k=5.0, cap=2.0):
    """Upslope enhancement 1 + k * max(0, V . grad h) (V.grad h in m/s), capped."""
    dx, dy = grid_spacing_km(lat, lon)
    dhdx = np.gradient(orog, axis=-1) / (dx * 1000.0)
    dhdy = np.gradient(orog, axis=-2) / (dy * 1000.0)
    w = u * dhdx + v * dhdy
    return np.clip(1.0 + k * np.maximum(w, 0.0), 1.0, cap)


def place_rain(rain_rate_mmh, mfc, oro_factor, hours=6.0):
    """Injected rain accumulation [mm per `hours`]: only where MFC > 0, orographically
    enhanced, and never negative."""
    acc = rain_rate_mmh * hours * (mfc > 0) * oro_factor
    return np.maximum(acc, 0.0)


def cap_rh(rh):
    return np.clip(rh, 0.0, 100.0)
