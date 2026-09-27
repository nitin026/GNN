"""Analytic event injectors. Each returns field INCREMENTS on a (lat, lon) grid.

All distances use a local equirectangular projection around the event centre, which is
accurate to a few percent over the ~1500 km an event covers.
"""
import numpy as np

KM_PER_DEG = 111.195
RHO_AIR = 1.15          # kg m-3, boundary-layer air density
OMEGA = 7.2921e-5
E = np.e


def local_xy(lat2d, lon2d, lat0, lon0):
    """Return (x, y) in km from (lat0, lon0)."""
    x = (lon2d - lon0) * KM_PER_DEG * np.cos(np.deg2rad(lat0))
    y = (lat2d - lat0) * KM_PER_DEG
    return x, y


def coriolis(lat):
    return 2 * OMEGA * np.sin(np.deg2rad(lat))


# --------------------------------------------------------------------------- cyclone
def holland_B(vmax, dp):
    """Holland B from surface Vmax [m/s] and pressure deficit dp [Pa] (gradient wind = Vmax/0.9)."""
    vg = vmax / 0.9
    return float(np.clip(RHO_AIR * E * vg ** 2 / max(dp, 100.0), 1.0, 2.5))


def holland_dp(vmax, B=1.8):
    vg = vmax / 0.9
    return RHO_AIR * E * vg ** 2 / B


def tropical_cyclone(lat2d, lon2d, lat0, lon0, vmax, rmw_km, dp=None, B=None,
                     motion=(0.0, 0.0), land=None, rain_scale=1.0):
    """Holland (1980) vortex.

    Parameters
    ----------
    vmax : maximum 10-m wind [m/s]; rmw_km : radius of maximum wind [km]
    dp : central pressure deficit [Pa] (derived from vmax and B if None)
    motion : storm translation (u, v) [m/s]; adds a 0.5*motion asymmetry (right-front max)
    land : optional boolean grid; inflow angle 20 deg over sea, 35 deg over land

    Returns dict with dmsl [Pa] (<=0), du10, dv10 [m/s], rain_rate [mm/h] BEFORE physics
    gating, and core weight (0..1) used for moistening.
    """
    if B is None:
        B = 1.8
    if dp is None:
        dp = holland_dp(vmax, B)
    x, y = local_xy(lat2d, lon2d, lat0, lon0)
    r = np.sqrt(x ** 2 + y ** 2) + 1e-3                     # km
    th = np.arctan2(y, x)
    rr = (rmw_km / r) ** B
    dmsl = -dp * (1.0 - np.exp(-rr))                        # P(r) = Pc + dp exp(-(Rm/r)^B)
    f = abs(coriolis(lat0))
    rm = r * 1000.0
    vg = np.sqrt(B / RHO_AIR * dp * rr * np.exp(-rr) + (rm * f / 2) ** 2) - rm * f / 2
    vg_max = vg.max() if vg.size else 1.0
    v10 = vg * (vmax / max(vg_max, 1e-6))                  # scale so the peak equals vmax
    alpha = np.deg2rad(20.0) if land is None else np.where(land, np.deg2rad(35.0), np.deg2rad(20.0))
    # cyclonic (anticlockwise, NH) tangential + inflow toward centre
    tx, ty = -np.sin(th), np.cos(th)
    rx, ry = -np.cos(th), -np.sin(th)
    du = v10 * (np.cos(alpha) * tx + np.sin(alpha) * rx)
    dv = v10 * (np.cos(alpha) * ty + np.sin(alpha) * ry)
    w = np.clip(v10 / max(vmax, 1e-6), 0, 1)
    du = du + 0.5 * motion[0] * w
    dv = dv + 0.5 * motion[1] * w
    # rain: eye, eyewall peak near Rm, exponential decay, 2-arm log-spiral bands
    r_e = 150.0 + 2.0 * rmw_km
    peak = rain_scale * (3.0 + 0.3 * vmax)                  # mm/h at the eyewall
    radial = np.where(r < rmw_km, (r / rmw_km) ** 2, np.exp(-(r - rmw_km) / r_e))
    radial = np.where(r < 0.4 * rmw_km, 0.05 * radial, radial)          # eye
    pitch = np.deg2rad(15.0)
    band = 1.0 + 0.8 * np.cos(2 * (th - np.log(np.maximum(r, 1) / rmw_km) / np.tan(pitch)))
    bw = np.clip((r - 1.5 * rmw_km) / (2 * rmw_km), 0, 1)   # bands only outside the core
    rain = peak * radial * ((1 - bw) + bw * np.maximum(band, 0))
    rain = np.where(r > 1500, 0.0, rain)
    core = np.exp(-(r / (4 * rmw_km + 150.0)) ** 2)
    return {"dmsl": dmsl.astype(np.float32), "du10": du.astype(np.float32),
            "dv10": dv.astype(np.float32), "rain_rate": rain.astype(np.float32),
            "core": core.astype(np.float32), "vmax": float(vmax)}


def lifecycle_vmax(t_h, t_genesis, t_peak, v0, vpeak, landfall_h=None, vb=13.8, alpha=0.095):
    """Intensity lifecycle: genesis v0 -> peak (smooth rise), then Kaplan-DeMaria (1995)
    exponential inland decay V(t) = vb + (V_L - vb) exp(-alpha t) after landfall."""
    t = np.asarray(t_h, float)
    rise = np.clip((t - t_genesis) / max(t_peak - t_genesis, 1), 0, 1)
    v = v0 + (vpeak - v0) * np.sin(0.5 * np.pi * rise) ** 2
    v = np.where(t < t_genesis, 0.0, v)
    if landfall_h is not None:
        vl = np.interp(landfall_h, t, v)
        dt = t - landfall_h
        v = np.where(dt > 0, vb + (vl - vb) * np.exp(-alpha * dt), v)
    return v


# --------------------------------------------------------------------------- heat dome / cold wave
def blob(lat2d, lon2d, lat0, lon0, sx_km, sy_km, angle_deg, distortion=None):
    """Rotated elliptical Gaussian shape (peak 1), optionally distorted by a smooth field."""
    x, y = local_xy(lat2d, lon2d, lat0, lon0)
    a = np.deg2rad(angle_deg)
    xr = x * np.cos(a) + y * np.sin(a)
    yr = -x * np.sin(a) + y * np.cos(a)
    s = np.exp(-0.5 * ((xr / sx_km) ** 2 + (yr / sy_km) ** 2))
    if distortion is not None:
        s = s * np.clip(1.0 + 0.3 * distortion, 0.4, 1.6)
        s = s / max(s.max(), 1e-6) * min(1.0, s.max())
    return s.astype(np.float32)


def amplitude_envelope(t_h, t_start, duration_h, ramp_h=36.0):
    """0 -> 1 -> 0 envelope with smooth ramps (event 'on' between t_start and t_start+duration)."""
    t = np.asarray(t_h, float)
    up = np.clip((t - t_start) / ramp_h, 0, 1)
    down = np.clip((t_start + duration_h - t) / ramp_h, 0, 1)
    return (np.sin(0.5 * np.pi * np.minimum(up, down)) ** 2).astype(np.float32)


def geostrophic_wind(dp_field, lat2d, dx_km, dy_km, friction=0.6):
    """Near-surface wind increment from a pressure increment (geostrophic x friction factor)."""
    dpdy = np.gradient(dp_field, axis=0) / (dy_km * 1000.0)
    dpdx = np.gradient(dp_field, axis=1) / (np.asarray(dx_km) * 1000.0)
    f = coriolis(np.maximum(lat2d, 5.0))
    return (-friction * dpdy / (RHO_AIR * f)).astype(np.float32), \
           (friction * dpdx / (RHO_AIR * f)).astype(np.float32)


def heat_dome(lat2d, lon2d, lat0, lon0, amp_K, sx_km, sy_km, angle_deg, hour_utc,
              distortion=None, dmsl_amp=200.0, land_w=None):
    """Heat-dome increments: dT2m (with a diurnal cycle peaking ~09 UTC = 14:30 IST),
    a broad weak surface high and rain suppression factor."""
    s = blob(lat2d, lon2d, lat0, lon0, sx_km, sy_km, angle_deg, distortion)
    if land_w is not None:
        s = s * land_w
    diurnal = (1.0 + 0.2 * np.cos(2 * np.pi * (hour_utc - 9.0) / 24.0)) / 1.2   # peak = amp_K
    dT = amp_K * diurnal * s
    broad = blob(lat2d, lon2d, lat0, lon0, 1.6 * sx_km, 1.6 * sy_km, angle_deg)
    return {"dt2m": dT.astype(np.float32), "dmsl": (dmsl_amp * broad).astype(np.float32),
            "shape": s, "rain_factor": (1.0 - 0.8 * s).astype(np.float32)}


def cold_wave(lat2d, lon2d, lat0, lon0, amp_K, sx_km, sy_km, angle_deg, hour_utc,
              plains_w, distortion=None, dmsl_amp=400.0):
    """Cold-wave increments: dT2m < 0 confined to the plains (plains_w), stronger at night
    (minimum ~00 UTC = 05:30 IST), a surface high to the NW, and rain suppression."""
    s = blob(lat2d, lon2d, lat0, lon0, sx_km, sy_km, angle_deg, distortion) * plains_w
    diurnal = (1.0 + 0.2 * np.cos(2 * np.pi * (hour_utc - 0.0) / 24.0)) / 1.2   # peak = amp_K
    dT = -abs(amp_K) * diurnal * s
    hi = blob(lat2d, lon2d, lat0 + 3.0, lon0 - 4.0, 1.5 * sx_km, 1.5 * sy_km, angle_deg)
    return {"dt2m": dT.astype(np.float32), "dmsl": (dmsl_amp * hi).astype(np.float32),
            "shape": s, "rain_factor": (1.0 - 0.7 * s).astype(np.float32)}
