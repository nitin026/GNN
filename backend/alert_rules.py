"""Alert categorisation: ensemble probability x IMD thresholds (BRIEF3; see docs/ALERT_RULES.md).

The physical thresholds are IMD's:
  cyclone wind (IMD cyclone classification, 3-min sustained wind; the 10-min synthetic/ERA5 10 m
  wind is used as is): cyclonic storm >= 62 km/h, severe cyclonic storm >= 89 km/h,
  very severe cyclonic storm >= 118 km/h
  heat wave  : Tmax departure >= 4.5 C (severe >= 6.5 C) with the regional absolute criterion
  cold wave  : Tmin departure <= -4.5 C (severe <= -6.5 C) with the regional absolute criterion
  rain (24 h): heavy >= 64.5 mm, very heavy >= 115.6 mm, extremely heavy >= 204.5 mm
The PROBABILITY cut-offs that turn P(exceedance) into low / moderate / severe are this
project's design choice (not an IMD rule), stated once here and used everywhere:
  severe   : P(severe threshold) >= 0.5
  moderate : P(severe threshold) >= 0.2  or  P(moderate threshold) >= 0.5
  low      : P(moderate threshold) >= 0.2 or  P(low threshold) >= 0.5
IMD colour code: low = yellow, moderate = orange, severe = red.

Probabilities are NEIGHBOURHOOD exceedance probabilities: the fraction of members that exceed
the threshold anywhere within NBHD_KM (50 km) of the cell. This allows for ensemble position
spread (a strong but displaced vortex still counts). The plain per-cell probability is also
reported with every alert. The 50 km radius is a design choice, see docs/ALERT_RULES.md.
"""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

NBHD_KM = 50.0

CATEGORIES = ["none", "low", "moderate", "severe"]
COLOURS = {"low": "#FFD400", "moderate": "#FF8C00", "severe": "#D7191C"}   # IMD yellow/orange/red
P_SEVERE, P_MID, P_HIGH = 0.5, 0.2, 0.5

# hazard -> (variable label, unit, (low, moderate, severe) thresholds, comparison)
THRESHOLDS = {
    "wind": ("wind", "km/h", (62.0, 89.0, 118.0), "ge"),
    "heat": ("Tmax departure", "C", (4.5, 5.5, 6.5), "ge"),
    "cold": ("Tmin departure", "C", (-4.5, -5.5, -6.5), "le"),
    "rain": ("24 h rain", "mm", (64.5, 115.6, 204.5), "ge"),
}
# heat/cold 'moderate' (5.5 C) is the midpoint between IMD's heat/cold wave (4.5) and severe (6.5);
# IMD itself has two levels there. This is stated in docs/ALERT_RULES.md.


def exceed(members, thr, how):
    """Fraction of members exceeding thr (members: (M, ...))."""
    m = np.asarray(members)
    return (m >= thr).mean(0) if how == "ge" else (m <= thr).mean(0)


def categorise(p_low, p_mod, p_sev):
    """Per-cell category index 0..3 from exceedance probabilities."""
    cat = np.zeros(np.shape(p_low), np.int8)
    cat = np.where((p_mod >= P_MID) | (p_low >= P_HIGH), 1, cat)
    cat = np.where((p_sev >= P_MID) | (p_mod >= P_HIGH), 2, cat)
    cat = np.where(p_sev >= P_SEVERE, 3, cat)
    return cat


def footprint(dx_km=13.34, radius_km=NBHD_KM):
    r = int(np.ceil(radius_km / dx_km))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (np.hypot(y, x) * dx_km) <= radius_km


def probabilities(kind, members, valid=None, neighbourhood=True, dx_km=13.34):
    """(P_low, P_mod, P_sev) for a hazard kind from members (M, y, x). valid masks cells where
    the IMD absolute criteria hold per member (heat/cold). With neighbourhood=True each member's
    field is replaced by its max (min for 'le') within NBHD_KM before counting."""
    _, _, (t1, t2, t3), how = THRESHOLDS[kind]
    m = np.asarray(members, float)
    if valid is not None:
        m = np.where(valid, m, -np.inf if how == "ge" else np.inf)
    if neighbourhood and m.ndim == 3:
        fp = footprint(dx_km)
        filt = maximum_filter if how == "ge" else minimum_filter
        m = np.array([filt(x, footprint=fp, mode="nearest") for x in m])
    return exceed(m, t1, how), exceed(m, t2, how), exceed(m, t3, how)


def reason(kind, level, p):
    label, unit, thr, how = THRESHOLDS[kind]
    t = thr[CATEGORIES.index(level) - 1]
    op = ">=" if how == "ge" else "<="
    return f"P({label} {op} {t:g} {unit}) = {p:.2f}"
