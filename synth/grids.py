"""Common grids over the India box.

G12: 0.12 deg (~12 km, NEPS-G native), 333 x 333 cells, edges 0..39.96N, 60..99.96E.
G5 : 0.04 deg (~4.4 km), 999 x 999 cells, same edges. Exactly 3x3 G5 cells per G12 cell,
     so a conservative block average avgpool(G5) == G12 holds exactly.
(The brief suggested 0.045 deg for G5, but 0.12/0.045 is not an integer, which makes an
exact block average impossible; 0.04 deg is the nearest exact-ratio choice.)
"""
import numpy as np

LAT0, LON0 = 0.0, 60.0
D12, D5, FACTOR = 0.12, 0.04, 3
N12, N5 = 333, 999


def centers(d, n, origin):
    return (origin + d * (np.arange(n) + 0.5)).round(6)


LAT12, LON12 = centers(D12, N12, LAT0), centers(D12, N12, LON0)
LAT5, LON5 = centers(D5, N5, LAT0), centers(D5, N5, LON0)


def grid(name):
    return {"g12": (LAT12, LON12), "g5": (LAT5, LON5)}[name]


def avgpool(a, k=FACTOR):
    """Conservative block average over the last two axes (equal-angle cells)."""
    *lead, ny, nx = a.shape
    return a.reshape(*lead, ny // k, k, nx // k, k).mean(axis=(-3, -1))


def regrid(da, name, method="linear"):
    """Interpolate an xarray DataArray/Dataset with latitude/longitude coords onto G12/G5."""
    lat, lon = grid(name)
    return da.interp(latitude=lat, longitude=lon, method=method)
