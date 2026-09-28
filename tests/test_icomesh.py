"""BRIEF4 Phase 2: icosahedral multi-mesh (pipeline/icomesh.py)."""
import numpy as np
import pytest

from pipeline.icomesh import (ICO_EDGE_KM, arc_km, bipartite, build_mesh, finest_edge_km, grid_to_mesh_mean,
                              icosahedron, mesh_to_grid_mean, n_vertices, subdivide)


def test_global_node_counts_per_level():
    v, f = icosahedron()
    for k in range(5):
        assert len(v) == n_vertices(k) and len(f) == 20 * 4 ** k
        v2, f2 = subdivide(v, f)
        assert np.allclose(v2[:len(v)], v)                     # vertex ids are stable
        v, f = v2, f2
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


@pytest.mark.parametrize("level", [5, 6, 7])
def test_regional_mesh_edges(level):
    m = build_mesh(level)
    # nested vertex sets: counts grow ~4x per level inside the region
    c = [m.counts[k] for k in range(3, level + 1)]
    assert all(3.0 < b / a < 5.0 for a, b in zip(c, c[1:]))
    # no duplicate or self edges
    assert len(np.unique(m.edges, axis=0)) == len(m.edges)
    assert (m.edges[:, 0] < m.edges[:, 1]).all()
    # every edge of level k has the expected length (icosahedron edge / 2**k, within +-40 %)
    for k in range(level + 1):
        e = m.edges[m.edge_level == k]
        if len(e) == 0:
            continue
        d = arc_km(m.xyz[e[:, 0]], m.xyz[e[:, 1]])
        nominal = ICO_EDGE_KM / 2 ** k
        assert d.min() > 0.6 * nominal and d.max() < 1.4 * nominal, (k, d.min(), d.max(), nominal)
    # finest edges: ~55 km at level 7, halving per level
    assert abs(np.median(finest_edge_km(m)) - ICO_EDGE_KM / 2 ** level) / (ICO_EDGE_KM / 2 ** level) < 0.25


def test_constant_field_round_trip():
    lat = np.arange(0.06, 40, 0.48)
    lon = np.arange(60.06, 100, 0.48)
    m = build_mesh(5)
    g2m, m2g, r_km, _ = bipartite(m, lat, lon)
    assert 0.5 * finest_edge_km(m).max() < r_km < 0.7 * finest_edge_km(m).max()
    assert set(np.unique(m2g[:, 1])) == set(range(len(lat) * len(lon)))     # every cell decoded
    x = np.full(len(lat) * len(lon), 7.5)
    back = mesh_to_grid_mean(grid_to_mesh_mean(x, g2m, m.n), m2g, len(x))
    assert np.allclose(back, 7.5, atol=1e-9)
    # a smooth field survives the round trip approximately
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    s = np.sin(np.deg2rad(LA * 4)).ravel()
    back = mesh_to_grid_mean(grid_to_mesh_mean(s, g2m, m.n), m2g, len(s))
    assert np.abs(back - s).mean() < 0.1
