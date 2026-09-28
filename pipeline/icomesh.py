"""Icosahedral multi-mesh over the India box + halo (BRIEF4 Phase 2), pure numpy / scipy.

GraphCast-style:
  * an icosahedron is refined L times by edge-midpoint subdivision (each triangle -> 4), vertices
    re-projected to the unit sphere. Vertex ids are stable: level-k vertices keep their ids at k+1,
    so the level-k vertex set is ids < n_vertices(k) = 10 * 4**k + 2.
  * the MULTI-MESH edge set is the union of the triangle edges of every level 0..L, expressed on
    the finest vertices (long coarse edges carry information far in few hops).
  * the regional mesh keeps the vertices inside the India box plus a halo (default 0-40N, 60-100E
    +/- 12 deg), and the edges whose two ends are both kept.
  * grid->mesh edges: each grid cell -> every mesh node within R_G2M = 0.6 x the longest finest-
    level edge (haversine / chord on the unit sphere); mesh->grid edges: each grid cell <- every
    mesh node within the same radius (at least the nearest one), so every cell is decoded.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6371.0
BOX = (0.0, 40.0, 60.0, 100.0)             # lat0, lat1, lon0, lon1
ICO_EDGE_KM = 1.1071487177940904 * R_EARTH  # arc length of an icosahedron edge on the sphere


def icosahedron():
    p = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0], [0, -1, p], [0, 1, p],
                  [0, -1, -p], [0, 1, -p], [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v, f


def subdivide(v, f):
    """One midpoint refinement. New vertices are appended (old ids unchanged)."""
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    mid = v[ue[:, 0]] + v[ue[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    mid_id = len(v) + np.arange(len(ue))
    nf = len(f)
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    ab, bc, ca = mid_id[inv[:nf]], mid_id[inv[nf:2 * nf]], mid_id[inv[2 * nf:]]
    f2 = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1),
                         np.stack([c, ca, bc], 1), np.stack([ab, bc, ca], 1)])
    return np.vstack([v, mid]), f2


def n_vertices(level):
    return 10 * 4 ** level + 2


def to_xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def to_latlon(xyz):
    return np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1, 1))), np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))


def arc_km(a, b):
    """Great-circle distance between unit vectors (rows)."""
    return R_EARTH * 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))


def face_edges(f):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


@dataclass
class Mesh:
    level: int
    xyz: np.ndarray                 # (N, 3) regional nodes
    lat: np.ndarray
    lon: np.ndarray
    node_level: np.ndarray          # coarsest level at which each node exists
    edges: np.ndarray               # (E, 2) undirected multi-mesh edges, i < j, regional ids
    edge_level: np.ndarray          # level of the triangle the edge belongs to
    global_ids: np.ndarray
    counts: dict = field(default_factory=dict)   # level -> regional vertex count

    @property
    def n(self):
        return len(self.xyz)


def build_mesh(level=6, box=BOX, halo_deg=12.0, min_level=0):
    v, f = icosahedron()
    faces = [f]
    for _ in range(level):
        v, f = subdivide(v, f)
        faces.append(f)
    lat, lon = to_latlon(v)
    la0, la1, lo0, lo1 = box
    keep = (lat >= la0 - halo_deg) & (lat <= la1 + halo_deg) & (lon >= lo0 - halo_deg) & (lon <= lo1 + halo_deg)
    gid = np.nonzero(keep)[0]
    remap = np.full(len(v), -1)
    remap[gid] = np.arange(len(gid))
    E, EL = [], []
    for k in range(min_level, level + 1):
        e = face_edges(faces[k])
        ok = keep[e[:, 0]] & keep[e[:, 1]]
        E.append(remap[e[ok]])
        EL.append(np.full(ok.sum(), k))
    E = np.concatenate(E)
    EL = np.concatenate(EL)
    E = np.sort(E, axis=1)
    E, first = np.unique(E, axis=0, return_index=True)       # an edge can repeat across levels? no, but safe
    EL = EL[first]
    node_level = np.zeros(len(gid), int)
    for k in range(level, -1, -1):
        node_level[gid < n_vertices(k)] = k
    counts = {k: int((gid < n_vertices(k)).sum()) for k in range(level + 1)}
    return Mesh(level, v[gid], lat[gid], lon[gid], node_level, E, EL, gid, counts)


def finest_edge_km(mesh):
    e = mesh.edges[mesh.edge_level == mesh.level]
    return arc_km(mesh.xyz[e[:, 0]], mesh.xyz[e[:, 1]])


def bipartite(mesh, grid_lat, grid_lon, ratio=0.6):
    """grid->mesh and mesh->grid edges by radius (0.6 x the longest finest-level edge).
    Returns g2m (E, 2) [grid_id, mesh_id], m2g (E, 2) [mesh_id, grid_id] and the radius (km).
    Grid ids are row-major over (grid_lat, grid_lon)."""
    LA, LO = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    gx = to_xyz(LA.ravel(), LO.ravel())
    r_km = ratio * float(finest_edge_km(mesh).max())
    chord = 2 * np.sin(r_km / R_EARTH / 2)
    tree = cKDTree(mesh.xyz)
    nb = tree.query_ball_point(gx, chord)
    gi = np.repeat(np.arange(len(gx)), [len(x) for x in nb])
    mi = np.concatenate([np.asarray(x, int) for x in nb]) if len(gi) else np.zeros(0, int)
    lone = np.nonzero(np.array([len(x) == 0 for x in nb]))[0]
    if len(lone):                                   # guarantee every cell is decoded
        _, nn = tree.query(gx[lone])
        gi = np.concatenate([gi, lone])
        mi = np.concatenate([mi, nn])
    g2m = np.stack([gi, mi], 1)
    return g2m, g2m[:, ::-1].copy(), r_km, gx


def edge_features(src_xyz, dst_xyz):
    """Displacement in the receiver's local (east, north, up) frame + great-circle length
    (normalised by R_EARTH), as in GraphCast."""
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, dst_xyz)
    east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-12)
    north = np.cross(dst_xyz, east)
    d = src_xyz - dst_xyz
    return np.stack([(d * east).sum(1), (d * north).sum(1), (d * dst_xyz).sum(1),
                     arc_km(src_xyz, dst_xyz) / R_EARTH * 10], 1).astype(np.float32)


def grid_to_mesh_mean(field_flat, g2m, n_mesh):
    s = np.bincount(g2m[:, 1], weights=field_flat[g2m[:, 0]], minlength=n_mesh)
    c = np.bincount(g2m[:, 1], minlength=n_mesh)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def mesh_to_grid_mean(mesh_vals, m2g, n_grid):
    ok = np.isfinite(mesh_vals[m2g[:, 0]])
    s = np.bincount(m2g[ok, 1], weights=mesh_vals[m2g[ok, 0]], minlength=n_grid)
    c = np.bincount(m2g[ok, 1], minlength=n_grid)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)
