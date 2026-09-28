"""Icosahedral mesh GNN: per-cell hazard SEGMENTATION on the 12 km grid (BRIEF4 Phase 2).

    grid features --encoder MLP--> grid latent --(grid->mesh edges, edge MLP, mean)--> mesh latent
    --(K interaction-network layers on the multi-mesh)--> --(mesh->grid edges)--> grid latent
    --decoder MLP--> 3 logits per cell: P(cyclone), P(heat wave), P(cold wave)

Plain PyTorch (index_add message passing), GraphCast-style but tiny (hidden 32, 4 processor
layers). Inputs per grid cell (N_IN = 13 channels, `features()`):
  z(MSLP), z(T2m) against the ERA5 1990-2019 climatology; 10 m wind speed / 20; log1p(6 h rain) / 3;
  850 hPa moisture-flux-convergence PROXY -div(RH850/100 * V10) (the synthetic cases carry RH at
  850 hPa but not q/u/v at 850 hPa, so the 10 m wind stands in for the 850 hPa wind); EFI of low
  MSLP and EFI of T2m (signed) from the whole ensemble; DEM / 3000 m; land-sea mask; unit vector
  (x, y, z) on the sphere; lead / 240 h.
A same-parameter-count grid CNN (`GridCNN`) is the ablation baseline.
"""
import numpy as np
import torch
from torch import nn

from .icomesh import bipartite, build_mesh, edge_features, to_xyz

N_IN = 13
HAZ = ["tropical_cyclone", "heat_dome", "cold_wave"]
FEATURES = ["z_msl", "z_t2m", "wind/20", "log1p(tp)/3", "qconv_proxy", "efi_lowmsl", "efi_t2m", "dem/3000",
            "land", "x", "y", "z", "lead/240"]


def mlp(i, h, o, n=2, norm=True):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    if norm:
        layers.append(nn.LayerNorm(o))
    return nn.Sequential(*layers)


def scatter_mean(src, index, n, dim_size_like):
    """Mean of src rows (B, E, H) into n receivers along dim 1."""
    out = torch.zeros(dim_size_like.shape[0], n, src.shape[-1], dtype=src.dtype)
    out.index_add_(1, index, src)
    cnt = torch.zeros(n, dtype=src.dtype).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1)[None, :, None]


class Graph:
    """Static tensors of one mesh level on the G12 grid."""

    def __init__(self, level, lat, lon):
        self.mesh = build_mesh(level)
        g2m, m2g, self.radius_km, gx = bipartite(self.mesh, lat, lon)
        mx = self.mesh.xyz
        e = self.mesh.edges
        e2 = np.concatenate([e, e[:, ::-1]])                          # both directions
        self.n_grid, self.n_mesh = len(gx), self.mesh.n
        t = lambda a: torch.tensor(a, dtype=torch.long)
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        self.g2m_src, self.g2m_dst = t(g2m[:, 0]), t(g2m[:, 1])
        self.m2g_src, self.m2g_dst = t(m2g[:, 0]), t(m2g[:, 1])
        self.mm_src, self.mm_dst = t(e2[:, 0]), t(e2[:, 1])
        self.g2m_ef = f(edge_features(gx[g2m[:, 0]], mx[g2m[:, 1]]))
        self.m2g_ef = f(edge_features(mx[m2g[:, 0]], gx[m2g[:, 1]]))
        self.mm_ef = f(edge_features(mx[e2[:, 0]], mx[e2[:, 1]]))
        self.mesh_x = f(mx)
        self.level = level


class InteractionLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.edge = mlp(3 * h, h, h)
        self.node = mlp(2 * h, h, h)

    def forward(self, hn, he, src, dst):
        m = self.edge(torch.cat([hn[:, src], hn[:, dst], he], -1))
        agg = scatter_mean(m, dst, hn.shape[1], hn)
        return hn + self.node(torch.cat([hn, agg], -1)), he + m


class MeshGNN(nn.Module):
    def __init__(self, graph: Graph, hidden=32, layers=4, n_in=N_IN, n_out=3):
        super().__init__()
        self.g = graph
        h = hidden
        self.grid_enc = mlp(n_in, h, h)
        self.mesh_enc = mlp(3, h, h)
        self.g2m_edge = mlp(2 * h + 4, h, h)
        self.g2m_node = mlp(2 * h, h, h)
        self.mm_enc = mlp(4, h, h)
        self.proc = nn.ModuleList([InteractionLayer(h) for _ in range(layers)])
        self.m2g_edge = mlp(2 * h + 4, h, h)
        self.dec = mlp(2 * h, h, n_out, norm=False)

    def forward(self, x):
        """x: (B, n_grid, N_IN) -> logits (B, n_grid, 3)."""
        g = self.g
        B = x.shape[0]
        hg = self.grid_enc(x)
        hm = self.mesh_enc(g.mesh_x)[None].expand(B, -1, -1)
        m = self.g2m_edge(torch.cat([hg[:, g.g2m_src], hm[:, g.g2m_dst], g.g2m_ef[None].expand(B, -1, -1)], -1))
        hm = hm + self.g2m_node(torch.cat([hm, scatter_mean(m, g.g2m_dst, g.n_mesh, hm)], -1))
        he = self.mm_enc(g.mm_ef)[None].expand(B, -1, -1)
        for layer in self.proc:
            hm, he = layer(hm, he, g.mm_src, g.mm_dst)
        m = self.m2g_edge(torch.cat([hm[:, g.m2g_src], hg[:, g.m2g_dst], g.m2g_ef[None].expand(B, -1, -1)], -1))
        return self.dec(torch.cat([hg, scatter_mean(m, g.m2g_dst, g.n_grid, hg)], -1))


class GridCNN(nn.Module):
    """Ablation: a plain conv net on the lat-lon grid (no mesh), with a similar parameter count."""

    def __init__(self, shape=(333, 333), width=30, n_in=N_IN, n_out=3):
        super().__init__()
        self.shape = shape
        w = width
        c = lambda i, o, d=1: nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.SiLU())
        self.net = nn.Sequential(c(n_in, w), c(w, w, 2), c(w, w, 4), c(w, w, 8), c(w, w, 16), c(w, w, 1),
                                 nn.Conv2d(w, n_out, 1))

    def forward(self, x):
        B = x.shape[0]
        y = self.net(x.transpose(1, 2).reshape(B, -1, *self.shape))
        return y.reshape(B, y.shape[1], -1).transpose(1, 2)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- features
class Static:
    def __init__(self, lat, lon):
        import xarray as xr
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        orog = xr.open_dataset(root / "data/real/dem/dem_g12.nc").orog.values.astype(np.float32)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        xyz = to_xyz(LA, LO).astype(np.float32)
        self.base = np.stack([orog / 3000.0, (orog > 1.0).astype(np.float32), xyz[..., 0], xyz[..., 1], xyz[..., 2]])
        dy = (lat[1] - lat[0]) * 111.195e3
        self.dx = dy * np.cos(np.deg2rad(lat))[:, None]
        self.dy = dy


def qconv_proxy(r850, u10, v10, st):
    """-div(RH/100 * V) [1/s] x 1e5 (moisture-flux convergence proxy)."""
    q = np.clip(r850, 0, 100) / 100.0
    fu, fv = q * u10, q * v10
    div = np.gradient(fu, axis=-1) / st.dx + np.gradient(fv, axis=-2) / st.dy
    return -div * 1e5


def efi_fields(msl_ens, t2m_ens, times, clim):
    """EFI of low MSLP and of T2m for every lead from the whole ensemble: (T, 2, y, x)."""
    from .efi import efi_gaussian
    out = np.zeros((len(times), 2) + msl_ens.shape[-2:], np.float32)
    for t, vt in enumerate(times):
        out[t, 0] = -efi_gaussian(msl_ens[:, t], *clim.get("msl", vt))
        out[t, 1] = efi_gaussian(t2m_ens[:, t], *clim.get("t2m", vt))
    return out


def features(fields, t, vt, lead_h, efi_t, clim, st):
    """fields: dict of (T, y, x) arrays of one run (t2m K, u10, v10, msl Pa, tp mm/6h, r850 %).
    Returns (N_IN, y, x) float32."""
    msl, t2m = fields["msl"][t], fields["t2m"][t]
    u, v = fields["u10"][t], fields["v10"][t]
    zm = clim.z("msl", msl, vt)
    zt = clim.z("t2m", t2m, vt)
    wind = np.hypot(u, v) / 20.0
    rain = np.log1p(np.maximum(fields["tp"][t], 0)) / 3.0
    qc = qconv_proxy(fields["r850"][t], u, v, st) if "r850" in fields else np.zeros_like(msl)
    lead = np.full_like(msl, lead_h / 240.0)
    x = np.concatenate([np.stack([zm, zt, wind, rain, np.clip(qc, -5, 5)]), efi_t, st.base, lead[None]])
    return np.nan_to_num(np.clip(x, -10, 10)).astype(np.float32)
