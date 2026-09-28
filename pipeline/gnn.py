"""Spatio-temporal GNN tracker (BRIEF2 Phase 2), plain PyTorch (no torch_geometric needed).

Graph: nodes = candidate objects (member, lead); undirected edges of two types
(temporal: same member t->t+1/t+2; cross-member: same lead). Each layer:
    m_ij^r = MLP_r([h_i, h_j, e_ij])          per edge type r, both directions
    a_i^r  = mean_j m_ij^r
    h_i   <- LayerNorm(h_i + MLP_u([h_i, a_i^temporal, a_i^cross]))
Heads: node event logit MLP(h_i); edge link logit MLP([h_i + h_j, |h_i - h_j|, e_ij]).
Ablations switch edge types off (use_temporal / use_cross); with both off it is a node MLP.
readout=True (BRIEF4 Phase 3) adds a graph-level consensus readout: the mean of h over all nodes at
the same lead (all members) is fed into every node update, so each node sees what the whole
ensemble is doing at that lead.
"""
import torch
import torch.nn as nn


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class GNNTracker(nn.Module):
    def __init__(self, n_node, n_edge, hidden=64, layers=3, use_temporal=True, use_cross=True,
                 dropout=0.1, readout=False):
        super().__init__()
        self.use = {"temporal": use_temporal, "cross": use_cross}
        self.readout = readout
        self.enc = mlp(n_node, hidden, hidden)
        self.eenc = mlp(n_edge, hidden // 2, hidden // 2)
        self.msg = nn.ModuleList([nn.ModuleDict({r: mlp(2 * hidden + hidden // 2, hidden, hidden)
                                                 for r in ("temporal", "cross")})
                                  for _ in range(layers)])
        self.upd = nn.ModuleList([mlp((4 if readout else 3) * hidden, hidden, hidden) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.node_head = mlp(hidden, hidden, 1)
        self.edge_head = mlp(2 * hidden + hidden // 2, hidden, 1)

    def forward(self, x, edges, efeat, etype, lead=None):
        """x (N,F); edges (E,2) long; efeat (E,Fe); etype (E,) bool, True = temporal;
        lead (N,) long lead index (needed only with readout=True)."""
        h = self.enc(x)
        e = self.eenc(efeat)
        N = h.shape[0]
        src = torch.cat([edges[:, 0], edges[:, 1]])
        dst = torch.cat([edges[:, 1], edges[:, 0]])
        e2 = torch.cat([e, e])
        t2 = torch.cat([etype, etype])
        for layer, upd, norm in zip(self.msg, self.upd, self.norm):
            aggs = []
            for r, sel in (("temporal", t2), ("cross", ~t2)):
                agg = torch.zeros(N, h.shape[1], device=h.device)
                if self.use[r] and sel.any():
                    s, d = src[sel], dst[sel]
                    m = layer[r](torch.cat([h[s], h[d], e2[sel]], 1))
                    agg = agg.index_add(0, d, m)
                    deg = torch.zeros(N, device=h.device).index_add(
                        0, d, torch.ones(len(d), device=h.device)).clamp(min=1)
                    agg = agg / deg[:, None]
                aggs.append(agg)
            if self.readout:
                nl = int(lead.max()) + 1
                pooled = torch.zeros(nl, h.shape[1], device=h.device).index_add(0, lead, h)
                cnt = torch.zeros(nl, device=h.device).index_add(0, lead, torch.ones_like(lead, dtype=h.dtype))
                aggs.append((pooled / cnt.clamp(min=1)[:, None])[lead])
            h = norm(h + self.drop(upd(torch.cat([h] + aggs, 1))))
        node_logit = self.node_head(h).squeeze(-1)
        hi, hj = h[edges[:, 0]], h[edges[:, 1]]
        edge_logit = self.edge_head(torch.cat([hi + hj, (hi - hj).abs(), e], 1)).squeeze(-1)
        return node_logit, edge_logit
