"""GNN tracker: forward pass / ablation switches, decoder, TempestExtremes-style detection."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.gnn import GNNTracker  # noqa: E402
from pipeline.gnn_eval import decode  # noqa: E402
from pipeline.te_style import te_cyclones  # noqa: E402


def test_forward_shapes_and_ablation():
    torch.manual_seed(0)
    x = torch.randn(10, 18)
    e = torch.tensor([[0, 1], [1, 2], [0, 5], [3, 4]])
    ef = torch.randn(4, 6)
    et = torch.tensor([True, True, False, False])
    for ut, uc in ((1, 1), (1, 0), (0, 1), (0, 0)):
        m = GNNTracker(18, 6, hidden=16, use_temporal=bool(ut), use_cross=bool(uc)).eval()
        nl, el = m(x, e, ef, et)
        assert nl.shape == (10,) and el.shape == (4,)
    # with no edge types, a node's output must not depend on its neighbours
    m = GNNTracker(18, 6, hidden=16, use_temporal=False, use_cross=False).eval()
    x2 = x.clone()
    x2[1] += 5.0
    assert torch.allclose(m(x, e, ef, et)[0][0], m(x2, e, ef, et)[0][0])


def _toy_graph():
    # member 0: nodes 0..3 at t=0..3 on one track; node 4 at t=1 is an unrelated object
    t = np.array([0, 1, 2, 3, 1])
    cells = [np.array([k], np.int32) for k in range(5)]
    E = np.array([[0, 1], [1, 2], [2, 3], [0, 4]])
    EF = np.zeros((4, 6), np.float32)
    EF[:, -1] = 1.0
    return {"member": np.zeros(5, np.int16), "t": t, "E": E, "EF": EF,
            "lat": np.full(5, 15.0), "lon": np.full(5, 85.0), "clat": np.full(5, 15.0),
            "clon": np.full(5, 85.0), "cells": np.concatenate(cells),
            "cell_offsets": np.arange(6, dtype=np.int64)}


def test_decode_links_high_probability_chain():
    g = _toy_graph()
    pn = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    pe = np.array([0.9, 0.8, 0.9, 0.1])
    tr = decode(g, pn, pe, 0, "tropical_cyclone", tau_n=0.5, tau_e=0.5, min_len=3)
    assert len(tr) == 1 and [o["node"] for _, o in tr[0]] == [0, 1, 2, 3]
    assert decode(g, np.full(5, 0.1), pe, 0, "tropical_cyclone") == []


def test_te_style_finds_idealised_cyclone():
    from synth.events import tropical_cyclone
    lat = np.arange(333) * 0.12 + 0.06
    lon = lat + 60
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    series = []
    for k in range(6):
        v = tropical_cyclone(LA, LO, 14 + 0.3 * k, 88 - 0.3 * k, 40.0, 40.0)
        series.append(101000.0 + v["dmsl"])
    tracks = te_cyclones(np.array(series), lat, lon)
    assert len(tracks) == 1 and len(tracks[0]) == 6
    t, o = tracks[0][-1]
    assert abs(o["center_lat"] - 15.5) < 0.2 and abs(o["center_lon"] - 86.5) < 0.2
