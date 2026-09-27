"""Streaming accumulators in pipeline/downscale_eval.py equal direct (in-memory) computations."""
import numpy as np
import pytest

pytest.importorskip("torch")
from pipeline.downscale_eval import Acc  # noqa: E402


def test_online_metrics_match_direct():
    rng = np.random.default_rng(0)
    t = rng.gamma(0.6, 4.0, size=(6, 24, 24))
    s = t[None] + rng.normal(0, 1.5, size=(8,) + t.shape)
    p = s.mean(0)
    a = Acc("tp")
    for k in range(0, 6, 2):                       # streamed in 3 batches
        a.add(p[k:k + 2], t[k:k + 2], samples=s[:, k:k + 2], spectra=False)
    r = a.result("tp")
    assert np.isclose(r["rmse"], np.sqrt(((p - t) ** 2).mean()))
    assert np.isclose(r["mae"], np.abs(p - t).mean())
    assert abs(r["p99_truth"] - np.percentile(t, 99)) < 0.05       # histogram bin 0.02
    # fair CRPS reference (O(S^2))
    S = s.shape[0]
    t1 = np.abs(s - t[None]).mean(0)
    t2 = np.abs(s[:, None] - s[None]).sum((0, 1)) / (S * (S - 1))
    assert np.isclose(r["crps"], (t1 - 0.5 * t2).mean())
    assert np.isclose(sum(r["rank_hist"]), 1.0) and len(r["rank_hist"]) == S + 1
    assert np.isclose(r["spread_skill"], np.sqrt(s.var(0, ddof=1).mean()) / r["rmse"])
