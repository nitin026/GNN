"""Lead-time-dependent calibration of the object-GNN event probability (BRIEF4 Phase 3).

The GNN node probability p(event) is verified against the TRUTH (node overlaps the truth event
mask: y_truth in the graph). Per lead band (0-72, 78-168, 174-240 h) two calibrators are fitted on
VAL only:
  * temperature scaling: p' = sigmoid(logit(p) / T), T by 1-D NLL minimisation;
  * isotonic regression (sklearn), monotone, clipped.
The calibrator per band is chosen by leave-one-VAL-case-out Brier score (so the isotonic fit is not
scored in-sample). TEST is only reported: reliability diagram, reliability slope (weighted least
squares of observed frequency on forecast probability over bins), Brier score and Brier skill
score (BSS) against (a) the VAL climatological base rate and (b) the raw ensemble frequency
(fraction of members with a candidate object overlapping the node's cells at that lead).
"""
import json

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

BANDS = [("0-72h", 0, 72), ("78-168h", 78, 168), ("174-240h", 174, 240)]
EPS = 1e-6


def band_of(lead_h):
    for name, a, b in BANDS:
        if a <= lead_h <= b:
            return name
    return BANDS[-1][0]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Temperature:
    kind = "temperature"

    def fit(self, p, y):
        z = logit(p)

        def nll(t):
            q = np.clip(1 / (1 + np.exp(-z / t)), EPS, 1 - EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        self.T = float(minimize_scalar(nll, bounds=(0.05, 50), method="bounded").x)
        return self

    def __call__(self, p):
        return 1 / (1 + np.exp(-logit(p) / self.T))

    def to_json(self):
        return {"kind": self.kind, "T": self.T}


class Isotonic:
    kind = "isotonic"

    def fit(self, p, y):
        self.m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        return self

    def __call__(self, p):
        return self.m.predict(p)

    def to_json(self):
        return {"kind": self.kind, "x": self.m.X_thresholds_.tolist(), "y": self.m.y_thresholds_.tolist()}


def from_json(d):
    if d["kind"] == "temperature":
        c = Temperature()
        c.T = d["T"]
        return c
    c = Isotonic()
    c.m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    c.m.fit(np.asarray(d["x"]), np.asarray(d["y"]))
    return c


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        k = idx == b
        if k.sum():
            rows.append((float(p[k].mean()), float(y[k].mean()), int(k.sum())))
    return rows


def slope(rows):
    """Weighted least-squares slope of observed frequency on mean forecast probability."""
    if len(rows) < 2:
        return None
    x = np.array([r[0] for r in rows])
    yv = np.array([r[1] for r in rows])
    w = np.array([r[2] for r in rows], float)
    xm, ym = np.average(x, weights=w), np.average(yv, weights=w)
    den = np.sum(w * (x - xm) ** 2)
    return float(np.sum(w * (x - xm) * (yv - ym)) / den) if den > 0 else None


def fit_band(p_by_case, y_by_case):
    """Choose temperature vs isotonic by leave-one-case-out Brier on the VAL cases of a band."""
    cases = [c for c in p_by_case if len(p_by_case[c])]
    scores = {}
    for C in (Temperature, Isotonic):
        errs = []
        for c in cases:
            tr = [k for k in cases if k != c]
            if not tr:
                continue
            P = np.concatenate([p_by_case[k] for k in tr])
            Y = np.concatenate([y_by_case[k] for k in tr])
            if Y.min() == Y.max():
                continue
            errs.append(brier(C().fit(P, Y)(p_by_case[c]), y_by_case[c]) * len(p_by_case[c]))
        n = sum(len(p_by_case[c]) for c in cases)
        scores[C.kind] = float(np.sum(errs) / max(n, 1)) if errs else np.inf
    best = min(scores, key=scores.get)
    P = np.concatenate([p_by_case[c] for c in cases])
    Y = np.concatenate([y_by_case[c] for c in cases])
    cal = (Temperature if best == "temperature" else Isotonic)().fit(P, Y)
    return cal, scores, float(Y.mean())


def ens_frequency(g, pn=None):
    """For each node: fraction of the ensemble's members that have a candidate object overlapping
    the node's cells at the same lead (the raw ensemble frequency baseline)."""
    cells = [g["cells"][g["cell_offsets"][i]:g["cell_offsets"][i + 1]] for i in range(len(g["X"]))]
    mem, t = g["member"], g["t"]
    M = int(mem[mem >= 0].max()) + 1 if (mem >= 0).any() else 1
    out = np.zeros(len(cells))
    by_t = {}
    for i in range(len(cells)):
        by_t.setdefault(int(t[i]), []).append(i)
    for tt, nodes in by_t.items():
        ens_nodes = [j for j in nodes if mem[j] >= 0]
        for i in nodes:
            hit = {int(mem[j]) for j in ens_nodes
                   if len(np.intersect1d(cells[i], cells[j], assume_unique=True))}
            out[i] = len(hit) / M
    return out


def save(path, cals, meta):
    path.write_text(json.dumps({"bands": {b: c.to_json() for b, c in cals.items()}, **meta}, indent=1))


def load(path):
    d = json.loads(path.read_text())
    return {b: from_json(c) for b, c in d["bands"].items()}


def apply(cals, p, lead_h):
    out = np.empty_like(np.asarray(p, float))
    lead_h = np.asarray(lead_h)
    for name, a, b in BANDS:
        k = (lead_h >= a) & (lead_h <= b)
        if k.any():
            out[k] = cals[name](np.asarray(p, float)[k])
    return out
