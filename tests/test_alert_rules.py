"""Alert categorisation rules (backend/alert_rules.py, docs/ALERT_RULES.md)."""
import numpy as np

from backend.alert_rules import (CATEGORIES, THRESHOLDS, categorise, footprint, probabilities,
                                 reason)


def test_category_boundaries():
    p = lambda *a: np.array(a, float)
    #            P_low          P_mod          P_sev
    cat = categorise(p(0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0),
                     p(0.0, 0.0, 0.2, 0.5, 0.0, 0.0, 1.0),
                     p(0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 1.0))
    assert [CATEGORIES[c] for c in cat] == ["none", "low", "low", "moderate", "moderate",
                                            "severe", "severe"]


def test_imd_thresholds_are_the_documented_ones():
    assert THRESHOLDS["wind"][2] == (62.0, 89.0, 118.0)
    assert THRESHOLDS["rain"][2] == (64.5, 115.6, 204.5)
    assert THRESHOLDS["heat"][2][0] == 4.5 and THRESHOLDS["heat"][2][2] == 6.5
    assert THRESHOLDS["cold"][2][0] == -4.5 and THRESHOLDS["cold"][2][2] == -6.5


def test_neighbourhood_counts_displaced_members():
    M, n = 10, 21
    f = np.zeros((M, n, n))
    for m in range(M):
        f[m, 10, 8 + (m % 5)] = 150.0            # each member's peak displaced by 0-4 cells
    cell = probabilities("wind", f, neighbourhood=False)[2]
    nb = probabilities("wind", f)[2]
    assert cell.max() <= 0.2                     # no single cell has many members
    assert nb[10, 10] == 1.0                     # all peaks lie within 50 km (<= 3.7 cells)


def test_heat_validity_mask_blocks_exceedance():
    dep = np.full((4, 5, 5), 7.0)                # departure 7 C everywhere
    ok = np.zeros((4, 5, 5), bool)
    ok[:2] = True                                # only 2 of 4 members meet the absolute criterion
    p1, _, p3 = probabilities("heat", dep, ok, neighbourhood=False)
    assert np.allclose(p1, 0.5) and np.allclose(p3, 0.5)


def test_cold_is_lower_tail_and_reason_text():
    dep = np.array([[[-7.0]], [[-3.0]]])
    p1, _, p3 = probabilities("cold", dep, neighbourhood=False)
    assert p1[0, 0] == 0.5 and p3[0, 0] == 0.5
    assert reason("wind", "severe", 0.72) == "P(wind >= 118 km/h) = 0.72"
    assert reason("cold", "low", 0.4) == "P(Tmin departure <= -4.5 C) = 0.40"
    assert footprint().sum() > 1
