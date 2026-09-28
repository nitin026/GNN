"""BRIEF2 Phase 5: demo script and the representative-member rule used by the exported products."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(points, p=0.9):
    return [(t, {"center_lat": la, "center_lon": lo, "p": p}) for t, la, lo in points]


def test_representative_member_is_closest_to_consensus():
    ep = _load("export_products")
    cn = [(t, 15.0, 88.0, 50.0, 3) for t in range(5)]
    far = [_track([(t, 25.0, 70.0) for t in range(5)])]           # full length but far away
    near = [_track([(t, 15.1, 88.1) for t in range(5)])]          # full length and close
    short = [_track([(t, 15.0, 88.0) for t in range(2)])]         # closest but short
    assert ep.representative_member([far, near, short], cn) == 1
    assert ep.representative_member([[], []], []) == 0


def test_demo_skip_cycle_runs_without_server():
    demo = _load("demo")
    if not (ROOT / "backend/products/amphan_replay/meta.json").exists():
        pytest.skip("run scripts/export_products.py first")
    assert demo.main(["--case", "amphan", "--skip-cycle", "--no-serve"]) == 0
    assert demo.ALIASES["amphan"] == "amphan_replay"
