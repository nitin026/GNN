"""BRIEF4 Phase 1: locked real split, loaders, climatology guard, IBTrACS matching."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_real_split_is_locked_and_time_based():
    s = json.loads((ROOT / "data/real/SPLITS.json").read_text())
    assert s["locked"] is True
    for eid, ev in s["events"].items():
        year = int(eid.split("_")[-1])
        assert ev["split"] == ("REAL-VAL" if year <= 2021 else "REAL-TEST"), eid
    assert set(s["REAL-VAL"]).isdisjoint(s["REAL-TEST"])
    assert "amphan_2020" in s["REAL-VAL"] and len(s["REAL-TEST"]) >= 3


def test_wb2_loader_interface():
    from pipeline.loaders import load
    files = sorted((ROOT / "data/real/ensembles").glob("ifs_ens_1p5_*.nc"))
    if not files:
        pytest.skip("no ensemble files")
    e = load(files[0])
    assert e.msl.ndim == 4 and e.members == 50 and not e.synthetic
    assert len(e.times) == e.msl.shape[1] and e.times[0] == e.init


def test_climatology_fails_loudly_outside_coverage():
    from pipeline.anomaly import Climatology
    c = Climatology(extra=ROOT / "does_not_exist.nc")
    assert c.get("msl", "2020-05-18T06")[0].shape == (333, 333)
    with pytest.raises(KeyError):
        c.get("msl", "2020-08-15T00")            # no base climatology in mid-August


def test_match_track_picks_closest_and_rejects_far():
    from pipeline.real_eval import match_track
    times = pd.date_range("2020-05-16", periods=3, freq="6h")
    ib = pd.DataFrame({"time": times, "LAT": [12.0, 13.0, 14.0], "LON": [86.0, 86.0, 86.5]})
    near = [(i, {"center_lat": 12.0 + i + 0.1, "center_lon": 86.0}) for i in range(3)]
    far = [(i, {"center_lat": 25.0, "center_lon": 70.0}) for i in range(3)]
    got = match_track([far, near], times, ib)
    assert got is not None and abs(got[times[1]][0] - 13.1) < 1e-9
    assert match_track([far], times, ib) is None
