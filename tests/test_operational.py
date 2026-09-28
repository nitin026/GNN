"""BRIEF4 Phase 5: operational pipeline helpers (idempotent run id, crop, resumable stages)."""
from pipeline.run_operational import Stages, crop_of, run_id


def test_run_id_is_content_hash(tmp_path):
    a, b, c = tmp_path / "a.nc", tmp_path / "b.nc", tmp_path / "c.nc"
    a.write_bytes(b"CDF\x01same")
    b.write_bytes(b"CDF\x01same")
    c.write_bytes(b"CDF\x01diff")
    assert run_id(a) == run_id(b) != run_id(c)
    assert run_id(a).startswith("op_")


def test_crop_contains_box_plus_margin_and_is_multiple_of_4():
    box = {"lat_min": 10.0, "lat_max": 20.0, "lon_min": 82.0, "lon_max": 90.0}
    i0, i1, j0, j1 = crop_of(box, margin_km=100.0)
    lat = lambda i: 0.06 + 0.12 * i
    lon = lambda j: 60.06 + 0.12 * j
    assert lat(i0) <= 10.0 - 0.8 and lat(i1 - 1) >= 20.0 and lon(j0) <= 82.0 - 0.8 and lon(j1 - 1) >= 90.0
    assert (i1 - i0) % 4 == 0 and (j1 - j0) % 4 == 0 and 0 <= i0 < i1 <= 333 and 0 <= j0 < j1 <= 333
    edge = crop_of({"lat_min": 38.0, "lat_max": 39.9, "lon_min": 98.0, "lon_max": 99.9})
    assert edge[1] <= 333 and edge[3] <= 333


def test_stages_checkpoint_and_resume(tmp_path):
    calls = []
    s = Stages(tmp_path)
    assert s.run("1 load x", lambda: calls.append(1) or {"v": 42}) == {"v": 42}
    s2 = Stages(tmp_path)                                  # a new process after an interruption
    assert s2.run("1 load x", lambda: calls.append(2) or {"v": 0}) == {"v": 42}
    assert calls == [1] and s2.rec[0]["resumed"] is True
