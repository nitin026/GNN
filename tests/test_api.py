"""BRIEF3 Phase B: every REST endpoint (FastAPI TestClient; POST /run with a fake runner)."""
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "backend" / "products"
pytest.importorskip("fastapi")
pytestmark = pytest.mark.skipif(not (P / "amphan_replay" / "meta.json").exists(),
                                reason="run scripts/export_products.py first")
from fastapi.testclient import TestClient  # noqa: E402


def fake_runner(job, progress):
    progress("load")
    progress("GNN tracking")
    return {"case": job["output_case"], "alerts": 0, "severe_alerts": 0}


@pytest.fixture(scope="module")
def client():
    from backend.api.app import create_app
    return TestClient(create_app(runner=fake_runner))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "X-Response-Time-ms" in r.headers


def test_cases_have_badges(client):
    cs = {c["case"]: c for c in client.get("/cases").json()}
    assert {"amphan_replay", "cyc_04", "heat_04", "cold_04", "amphan_era5_real"} <= set(cs)
    for c in cs.values():
        if c["synthetic"]:
            assert c["badge"] == "SYNTHETIC"
    assert cs["amphan_era5_real"]["badge"] == "REAL (ERA5 reanalysis)"
    assert client.get("/cases/amphan_replay").json()["n_members"] == 20
    assert client.get("/cases/nope").status_code == 404
    assert client.get("/cases/..%2Fsecret").status_code == 404


def test_tracks_and_lead_cut(client):
    full = client.get("/cases/amphan_replay/tracks").json()
    assert full["type"] == "FeatureCollection" and full["badge"] == "SYNTHETIC"
    at = client.get("/cases/amphan_replay/tracks", params={"lead": 48}).json()
    for f in at["features"]:
        p = f["properties"]
        if p["kind"] in ("member_track", "consensus"):
            assert max(p["leads_h"]) <= 48
    kinds = {f["properties"]["kind"] for f in at["features"]}
    assert "member_track_position" in kinds and "bbox4d" in kinds
    assert client.get("/cases/amphan_replay/tracks", params={"lead": 7}).status_code == 422


def test_bbox4d(client):
    b = client.get("/cases/cyc_04/bbox4d").json()
    assert b["synthetic"] is True and b["bbox4d"]
    box = b["bbox4d"][0]
    assert box["lat_min"] < box["lat_max"] and box["lead_start_h"] <= box["lead_end_h"]
    assert {"level_hPa_bottom", "level_hPa_top", "t_start", "t_end"} <= set(box)


def test_fields_png_and_json(client):
    r = client.get("/cases/amphan_replay/fields/wind", params={"res": "12km", "lead": 96})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n" and len(r.headers["X-Bounds"].split(",")) == 4
    j = client.get("/cases/amphan_replay/fields/tp", params={"res": "5km", "lead": 96, "format": "json"}).json()
    assert j["legend"]["units"] == "mm/6h" and client.get(j["url"]).status_code == 200
    assert client.get("/cases/amphan_replay/fields/strike", params={"lead": 96}).status_code == 200
    assert client.get("/cases/amphan_replay/fields/tp", params={"res": "5km", "lead": 6}).status_code == 422
    assert client.get("/cases/amphan_replay/fields/anom", params={"res": "5km", "lead": 12}).status_code == 422
    assert client.get("/cases/amphan_replay/fields/xyz").status_code == 422


def test_alerts_sorted_and_point_query(client):
    js = client.get("/alerts", params={"case": "amphan_replay", "lead": 96}).json()
    al = js["alerts"]
    assert js["count"] == len(al) > 0
    order = {"low": 1, "moderate": 2, "severe": 3}
    assert all(order[a["category"]] >= order[b["category"]] for a, b in zip(al, al[1:]))
    a = al[0]
    for k in ("pinpoint", "category", "impact_polygon", "valid_time", "probability", "reason"):
        assert k in a
    assert a["reason"].startswith("P(") and a["impact_radius_km"] == 5.0
    hit = client.get("/alerts", params={"case": "amphan_replay", "lead": 96, "lat": a["pinpoint"]["lat"],
                                        "lon": a["pinpoint"]["lon"]}).json()["alerts"]
    assert any(h["id"] == a["id"] and h["within_impact_radius"] for h in hit)
    far = client.get("/alerts", params={"case": "amphan_replay", "lat": 36.0, "lon": 62.0}).json()
    assert far["count"] == 0
    sev = client.get("/alerts", params={"case": "amphan_replay", "min_category": "severe"}).json()["alerts"]
    assert all(x["category"] == "severe" for x in sev)
    assert client.get("/alerts", params={"lat": 20.0}).status_code == 422


@pytest.mark.skipif(not (P / "amphan_replay" / "alert_grid.npz").exists(), reason="no alert_grid.npz")
def test_district_rollup(client):
    r = client.get("/alerts/districts", params={"case": "amphan_replay"}).json()
    assert r["synthetic"] is True and "GADM" in r["unit"]
    rows = r["districts"]
    assert rows, "Amphan replay should warn some districts over all leads"
    order = {"low": 1, "moderate": 2, "severe": 3}
    assert all(order[a["category"]] >= order[b["category"]] for a, b in zip(rows, rows[1:]))
    for x in rows:
        assert 0 <= x["probability"] <= 1 and 0 < x["fraction_of_district"] <= 1
        assert x["first_lead_h"] <= x["last_lead_h"]
    one = client.get("/alerts/districts", params={"case": "amphan_replay", "lead": 96}).json()
    assert all(x["first_lead_h"] == 96 for x in one["districts"])


def _wait(client, jid):
    for _ in range(100):
        j = client.get(f"/run/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_run_selected_case_and_upload(client, tmp_path):
    r = client.post("/run", params={"case": "cyc_04"})
    assert r.status_code == 202
    j = _wait(client, r.json()["job_id"])
    assert j["status"] == "done" and [s["stage"] for s in j["stages"]] == ["load", "GNN tracking"]
    assert j["total_seconds"] is not None
    import xarray as xr
    f = tmp_path / "ens.nc"
    xr.Dataset({"msl": (("number", "step", "latitude", "longitude"), np.full((2, 2, 3, 3), 1e5, np.float32))},
               coords={"step": [0, 6], "latitude": [10.0, 10.12, 10.24], "longitude": [80.0, 80.12, 80.24]},
               attrs={"init_time": "2020-05-16T00:00"}).to_netcdf(f)
    r = client.post("/run", params={"hazard": "tropical_cyclone"}, content=f.read_bytes(),
                    headers={"Content-Type": "application/x-netcdf"})
    assert r.status_code == 202
    assert _wait(client, r.json()["job_id"])["status"] == "done"
    assert client.post("/run", content=b"CDF\x01xx").status_code == 422          # no hazard
    assert client.post("/run", params={"hazard": "heat_dome"}, content=b"hello").status_code == 422
    assert client.post("/run").status_code == 422
    assert client.get("/run/doesnotexist").status_code == 404
    assert len(client.get("/runs").json()) >= 2


def test_reports(client):
    r = client.get("/reports/gnn_results.json")
    assert r.status_code == 200 and "cases" in r.json()
    assert client.get("/reports/..%2F.env").status_code == 404
    assert client.get("/reports/RESULTS.md").status_code == 404
