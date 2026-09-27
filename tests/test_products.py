"""Exported web products (backend/products): flags, badges, image sizes, GeoJSON, alerts."""
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "backend" / "products"
cases = sorted(p for p in P.iterdir() if (p / "meta.json").exists()) if P.exists() else []
pytestmark = pytest.mark.skipif(not cases, reason="run scripts/export_products.py first")


def test_badges_and_flags():
    for c in cases:
        m = json.loads((c / "meta.json").read_text())
        if c.name == "amphan_era5_real":
            assert m["synthetic"] is False and m["badge"] == "REAL (ERA5 reanalysis)"
        else:
            assert m["synthetic"] is True and m["badge"] == "SYNTHETIC"
        assert len(m["leads_h"]) == 41 if m["synthetic"] else len(m["leads_h"]) > 0
        for v, lg in m["legends"].items():
            assert lg["units"] and len(lg["stops"]) == 9


def test_images_and_resolutions():
    for c in cases:
        m = json.loads((c / "meta.json").read_text())
        assert Image.open(c / "fields_12km" / "t2m" / "000.png").size == (333, 333)
        lead5 = m["leads_5km_h"][1]
        assert Image.open(c / "fields_5km" / "t2m" / f"{lead5:03d}.png").size == (999, 999)
        assert (c / "strike_prob" / "000.png").exists()


def test_tracks_and_alerts_structure():
    for c in cases:
        g = json.loads((c / "tracks.geojson").read_text())
        assert g["type"] == "FeatureCollection"
        kinds = {f["properties"]["kind"] for f in g["features"]}
        assert kinds <= {"member_track", "consensus", "bbox4d"}
        a = json.loads((c / "alerts.json").read_text())
        for x in a["alerts"]:
            assert x["category"] in ("low", "moderate", "severe")
            assert 0 <= x["probability"] <= 1 and x["reason"].startswith("P(")
            ring = x["impact_polygon"]["coordinates"][0]
            assert ring[0] == ring[-1] and len(ring) == 33
    assert sum(p.stat().st_size for p in P.rglob("*") if p.is_file()) < 500e6
