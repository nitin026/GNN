import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SYN = ROOT / "data" / "synthetic"


def case_dirs():
    return sorted(p for p in SYN.iterdir() if (p / "labels.json").exists()) if SYN.exists() else []


@pytest.fixture(scope="session")
def cases():
    c = case_dirs()
    if not c:                      # CI / fresh clone: the NetCDF cases are reproducible, not in git
        pytest.skip("no synthetic cases found - run `make data-synth`")
    return c


def pytest_collection_modifyitems(config, items):
    """Tests marked `needs_data` (or that open data/ files) skip when the large data are absent."""
    have = (ROOT / "data/real/clim/era5_clim_g12.nc").exists()
    skip = pytest.mark.skip(reason="large reproducible data not present (CI): run `make all`")
    for it in items:
        if "needs_data" in it.keywords and not have:
            it.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: needs data/real or data/synthetic NetCDF files")
