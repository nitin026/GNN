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
    assert c, "no synthetic cases found - run `make data-synth`"
    return c
