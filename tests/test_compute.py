"""The CPU-only path must work when torch cannot be imported (e.g. blocked DLLs)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_torch_fallback():
    code = ("import sys; sys.modules['torch'] = None\n"
            "import pipeline.compute as c\n"
            "assert not c.HAS_TORCH and c.device() is None\n"
            "c.seed_all(0)\n"
            "import pipeline.track, pipeline.efi, pipeline.anomaly, pipeline.evaluate\n"
            "print(c.describe())")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "CPU-only" in r.stdout


def test_splits_are_disjoint():
    from pipeline.splits import TEST, TRAIN, VAL
    assert not (set(TRAIN) & set(VAL) or set(TRAIN) & set(TEST) or set(VAL) & set(TEST))
    assert "amphan_replay" in TEST
