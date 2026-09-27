"""Run Makefile targets without GNU make (Windows). Usage: python scripts/run_all.py [target ...]"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "Makefile").read_text()
py = os.environ.get("PY", sys.executable)
targets, cur = {}, None
for ln in text.splitlines():
    m = re.match(r"^([a-z-]+):\s*(.*)$", ln)
    if m and not ln.startswith("\t"):
        cur = m.group(1)
        targets[cur] = {"deps": m.group(2).split(), "cmds": []}
    elif ln.startswith("\t") and cur:
        targets[cur]["cmds"].append(ln.strip().replace("$(PY)", py))


def run(t, done=set()):
    if t in done:
        return
    for d in targets[t]["deps"]:
        run(d)
    for c in targets[t]["cmds"]:
        print(f"[{t}] {c}", flush=True)
        subprocess.run(c, shell=True, check=True, cwd=ROOT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    done.add(t)


for t in sys.argv[1:] or ["all"]:
    run(t)
