"""Check every deliverable in BRIEF.md. Prints ALL CHECKS PASSED or lists what is missing."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import netCDF4

ROOT = Path(__file__).resolve().parents[1]
fails, notes = [], []


def check(cond, msg):
    (notes if cond else fails).append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)


def section(name):
    print(f"\n== {name}")


# --------------------------------------------------------------------------- Phase 1
section("Phase 1: research")
pw = ROOT / "research" / "PRIOR_WORK.md"
check(pw.exists(), "research/PRIOR_WORK.md exists")
if pw.exists():
    txt = pw.read_text(encoding="utf-8")
    groups = re.split(r"^## ", txt, flags=re.M)
    counts = {}
    for g in groups:
        head = g.splitlines()[0] if g else ""
        m = re.match(r"([a-d])\)", head)
        if m:
            counts[m.group(1)] = len(re.findall(r"^\|.*https?://", g, flags=re.M))
    total = sum(counts.values())
    check(total >= 20, f">= 20 entries with links (found {total}: {counts})")
    check(len(counts) == 4 and all(v > 0 for v in counts.values()), "all 4 groups a-d populated")
    diff = txt.split("## Our differentiation")[-1] if "## Our differentiation" in txt else ""
    nb = len(re.findall(r"^\s*[-*] ", diff, flags=re.M))
    check(5 <= nb <= 8, f"'Our differentiation' has 5-8 bullets (found {nb})")
    urls = set(u.rstrip(".,;") for u in re.findall(r"https?://[^\s|)<>\]]+", txt))
    lc = ROOT / "research" / "LINKS_CHECK.csv"
    check(lc.exists(), "research/LINKS_CHECK.csv exists")
    if lc.exists():
        rows = list(csv.DictReader(lc.open(encoding="utf-8")))
        st = {r["url"]: r["http_status"] for r in rows}
        good = [u for u, s in st.items() if s.isdigit() and 200 <= int(s) < 400]
        check(set(urls) <= set(st), f"every PRIOR_WORK link is in LINKS_CHECK.csv ({len(urls)} links)")
        check(len(good) == len(st), f"all checked links 2xx/3xx (good={len(good)}, bad={len(st) - len(good)})")
        check(all(r.get("checked_at") for r in rows), "every link has checked_at")

# --------------------------------------------------------------------------- Phase 2
section("Phase 2: real data")
src = ROOT / "data" / "SOURCES.md"
check(src.exists(), "data/SOURCES.md exists")
if src.exists():
    s = src.read_text(encoding="utf-8")
    rows = [ln for ln in s.splitlines() if ln.startswith("| ") and not ln.startswith("| source")]
    stat = [ln.split("|")[3].strip() for ln in rows]
    check(rows and all(x in ("OK", "FAILED", "SKIPPED") for x in stat),
          f"every source row has status OK/FAILED/SKIPPED ({len(rows)} rows)")
    era5_ok = any("ERA5 amphan" in ln and "| OK |" in ln for ln in rows)
    ib_ok = any("IBTrACS" in ln and "| OK |" in ln for ln in rows)
    check(era5_ok or "synthetic_climatology" in s, "ERA5 Amphan India-box OK (or fallback logged)")
    check(ib_ok, "IBTrACS Amphan track OK")
check((ROOT / "data/real/README.md").exists(), "data/real/README.md exists")
for p in ("era5/amphan_era5_g12.nc", "era5/amphan_era5_g5.nc", "ibtracs/amphan_2020_ibtracs.csv",
          "clim/era5_clim_g12.nc"):
    check((ROOT / "data/real" / p).exists(), f"data/real/{p} exists")

# --------------------------------------------------------------------------- Phase 3
section("Phase 3: synthetic cases")
syn = ROOT / "data" / "synthetic"
cases = sorted(p for p in syn.iterdir() if p.is_dir()) if syn.exists() else []
hz_count, replay_ok = {}, False
for c in cases:
    files = [c / "fcst_12km.nc", c / "truth_5km.nc", c / "labels.json"]
    if not all(f.exists() for f in files):
        check(False, f"{c.name}: missing one of fcst_12km.nc/truth_5km.nc/labels.json")
        continue
    lab = json.loads(files[2].read_text())
    ok = lab.get("synthetic") == "true"
    for f in files[:2]:
        d = netCDF4.Dataset(f)
        ok &= d.getncattr("synthetic") == "true"
        if f.name == "fcst_12km.nc":
            steps = list(d.variables["step"][:])
            ok &= d.dimensions["number"].size == 20 and steps == list(range(0, 241, 6))
            if c.name == "amphan_replay":
                replay_ok = "ERA5" in d.getncattr("background") and "real" in d.getncattr("background")
        d.close()
    check(ok, f"{c.name}: 3 files, synthetic=\"true\", 20 members, lead 0-240 h/6 h")
    hz_count[lab["hazard"]] = hz_count.get(lab["hazard"], 0) + (c.name != "amphan_replay")
check(len(cases) >= 13, f">= 13 cases (found {len(cases)})")
for hz in ("tropical_cyclone", "heat_dome", "cold_wave"):
    check(hz_count.get(hz, 0) >= 4, f">= 4 {hz} cases (found {hz_count.get(hz, 0)})")
check(replay_ok, "amphan_replay case exists on a real ERA5 background")

# --------------------------------------------------------------------------- Phase 4/5 reports
section("Phase 4/5: reports")
for rep in ("SYNTH_VALIDATION.md", "BASELINE_RESULTS.md"):
    p = ROOT / "reports" / rep
    check(p.exists(), f"reports/{rep} exists")
    if p.exists():
        t = p.read_text(encoding="utf-8")
        figs = re.findall(r"\]\((figures/[^)]+\.png)\)", t)
        check(figs and all((ROOT / "reports" / f).exists() for f in figs),
              f"{rep}: {len(figs)} figures referenced and present")
        check(len(re.findall(r"^\|.*\d+\.\d+.*\|$", t, flags=re.M)) >= 3,
              f"{rep}: has numeric tables")

section("Phase 5: code and tests")
for p in ("pipeline/anomaly.py", "pipeline/efi.py", "pipeline/track.py", "Makefile",
          "requirements.txt"):
    check((ROOT / p).exists(), f"{p} exists")
mk = (ROOT / "Makefile").read_text() if (ROOT / "Makefile").exists() else ""
for t in ("research", "data-real", "data-synth", "validate", "baseline", "test", "all"):
    check(re.search(rf"^{t}:", mk, flags=re.M) is not None, f"Makefile target '{t}'")
req = (ROOT / "requirements.txt").read_text().splitlines() if (ROOT / "requirements.txt").exists() else []
check(req and all("==" in r for r in req if r.strip() and not r.startswith(("#", "-"))),
      "requirements.txt fully pinned")
tests = " ".join(p.read_text() for p in (ROOT / "tests").glob("test_*.py"))
for name, pat in (("conservation avgpool", r"def test_case_files_conserve"),
                  ("EFI sanity", r"def test_efi_climate_ensemble_is_zero"),
                  ("rain >= 0", r"def test_rain_nonnegative"),
                  ("tracker IoU on synthetic case", r"def test_tracker_iou_on_synthetic_case")):
    check(re.search(pat, tests) is not None, f"test present: {name}")

# --------------------------------------------------------------------------- BRIEF2
section("BRIEF2")
for pth in ("docs/COMPUTE.md", "notebooks/train_colab.ipynb", "scripts/package_training_data.py",
            "reports/IMPROVEMENTS.md", "pipeline/tracker2.py", "reports/GNN_RESULTS.md",
            "pipeline/gnn.py", "scripts/train_tracker.py"):
    check((ROOT / pth).exists(), f"{pth} exists")
if (ROOT / "reports/GNN_RESULTS.md").exists():
    t = (ROOT / "reports/GNN_RESULTS.md").read_text(encoding="utf-8")
    figs = re.findall(r"\]\((figures/[^)]+\.png)\)", t)
    check(figs and all((ROOT / "reports" / f).exists() for f in figs),
          f"GNN_RESULTS.md: {len(figs)} figures referenced and present")
    for v in ("full", "temporal", "cross", "none"):
        check((ROOT / "models/tracker" / v / "config.json").exists(), f"GNN variant '{v}' trained")

# --------------------------------------------------------------------------- BRIEF4
section("BRIEF4")
check((ROOT / "reports/TUNING_LOG.md").exists(), "reports/TUNING_LOG.md exists")
dsr = ROOT / "reports/DOWNSCALE_RESULTS.md"
check(dsr.exists(), "reports/DOWNSCALE_RESULTS.md exists (Phase 0)")
if dsr.exists():
    t = dsr.read_text(encoding="utf-8")
    figs = re.findall(r"\]\((figures/[^)]+\.png)\)", t)
    check(len(figs) >= 3 and all((ROOT / "reports" / f).exists() for f in figs),
          f"DOWNSCALE_RESULTS.md: {len(figs)} figures present")
    check("## What did not work" in t, "DOWNSCALE_RESULTS.md has a 'What did not work' section")
    js = json.loads((ROOT / "reports/downscaling_results.json").read_text())
    check(set(js["metrics"]) >= {"bicubic+lapse", "unet", "diffusion_mean", "diffusion_sample"},
          "downscaling metrics for all 4 models")
    check(js["n_samples"] >= 8 and "crps" in js["metrics"]["diffusion_mean"]["tp"],
          "diffusion CRPS with >= 8 samples")
    check(js["peak_rss_mb"] > 0, f"peak RSS recorded ({js['peak_rss_mb']:.0f} MB)")
    check("imd_perfect_model" in js, "IMD perfect-model real-data check present")

# --------------------------------------------------------------------------- constraints
section("Constraints")
size = sum(p.stat().st_size for p in (ROOT / "data").rglob("*") if p.is_file()) / 1e9
check(size < 15, f"total data size {size:.2f} GB < 15 GB")
try:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    check(not any(Path(t).name.startswith(".env") for t in tracked), "no .env tracked in git")
    secret = re.compile(r"(AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|"
                        r"(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9/+_-]{20,}['\"])", re.I)
    leaks = [t for t in tracked if (ROOT / t).is_file() and (ROOT / t).stat().st_size < 5e6
             and secret.search((ROOT / t).read_text(errors="ignore"))]
    check(not leaks, f"no secret-looking strings in tracked files {leaks if leaks else ''}")
    big = [t for t in tracked if (ROOT / t).is_file() and (ROOT / t).stat().st_size > 50e6]
    check(not big, "no files > 50 MB tracked in git")
    log = subprocess.run(["git", "log", "--format=%s"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    for ph in range(1, 6):
        check(re.search(rf"^Phase {ph}\b", log, flags=re.M) is not None, f"git commit for Phase {ph}")
    for ph in range(0, 2):
        check(re.search(rf"^BRIEF2 Phase {ph}\b", log, flags=re.M) is not None,
              f"git commit for BRIEF2 Phase {ph}")
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    check(False, f"git available ({e})")
check((ROOT / "SUMMARY.md").exists(), "SUMMARY.md exists")

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"ALL CHECKS PASSED ({len(notes)} checks)")
