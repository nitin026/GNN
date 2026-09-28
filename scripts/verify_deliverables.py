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

# BRIEF2 Phase 5 (demo) and Phase 6 (pitch)
for pth in ("scripts/demo.py", "backend/static/index.html", "backend/static/india_outline.geojson",
            "reports/RESULTS.md", "docs/PITCH.md", "scripts/make_results.py"):
    check((ROOT / pth).exists(), f"{pth} exists")
tj = ROOT / "reports/demo_timing.json"
check(tj.exists(), "reports/demo_timing.json exists (inference time per forecast cycle)")
if tj.exists():
    tm = json.loads(tj.read_text())
    ok = [r for r in tm.values() if r.get("graph") == "built"]
    desc = ", ".join(f"{r['case']} {r['total_seconds']:.0f} s" for r in ok)
    check(bool(ok) and all(r["total_seconds"] < 300 for r in ok),
          f"demo forecast cycle < 5 min on CPU ({desc})")
if (ROOT / "reports/RESULTS.md").exists():
    t = (ROOT / "reports/RESULTS.md").read_text(encoding="utf-8")
    check("What did not work" in t and "PRIOR_WORK" in t, "RESULTS.md cites PRIOR_WORK and has 'What did not work'")
if (ROOT / "docs/PITCH.md").exists():
    n = len(re.findall(r"^## \d+\.", (ROOT / "docs/PITCH.md").read_text(encoding="utf-8"), flags=re.M))
    check(n >= 10, f"docs/PITCH.md has {n} slides (>= 10)")

# --------------------------------------------------------------------------- BRIEF3
section("BRIEF3")
for pth in ("backend/api/app.py", "backend/api/store.py", "backend/api/jobs.py", "docs/API.md", "docs/openapi.json",
            "docs/ALERT_RULES.md", "tests/test_api.py", "tests/test_alert_rules.py", "scripts/build_districts.py",
            "backend/static/districts.json", "frontend/package.json", "frontend/src/App.tsx",
            "frontend/src/views/Operations.tsx", "frontend/src/views/Downscaling.tsx", "frontend/src/views/Alerts.tsx",
            "frontend/src/views/Performance.tsx", "frontend/src/test/app.test.tsx", "frontend/e2e/smoke.spec.ts",
            "reports/SYSTEM_PERF.md", "README.md"):
    check((ROOT / pth).exists(), f"{pth} exists")
for c in ("amphan_replay", "cyc_04", "heat_04", "cold_04", "amphan_era5_real"):
    check((ROOT / "backend/products" / c / "alert_grid.npz").exists(), f"alert_grid.npz for demo case {c}")
if (ROOT / "docs/openapi.json").exists():
    paths = set(json.loads((ROOT / "docs/openapi.json").read_text())["paths"])
    need = {"/health", "/cases", "/cases/{case}/tracks", "/cases/{case}/bbox4d", "/cases/{case}/fields/{var}",
            "/alerts", "/alerts/districts", "/run", "/run/{job_id}"}
    check(need <= paths, f"OpenAPI has all BRIEF3 endpoints (missing {sorted(need - paths)})")
sp = ROOT / "reports/system_perf.json"
if sp.exists():
    js = json.loads(sp.read_text())
    check(js["end_to_end"] and js["end_to_end"]["status"] == "done", "end-to-end POST /run job measured")
    check(all("p95_ms" in r for r in js["api_latency"]), f"API p95 latency measured ({len(js['api_latency'])} endpoints)")
for f in ("ui_operations", "ui_downscaling", "ui_alerts", "ui_performance"):
    check((ROOT / "docs/figures" / f"{f}.png").exists(), f"screenshot docs/figures/{f}.png")

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

# BRIEF4 Phase 1: real-data evaluation set
sj = ROOT / "data/real/SPLITS.json"
check(sj.exists(), "data/real/SPLITS.json exists")
if sj.exists():
    spl = json.loads(sj.read_text())
    check(spl.get("locked") is True, "real split is locked")
    check(all(e["split"] == ("REAL-VAL" if int(k.split("_")[-1]) <= 2021 else "REAL-TEST")
              for k, e in spl["events"].items()), "real split is by time (<= 2021 VAL, >= 2022 TEST)")
    missing = []
    for k, e in spl["events"].items():
        if not list(ROOT.glob(e["files"])):
            missing.append(k)
    check(not missing, f"ERA5 files present for every real event {missing if missing else ''}")
    ncyc = sum(e["hazard"] == "tropical_cyclone" for e in spl["events"].values())
    check(ncyc >= 5 and all((ROOT / "data/real/ibtracs/events" / f"{k}.csv").exists()
                            for k, e in spl["events"].items() if e["hazard"] == "tropical_cyclone"),
          f"{ncyc} real cyclones with IBTrACS best tracks")
ens = list((ROOT / "data/real/ensembles").glob("*.nc"))
check(len(ens) >= 1, f"at least one REAL ensemble forecast ({len(ens)} files)")
for pth in ("pipeline/loaders.py", "docs/NEPS_G.md", "pipeline/real_eval.py", "reports/real_results.json",
            "reports/real_cache/era5_real_scores.json"):
    check((ROOT / pth).exists(), f"{pth} exists")
rr = ROOT / "reports/REAL_RESULTS.md"
check(rr.exists(), "reports/REAL_RESULTS.md exists")
if rr.exists():
    t = rr.read_text(encoding="utf-8")
    for h in ("Track error by lead time", "P ≥ 0.5 first-flag lead time", "What did not work"):
        check(h in t, f"REAL_RESULTS.md reports '{h}'")
    check(all(f"| {L} h" in t or f" {L} h |" in t for L in (24, 48, 72, 120, 168, 240)), "track error at 24/48/72/120/168/240 h")

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
    for ph in (0, 1):
        check(re.search(rf"^BRIEF4 Phase {ph}\b", log, flags=re.M) is not None, f"git commit for BRIEF4 Phase {ph}")
    for ph in "ABCD":
        check(re.search(rf"^BRIEF3 Phase {ph}\b", log, flags=re.M) is not None, f"git commit for BRIEF3 Phase {ph}")
    for ph in (0, 1, 2, 3, 5, 6):
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
