"""BRIEF3 Phase D: API latency (p50/p95) and end-to-end inference time -> reports/SYSTEM_PERF.md

    python -m backend.api &            # the API must be running (real HTTP, uvicorn)
    python scripts/measure_perf.py [--api http://127.0.0.1:8000] [--n 40] [--no-run]

1. Latency: each GET endpoint is called N times over HTTP (keep-alive session), first call dropped
   as warm-up; p50 / p95 / max in ms.
2. End-to-end: POST /run?case=cyc_04 runs the full pipeline in the API worker (load -> candidate
   graph -> GNN -> downscaling + images -> alerts -> products); the stage times come from the job.
   The job's output case is deleted afterwards so the committed index is unchanged.
3. The CPU forecast cycle from scripts/demo.py (reports/demo_timing.json) is included.
"""
import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.jsonutil import dumps  # noqa: E402

ENDPOINTS = [
    ("GET /health", "/health"),
    ("GET /cases", "/cases"),
    ("GET /cases/{id}", "/cases/amphan_replay"),
    ("GET /cases/{id}/tracks?lead=96", "/cases/amphan_replay/tracks?lead=96"),
    ("GET /cases/{id}/bbox4d", "/cases/amphan_replay/bbox4d"),
    ("GET /cases/{id}/fields/wind (PNG 12 km)", "/cases/amphan_replay/fields/wind?res=12km&lead=96"),
    ("GET /cases/{id}/fields/wind (PNG 5 km)", "/cases/amphan_replay/fields/wind?res=5km&lead=96"),
    ("GET /alerts?case&lead", "/alerts?case=amphan_replay&lead=96"),
    ("GET /alerts?lat&lon (point query)", "/alerts?case=amphan_replay&lat=15.5&lon=87.5"),
    ("GET /alerts (all cases)", "/alerts?limit=100"),
    ("GET /alerts/districts?lead", "/alerts/districts?case=amphan_replay&lead=96"),
    ("GET /alerts/districts (all leads)", "/alerts/districts?case=amphan_replay"),
]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def latency(api, n):
    s = requests.Session()
    out = []
    for name, path in ENDPOINTS:
        ms = []
        for i in range(n + 1):
            t0 = time.perf_counter()
            r = s.get(api + path, timeout=60)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                raise SystemExit(f"{path}: HTTP {r.status_code} {r.text[:200]}")
            if i:
                ms.append(dt)
        out.append({"endpoint": name, "path": path, "n": n, "p50_ms": round(statistics.median(ms), 1),
                    "p95_ms": round(pct(ms, 0.95), 1), "max_ms": round(max(ms), 1), "bytes": len(r.content)})
        print(f"{name:<45s} p50 {out[-1]['p50_ms']:7.1f} ms  p95 {out[-1]['p95_ms']:7.1f} ms", flush=True)
    return out


def end_to_end(api, case="cyc_04"):
    name = "perf_e2e_" + case
    t0 = time.perf_counter()
    r = requests.post(f"{api}/run", params={"case": case, "name": name}, timeout=60)
    r.raise_for_status()
    jid = r.json()["job_id"]
    while True:
        j = requests.get(f"{api}/run/{jid}", timeout=30).json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(2)
    wall = time.perf_counter() - t0
    out_dir = ROOT / "backend/products" / name
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6 if out_dir.exists() else 0
    shutil.rmtree(out_dir, ignore_errors=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ep", ROOT / "scripts/export_products.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    ep.write_index()
    return {"case": case, "status": j["status"], "error": j.get("error"), "stages": j["stages"],
            "job_seconds": j["total_seconds"], "wall_seconds": round(wall, 1), "products_mb": round(size, 1),
            "result": j.get("result")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--operational", nargs="*", default=[], help="run.json files of pipeline/run_operational.py")
    ap.add_argument("--vm-usd-per-hour", type=float, default=0.50,
                    help="ASSUMED on-demand price of a 12-vCPU / 16 GB cloud VM, used only for the cost estimate")
    ap.add_argument("--latency-only", action="store_true", help="keep the stored end-to-end and operational numbers")
    a = ap.parse_args()
    prev = json.loads((ROOT / "reports/system_perf.json").read_text()) if (ROOT / "reports/system_perf.json").exists() else {}
    lat = latency(a.api, a.n) if a.api != "none" else prev.get("api_latency", [])
    e2e = prev.get("end_to_end") if (a.no_run or a.latency_only or a.api == "none") else end_to_end(a.api)
    ops = [json.loads(Path(f).read_text()) for f in a.operational] or prev.get("operational", [])
    for o in ops:
        o["cost_usd_per_cycle_assumed_vm"] = round(o["total_seconds"] / 3600 * a.vm_usd_per_hour, 3)
        o["vm_usd_per_hour_assumed"] = a.vm_usd_per_hour
    demo = json.loads((ROOT / "reports/demo_timing.json").read_text()) if (ROOT / "reports/demo_timing.json").exists() else {}
    res = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": f"{platform.processor()} ({platform.system()}), CPU only, no GPU",
           "api_latency": lat, "end_to_end": e2e, "demo_cycle": demo, "operational": ops}
    (ROOT / "reports/system_perf.json").write_text(dumps(res, indent=1))
    write_md(res)


def write_md(res):
    L = ["# System performance (BRIEF3 Phase D)", "",
         f"Measured {res['measured_at']} on {res['machine']} by `scripts/measure_perf.py` "
         "(API served by uvicorn, called over real HTTP from the same machine).", "",
         "## API latency", "", "| endpoint | n | p50 (ms) | p95 (ms) | max (ms) | response size |", "|---|---|---|---|---|---|"]
    for r in res["api_latency"]:
        L.append(f"| {r['endpoint']} | {r['n']} | {r['p50_ms']} | {r['p95_ms']} | {r['max_ms']} | {r['bytes'] / 1e3:.1f} kB |")
    p95 = max(r["p95_ms"] for r in res["api_latency"])
    L += ["", f"Worst endpoint p95: **{p95:.0f} ms**.", ""]
    e = res["end_to_end"]
    if e:
        L += ["## End-to-end inference through the API (POST /run)", "",
              f"Job on `{e['case']}` (20 members x 41 leads, 12 km): status **{e['status']}**, "
              f"job time **{e['job_seconds']:.0f} s** (wall incl. polling {e['wall_seconds']:.0f} s), "
              f"{e['products_mb']} MB of products written.", "",
              "| stage | seconds |", "|---|---|"]
        L += [f"| {s['stage']} | {s['seconds']:.1f} |" for s in e["stages"]]
        L.append("")
    if res["demo_cycle"]:
        L += ["## Forecast cycle without image export (scripts/demo.py)", "",
              "| case | members x leads | total (s) | peak RSS (MB) | stages |", "|---|---|---|---|---|"]
        for k, r in res["demo_cycle"].items():
            st = "; ".join(f"{s['stage']} {s['seconds']:.0f}s" for s in r["stages"])
            L.append(f"| {k} | {r['members']} x {r['leads']} | {r['total_seconds']:.0f} | {r['peak_rss_mb']} | {st} |")
        L.append("")
    if res.get("operational"):
        L += ["## Operational pipeline (pipeline/run_operational.py, BRIEF4 Phase 5)", "",
              "Full chain: load -> anomaly/EFI + mesh GNN (3 hazards) -> objects + object GNN tracks -> calibration -> "
              "crop (4-D box + 100 km) -> diffusion downscaling (U-Net mean + residual sample per member, 25 DDIM steps) -> "
              "5 km alerts -> products.", "",
              "| input | members x leads | total (s) | peak RAM (MB) | alerts | cost / cycle (assumed VM price) |",
              "|---|---|---|---|---|---|"]
        for o in res["operational"]:
            mx = f"{o.get('members', '?')} x {o.get('leads', '?')}"
            L.append(f"| {o['input']} | {mx} | {o['total_seconds']:.0f} | {o['peak_rss_mb']} | {o['alerts']} | "
                     f"${o['cost_usd_per_cycle_assumed_vm']:.3f} at ${o['vm_usd_per_hour_assumed']:.2f}/h (assumption) |")
        L += ["", "| stage | " + " | ".join(Path(o["input"]).name for o in res["operational"]) + " |",
              "|---|" + "---|" * len(res["operational"])]
        names = []
        for o in res["operational"]:
            for s_ in o["stages"]:
                if s_["stage"] not in names:
                    names.append(s_["stage"])
        for n in names:
            L.append(f"| {n} | " + " | ".join(next((f"{s_['seconds']:.1f}" for s_ in o["stages"] if s_["stage"] == n), "-")
                                              for o in res["operational"]) + " |")
        L += ["", "**GPU claim.** The PS text says a cycle takes 'seconds on a cloud GPU'. No GPU was available here, so this "
              "is NOT measured: the numbers above are CPU-only. The notebook notebooks/train_colab.ipynb has a cell that times "
              "the same command on a Colab T4; until it is run, the honest statement is the CPU time above.", ""]
    L += ["## What did not work", "",
          "- Everything runs on one CPU (no GPU); a Colab T4 timing is BRIEF4 Phase 5.",
          "- Peak RAM of a forecast cycle is several GB because the whole 20-member ensemble is held in "
          "memory; streaming per member/lead is BRIEF4 Phase 5.",
          "- The API runs one pipeline job at a time; a second POST /run queues."]
    if e and e["status"] != "done":
        L.append(f"- The end-to-end job FAILED: {e['error']}")
    (ROOT / "reports/SYSTEM_PERF.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/SYSTEM_PERF.md")


if __name__ == "__main__":
    main()
