"""One-command demo (BRIEF2 Phase 5).

    python scripts/demo.py --case amphan            # timed forecast cycle + static dashboard
    python scripts/demo.py --case cyc_04 --no-browser
    python scripts/demo.py --case amphan --cached   # reuse the cached candidate graph
    python scripts/demo.py --case amphan_replay --skip-cycle   # just API + dashboard

What it does
1. Runs ONE forecast cycle of the pipeline on the case's 12 km ensemble NetCDF and times every
   stage (this is the "inference time per forecast cycle" number for NCMRWF):
     load 12 km ensemble -> z-score anomaly + EFI -> candidate objects + graph -> GNN node/edge
     inference + track decoding -> U-Net 12 -> 5 km downscaling (representative member, every lead) -> alerts
   Nothing is written to backend/products here; the timing goes to reports/demo_timing.json.
2. Starts the REST API (python -m backend.api, port 8000) and the React dashboard (frontend/, Vite
   dev server, port 5173) and opens the browser (BRIEF3 Phase D). With --static, or if
   frontend/node_modules is missing, it serves the BRIEF2 static dashboard
   (backend/static/index.html) instead. Both read the precomputed products of
   scripts/export_products.py (cached model outputs).

`--case amphan` means the SYNTHETIC Amphan replay (20-member ensemble on the real ERA5 Amphan
background); the REAL ERA5 Amphan run is `amphan_era5_real` (single run, no ensemble).
Runs on CPU; the target is < 5 min end to end.
"""
import argparse
import functools
import http.server
import importlib.util
import json
import platform
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALIASES = {"amphan": "amphan_replay", "amphan_real": "amphan_era5_real"}
TIMING = ROOT / "reports" / "demo_timing.json"


def peak_rss_mb():
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)   # peak_wset: Windows peak
    except Exception:                                              # pragma: no cover
        return None


def export_module():
    spec = importlib.util.spec_from_file_location("export_products", ROOT / "scripts/export_products.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Timer:
    def __init__(self):
        self.stages = []

    def __call__(self, name):
        timer = self

        class _S:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                timer.stages.append({"stage": name, "seconds": round(time.perf_counter() - self.t0, 2)})
                print(f"  {name:<46s} {timer.stages[-1]['seconds']:8.2f} s", flush=True)
        return _S()


def run_cycle(case, cached=False):
    """Time one forecast cycle. Returns the timing record."""
    from pipeline.anomaly import Climatology
    from pipeline.efi import efi_gaussian
    from pipeline.gnn_eval import decode, infer, load_graph, load_model
    from pipeline.graphs import build_graph
    from pipeline.tracker2 import regions
    import xarray as xr
    from synth.grids import LAT12, LON12

    ep = export_module()
    T = Timer()
    t_all = time.perf_counter()
    with T("setup: climatology, DEM, masks, models"):
        clim = Climatology()
        ds = ep.Downscaler()
        orog = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float64)
        reg = regions(orog, LAT12, LON12)
        india = xr.open_dataset(ROOT / "data/real/boundary/india_mask_g12.nc").india.values.astype(bool)
        dec = json.loads((ROOT / "reports/gnn_results.json").read_text())["decode"]
    with T("1 load 12 km ensemble NetCDF"):
        c = ep.load(case)
    hz, times = c["hazard"], c["times"]
    var = "msl" if hz == "tropical_cyclone" else "t2m"
    M, NT = c["ens"][var].shape[:2]
    with T("2 z-score anomaly + EFI (all leads)"):
        sign = -1.0 if hz in ("tropical_cyclone", "cold_wave") else 1.0
        ens = c["ens"][var].astype(np.float64)
        z = np.array([clim.z(var, ens[:, t], times[t]) for t in range(NT)], np.float32)
        efi = np.array([sign * efi_gaussian(ens[:, t], *clim.get(var, times[t])) for t in range(NT)],
                       np.float32)
    real = not c["synthetic"]
    variant = "temporal" if real else "full"
    if cached:
        with T("3 candidate objects + graph (CACHED data/graphs)"):
            g = load_graph(case)
    else:
        with T("3 candidate objects + graph"):
            g = build_graph(hz, ens, times, LAT12, LON12, clim)
    del ens
    with T("4 GNN inference + track decoding"):
        model, st, _ = load_model(variant)
        pn, pe = infer(model, st, g)
        dp = dec[variant][hz]
        member_tracks = [decode(g, pn, pe, m, hz, **dp) for m in range(M)]
    n_tracks = sum(len(x) for x in member_tracks)
    from pipeline.gnn_eval import cone
    rep = ep.representative_member(member_tracks, cone({"lat": LAT12, "lon": LON12}, member_tracks))
    with T(f"5 U-Net downscaling 12->5 km (1 member, {NT} leads)"):
        ds5 = {"wind": [], "rain": [], "heat": [], "cold": []}
        tp5 = []
        for t in range(NT):
            f5 = ds(**{v: c["ens"][v][rep, t].astype(np.float64) for v in ("t2m", "u10", "v10", "msl", "tp")})
            ds5["wind"].append((np.hypot(f5["u10"], f5["v10"]) * 3.6).astype(np.float32))
            tp5.append(f5["tp"].astype(np.float32))
            ds5["rain"].append(np.sum(tp5[t - 3:t + 1], 0) if t >= 3 else None)
            ds5["heat"].append(f5["t2m"].astype(np.float32))
            ds5["cold"].append(ds5["heat"][-1])
            if t >= 4:
                tp5[t - 4] = None
    with T("6 alerts (ensemble probability x IMD thresholds)"):
        alerts = ep.compute_alerts(c, clim, reg, india, ds5)
    total = round(time.perf_counter() - t_all, 2)
    rec = {"case": case, "synthetic": c["synthetic"], "hazard": hz, "members": int(M), "leads": int(NT),
           "graph": "cached" if cached else "built", "nodes": int(len(g["X"])), "edges": int(len(g["E"])),
           "tracks": int(n_tracks), "representative_member": int(rep), "alerts": len(alerts),
           "severe_alerts": sum(a["category"] == "severe" for a in alerts),
           "max_z": float(np.nanmax(np.abs(z))), "max_efi": float(np.nanmax(efi)),
           "stages": T.stages, "total_seconds": total, "peak_rss_mb": peak_rss_mb(),
           "machine": f"{platform.processor() or platform.machine()} ({platform.system()}), CPU only",
           "tracker": f"GNN {variant}", "downscaler": "U-Net + exact conservation projection",
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"  {'TOTAL (one forecast cycle)':<46s} {total:8.2f} s   peak RSS {rec['peak_rss_mb']} MB")
    return rec


def save_timing(rec):
    d = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    d[rec["case"] + ("_cached" if rec["graph"] == "cached" else "")] = rec
    from pipeline.jsonutil import dumps
    TIMING.write_text(dumps(d, indent=1))


def serve(port, case, browser=True):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "backend"))
    handler.log_message = lambda *a, **k: None
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/static/index.html?case={case}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"dashboard: {url}  (Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    return httpd


def wait_http(url, timeout=90):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def serve_full(case, api_port=8000, ui_port=5173, browser=True):
    """BRIEF3 Phase D: start the REST API (uvicorn) and the React dashboard (Vite dev server)."""
    import shutil
    import subprocess
    procs = []
    api = f"http://127.0.0.1:{api_port}"
    if not wait_http(f"{api}/health", timeout=1):
        procs.append(subprocess.Popen([sys.executable, "-m", "backend.api", "--port", str(api_port)], cwd=ROOT))
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    ui = f"http://127.0.0.1:{ui_port}"
    if not wait_http(ui, timeout=1):
        procs.append(subprocess.Popen([npx, "vite", "--port", str(ui_port), "--strictPort"], cwd=ROOT / "frontend",
                                      env={**__import__("os").environ, "VITE_API": api}))
    ok_api, ok_ui = wait_http(f"{api}/health"), wait_http(ui)
    print(f"API {'up' if ok_api else 'NOT reachable (dashboard runs in offline demo mode)'}: {api}/docs")
    url = f"{ui}/?case={case}"
    print(f"dashboard {'up' if ok_ui else 'NOT reachable'}: {url}  (Ctrl+C to stop)")
    if browser and ok_ui:
        webbrowser.open(url)
    return procs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="amphan")
    ap.add_argument("--cached", action="store_true", help="reuse the cached candidate graph")
    ap.add_argument("--skip-cycle", action="store_true", help="only open the dashboard")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-serve", action="store_true", help="do not start the dashboard server")
    ap.add_argument("--port", type=int, default=8765, help="port of the --static dashboard")
    ap.add_argument("--static", action="store_true",
                    help="serve the BRIEF2 static dashboard instead of API + React dashboard")
    a = ap.parse_args(argv)
    case = ALIASES.get(a.case, a.case)
    t0 = time.perf_counter()
    if not (ROOT / "backend/products" / case / "meta.json").exists():
        sys.exit(f"no precomputed products for {case}: run  python scripts/export_products.py {case}")
    if not a.skip_cycle:
        print(f"forecast cycle for {case} (CPU):")
        rec = run_cycle(case, cached=a.cached)
        save_timing(rec)
        print(f"{rec['tracks']} member tracks, {rec['alerts']} alerts ({rec['severe_alerts']} severe); "
              f"timing -> {TIMING.relative_to(ROOT)}")
    print(f"demo ready in {time.perf_counter() - t0:.0f} s")
    if a.no_serve:
        return 0
    use_static = a.static or not (ROOT / "frontend/node_modules").exists()
    if use_static:
        httpd = serve(a.port, case, browser=not a.no_browser)
    else:
        procs = serve_full(case, browser=not a.no_browser)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if use_static:
            httpd.shutdown()
        else:
            for p in procs:
                p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
