"""POST /run jobs: run the pipeline on an uploaded or selected 12 km ensemble NetCDF (BRIEF3 Phase B).

One worker thread runs jobs one at a time (a cycle needs several GB of RAM). Each job records the
wall time of every stage; the log goes to logs/api_runs.jsonl ("log the inference time").
The job writes its products to backend/products/{job case id}/ with the same export code as
scripts/export_products.py, so the dashboard can open it like any other case.
"""
import importlib.util
import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOADS = ROOT / "data" / "uploads"
LOG = ROOT / "logs" / "api_runs.jsonl"
HAZARDS = ("tropical_cyclone", "heat_dome", "cold_wave")


def _export_module():
    spec = importlib.util.spec_from_file_location("export_products", ROOT / "scripts/export_products.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_runner(job, progress):
    """Real pipeline: load -> candidate graph -> GNN -> downscaling -> alerts -> products."""
    ep = _export_module()
    import numpy as np
    import xarray as xr
    from pipeline.anomaly import Climatology
    from pipeline.tracker2 import regions
    from synth.grids import LAT12, LON12

    progress("setup")
    clim = Climatology()
    ds = ep.Downscaler()
    orog = xr.open_dataset(ROOT / "data/real/dem/dem_g12.nc").orog.values.astype(np.float64)
    reg = regions(orog, LAT12, LON12)
    india = xr.open_dataset(ROOT / "data/real/boundary/india_mask_g12.nc").india.values.astype(bool)
    dec = json.loads((ROOT / "reports/gnn_results.json").read_text())["decode"]
    if job["input"]["type"] == "upload":
        progress("load")
        c = ep.load_file(job["input"]["path"], job["hazard"], job["output_case"])
    else:
        progress("load")
        c = ep.load(job["input"]["case"])
        c["case"] = job["output_case"]
        c["badge"] = ("SYNTHETIC" if c["synthetic"] else "REAL (ERA5 reanalysis)")
        c["source"] = c["source"] + f" [re-run of {job['input']['case']} by POST /run]"
    ep.export_case(job["output_case"], clim, ds, india, reg, dec, c=c, stage=progress)
    ep.write_index()
    alerts = json.loads((ep.OUT / job["output_case"] / "alerts.json").read_text())["alerts"]
    return {"case": job["output_case"], "alerts": len(alerts),
            "severe_alerts": sum(a["category"] == "severe" for a in alerts)}


class JobManager:
    def __init__(self, runner=default_runner):
        self.runner = runner
        self.jobs = {}
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, input_, hazard, name=None):
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "status": "queued", "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "input": input_, "hazard": hazard, "output_case": name or f"run_{jid}",
               "stages": [], "result": None, "error": None, "total_seconds": None}
        with self.lock:
            self.jobs[jid] = job
        self.q.put(jid)
        return job

    def get(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            return json.loads(json.dumps(j)) if j else None

    def list(self):
        with self.lock:
            return [json.loads(json.dumps(j)) for j in self.jobs.values()]

    def _loop(self):
        while True:
            jid = self.q.get()
            job = self.jobs[jid]
            t0 = time.perf_counter()
            last = [None, t0]

            def progress(name):
                now = time.perf_counter()
                with self.lock:
                    if last[0] is not None:
                        job["stages"].append({"stage": last[0], "seconds": round(now - last[1], 2)})
                    job["current_stage"] = name
                last[0], last[1] = name, now

            with self.lock:
                job["status"] = "running"
            try:
                res = self.runner(job, progress)
                progress(None)
                with self.lock:
                    job.update(status="done", result=res)
            except Exception as e:                     # the job fails, the API keeps running
                progress(None)
                with self.lock:
                    job.update(status="failed", error=f"{type(e).__name__}: {e}",
                               traceback=traceback.format_exc()[-2000:])
            with self.lock:
                job["total_seconds"] = round(time.perf_counter() - t0, 2)
                job.pop("current_stage", None)
            try:
                LOG.parent.mkdir(parents=True, exist_ok=True)
                with LOG.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({k: job[k] for k in ("id", "status", "input", "hazard", "output_case",
                                                             "stages", "total_seconds", "error")}) + "\n")
            except OSError:
                pass
