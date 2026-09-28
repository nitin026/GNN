"""Alerting REST API (BRIEF3 Phase B). FastAPI; OpenAPI docs at /docs and /openapi.json.

    python -m backend.api            # http://127.0.0.1:8000  (uvicorn)

All data comes from backend/products/ (scripts/export_products.py or POST /run). Synthetic cases
carry synthetic=true and badge "SYNTHETIC" in every response that describes a case.
"""
import json
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.jobs import HAZARDS, UPLOADS, JobManager
from pipeline.jsonutil import clean
from backend.api.store import PRODUCTS, STATIC, VARS_5, VARS_12, Store, district_rollup

API_VERSION = "1.0.0"
REPORTS = Path(__file__).resolve().parents[2] / "reports"
MAX_UPLOAD_MB = 2048


def create_app(products=PRODUCTS, runner=None):
    store = Store(products)
    jobs = JobManager(runner) if runner else JobManager()
    app = FastAPI(title="SIH 26078 extreme-weather alerting API", version=API_VERSION,
                  description="Ensemble anomaly tracking (GNN), 12->5 km downscaling and IMD-threshold alerts. "
                              "SYNTHETIC cases are flagged in every response.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
                       expose_headers=["X-Bounds", "X-Units", "X-Valid-Time", "X-Synthetic"])
    app.state.store, app.state.jobs = store, jobs
    lat_ms = []                                      # recent request latencies (ms) for /health

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        lat_ms.append(ms)
        del lat_ms[:-1000]
        resp.headers["X-Response-Time-ms"] = f"{ms:.1f}"
        return resp

    def need_case(case):
        if not store.has(case):
            raise HTTPException(404, f"unknown case '{case}' (GET /cases lists them)")
        return store.meta(case)

    def need_lead(meta, lead):
        if lead is not None and lead not in meta["leads_h"]:
            raise HTTPException(422, f"lead {lead} h not available; leads are {meta['leads_h'][0]}.."
                                     f"{meta['leads_h'][-1]} h every 6 h")

    @app.get("/health", tags=["meta"])
    def health():
        s = sorted(lat_ms)
        return {"status": "ok", "version": API_VERSION, "cases": len(store.index()),
                "jobs": {"queued": sum(j["status"] == "queued" for j in jobs.list()),
                         "running": sum(j["status"] == "running" for j in jobs.list())},
                "latency_ms": {"n": len(s), "p50": round(s[len(s) // 2], 1) if s else None,
                               "p95": round(s[int(0.95 * (len(s) - 1))], 1) if s else None}}

    @app.get("/cases", tags=["cases"])
    def cases():
        """All cases with hazard, synthetic flag, badge and date range."""
        return store.index()

    @app.get("/cases/{case}", tags=["cases"])
    def case_meta(case: str):
        """Full meta.json: times, bounds, legends (units), 4-D boxes, metrics copied from reports/*.json."""
        return need_case(case)

    @app.get("/cases/{case}/tracks", tags=["cases"])
    def tracks(case: str, lead: Optional[int] = Query(None, ge=0, le=360, description="lead time (h)")):
        """GeoJSON: member tracks, consensus track (with spread) and 4-D boxes. With `lead`, tracks are
        cut at that lead, current positions are added as Points and boxes get `active`."""
        m = need_case(case)
        need_lead(m, lead)
        fc = store.tracks_at(case, lead)
        return {**fc, "case": case, "synthetic": m["synthetic"], "badge": m["badge"]}

    @app.get("/cases/{case}/bbox4d", tags=["cases"])
    def bbox4d(case: str):
        """4-D bounding boxes (lat, lon, level, time) of the tracked event (GNN consensus)."""
        m = need_case(case)
        return {"case": case, "synthetic": m["synthetic"], "badge": m["badge"], "bbox4d": m["bbox4d"],
                "label_bbox4d": m.get("label_bbox4d")}

    @app.get("/cases/{case}/fields/{var}", tags=["cases"])
    def field(case: str, var: str, res: Literal["12km", "5km"] = "12km",
              lead: int = Query(0, ge=0, le=360), format: Literal["png", "json"] = "png"):
        """A field as a paletted PNG (north up, `bounds` = [lon_min, lat_min, lon_max, lat_max] cell
        edges), or with format=json its URL, bounds and legend. var: t2m, wind, msl, tp, anom (12 km
        only) or strike (strike probability). 5 km fields exist every 12 h."""
        m = need_case(case)
        need_lead(m, lead)
        ok = VARS_12 + ("strike",) if res == "12km" else VARS_5
        if var not in ok:
            raise HTTPException(422, f"var must be one of {list(ok)} at {res}")
        if res == "5km" and lead not in m["leads_5km_h"]:
            raise HTTPException(422, f"5 km fields exist at leads {m['leads_5km_h'][:3]}... (every 12 h)")
        p = store.field_path(case, var, res, lead)
        if not p.exists():
            raise HTTPException(404, "field image not exported")
        vt = m["valid_times"][m["leads_h"].index(lead)]
        leg = m["legends"]["strike" if var == "strike" else var]
        if format == "json":
            rel = p.relative_to(store.root).as_posix()
            return {"case": case, "var": var, "res": res, "lead_h": lead, "valid_time": vt, "url": f"/products/{rel}",
                    "bounds": m["bounds"], "legend": leg, "synthetic": m["synthetic"], "badge": m["badge"]}
        return FileResponse(p, media_type="image/png", headers={
            "X-Bounds": ",".join(str(b) for b in m["bounds"]), "X-Units": leg["units"], "X-Valid-Time": vt,
            "X-Synthetic": str(m["synthetic"]).lower(), "Cache-Control": "public, max-age=3600"})

    @app.get("/alerts", tags=["alerts"])
    def alerts(case: Optional[str] = None, lead: Optional[int] = Query(None, ge=0, le=360),
               lat: Optional[float] = Query(None, ge=-90, le=90), lon: Optional[float] = Query(None, ge=-180, le=180),
               min_category: Literal["low", "moderate", "severe"] = "low", limit: int = Query(500, ge=1, le=5000)):
        """Alerts, most severe first. With lat+lon: only alerts whose 12 km alert region covers that point,
        with the distance to the alert's pinpoint. Each alert has the pinpoint core coordinate (5 km grid),
        category low/moderate/severe, a 5 km impact-radius polygon, valid time, ensemble probability and
        the reason, e.g. "P(wind >= 118 km/h) = 0.72 within 50 km"."""
        if (lat is None) != (lon is None):
            raise HTTPException(422, "give both lat and lon")
        if case:
            need_case(case)
        out = store.alerts_filtered(case, lead, lat, lon, min_category)
        return {"count": len(out), "truncated": len(out) > limit, "rules": "docs/ALERT_RULES.md",
                "alerts": out[:limit]}

    @app.get("/alerts/districts", tags=["alerts"])
    def districts(case: str, lead: Optional[int] = Query(None, ge=0, le=360)):
        """District roll-up (GADM 4.1 level-2 districts rasterised to the 12 km grid): the highest
        category per district and hazard kind, its probability, and the fraction of the district
        covered. Without `lead`, all leads are aggregated and first/last warned leads are given."""
        m = need_case(case)
        need_lead(m, lead)
        r = district_rollup(store, case, lead)
        if r is None:
            raise HTTPException(404, "no alert_grid.npz for this case (python scripts/export_products.py "
                                     f"--grids-only {case}) or no district raster (scripts/build_districts.py)")
        return {**r, "synthetic": m["synthetic"], "badge": m["badge"]}

    @app.get("/reports/{name}", tags=["meta"])
    def report(name: str):
        """A pipeline results file from reports/ (JSON only), e.g. gnn_results.json,
        downscaling_results.json, demo_timing.json. These feed the model-performance view."""
        allowed = {p.name: p for p in REPORTS.glob("*.json")}
        if name not in allowed:
            raise HTTPException(404, f"available: {sorted(allowed)}")
        return clean(json.loads(allowed[name].read_text(encoding="utf-8")))     # strict JSON (NaN -> null)

    @app.post("/run", tags=["run"], status_code=202)
    async def run(request: Request, case: Optional[str] = None,
                  hazard: Optional[Literal["tropical_cyclone", "heat_dome", "cold_wave"]] = None,
                  name: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9_-]{1,40}$")):
        """Run the pipeline asynchronously. Either select an existing case (`?case=cyc_04`) or upload a
        12 km ensemble NetCDF as the raw request body (`Content-Type: application/x-netcdf`, with
        `?hazard=`). Returns a job id; poll GET /run/{job_id}. Stage times are logged."""
        body = await request.body()
        if body:
            if not hazard:
                raise HTTPException(422, f"an uploaded file needs ?hazard= one of {list(HAZARDS)}")
            if len(body) > MAX_UPLOAD_MB * 1e6:
                raise HTTPException(413, f"upload larger than {MAX_UPLOAD_MB} MB")
            if body[:3] not in (b"CDF", b"\x89HD"):
                raise HTTPException(422, "the body is not a NetCDF (classic or NetCDF-4/HDF5) file")
            UPLOADS.mkdir(parents=True, exist_ok=True)
            path = UPLOADS / f"{uuid.uuid4().hex[:12]}.nc"
            path.write_bytes(body)
            job = jobs.submit({"type": "upload", "path": str(path), "bytes": len(body)}, hazard, name)
        elif case:
            m = need_case(case)
            job = jobs.submit({"type": "case", "case": case}, hazard or m["hazard"], name)
        else:
            raise HTTPException(422, "give ?case= or upload a NetCDF body")
        return {"job_id": job["id"], "status": job["status"], "status_url": f"/run/{job['id']}",
                "output_case": job["output_case"]}

    @app.get("/run/{job_id}", tags=["run"])
    def run_status(job_id: str):
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        j.pop("traceback", None)
        return j

    @app.get("/runs", tags=["run"])
    def runs():
        return [{k: j[k] for k in ("id", "status", "output_case", "total_seconds")} for j in jobs.list()]

    app.mount("/products", StaticFiles(directory=str(Path(products))), name="products")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app


app = create_app()
