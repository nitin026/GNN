# BRIEF3 — Alerting API + Visualization Dashboard (PS deliverables 3 & 4)
Read SUMMARY.md, reports/GNN_RESULTS.md, pipeline/gnn.py, pipeline/downscale.py first.
Rules from BRIEF.md still apply. Every synthetic case shown in the UI must carry a visible
"SYNTHETIC" badge, and real cases (Amphan ERA5) must be labelled "REAL (ERA5 reanalysis)".
Never show a number in the UI that the pipeline didn't produce.

Environment: Windows, Python 3.12, Node 24 / npm 11. Windows Application Control may block new
binary wheels. First try `pip install fastapi uvicorn`. If pydantic-core is blocked, build the
API on Flask (pure Python) or the stdlib http.server with the same routes. Record the
choice in docs/COMPUTE.md.

## Phase A — Precompute products (backend/products/)
- scripts/export_products.py: for every case, run the GNN tracker + downscaler once and write
  light web-ready files:
  * tracks.geojson: ensemble member tracks, the consensus track and the 4-D bounding boxes
    (bbox + t_start/t_end), each with hazard, peak intensity and member count.
  * strike_prob/{lead}.png + a bounds JSON (or Cloud-Optimized GeoTIFF): probability overlays.
  * fields_12km/{var}/{lead}.png and fields_5km/{var}/{lead}.png on the same colour scale,
    so the 12 km vs 5 km comparison is honest.
  * alerts.json: see Phase B.
  * meta.json: synthetic flag, source, time range, metrics from reports/*.json.
- Keep the total under 500 MB. Add the output folder to .gitignore, and commit a small demo
  subset (amphan_replay + 1 case per hazard).

## Phase B — Alerting REST API (backend/api/)
Endpoints (OpenAPI spec in docs/API.md):
- GET /health
- GET /cases → list with hazard, synthetic flag, date range
- GET /cases/{id}/tracks?lead=H → GeoJSON
- GET /cases/{id}/bbox4d → 4-D bounding boxes
- GET /cases/{id}/fields/{var}?res=12km|5km&lead=H → tile or PNG + bounds
- GET /alerts?case=&lead=&lat=&lon= → the alerts that cover that point: the core pinpoint
  coordinate, category low/moderate/severe, the 5 km impact radius polygon, valid time,
  ensemble probability and the reason, e.g. "P(wind>118 km/h)=0.72".
- GET /alerts/districts?case=&lead= → district-level roll-up (Natural Earth / GADM admin-2
  if available offline; otherwise a coarse grid, stated as such).
- POST /run → runs the pipeline on an uploaded or selected 12 km ensemble NetCDF (async job
  and a status endpoint). Log the inference time.
Alert categories come from ensemble probability × IMD thresholds (cyclone wind/MSLP, heat
departure ≥4.5 / ≥6.5 °C, cold departure ≤ −4.5 / ≤ −6.5 °C, rain 64.5/115.6/204.5 mm/day).
Document the rule in docs/ALERT_RULES.md. Add pytest tests for every endpoint and for
the alert categorisation rules.

## Phase C — Dashboard (frontend/, React + Vite + TypeScript)
Map: MapLibre GL (free OSM/Carto basemap, no API key) + deck.gl layers.
Screens:
1. **Operations view**: India map with a case picker and a lead-time slider (0–240 h, 6 h
   steps, play button). Layers: anomaly field, strike-probability overlay, ensemble spaghetti
   tracks + consensus + uncertainty cone, and the 4-D bounding box (a 3-D time-extruded box,
   togglable).
2. **Downscaling view**: 12 km vs 5 km side-by-side with a swipe slider. A power spectrum and a
   p99 amplitude chart show that peaks are preserved (data from downscale_eval outputs).
3. **Alerts view**: alert pins with 5 km radius circles coloured by severity (IMD
   yellow/orange/red), a table sorted by severity, and click-to-show reasoning and
   probability. A "Check my location" box calls /alerts?lat&lon.
4. **Model performance view**: metric cards and charts from reports/*.json (GNN vs baseline
   vs TempestExtremes-style, Amphan track error vs IBTrACS).
UX: dark "ops-centre" theme with a light mode, responsive down to mobile width,
SYNTHETIC/REAL badges on every screen, a loading skeleton, and an offline demo mode that
reads the precomputed files if the API is down. Use a colourblind-safe palette and a legend
with units on every layer.

## Phase D — Integration & demo
- scripts/demo.py --case amphan_replay: starts the API and the frontend dev server and opens
  the browser.
- Playwright (or Vitest + React Testing Library if Playwright's browsers are blocked) smoke
  test: load the page, move the slider, see an alert.
- Measure the API p95 latency and the end-to-end inference time. Put both in
  reports/SYSTEM_PERF.md.
- Take screenshots of all 4 views for docs/PITCH.md.
- Update README.md with run instructions, SUMMARY.md, and verify_deliverables.py (new checks).
  pytest and the frontend build must pass.

Stop after each phase and report what works, what doesn't, and the numbers.
