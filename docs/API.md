# Alerting REST API (BRIEF3 Phase B)

`backend/api/` is built on FastAPI. `pip install fastapi uvicorn` worked on this machine and
pydantic-core was not blocked (see docs/COMPUTE.md).

```
python -m backend.api                 # http://127.0.0.1:8000, interactive docs at /docs
```

- The full OpenAPI 3 spec is in [`docs/openapi.json`](openapi.json), exported from the app.
- Every response comes from files in `backend/products/`, written by `scripts/export_products.py`
  or by a `POST /run` job.
- Every response that describes a case carries `synthetic` and `badge`: `SYNTHETIC`,
  `REAL (ERA5 reanalysis)` or `UPLOADED FORECAST (unverified)`.
- CORS is open for GET and POST, so the dashboard dev server can call the API.
- Every response has an `X-Response-Time-ms` header, and `/health` reports p50 and p95 over the
  last 1000 requests.

## Endpoints

| method | path | parameters | returns |
|---|---|---|---|
| GET | `/health` | – | `status`, `version`, number of cases, queued/running jobs, request latency p50/p95 (ms) |
| GET | `/cases` | – | list of `{case, hazard, synthetic, badge, init_time, start, end, source}` |
| GET | `/cases/{id}` | – | full `meta.json`: valid times, leads, bounds, colour legends with units, 4-D boxes, metrics copied from `reports/*.json` |
| GET | `/cases/{id}/tracks` | `lead` (h, optional) | GeoJSON FeatureCollection (see below) |
| GET | `/cases/{id}/bbox4d` | – | GNN-consensus 4-D boxes (lat/lon/level/time), plus the label box for synthetic cases |
| GET | `/cases/{id}/fields/{var}` | `res` = `12km` or `5km`, `lead`, `format` = `png` or `json` | a paletted PNG (headers `X-Bounds`, `X-Units`, `X-Valid-Time`, `X-Synthetic`), or with `format=json` `{url, bounds, legend, valid_time}` |
| GET | `/alerts` | `case`, `lead`, `lat` + `lon`, `min_category`, `limit` | alerts, most severe first (see below) |
| GET | `/alerts/districts` | `case` (required), `lead` (optional) | district roll-up (see below) |
| GET | `/reports/{name}` | – | a pipeline results JSON from `reports/` (e.g. `gnn_results.json`, `downscaling_results.json`, `demo_timing.json`) |
| POST | `/run` | `case`, or a NetCDF body with `hazard`; optional `name` | `202 {job_id, status_url, output_case}` |
| GET | `/run/{job_id}` | – | job status: `queued`, `running`, `done` or `failed`; per-stage wall times; result |
| GET | `/runs` | – | all jobs of this API process |

Static files are also served:
- `/products/...` holds the images, GeoJSON and JSON that the endpoints reference.
- `/static/...` holds the India outline and the GADM district outlines.

### `fields/{var}`

- `var` is one of `t2m`, `wind`, `msl`, `tp`, `anom` (12 km only; the hazard z-score) or `strike`
  (strike probability, 12 km).
- The PNG is north-up and covers `bounds = [lon_min, lat_min, lon_max, lat_max]` (cell edges,
  `[60, 0, 99.96, 39.96]`).
- 12 km and 5 km images of the same variable share one colour scale.
- 5 km images exist every 12 h; other leads return 422 with the list of available leads.

### `tracks?lead=H`

- Member tracks and the consensus track are cut at lead H.
- The current positions are added as Point features (`member_track_position`,
  `consensus_position` with `spread_km` and `member_count`).
- Each 4-D box gets `active: true` if H is inside its lead range.
- Without `lead`, the whole file is returned.

### `alerts`

Each alert contains:
- `pinpoint`: the core coordinate on the 5 km grid, the most extreme 5 km cell of the downscaled
  representative member inside the alert region.
- `category`: `low`, `moderate` or `severe`, with its IMD colour.
- `impact_polygon`: a 5 km-radius circle around the pinpoint, and `impact_radius_km`.
- `valid_time`, `lead_h`.
- `probability`: the ensemble neighbourhood probability. `probability_cell` is the per-cell
  probability.
- `reason`: for example `"P(wind >= 118 km/h) = 0.72 within 50 km"`.
- `region_bbox`, `in_india`, `n_members`, `synthetic`.

**Point queries.** With `lat` and `lon`, only alerts whose 12 km alert *region* covers the point are
returned:
- The point must be inside the region's bounding box, and its 12 km cell must have
  category ≥ low in `alert_grid.npz` for that kind and lead.
- Each returned alert gets `distance_to_pinpoint_km` and `within_impact_radius`.

The rules are in [ALERT_RULES.md](ALERT_RULES.md).

### `alerts/districts`

- Districts are the 676 GADM 4.1 level-2 districts of India, rasterised to the 12 km grid by
  `scripts/build_districts.py`. The 5 small districts that own no 12 km cell centre use their
  centroid cell.
- Per district and hazard kind, the roll-up gives:
  - the highest category over the district's cells;
  - the neighbourhood probability of the rule that set that category, as a reason string;
  - `fraction_of_district`, the fraction of its cells at ≥ low;
  - `first_lead_h` and `last_lead_h`.
- Without `lead`, all leads are aggregated.
- GADM boundaries are shown as given. They are not a Survey of India product, and the GADM
  licence is for non-commercial use.

### `POST /run`

- To re-run an existing case: `POST /run?case=cyc_04`.
- To upload a 12 km ensemble NetCDF as the raw body:
  `curl -X POST "http://127.0.0.1:8000/run?hazard=tropical_cyclone&name=my_run" -H "Content-Type: application/x-netcdf" --data-binary @ens.nc`
- The file layout is the one of `fcst_12km.nc`:
  - dims `(number, step [h], latitude, longitude)`;
  - variables `msl` [Pa] (required), `u10`, `v10`, `t2m`, `tp`;
  - global attribute `init_time`.
- Other grids are bilinearly regridded to G12. Missing variables are filled (t2m from the ERA5
  climatology, the others with 0), and the fill is written in the case `source`.
- One worker thread runs the jobs one at a time. A cycle needs several GB of RAM.
- The stage times (`load`, `candidate graph`, `GNN tracking`, `downscaling + images`, `alerts`)
  are returned by `GET /run/{id}` and appended to `logs/api_runs.jsonl`.
- The output becomes a normal case (`/cases/{output_case}`).

## Errors

| status | when |
|---|---|
| 404 | unknown case, job or report |
| 422 | invalid or unavailable `lead`, `var` or `res`; `lat` without `lon`; an upload without `hazard`; a body that is not NetCDF |
| 413 | upload larger than 2 GB |
| 500 | not expected: bare NaN values in result files are converted to `null` before they are served |

Rate limits and input hardening beyond these checks are BRIEF4 Phase 6.

## Tests

`tests/test_api.py` covers every endpoint through the FastAPI TestClient. `POST /run` uses a fake
runner there, so the test takes seconds. `tests/test_alert_rules.py` covers the categorisation rules.
