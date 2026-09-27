# BRIEF4 — Close every gap, prove it on real data, and make it a winning submission
SIH PS 26078 (MoES / NCMRWF). Run this after BRIEF3 (API + dashboard) is finished.

Read these first: BRIEF.md, BRIEF2.md, BRIEF3.md, SUMMARY.md, and every file in reports/
(GNN_RESULTS.md, IMPROVEMENTS.md, SYNTH_VALIDATION.md, BASELINE_RESULTS.md, *.json),
docs/COMPUTE.md, pipeline/gnn.py, pipeline/graphs.py, pipeline/downscale.py and
pipeline/downscale_eval.py.

## Standing rules (apply to every phase)
- All the rules of BRIEF.md still apply: no invented citations or URLs, synthetic data is
  always flagged (`synthetic="true"`, a SYNTHETIC badge in the UI), fixed seeds, no secrets in
  git, and one commit per phase.
- Data budget: stay under 15 GB in total. Currently about 11 GB is used. If a phase needs more,
  delete reproducible intermediates first, and log what was deleted in data/SOURCES.md.
- **Splits are locked.**
  - Synthetic: train = cyc_01, cyc_02, heat_01, heat_02, cold_01, cold_02;
    val = cyc_03, heat_03, cold_03; test = cyc_04, heat_04, cold_04, amphan_replay.
  - The new REAL test set from Phase 1 is locked the moment it is created.
  - Never tune thresholds, checkpoints or calibration on any test case. Record every tuning
    decision in reports/TUNING_LOG.md.
- Memory: the machine is Windows with no GPU and limited RAM, and the downscaling evaluation
  was killed once for running out of memory. Everything must stream: dask chunks, per-case and
  per-lead loops, float32 or float16, np.memmap, `del` + `gc.collect()` between cases.
  Print peak RSS for each script (psutil if available, otherwise tracemalloc).
- If a phase would take more than about 2 h of CPU, move that step to notebooks/train_colab.ipynb
  (a GPU on Colab/Kaggle) and keep a small CPU version locally. Say which one was used.
- Windows Application Control blocks new binary wheels. Try an install once; if it's blocked,
  use a pure-Python fallback and write it down in docs/COMPUTE.md. Do not stall.
- A missed target is fine. A hidden one is not. Every report ends with a "What did not work"
  section that gives the numbers.
- When a phase ends: pytest -q passes, verify_deliverables.py passes (with new checks added for
  that phase), the commit is made, and you stop and report a numbers table.

---------------------------------------------------------------------------------------------
## Phase 0 — Finish what is half-done (BRIEF2 Phase 3 downscaling evaluation)
The downscaling models are trained, but there are no test metrics because the evaluation ran
out of memory.
- Rewrite pipeline/downscale_eval.py to stream: one case, one variable and one lead batch at a
  time, and accumulate the metrics online (running sums, streaming histograms and quantile
  sketches, per-patch spectra averaged together).
- Compare bicubic+lapse-rate, the U-Net and the CorrDiff-style diffusion model on the TEST cases:
  - RMSE and MAE;
  - radially averaged power spectrum ratio at 10, 25 and 50 km;
  - p99 and p99.9 amplitude ratio (pred / truth) and peak rain / peak wind error;
  - CRPS, a rank histogram and the spread–skill ratio for diffusion
    (at least 8 samples, using DDIM);
  - conservation error (max |avgpool(pred) − x12|) and the minimum rain value.
- Add a real sanity check: coarsen IMD 0.25° rain for 2020 by 3× (to 0.75°), downscale it back
  and score it against the real 0.25° values. Mark it clearly as a perfect-model test on real
  data.
- Write reports/DOWNSCALE_RESULTS.md with figures: spectra, a p99 bar chart, and example panels
  (12 km | bicubic | U-Net | diffusion sample | diffusion mean | truth).
- **Headline claim to test:** diffusion keeps extremes where the U-Net smooths them. The target
  is a p99 ratio of at least 0.9 for diffusion against a lower value for the U-Net. If it does
  not hold, say so.

## Phase 1 — Real-data evaluation set (the biggest credibility gain)
Right now only one real event (Amphan) is scored, with 21 fixes. Judges will ask
"does it work on real data?"
- Fetch ERA5 (WB2 or ARCO over HTTPS, as in Phase 2 of BRIEF.md) surface data plus 850 hPa
  daily for more real North Indian Ocean cyclones. Candidates: Fani 2019, Vayu 2019, Nisarga
  2020, Nivar 2020, Tauktae 2021, Yaas 2021, Gulab 2021, Mocha 2023 and Biparjoy 2023.
  Take whatever is available in the stores; each needs its IBTrACS best track.
- Add real heat and cold events, with IMD rain and temperature as truth where available: the
  2019 and 2022 heatwaves (the 2024 one is already fetched) and one more cold wave.
- Split by time: events up to 2021 are REAL-VAL, events from 2022 on are REAL-TEST. Write the
  split to data/real/SPLITS.json and lock it.
- Find a real ensemble forecast (verify every URL first and record it in SOURCES.md):
  - WeatherBench 2 public IFS ENS (ifs_ens) for 2020, which covers Amphan and Nivar;
  - WB2 GraphCast / Pangu / other AI-model forecasts for 2020, if they are public;
  - TIGGE NEPS-G / ECMWF ENS, only if keys are in .env. If they are not, add a short
    "how to plug in NEPS-G" section and a loader stub with the same interface.
- Deliverable: the tracker runs on at least one REAL ensemble forecast of Amphan, and track
  error against IBTrACS is reported by lead time (24/48/72/120/168/240 h). Also record the lead
  time at which the event was first flagged at P ≥ 0.5.

## Phase 2 — Icosahedral mesh GNN (make the code match the PS claim)
The PS promises an icosahedral mesh GNN. The current GNN is an object-graph model. Build the
mesh stage and keep the object GNN as the linker, so the system has two levels.
- pipeline/icomesh.py: a pure-numpy icosahedron subdivision. Keep refinement levels 5–7 over
  the India box plus a halo (GraphCast-style), with multi-level mesh edges. Build the
  grid→mesh and mesh→grid bipartite edges by haversine radius. Unit tests: node count per
  level, all edge lengths within the expected range, no duplicate edges, and a constant field
  round-trips grid→mesh→grid within tolerance.
- pipeline/mesh_gnn.py: encoder → processor → decoder (plain PyTorch; torch_geometric is
  optional because it already works).
  - Inputs per grid cell: z-anomalies of MSLP, T2m, 10 m wind, rain and 850 hPa q-flux
    convergence; EFI; DEM; land-sea mask; lat/lon on the sphere (3-D unit vector); lead time.
  - Output: a per-cell event probability for each hazard. This is a real SEGMENTATION output,
    which fixes the IoU problem: IoU was 0.19 because objects were never re-segmented.
- New pipeline: mesh GNN probability → hysteresis connected components on the mesh →
  objects → the existing object GNN links them across time and members → 4-D boxes.
- Train on synthetic TRAIN and choose on VAL, then report on synthetic TEST and REAL-VAL.
  REAL-TEST is kept for Phase 8.
- Targets on TEST: IoU ≥ 0.40 on every hazard (it was 0.21), CSI kept ≥ 0.65, FAR ≤ 0.20.
- Ablations: grid CNN vs mesh GNN at the same parameter count, and mesh level 5 vs 6 vs 7.
  This gives the evidence for the "no polar/projection distortion" claim.

## Phase 3 — Fix the weaknesses of the object GNN
Each of these comes from reports/GNN_RESULTS.md:
- **Links barely help** (node-only ≈ full). Give edges real information: displacement vs a
  Kalman-predicted position, intensity change, shape overlap, steering-flow consistency
  (850 hPa wind at the source node), and member-agreement counts. Add a graph-level consensus
  readout. Re-run the 4 ablations. If links still add nothing, say so and simplify the model
  (fewer parameters, faster).
- **Over-confidence** (a score of 1 verifies about 60 % of the time at days 7–10). Fit
  lead-time-dependent calibration on VAL only (temperature scaling vs isotonic, per lead band
  0–72, 72–168 and 168–240 h). Report reliability diagrams, Brier score and Brier skill score
  against climatology and against raw ensemble frequency. Target: reliability slope in
  0.8–1.2 at every lead band.
- **No lead-time gain.** Add an early-warning mode that raises an alert when the calibrated
  probability of an event somewhere in a region within 72 h exceeds a threshold. Measure
  hours gained over tracker v2 and over EFI-only. Report it honestly, even if it is zero.
- **Heat/cold label mismatch** (IMD daily criteria vs instantaneous labels). Regenerate the
  heat and cold labels as daily Tmax/Tmin departures that match IMD's operational definition.
  Keep the old labels for backward compatibility. Also fix heat_04 so that it actually meets
  the 40 °C IMD threshold, or document why it does not.
- **Rough domain mask** (cold_04 found a track near Herat): use the Natural Earth India polygon
  plus a 250 km buffer everywhere, in the tracker, the GNN and the alerts.
- **TempestExtremes-style comparator is incomplete.** Add the warm-core criterion (the
  300–500 hPa thickness or temperature anomaly if it is available; otherwise document the
  proxy) and the wind-speed criterion, so the baseline comparison is fair.
- **Spurious tracks < 10 per case** was met in only 8/13 cases. The target is now 13/13.

## Phase 4 — Physics-informed downscaling (the PS claims this; make it true)
- Add the constraints to the loss, with a weight for each term chosen on VAL:
  - penalise rain > threshold where the 850 hPa moisture-flux convergence ≤ 0
    (the PS example);
  - RH ≤ 100 % / q ≤ q_sat(T, p) using Clausius–Clapeyron (MetPy if it installs; otherwise
    write the formula);
  - a wind-divergence smoothness term, and a lapse-rate consistency term between T2m and the
    DEM;
  - the hard conservation projection stays as it is.
- Report the physics-violation rate (% of cells that break each constraint) for bicubic,
  U-Net, diffusion and diffusion+physics, on TEST and on the real IMD check.
- Downscaling realism: synthetic fine scales carry about 19× ERA5's power at 25 km. Retune the
  generator's k^(−5/3) noise amplitude against the real IMD/ERA5 spectrum, regenerate only the
  affected fields, and re-run Phase 4 of BRIEF.md and Phase 0 of this brief. Show the spectrum
  before and after.
- Crop-aware inference: downscale only the 4-D box from the tracker plus a 100 km margin
  (as in the PS: "pipes the isolated region"). Measure time and memory saved against
  full-domain inference.

## Phase 5 — End-to-end operational pipeline
- pipeline/run_operational.py: input is a 12 km ensemble NetCDF (or GRIB2 through cfgrib if it
  installs; if eccodes is blocked, NetCDF only, stated clearly). Steps: anomaly/EFI → mesh
  GNN → object GNN tracks + 4-D boxes → calibrated probabilities → crop → diffusion
  downscaling (N samples) → alerts. Outputs are the exact files the BRIEF3 API and dashboard
  read.
- Alerts: take P(exceed IMD threshold) from the diffusion samples at 5 km and map it to
  low/moderate/severe with the rules in docs/ALERT_RULES.md. Each alert has a pinpoint core,
  a 5 km impact radius, a valid time, a probability and a reason string.
- Make it idempotent and resumable, with a --watch mode that processes new files dropped in a
  folder (this is the "continuously processes NWP streams" claim). Add a Dockerfile plus a
  docker-compose file for the API + frontend + worker; they do not have to be built locally.
- Measure and report in reports/SYSTEM_PERF.md: wall time per stage on this CPU and on the
  Colab GPU (T4), peak RAM, and cost per forecast cycle. The PS says "seconds on a cloud GPU";
  either prove it or restate it with the measured number.

## Phase 6 — Dashboard and API upgrade (builds on BRIEF3)
- Wire in the new outputs: mesh-GNN probability layer, calibrated strike probability, a
  diffusion "scenario" picker (sample 1..N plus mean plus p90), a physics-violation overlay,
  and real-event pages for the Phase 1 storms with the IBTrACS best track overlaid.
- Add a "Forecaster view": compare, side by side at the same lead, the raw ensemble, the EFI,
  our tracker and the IBTrACS/IMD truth.
- Add an "Explain this alert" panel: which variables and members drove it, the probability and
  the calibration curve at that lead.
- Add a "District bulletin" export as PDF or HTML, IMD-style, in English and Hindi. Hindi
  strings go in an i18n file; do not machine-translate numbers or place names wrongly.
- Checks: Lighthouse score ≥ 90 for performance and accessibility (or explain which item
  fails), keyboard navigation, colour-blind-safe palettes, and a smoke test that passes.
- The API gets rate limits, input validation, a health check with model versions and data
  timestamps, and OpenAPI docs.

## Phase 7 — Engineering quality
- Test coverage ≥ 70 % for pipeline/ and backend/ (pytest-cov if it installs; otherwise report
  the line count of tested modules).
- Add a CI config (.github/workflows/ci.yml): lint with ruff, run pytest on a tiny fixture
  dataset, and build the frontend. Put the tiny fixtures in tests/fixtures/, under 20 MB.
- Model cards in docs/MODEL_CARDS.md for the mesh GNN, the object GNN, the U-Net and the
  diffusion model. Each covers training data (synthetic vs real), intended use, metrics,
  known failure modes, and that it is not for operational use without NEPS-G validation.
- README.md with a quick start (3 commands), an architecture diagram (Mermaid), and the
  folder map.
- Clean up dead code (evaluate.py vs evaluate2.py, tune scripts) so that each concept has one
  module.

## Phase 8 — Final blind evaluation (run exactly once)
- Freeze everything: git tag `v1.0-frozen`, then run the full pipeline once on synthetic TEST
  and REAL-TEST. Nothing may be tuned after this point.
- reports/FINAL_RESULTS.md: one headline table (our system vs tracker v2 vs TE-style vs
  EFI-only vs the public SIH-26078 repos' reported numbers, cited from research/PRIOR_WORK.md,
  noting where their numbers came from synthetic data). Give bootstrap 95 % confidence
  intervals (resampling over cases and members) for every headline number, and add
  per-hazard and per-lead breakdowns.
- State plainly which results are synthetic and which are real.

## Phase 9 — Pitch package
- docs/PITCH.md, a 10–12 slide outline:
  problem → why current tools fail (spectral smoothing, with our U-Net vs diffusion figure as
  proof) → architecture (mesh GNN → object GNN → physics diffusion → alerts) → real-data
  results → demo screenshots → operational fit with NCMRWF NEPS-G → cost and speed →
  limitations and roadmap.
- docs/PS_ALIGNMENT.md: a table of every claim in the PS text (the Proposed Solution,
  Methodology, Tools and Deliverables sections) against the file that implements it and the
  evidence. Mark each one DONE, PARTIAL or NOT DONE. Where the tools differ (e.g. plain
  PyTorch instead of DGL, or our own DDPM instead of HF Diffusers), say so and give the
  reason. Judges reward honesty; they punish claims they can disprove.
- docs/JUDGE_QA.md: the 20 hardest questions judges could ask ("Isn't this all synthetic?",
  "Why a GNN over a CNN?", "How is it calibrated?", "What happens with real NEPS-G?",
  "Latency?", "What if the tracker misses a storm?") with answers backed by numbers.
- docs/DEMO_SCRIPT.md: a 3-minute live demo script and a 5-minute video storyboard, with a
  fallback to precomputed files if the network is down.
- Final: update SUMMARY.md and verify_deliverables.py, run pytest, build the frontend, commit
  and tag `v1.0-submission`.

---------------------------------------------------------------------------------------------
## Priority if time runs short
Do these first: Phase 0 → Phase 1 (real ensemble for Amphan) → Phase 2 (mesh GNN, IoU fix)
→ Phase 3 (calibration only) → Phase 4 (moisture-convergence loss only) → Phase 8 → Phase 9.
Everything else is a bonus. After each phase, report: what was done, a numbers table against
the previous best, the targets met or missed, and the next phase.
