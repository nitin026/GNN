# Pitch deck outline (10 slides): SIH PS 26078, MoES / NCMRWF

Every number below comes from `reports/RESULTS.md`, which `scripts/make_results.py` builds from
`reports/*.json`. SYNTHETIC and REAL results are labelled on every slide. BRIEF4 Phase 9 extends
this deck; this is the BRIEF2 version.

## 1. Problem
- NCMRWF's NEPS-G produces a 23-member, 12 km ensemble every day out to 10 days. Forecasters need
  to know **where an extreme event is, how sure the ensemble is, and how intense it is at a 5 km
  scale**. Reading raw fields by eye for 23 members × 40 leads is not feasible.
- Three hazards in scope: tropical cyclones, heat waves and cold waves (IMD criteria), plus heavy rain.

## 2. The gap
- Classical trackers (threshold + connected components, TempestExtremes-style) produce hundreds of
  false-alarm tracks per ensemble. Our re-implementation gives **177 spurious tracks per case** on
  SYNTHETIC TEST.
- Deterministic CNN/U-Net downscalers smooth fine scales. Our U-Net keeps only **0.25** of the true
  10 km rain power.
- Public SIH-26078 repos: one storm each, synthetic or illustrative data, no exact ground truth
  (table in RESULTS.md §5).

## 3. Our method
- **Data with exact truth.** 13 SYNTHETIC 20-member cases (cyclones on real IBTrACS tracks, heat
  domes, cold waves) on REAL ERA5 backgrounds. avgpool(5 km) == 12 km exactly.
- **Anomaly layer.** z-scores against the ERA5 1990–2019 climatology, plus the ECMWF EFI.
- **GNN tracker.** Nodes are candidate objects per member per lead; edges are temporal and
  cross-member links. It predicts event probability and link probability, then Hungarian decoding
  builds tracks. Outputs: 4-D boxes, strike probability and a consensus cone.
- **Downscaling.** U-Net and CorrDiff-style residual diffusion, DEM-conditioned, with an **exact
  conservation projection**.
- **Alerts.** Ensemble probability × IMD thresholds, giving low/moderate/severe alerts, each with a
  pinpoint, a 5 km impact radius and a reason string.

## 4. Results: tracking (SYNTHETIC TEST)
- The GNN reaches CSI **0.65** against 0.32 for tracker v2 and 0.09 for TE-style; FAR is **0.16**
  against 0.58.
- Spurious tracks fall to **6 per case**, from 468 for the original baseline and 177 for TE-style.
- Honest: IoU is **0.21**, below the original baseline's 0.33. The GNN does not re-segment objects;
  the fix is the mesh-GNN head in BRIEF4.

## 5. Results: real data
- Amphan 2020, REAL ERA5 against the IBTrACS best track: the GNN (temporal) mean track error is
  **38 km** (median 28 km) over 22 six-hourly fixes, against 45 km for tracker v2 and 55 km for TE-style.
- REAL IMD 0.25° perfect-model downscaling check: every model is within 0.1 mm/day RMSE of bicubic.
  This data is out of distribution for the learned models, and we say so.

## 6. Results: downscaling (SYNTHETIC TEST)
- The diffusion sample keeps **0.61** of the true 10 km rain power (U-Net 0.25, bicubic 0.06) and a
  patch-peak ratio of **1.05** (U-Net 0.86).
- Conservation error is ≤ 2e-5 for every model (projection layer).
- Honest: the p99 ratio does **not** separate the models (0.99–1.00 for all).
- Figure: `reports/figures/ds4_example.png` (12 km | bicubic | U-Net | diffusion | truth).

## 7. Demo
- `python scripts/demo.py --case amphan` runs one full forecast cycle on CPU, then opens the dashboard.
- Dashboard features: lead slider with play, anomaly and strike-probability layers, spaghetti
  tracks with consensus and cone, the 4-D box, 12 km vs 5 km side by side, and an IMD-colour
  alert bulletin.
- Screenshots: `docs/figures/demo_amphan_lead096.png` and `docs/figures/demo_cold04_lead120.png`.

## 8. Operational fit with NEPS-G
- **One forecast cycle takes 122 s on a laptop CPU** (20 members × 41 leads, from the 12 km NetCDF to
  alerts; `reports/demo_timing.json`). Peak RAM is 5.8 GB.
- Variable names and the member layout follow NEPS-G conventions, so real NEPS-G drops in through
  `pipeline/loaders.py` (see `docs/NEPS_G.md`).
- Alert rules are written down (`docs/ALERT_RULES.md`); probability cut-offs are a stated design
  choice, not IMD rules.

## 9. Limitations (said before the judges ask)
- Training and TEST skill are on SYNTHETIC events. Real skill so far is one storm in one reanalysis
  run. No real ensemble forecast has been scored yet, because there is no TIGGE key; public WB2 IFS ENS
  and GenCast are the next step.
- ERA5 underestimates cyclone intensity: 26 m/s against 66 m/s observed for Amphan.
- GNN probabilities are over-confident at days 7–10 and are not calibrated yet.
- IoU is limited by object-level (not pixel-level) segmentation.

## 10. Roadmap
1. Real-data evaluation set: 13 NIO cyclones (2019–2023), real heat and cold waves, WB2 IFS ENS and
   GenCast ensembles for Amphan, and a locked time-based split.
2. Icosahedral mesh GNN segmentation stage, with the object GNN kept as the linker.
3. Lead-dependent calibration and an early-warning mode.
4. Physics-informed diffusion loss (rain only where moisture-flux convergence > 0; q ≤ q_sat).
5. Operational worker with `--watch`, Docker, REST API and React dashboard.
