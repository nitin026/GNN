# BRIEF2 — SIH PS 26078: from baseline to a competition-winning system
Read BRIEF.md, SUMMARY.md, reports/BASELINE_RESULTS.md and reports/SYNTH_VALIDATION.md first.
Keep every rule in BRIEF.md: no fake citations, synthetic data is always flagged, seeds are fixed,
one commit per phase, data stays under 15 GB, and weak results are reported honestly.

## Phase 0 — Compute check (do this before anything else)
- Test whether torch (+CUDA) can be installed or imported. Windows Application Control
  blocked new wheels last time.
- If it's blocked, don't stall. Make every training script run unchanged in Google Colab/Kaggle:
  write notebooks/train_colab.ipynb with data upload/download steps and a
  scripts/package_training_data.py that makes a small (<2 GB) training bundle.
- Keep a CPU-only path (numpy/scipy/sklearn) so the whole demo still runs locally.
- Write docs/COMPUTE.md explaining which path was used and why.

## Phase 1 — Fix the baseline's weak spots (cheap wins first)
- Tracker false alarms: add hysteresis thresholds (high to start a track, low to continue it),
  minimum area and duration filters, a hazard-specific z-score, and a Kalman-filter motion
  prior in the Hungarian cost. Target: fewer than 10 spurious tracks per case with the same
  or better IoU.
- Cold waves and heat domes: use IMD operational criteria (departure from normal plus absolute
  thresholds) and multi-day persistence instead of single-step z-scores. Target IoU ≥ 0.4.
- Synthetic rain: calibrate cyclone rain peaks against the IMD Amphan distribution
  (quantile mapping), with p99 within 20% of IMD. Rerun Phase 4 validation.
- Download the missing 850 hPa q/u/v for the heat and cold windows at a coarser time step
  (every 24 h, or fewer levels) so it stays feasible.
- Show before/after numbers in reports/IMPROVEMENTS.md.

## Phase 2 — GNN spatio-temporal tracker (our headline model)
- Graph: nodes are candidate anomaly objects per member per lead time. Node features are
  intensity, area, centroid, EFI and shape moments. Edges connect objects across time and
  across members.
- Model: a small message-passing GNN (plain PyTorch if torch_geometric won't install) that
  predicts link probability (a tracking edge classifier) and event probability per node.
- Train on synthetic cases with exact labels, holding out whole cases (no leakage).
  Test on the real Amphan ERA5 case against IBTrACS.
- Compare against the Phase 1 baseline and TempestExtremes-style tracking on track error (km),
  IoU, POD, FAR, CSI and time-to-detection (hours of lead gained). Ablate the edge types.
- Output ensemble products: strike-probability maps, a track-spread cone and lead-time
  reliability diagrams.

## Phase 3 — Amplitude-preserving downscaling 12 km → 5 km
- Models: (a) bicubic + DEM lapse-rate baseline, (b) a U-Net, (c) a CorrDiff-style residual
  diffusion (U-Net mean + diffusion on the residual), with the DEM as a conditioning input.
- Losses: MSE + spectral loss + quantile/extreme loss + a conservation constraint
  (avgpool(5 km output) == 12 km input, enforced exactly by a projection layer).
- Metrics: RMSE, power spectra, p99/p99.9 amplitude ratio, CRPS (diffusion), peak-rain error
  and conservation error. The point is to show that ours keeps extremes where a U-Net smooths
  them out.
- Evaluate on synthetic held-out cases, and on real ERA5 → IMD 0.25° as a sanity check.

## Phase 4 — Real ensemble evaluation
- If TIGGE/ECMWF keys are in .env, pull NEPS-G/ECMWF ENS for Amphan. Otherwise use the WB2
  public IFS ENS forecasts, if a public store exists (verify it first).
- Run the full pipeline on a real ensemble: EFI, the GNN tracker, downscaling, and verification
  against IBTrACS/IMD. Report the lead time at which each hazard was first flagged.

## Phase 5 — Demo that wins judges
- A Streamlit (or static HTML) dashboard: pick a case, then use a lead-time slider to see
  anomaly maps, ensemble tracks, strike probability, and a 12 km vs 5 km side-by-side view.
  Add an "alert bulletin" panel with IMD-style colour-coded district warnings.
- A one-command demo: `python scripts/demo.py --case amphan` runs in under 5 minutes on CPU
  using cached model outputs.
- Measure and report inference time per forecast cycle (operational readiness for NCMRWF).

## Phase 6 — Pitch materials
- reports/RESULTS.md: one headline table comparing us against the baseline and the public
  SIH-26078 repos from research/PRIOR_WORK.md (their reported numbers only, cited).
- A 10-slide deck outline in docs/PITCH.md covering: problem, gap, our method, real-data
  results, demo screenshots, operational fit with NEPS-G, limitations, and roadmap.
- Update SUMMARY.md and verify_deliverables.py for the new deliverables. All tests must pass.

Stop and report after each phase with numbers. If a target isn't met, say so and explain why.
Don't tune on the test cases.
