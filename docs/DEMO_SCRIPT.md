# Demo script

## Before the judges arrive (5 minutes)
1. `python scripts/demo.py --case amphan_replay --skip-cycle`
   - This starts the API on :8000 and the dashboard on :5173, then opens the browser.
   - Check that the header shows **● API**.
   - If it shows **○ offline demo**, the API is down. Everything below still works from the
     precomputed files, **except** the district bulletin, the exact point query and `POST /run`.
2. Open four tabs in advance:
   - Operations, `amphan_replay` at +96 h;
   - Alerts, `amphan_replay` at +96 h;
   - Real events;
   - Model performance.
3. Fallback if the laptop or network fails: `docs/figures/ui_*.png` and `docs/figures/demo_*.png`
   show every view. The static dashboard (`python scripts/demo.py --static --skip-cycle`) needs only
   Python.

## 3-minute live demo

| time | screen | what to say (numbers from reports/) |
|---|---|---|
| 0:00-0:20 | Operations, `amphan_replay`, lead 0 | "A NEPS-G-style 20-member, 12 km, 10-day ensemble. The badge says SYNTHETIC: this event has exact ground truth on a real ERA5 background." |
| 0:20-0:50 | press ▶ | "The mesh GNN finds the anomaly in every member. The object GNN links members and leads into tracks: spaghetti, consensus and spread cone, and the yellow 4-D box." Toggle 3-D to show time as height. |
| 0:50-1:10 | strike selector: raw → calibrated | "Raw GNN probabilities were over-confident at days 7-10. These are calibrated per lead band on validation cases only (OBJECT_GNN_V3.md)." |
| 1:10-1:40 | Downscaling, swipe; scenario → sample 1 / p90 | "12 → 5 km with exact conservation: the average of the 5 km field equals the 12 km input. The U-Net smooths fine scales; diffusion samples keep them. The p90 scenario gives the rain value to plan for." |
| 1:40-2:15 | Alerts: click the red pin, then "Explain this alert" | "Every alert has a 5 km pinpoint, an impact radius, a probability and its reason. Here are the members that drove it, and the calibration curve at this lead." Then click **हिन्दी** for the district bulletin. |
| 2:15-2:40 | Real events → open an ERA5 storm | "Real data: 13 North Indian Ocean cyclones, 2019-2023, split by time. The white line is the IBTrACS best track. REAL-TEST storms were scored exactly once (FINAL_RESULTS.md)." |
| 2:40-3:00 | Model performance | "Every number on this page is read from reports/*.json. What did not work is in each report." End with the one-cycle CPU time from SYSTEM_PERF.md. |

## 5-minute video storyboard

1. **0:00-0:30. Problem.** Show a NEPS-G ensemble and 20 × 41 maps side by side. Voice-over: "Forecasters
   need where, how likely and how intense, at 5 km."
2. **0:30-1:15. Data with truth.** Show synthetic cases on real ERA5 backgrounds (`reports/figures`), then the
   exact labels. Say why exact labels matter for IoU and CSI.
3. **1:15-2:15. Method.** Animate the architecture from the README Mermaid diagram: icosahedral mesh →
   object GNN → calibration → crop → diffusion → alerts. Show Operations and Forecaster (raw ensemble / EFI /
   tracker / truth).
4. **2:15-3:15. Results.** Show the FINAL_RESULTS.md headline table with bootstrap confidence intervals, the
   REAL-TEST numbers, and the downscaling spectra (DOWNSCALE_RESULTS.md).
5. **3:15-4:15. Operations.** Show `python -m pipeline.run_operational --watch inbox/` picking up a file
   dropped in the folder. Show products appearing in the dashboard, the bulletin export, and the Docker
   compose file.
6. **4:15-5:00. Honesty slide.** List the limitations (docs/PS_ALIGNMENT.md NOT DONE / PARTIAL rows) and the
   roadmap: NEPS-G via TIGGE, and a GPU.

## If something breaks live
- **API error:** keep going. The dashboard switches to offline mode by itself.
- **Map tiles missing (no internet):** the India outline and every data layer are local; say so.
- **Slow machine:** use the pre-exported cases and do not run `POST /run` live. Quote the measured time from
  `reports/SYSTEM_PERF.md` instead.
