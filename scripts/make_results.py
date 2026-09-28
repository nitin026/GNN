"""BRIEF2 Phase 6: reports/RESULTS.md, the headline results table, built from reports/*.json.

    python scripts/make_results.py

Every number of OUR system is read from a pipeline output (reports/*.json); nothing is typed in.
The public SIH-26078 repositories' numbers are THEIR self-reported README numbers, copied from
research/PRIOR_WORK.md (section d), where each repo link was fetched and checked.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
TEST = ["cyc_04", "heat_04", "cold_04", "amphan_replay"]

# (repo, link, data, their reported tracking / downscaling numbers) -- research/PRIOR_WORK.md (d)
PUBLIC = [
    ("AERO-TRACK 4D (krishnendukoley2007-arch)", "https://github.com/krishnendukoley2007-arch/aero-track-4d",
     "REAL ERA5 + IBTrACS, Amphan only (13 hourly steps); Fani/Yaas on 'ERA5-equivalent synthetic' data",
     "mean track error 50.9 km (Amphan), landfall error 8.7 km", "peak wind 83.9 % of ERA5; CRPS 7.26 km/h; FSS 0.072",
     "one storm, 13 steps; the '5 km target' is 25 km ERA5, so no true 12->5 km pair; no conservation check"),
    ("Avarta / avarta_test (yadavayush834, diiviikk5)", "https://github.com/yadavayush834/avarta_test",
     "REAL NOAA GEFS 0.5 deg (control + 4 members, one init), IMD 0.25 deg, CHIRPS",
     "heavy-rain IoU 0.00 (missed); ensemble-mean peak 44.6 vs IMD 469.2 mm/day",
     "CNN vs bilinear peak error 61.5 vs 84.9 mm/day; IoU 0.37 vs 0.26",
     "5 members, one case; GNN and DDPM untrained"),
    ("ChakraNet (ErrGuhan)", "https://github.com/ErrGuhan/Chakra-Net",
     "SYNTHETIC 50-member ensemble built from the Phailin best track + IMERG",
     "no IoU or km-error tracking benchmark reported",
     "high-frequency power 98.7 % vs 50.5 % bilinear; p99 bias 5.72 % vs 4.87 % bilinear; CRPS 3.52 vs 4.21 mm",
     "one storm; conservation losses are stubs"),
    ("SIH26078-weather-anomaly-ai (sauravrajput2124-cmyk)",
     "https://github.com/sauravrajput2124-cmyk/SIH26078-weather-anomaly-ai",
     "SYNTHETIC 10-day demo JSON", "none reported", "none reported", "no ML model, no real data"),
    ("MEGHA-DRISHTI (Srujanmirji)", "https://github.com/Srujanmirji/MEGHA-DRISHTI",
     "'DEMO DATA - illustrative'", "none reported", "none reported", "front end only"),
]


def f(x, n=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def mean(cases, method, key):
    v = [cases[c][method][key] for c in TEST if method in cases[c]]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def main():
    g = json.loads((REP / "gnn_results.json").read_text())
    p1 = json.loads((REP / "phase1_results.json").read_text())
    ds = json.loads((REP / "downscaling_results.json").read_text())
    tm = json.loads((REP / "demo_timing.json").read_text()) if (REP / "demo_timing.json").exists() else {}
    C = g["cases"]
    L = ["# Results: headline table (BRIEF2 Phase 6)", "",
         "Built by `scripts/make_results.py` from `reports/*.json`. **Synthetic** rows are SYNTHETIC TEST cases "
         "(cyc_04, heat_04, cold_04, amphan_replay; exact labels, never used for training or tuning). "
         "**Real** rows are ERA5 reanalysis against the IBTrACS best track or IMD gridded rain.", ""]

    L += ["## 1. Tracking (SYNTHETIC TEST, mean of 4 cases)", "",
          "| method | IoU | CSI | POD | FAR | centroid err (km) | spurious tracks / case |",
          "|---|---|---|---|---|---|---|"]
    old = [p1["cases"][c]["old"] for c in TEST]
    L.append(f"| baseline (BRIEF Phase 5: z-score + label + Hungarian) | {f(np.mean([o['iou_mean'] for o in old]))} "
             f"| – | – | – | {f(np.mean([o['centroid_err_km'] for o in old]), 0)} | "
             f"{np.mean([o['ens_spurious'] for o in old]):.0f} |")
    names = {"v2": "tracker v2 (hysteresis + Kalman + IMD criteria)", "te_style": "TempestExtremes-style",
             "gnn_full": "**GNN tracker (full)**", "gnn_temporal": "GNN (temporal edges only)",
             "gnn_none": "node-only MLP (ablation)"}
    for m, name in names.items():
        L.append(f"| {name} | {f(mean(C, m, 'iou'))} | {f(mean(C, m, 'csi'))} | {f(mean(C, m, 'pod'))} | "
                 f"{f(mean(C, m, 'far'))} | {f(mean(C, m, 'centroid_err_km'), 0)} | {f(mean(C, m, 'spurious_tracks'), 1)} |")
    L += ["", "Baseline POD/FAR/CSI were not computed in BRIEF Phase 5 (it scored IoU and centroid error only).", ""]

    L += ["## 2. Real cyclone track: Amphan 2020, ERA5 G12 vs IBTrACS (REAL)", "",
          "| method | matched 6-h fixes | mean err (km) | median (km) | max (km) |", "|---|---|---|---|---|"]
    for m, name in (("v2", "tracker v2"), ("te_style", "TempestExtremes-style"), ("gnn_full", "GNN full"),
                    ("gnn_temporal", "**GNN temporal** (like-for-like: ERA5 is one run)")):
        r = g["amphan_real"][m]
        L.append(f"| {name} | {r['n_matched']} | {r['err_mean_km']:.0f} | {r['err_median_km']:.0f} | {r['err_max_km']:.0f} |")
    L.append("")

    L += ["## 3. Downscaling 12 -> 5 km (SYNTHETIC TEST, rain)", "",
          "| model | RMSE (mm/6h) | CRPS | p99 ratio | 10 km power ratio | patch-peak ratio | conservation err |",
          "|---|---|---|---|---|---|---|"]
    for m in ("bicubic+lapse", "unet", "diffusion_mean", "diffusion_sample"):
        r = ds["metrics"][m]["tp"]
        L.append(f"| {m} | {f(r['rmse'], 3)} | {f(r.get('crps', r['mae']), 3)} | {f(r['p99_ratio'], 3)} | "
                 f"{f(r['psd_ratio_10km'], 3)} | {f(r.get('peak_ratio'), 3)} | {r['conservation_err']:.1e} |")
    im = ds["imd_perfect_model"]
    L += ["", "REAL IMD 0.25 deg perfect-model check (coarsened 3x, downscaled back, JJAS 2020): RMSE "
          + ", ".join(f"{m} {im[m]['rmse']:.2f}" for m in im if isinstance(im[m], dict) and "rmse" in im[m])
          + " mm/day (out of distribution for all learned models).", ""]

    L += ["## 4. Inference time per forecast cycle (CPU)", ""]
    if tm:
        L += ["| case | members x leads | total (s) | slowest stage | peak RSS (MB) |", "|---|---|---|---|---|"]
        for k, r in tm.items():
            s = max(r["stages"], key=lambda x: x["seconds"])
            L.append(f"| {k} | {r['members']} x {r['leads']} | {r['total_seconds']:.0f} | {s['stage']} "
                     f"({s['seconds']:.0f} s) | {r['peak_rss_mb']} |")
        L += ["", "Measured by `python scripts/demo.py --case <case>` (`reports/demo_timing.json`).", ""]
    else:
        L += ["Not measured yet: run `python scripts/demo.py --case amphan`.", ""]

    L += ["## 5. Public SIH-26078 repositories (their self-reported numbers)", "",
          "Copied from `research/PRIOR_WORK.md` (section d; every link fetched). They are **not** directly "
          "comparable: different cases, data and metric definitions. The 'data' column says whether their "
          "numbers come from real or synthetic data.", "",
          "| repo | data | tracking (reported) | downscaling (reported) | caveat |", "|---|---|---|---|---|"]
    for name, link, data, trk, dsn, cav in PUBLIC:
        L.append(f"| [{name}]({link}) | {data} | {trk} | {dsn} | {cav} |")
    L += ["", "Like for like, only AERO-TRACK 4D reports a real Amphan track error (50.9 km over 13 hourly "
          "steps). Ours is " + f"{g['amphan_real']['gnn_temporal']['err_mean_km']:.0f} km over "
          f"{g['amphan_real']['gnn_temporal']['n_matched']} six-hourly fixes of the ERA5 run; "
          "different time sampling and a different matching rule, so the two numbers are indicative only.", ""]

    L += ["## What did not work", ""]
    gi, vi = mean(C, "gnn_full", "iou"), mean(C, "v2", "iou")
    bi = float(np.mean([o["iou_mean"] for o in old]))
    be = float(np.mean([o["centroid_err_km"] for o in old]))
    L.append(f"- Area overlap: GNN TEST IoU {gi:.2f} (v2 {vi:.2f}) is BELOW the original Phase 5 baseline's truth-run "
             f"IoU {bi:.2f} (centroid error {be:.0f} km vs {mean(C, 'gnn_full', 'centroid_err_km'):.0f} km). The baseline "
             "gets there with ~470 spurious ensemble tracks per case, but on overlap alone it is better. The GNN links "
             "candidate objects and does not re-segment them (a mesh-GNN segmentation head is BRIEF4 Phase 2).")
    L.append("- Edges add little: the node-only MLP matches the full GNN on TEST CSI "
             f"({mean(C, 'gnn_none', 'csi'):.2f} vs {mean(C, 'gnn_full', 'csi'):.2f}).")
    r = ds["metrics"]
    L.append(f"- Downscaling p99 contrast: diffusion rain p99 ratio {r['diffusion_sample']['tp']['p99_ratio']:.3f} vs U-Net "
             f"{r['unet']['tp']['p99_ratio']:.3f}; the advantage is only in 10 km power and patch peaks.")
    L.append("- GNN probabilities are over-confident against the truth at days 7-10 (reports/GNN_RESULTS.md).")
    L.append("- No real ensemble is scored in this table: tracking skill on real data is one ERA5 run of one "
             "storm. The multi-storm real evaluation and the real IFS ENS / GenCast ensembles are BRIEF4 Phase 1.")
    (REP / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/RESULTS.md")


if __name__ == "__main__":
    main()
