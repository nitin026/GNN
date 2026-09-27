"""BRIEF2 Phase 1 report: before/after numbers -> reports/IMPROVEMENTS.md (+ phase1_results.json).

Old = Phase 5 tracker (pipeline/track.py + evaluate.run_detection), v2 = pipeline/tracker2.py with
parameters chosen on TRAIN+VAL (reports/tracker2_params.json). Both run on the same synth-0.2
data with the same metric definitions (pipeline/evaluate2.py). TEST cases were not used for
any choice.

    python -m pipeline.phase1_report
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .anomaly import Climatology
from .evaluate2 import evaluate, load_case, orog12
from .splits import ALL, TEST, TRAIN, VAL
from .track import haversine_km
from .tracker2 import Tracker2

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def amphan_real(params, clim, oro):
    ds = xr.open_dataset(ROOT / "data/real/era5/amphan_era5_g12.nc")
    ib = pd.read_csv(ROOT / "data/real/ibtracs/amphan_2020_ibtracs.csv")
    ib["time"] = pd.to_datetime(ib.ISO_TIME)
    ib6 = ib[ib.time.dt.hour % 6 == 0].set_index("time")
    times = pd.to_datetime(ds.time.values)
    msl = ds.msl.values.astype(float)
    tk = Tracker2("tropical_cyclone", clim, oro, ds.latitude.values, ds.longitude.values, **params)
    tracks, _ = tk.run(msl, times)
    t0, t1 = ib.time.min(), ib.time.max()
    best = max(tracks, key=lambda tr: (sum(t0 <= times[t] <= t1 for t, _ in tr), len(tr))) \
        if tracks else None
    errs, first = [], None
    for t, o in best or []:
        if times[t] in ib6.index:
            r = ib6.loc[times[t]]
            errs.append(float(haversine_km(o["center_lat"], o["center_lon"], r.LAT, r.LON)))
            first = first or str(times[t])
    others = len(tracks) - (1 if best else 0)
    ds.close()
    return {"n_matched": len(errs), "n_ibtracs": int(((ib6.index >= times[0]) &
                                                      (ib6.index <= times[-1])).sum()),
            "err_mean_km": float(np.mean(errs)) if errs else np.nan,
            "err_median_km": float(np.median(errs)) if errs else np.nan,
            "err_max_km": float(np.max(errs)) if errs else np.nan,
            "first_matched_time": first, "other_tracks": others}


def f(x, n=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def main():
    clim, oro = Climatology(), orog12()
    chosen = json.loads((REP / "tracker2_params.json").read_text())["chosen"]
    res = {}
    for c in ALL:
        d = load_case(c)
        p = chosen[d["hazard"]]["params"]
        res[c] = {"hazard": d["hazard"], **evaluate(d, clim, oro, params=p)}
        o, v = res[c]["old"], res[c]["v2"]
        print(f"{c}: IoU {o['iou_mean']:.2f}->{v['iou_mean']:.2f} ens_spurious {o['ens_spurious']}"
              f"->{v['ens_spurious']}", flush=True)
    am = amphan_real(chosen["tropical_cyclone"]["params"], clim, oro)
    old_am = json.loads((REP / "baseline_results.json").read_text())["amphan_real"]
    before = json.loads((REP / "phase1_before.json").read_text())
    val = json.loads((REP / "synth_validation.json").read_text())
    (REP / "phase1_results.json").write_text(json.dumps(
        {"cases": res, "amphan_real_v2": am, "params": chosen}, indent=1, default=float))
    write_md(res, am, old_am, before, val, chosen)


def split_of(c):
    return "train" if c in TRAIN else "val" if c in VAL else "test"


def write_md(res, am, old_am, before, val, chosen):
    L = ["# Phase 1 improvements (BRIEF2): before / after", "",
         "All synthetic cases are **SYNTHETIC** (exact labels); Amphan ERA5 vs IBTrACS is real. "
         "\"Old\" is the Phase 5 tracker; \"v2\" is `pipeline/tracker2.py`. Both run on the same "
         "regenerated data (generator `synth-0.2`) with the same metric definitions "
         "(`pipeline/evaluate2.py`). v2 parameters were chosen on TRAIN+VAL cases only "
         "(`reports/tracker2_params.json`). TEST cases (cyc_04, heat_04, cold_04, amphan_replay) "
         "were not used for any choice.", "",
         "## 1. Tracker false alarms and IoU", "",
         "Spurious ensemble tracks = tracks, summed over the 20 members, that never overlap that "
         "member's own injected-event mask (the generator's deliberate false-alarm events count as "
         "real). Target: < 10 per case, with the same or better IoU.", "",
         "| case | split | hazard | IoU old | IoU v2 | centroid err old (km) | v2 (km) | spurious ens tracks old | v2 | truth-run spurious old | v2 | ens hit @peak old | v2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in res.items():
        o, v = r["old"], r["v2"]
        L.append(f"| {c} | {split_of(c)} | {r['hazard']} | {f(o['iou_mean'])} | {f(v['iou_mean'])} | "
                 f"{f(o['centroid_err_km'], 0)} | {f(v['centroid_err_km'], 0)} | {o['ens_spurious']} | "
                 f"{v['ens_spurious']} | {o['truth_spurious']} | {v['truth_spurious']} | "
                 f"{f(o['ens_hit_rate_at_peak'])} | {f(v['ens_hit_rate_at_peak'])} |")
    L += ["", "### Summary by hazard (mean over cases)", "",
          "| hazard | split | n | IoU old | IoU v2 | spurious/case old | v2 | cases meeting < 10 spurious (v2) |",
          "|---|---|---|---|---|---|---|---|"]
    for hz in ("tropical_cyclone", "heat_dome", "cold_wave"):
        for sp, ids in (("train+val", TRAIN + VAL), ("test", TEST)):
            rs = [res[c] for c in ids if res[c]["hazard"] == hz]
            if not rs:
                continue
            L.append(f"| {hz} | {sp} | {len(rs)} | {f(np.mean([r['old']['iou_mean'] for r in rs]))} | "
                     f"{f(np.mean([r['v2']['iou_mean'] for r in rs]))} | "
                     f"{f(np.mean([r['old']['ens_spurious'] for r in rs]), 0)} | "
                     f"{f(np.mean([r['v2']['ens_spurious'] for r in rs]), 0)} | "
                     f"{sum(r['v2']['ens_spurious'] < 10 for r in rs)}/{len(rs)} |")
    L += ["", "Chosen v2 parameters (TRAIN+VAL):", ""]
    for hz, ch in chosen.items():
        L.append(f"- {hz}: `{json.dumps(ch['params'])}` (train+val IoU {f(ch['trainval']['iou_trainval'])}, "
                 f"spurious/case {f(ch['trainval']['spurious_per_case_trainval'], 1)})")
    L += ["", "Phase 5 numbers on the *old* data (synth-0.1, `reports/BASELINE_RESULTS.md`) are kept for "
          "reference: the ensemble false-alarm column there used the truth mask and a slightly different "
          "background. The table above re-runs the old tracker on the new data so the comparison is fair.", "",
          "## 2. Real Amphan (ERA5 G12 vs IBTrACS)", "",
          "| tracker | matched 6-h fixes | mean err (km) | median (km) | max (km) | other tracks in window |",
          "|---|---|---|---|---|---|",
          f"| old (Phase 5) | {old_am['n_matched']}/{old_am['n_ibtracs_6h_fixes']} | {f(old_am['track_err_km_mean'], 0)} | "
          f"{f(old_am['track_err_km_median'], 0)} | {f(old_am['track_err_km_max'], 0)} | n/a |",
          f"| v2 | {am['n_matched']}/{am['n_ibtracs']} | {f(am['err_mean_km'], 0)} | {f(am['err_median_km'], 0)} | "
          f"{f(am['err_max_km'], 0)} | {am['other_tracks']} |", "",
          "## 3. Heat and cold waves: IMD criteria + persistence", "",
          "v2 replaces the single-step T2m z-score with the IMD operational criteria on daily Tmax/Tmin "
          "(trailing 24 h of 6-hourly values):",
          "- heat wave: departure >= 4.5 C with Tmax >= 40 C (plains), >= 37 C (coastal), >= 30 C (hills), "
          "or Tmax >= 45 C;",
          "- cold wave: departure <= -4.5 C with Tmin <= 10 C (plains), <= 15 C (coastal), <= 0 C (hills), "
          "or Tmin <= 4 C in the plains;",
          "- persistence of at least 2 days (8 six-hourly steps), as in IMD's two-day rule.",
          "",
          "Cells above 2500 m (Tibetan plateau) are excluded as outside the IMD domain. Target: IoU >= 0.4 (see table 1).", "",
          "## 4. Synthetic cyclone rain vs IMD (quantile mapping)", "",
          "Swath = daily land rain within 500 km of the storm centre at ~0.25 deg, wet cells. The map was "
          "fitted on the TRAIN cases cyc_01 and cyc_02 only. Target: p99 within 20 % of IMD Amphan.", "",
          "| case | split | p99 before | p99 after | after vs IMD | max before | max after |",
          "|---|---|---|---|---|---|---|"]
    imd = before["rain_swath"]["IMD"]
    sw = val["swath"]
    for c in ("cyc_01", "cyc_02", "cyc_03", "cyc_04", "amphan_replay"):
        b, a = before["rain_swath"][c], sw[c]
        L.append(f"| {c} | {split_of(c)} | {f(b['p99'], 0)} | {f(a['p99'], 0)} | "
                 f"{(a['p99'] / imd['p99'] - 1) * 100:+.0f} % | {f(b['max'], 0)} | {f(a['max'], 0)} |")
    L.append(f"| IMD Amphan (real) | - | {f(imd['p99'], 0)} | {f(imd['p99'], 0)} | +0 % | "
             f"{f(imd['max'], 0)} | {f(imd['max'], 0)} |")
    L += ["", "Two maps were tried:",
          "- **Option A**: global empirical QM (kept).",
          "- **Option B**: intensity-aware QM, with rain normalised by the generator's own g(V) = 3 + 0.3 Vmax "
          "before mapping (`reports/rain_qm_optionB_intensity.json`).",
          "",
          "B was worse on the TRAIN/VAL storms (raw p99 vs IMD cyc_01 +49 %, cyc_02 -43 %, "
          "cyc_03 -16 %, against A's +7 %, -49 %, -21 %), so A was kept. The choice used no test case. The full "
          "Phase 4 validation was rerun (`reports/SYNTH_VALIDATION.md`).", "",
          "## 5. 850 hPa q/u/v for the heat and cold windows", ""]
    src = (ROOT / "data" / "SOURCES.md").read_text(encoding="utf-8")
    for ln in src.splitlines():
        if "850 hPa q/u/v (daily" in ln:
            L.append(f"- {ln.split('|')[1].strip()}: **{ln.split('|')[3].strip()}**, "
                     f"{ln.split('|')[5].strip()} ({ln.split('|')[6].strip()[:90]})")
    L += ["", "## 6. Why targets were missed", "",
          "- **Heat and cold IoU went down in 7/13 cases.** The IMD criteria are daily and operational. "
          "The exact label is the instantaneous injected anomaly (>= 3 K), which shrinks at night by "
          "design (diurnal factor). The old z-score tracker follows that label more closely; v2 follows "
          "IMD's definition. Merging masks over each day (daily IoU in `reports/phase1_results.json`) "
          "narrows the gap but does not close it: on train+val, heat is old 0.51 vs v2 0.43 and cold is "
          "old 0.38 vs v2 0.32. The old tracker gets that IoU while producing 400-580 spurious tracks per "
          "case, against 5-14 for v2 (train+val).",
          "- **heat_04 (TEST): IoU 0.** At 18 N in June, no mask cell reaches Tmax >= 40 C, so the "
          "synthetic +6 K dome is not an IMD heat wave; Phase 4 already showed 0 % of its cells meeting "
          "IMD. The label and the operational definition disagree for this case. It was diagnosed after "
          "evaluation and not tuned on.",
          "- **cold_04 (TEST): IoU 0.** v2 does detect the event (a track at ~26 N 85 E). The main-track "
          "rule (largest summed area) instead picks a long-lived object near Herat (36 N 62 E), which "
          "the rough IMD-domain mask (lat <= 37 N, south of the Himalayan crest, no country boundary) "
          "does not exclude. A proper India boundary would fix this; it was not added, because it was "
          "found on a test case.",
          "- **Spurious tracks.** The misses are heat_02 (11), cold_02 (14), cold_03 (12), heat_04 (27) "
          "and cold_04 (27). Most of the rest come from outside India (Afghanistan/Iran) for the same "
          "domain reason.",
          "- **Rain.** One global map cannot fit storms of different intensity to one storm's (Amphan's) "
          "distribution. cyc_02, a weak Arabian Sea storm, ends up 49 % too dry, and amphan_replay went "
          "from +2 % to -32 %. Pooled p99 is within 20 % (train -6 %, held-out -18 %). Option B "
          "(intensity-aware) was worse on train/val.",
          "- **Amphan real.** Mean error rose 40 -> 45 km because v2 keeps 23 of 26 fixes against 21 "
          "(the added ones are the harder landfall/decay fixes); the median is unchanged at 28 km, and v2 "
          "produced 0 other tracks in the window.",
          "", "## Targets", ""]
    L += targets(res, sw, imd)
    (REP / "IMPROVEMENTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/IMPROVEMENTS.md")


def targets(res, sw, imd):
    out = []
    spur_ok = [c for c in res if res[c]["v2"]["ens_spurious"] < 10]
    iou_ok = [c for c in res if res[c]["v2"]["iou_mean"] >= res[c]["old"]["iou_mean"] - 1e-9]
    out.append(f"- **Spurious tracks < 10 per case:** met in {len(spur_ok)}/13 cases "
               f"(v2 total {sum(r['v2']['ens_spurious'] for r in res.values())} vs old "
               f"{sum(r['old']['ens_spurious'] for r in res.values())}).")
    out.append(f"- **Same or better IoU:** met in {len(iou_ok)}/13 cases; misses: "
               f"{', '.join(c for c in res if c not in iou_ok) or 'none'}.")
    hc = [c for c in res if res[c]["hazard"] in ("heat_dome", "cold_wave")]
    ok = [c for c in hc if res[c]["v2"]["iou_mean"] >= 0.4]
    out.append(f"- **Heat/cold IoU >= 0.4:** met in {len(ok)}/{len(hc)} cases ({', '.join(ok) or 'none'}).")
    rain_ok = [c for c in ("cyc_01", "cyc_02", "cyc_03", "cyc_04", "amphan_replay")
               if abs(sw[c]["p99"] / imd["p99"] - 1) <= 0.2]
    out.append(f"- **Rain p99 within 20 % of IMD:** met in {len(rain_ok)}/5 cyclone cases "
               f"({', '.join(rain_ok)}). Pooled over storms, see the Phase 1 summary.")
    return out


def from_json():
    """Rewrite IMPROVEMENTS.md from reports/phase1_results.json without re-running."""
    r = json.loads((REP / "phase1_results.json").read_text())
    old_am = json.loads((REP / "baseline_results.json").read_text())["amphan_real"]
    before = json.loads((REP / "phase1_before.json").read_text())
    val = json.loads((REP / "synth_validation.json").read_text())
    write_md(r["cases"], r["amphan_real_v2"], old_am, before, val, r["params"])


if __name__ == "__main__":
    import sys
    from_json() if "--from-json" in sys.argv else main()
