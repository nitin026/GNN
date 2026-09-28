"""docs/MODEL_CARDS.md from the model configs and results files (BRIEF4 Phase 7).

    python scripts/make_model_cards.py

Every number is read from models/*/config.json or reports/*.json; missing files are reported as such.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def jl(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=2):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


NOT_OPERATIONAL = ("**Not for operational use** without validation on real NCMRWF NEPS-G forecasts over several "
                   "seasons; all training data are SYNTHETIC events on real ERA5 backgrounds.")


def mesh_card(L):
    cfg, res = jl("models/mesh/mesh_l6/config.json"), jl("reports/mesh_results.json")
    L += ["## 1. Icosahedral mesh GNN (per-cell hazard segmentation)", "",
          "- **Code / weights:** `pipeline/mesh_gnn.py`, `pipeline/icomesh.py`, `models/mesh/mesh_l6/`.",
          "- **Architecture:** encoder -> grid->mesh -> 4 interaction-network layers on the icosahedral multi-mesh "
          "(levels 0-6 over the India box + 12 deg halo) -> mesh->grid -> 3 logits (cyclone, heat wave, cold wave) per 12 km cell."]
    if cfg:
        L += [f"- **Size / training:** {cfg['params']} parameters; {cfg['epochs']} epochs on {cfg['compute']}, "
              f"{cfg['train_seconds']:.0f} s; best epoch {cfg['best_epoch']} (VAL IoU {f(cfg['best_val_iou'], 3)}).",
              f"- **Inputs:** {', '.join(cfg['features'])}."]
    L += ["- **Training data:** SYNTHETIC TRAIN cases (cyc_01/02, heat_01/02, cold_01/02), member event masks; VAL cases "
          "for model selection and hysteresis thresholds.",
          "- **Intended use:** first stage of the two-level tracker: where is an extreme event at each lead and member."]
    if res and "mesh_l6" in res:
        r = res["mesh_l6"]["cases"]
        test = [c for c, v in r.items() if v["split"] == "test"]
        L.append("- **Metrics (SYNTHETIC TEST, mesh + object GNN pipeline):** " + "; ".join(
            f"{c} IoU {f(r[c]['pipeline']['iou'])} CSI {f(r[c]['pipeline']['csi'])}" for c in test) + ".")
    L += ["- **Known failure modes:** heat/cold waves without RH850 input (proxy channel is 0); the moisture-flux "
          "input is a 10 m-wind proxy; trained on 6 cases only; see reports/MESH_GNN_RESULTS.md.",
          f"- {NOT_OPERATIONAL}", ""]


def object_card(L):
    v3 = jl("models/tracker_v3/full/config.json")
    old = jl("models/tracker/full/config.json")
    res = jl("reports/object_gnn_v3.json")
    L += ["## 2. Object GNN tracker (linking objects across leads and members)", "",
          "- **Code / weights:** `pipeline/gnn.py`, `pipeline/graphs.py`, `models/tracker_v3/` (BRIEF4 Phase 3; "
          "`models/tracker/` is the BRIEF2 version).",
          "- **Architecture:** message passing over temporal (t->t+1, t+2) and cross-member edges, per-lead consensus "
          "readout; heads: node event probability, edge link probability; Hungarian decoding into tracks."]
    for name, cfg in (("v3", v3), ("BRIEF2", old)):
        if cfg:
            L.append(f"- **{name} training:** {cfg['n_node_features']} node / {cfg['n_edge_features']} edge features, "
                     f"hidden {cfg['hidden']}, best epoch {cfg['best_epoch']}, VAL AP (node+edge) {f(cfg['best_val_ap_sum'], 3)}, "
                     f"{cfg['train_seconds']:.0f} s on {cfg['compute']}.")
    L += ["- **Training data:** SYNTHETIC TRAIN graphs; heat/cold targets are IMD-consistent daily labels in v3."]
    if res:
        test = [c for c, v in res["cases"].items() if v["split"] == "test"]
        L.append("- **Metrics (SYNTHETIC TEST):** " + "; ".join(
            f"{c} CSI {f(res['cases'][c]['gnn_full']['csi'])} FAR {f(res['cases'][c]['gnn_full']['far'])}" for c in test) + ".")
        cal = res.get("calibration_test", {})
        if cal:
            L.append("- **Calibration (lead bands, VAL-fitted):** " + "; ".join(
                f"{b}: slope {f(v['slope_raw'])} -> {f(v['slope_cal'])}, BSS vs climatology {f(v['bss_cal_vs_clim'])}"
                for b, v in cal.items()) + ".")
    real = jl("reports/real_results.json")
    if real:
        e = real["era5_real_val"]
        L.append("- **REAL (ERA5 vs IBTrACS, REAL-VAL cyclones):** " + "; ".join(
            f"{k} {f(v.get('gnn_temporal', {}).get('err_mean_km'), 0)} km" for k, v in e.items() if "gnn_temporal" in v) + ".")
    L += ["- **Known failure modes:** probabilities are node-level and over-confident without calibration; single "
          "reanalysis runs have no cross-member edges; tracks can jump between neighbouring lows (tauktae_2021 in "
          "reports/REAL_RESULTS.md).", f"- {NOT_OPERATIONAL}", ""]


def ds_card(L, name, title, desc):
    cfg = jl(f"models/downscale/{name}/config.json")
    res = jl("reports/downscaling_results.json")
    L += [f"## {title}", "", f"- **Code / weights:** `pipeline/downscale.py`, `models/downscale/{name}/`.", f"- **Model:** {desc}"]
    if cfg:
        L.append(f"- **Training:** {cfg['epochs']} epochs, batch {cfg['batch']}, base width {cfg['base']}, "
                 f"{cfg['n_train_patches']} SYNTHETIC TRAIN patch pairs (12 km -> 5 km), {cfg['train_seconds']:.0f} s on "
                 f"{cfg['compute']}; loss: {cfg['loss']}.")
    key = {"unet": "unet", "diffusion": "diffusion_sample", "unet_phys": None}[name]
    if res and key:
        m = res["metrics"][key]["tp"]
        L.append(f"- **Metrics (SYNTHETIC TEST rain):** RMSE {f(m['rmse'], 3)} mm/6h, p99 ratio {f(m['p99_ratio'], 3)}, "
                 f"10 km power ratio {f(m['psd_ratio_10km'], 3)}, conservation error {m['conservation_err']:.1e}.")
        im = res["imd_perfect_model"][key]
        L.append(f"- **REAL IMD perfect-model check:** RMSE {f(im['rmse'])} mm/day (out of distribution).")
    phys = jl("reports/physics_results.json")
    if phys and name in ("unet", "unet_phys"):
        v = phys["violations_test"][name]
        L.append(f"- **Physics violations (TEST):** rain without convergence {f(v['rain_noconv'])} %, lapse-rate wrong sign "
                 f"{f(v['lapse'])} %, divergence above truth p99.9 {f(v['div'])} %.")
    L += ["- **Intended use:** 12 km -> 5 km fields inside the tracker's 4-D box; conservation avgpool(5 km) == 12 km is exact.",
          "- **Known failure modes:** sub-grid peaks well above the 12 km value are not recovered; trained on synthetic "
          "fine scales (spectral noise, not dynamics); diffusion spread is over-dispersed for t2m/msl (reports/DOWNSCALE_RESULTS.md).",
          f"- {NOT_OPERATIONAL}", ""]


def main():
    L = ["# Model cards", "", "Generated by `scripts/make_model_cards.py` from the model configs and `reports/*.json`.", ""]
    mesh_card(L)
    object_card(L)
    ds_card(L, "unet", "3. U-Net downscaler", "3-level U-Net predicting the residual over bicubic + DEM lapse rate, "
            "DEM-conditioned, exact conservation projection.")
    if (ROOT / "models/downscale/unet_phys/config.json").exists():
        ds_card(L, "unet_phys", "3b. Physics-informed U-Net", "the U-Net fine-tuned with the physics terms of "
                "pipeline/physics.py (rain only with convergence, divergence smoothness, lapse-rate consistency).")
    ds_card(L, "diffusion", "4. CorrDiff-style residual diffusion", "DDPM (T = 500) on the residual over the U-Net mean, "
            "DDIM sampling (25 steps), exact conservation projection per sample.")
    (ROOT / "docs/MODEL_CARDS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote docs/MODEL_CARDS.md")


if __name__ == "__main__":
    main()
