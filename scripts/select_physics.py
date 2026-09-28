"""Choose the physics-loss weights on VAL (BRIEF4 Phase 4) and install models/downscale/unet_phys.

    python scripts/select_physics.py models/downscale/unet_phys_a models/downscale/unet_phys_b ...

Rule (VAL only): among the fine-tuned candidates whose VAL data loss (MSE + spectral + extreme, at
their selected epoch) is within 5 % of the plain U-Net's, take the one with the lowest VAL physics
penalty (rain-without-convergence + divergence + lapse terms, unweighted sum); ties -> lower data
loss. If no candidate is within 5 %, the one with the lowest VAL data loss is taken. The choice is appended to reports/TUNING_LOG.md.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT / "models/downscale/unet/config.json").read_text())["best_val_loss"]
    rows = []
    for d in map(Path, sys.argv[1:]):
        cfg = json.loads((d / "config.json").read_text())
        hist = json.loads((d / "history.json").read_text())
        h = hist[cfg["best_epoch"]]
        phys = h.get("val_physics", {})
        rows.append({"dir": str(d), "weights": cfg["physics_weights"], "val_data_loss": h["val_loss"],
                     "val_physics": phys, "penalty": sum(phys.values())})
    ok = [r for r in rows if r["val_data_loss"] <= 1.05 * base]
    best = (min(ok, key=lambda r: (r["penalty"], r["val_data_loss"])) if ok
            else min(rows, key=lambda r: r["val_data_loss"]))            # none within 5 %: least data damage
    dst = ROOT / "models/downscale/unet_phys"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(best["dir"], dst)
    with (ROOT / "reports/TUNING_LOG.md").open("a", encoding="utf-8") as f:
        f.write("\n## BRIEF4 Phase 4: physics-loss weights (VAL)\n")
        f.write(f"- Plain U-Net VAL data loss {base:.4f}; candidates (fine-tuned 6 epochs from the U-Net):\n")
        for r in rows:
            f.write(f"  - {r['weights']}: VAL data loss {r['val_data_loss']:.4f}, VAL physics {r['val_physics']}\n")
        f.write(f"- Chosen: {best['weights']} (lowest VAL physics penalty with data loss <= 1.05 x plain).\n")
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
