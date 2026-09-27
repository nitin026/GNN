"""Train the GNN tracker on TRAIN case graphs, select on VAL (TEST never used here).

    python scripts/train_tracker.py [--graphs data/graphs|bundle] [--out models/tracker]
                                    [--variant full|temporal|cross|none] [--epochs 150]
Runs unchanged on CPU or a Colab/Kaggle GPU (device from pipeline/compute.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.compute import describe, device, seed_all  # noqa: E402
from pipeline.splits import TRAIN, VAL  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from pipeline.gnn import GNNTracker  # noqa: E402

VARIANTS = {"full": (True, True), "temporal": (True, False), "cross": (False, True),
            "none": (False, False)}


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def to_tensors(g, stats, dev):
    x = torch.tensor((g["X"] - stats["xm"]) / stats["xs"], dtype=torch.float32, device=dev)
    ef = torch.tensor((g["EF"] - stats["em"]) / stats["es"], dtype=torch.float32, device=dev)
    return {"x": x, "e": torch.tensor(g["E"], dtype=torch.long, device=dev), "ef": ef,
            "et": torch.tensor(g["EF"][:, -1] > 0.5, device=dev),
            "yn": torch.tensor(g["y_node"], device=dev), "ye": torch.tensor(g["y_edge"], device=dev),
            "ens": torch.tensor(g["member"] >= 0, device=dev)}


def losses(model, b, pw_n, pw_e):
    nl, el = model(b["x"], b["e"], b["ef"], b["et"])
    m = b["ens"]                                  # train on ensemble members only (not truth run)
    em = m[b["e"][:, 0]] & m[b["e"][:, 1]]
    ln = F.binary_cross_entropy_with_logits(nl[m], b["yn"][m], pos_weight=pw_n)
    le = F.binary_cross_entropy_with_logits(el[em], b["ye"][em], pos_weight=pw_e)
    return ln + le, nl, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default=str(ROOT / "data" / "graphs"))
    ap.add_argument("--out", default=str(ROOT / "models" / "tracker"))
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)
    dev = device()
    print("compute:", describe(), flush=True)
    gdir = Path(a.graphs)
    tr = [load(gdir / f"{c}.npz") for c in TRAIN]
    va = [load(gdir / f"{c}.npz") for c in VAL]
    Xa = np.concatenate([g["X"][g["member"] >= 0] for g in tr])
    Ea = np.concatenate([g["EF"] for g in tr])
    stats = {"xm": Xa.mean(0), "xs": Xa.std(0) + 1e-6, "em": Ea.mean(0), "es": Ea.std(0) + 1e-6}
    stats["xs"][-3:] = 1.0                        # keep one-hot hazard as is
    stats["xm"][-3:] = 0.0
    stats["es"][-1] = 1.0
    stats["em"][-1] = 0.0
    ytr = np.concatenate([g["y_node"][g["member"] >= 0] for g in tr])
    yetr = np.concatenate([g["y_edge"] for g in tr])
    pw_n = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-6), device=dev)
    pw_e = torch.tensor((1 - yetr.mean()) / max(yetr.mean(), 1e-6), device=dev)
    B_tr = [to_tensors(g, stats, dev) for g in tr]
    B_va = [to_tensors(g, stats, dev) for g in va]
    ut, uc = VARIANTS[a.variant]
    model = GNNTracker(Xa.shape[1], Ea.shape[1], hidden=a.hidden, use_temporal=ut,
                       use_cross=uc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    out = Path(a.out) / a.variant
    out.mkdir(parents=True, exist_ok=True)
    best, hist, t0 = None, [], time.time()
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        model.train()
        tl = 0.0
        for k in rng.permutation(len(B_tr)):
            opt.zero_grad()
            loss, _, _ = losses(model, B_tr[k], pw_n, pw_e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss)
        model.eval()
        with torch.no_grad():
            vl, yn, pn, ye, pe = 0.0, [], [], [], []
            for b in B_va:
                loss, nl, el = losses(model, b, pw_n, pw_e)
                vl += float(loss)
                m = b["ens"]
                em = m[b["e"][:, 0]] & m[b["e"][:, 1]]
                yn.append(b["yn"][m].cpu().numpy()); pn.append(torch.sigmoid(nl[m]).cpu().numpy())
                ye.append(b["ye"][em].cpu().numpy()); pe.append(torch.sigmoid(el[em]).cpu().numpy())
        ap_n = average_precision_score(np.concatenate(yn), np.concatenate(pn))
        ap_e = average_precision_score(np.concatenate(ye), np.concatenate(pe))
        hist.append({"epoch": ep, "train_loss": tl / len(B_tr), "val_loss": vl / len(B_va),
                     "val_ap_node": ap_n, "val_ap_edge": ap_e})
        score = ap_n + ap_e
        if best is None or score > best[0]:
            best = (score, ep)
            torch.save(model.state_dict(), out / "model.pt")
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"ep {ep:3d} train {tl / len(B_tr):.3f} val {vl / len(B_va):.3f} "
                  f"AP node {ap_n:.3f} edge {ap_e:.3f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out / "stats.npz", **stats)
    cfg = {"variant": a.variant, "hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
           "best_epoch": best[1], "best_val_ap_sum": best[0], "train_cases": TRAIN, "val_cases": VAL,
           "n_node_features": int(Xa.shape[1]), "n_edge_features": int(Ea.shape[1]),
           "train_seconds": time.time() - t0, "compute": describe(), "synthetic_training_data": "true"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(json.dumps({k: cfg[k] for k in ("variant", "best_epoch", "best_val_ap_sum", "train_seconds")}))


if __name__ == "__main__":
    main()
