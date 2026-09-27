"""Train the 12 km -> 5 km downscalers on the TRAIN patches of the training bundle; select on VAL.

    python scripts/train_downscaler.py --model unet      [--bundle bundle] [--epochs 30]
    python scripts/train_downscaler.py --model diffusion [--bundle bundle] [--epochs 40]
The diffusion model needs a trained U-Net (its mean). Device via pipeline/compute.py, so the
same command runs on CPU or a Colab/Kaggle GPU. TEST patches are never used here.
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

import torch  # noqa: E402

from pipeline.downscale import (Normalizer, ResidualDiffusion, UNetDownscaler, cond_input,  # noqa: E402
                                downscale_loss)


def load_split(bundle, split, dev):
    z = np.load(Path(bundle) / f"downscale_{split}.npz")
    t = lambda k: torch.tensor(z[k].astype(np.float32))
    return {"x12": t("x12"), "y5": t("y5"), "dem12": t("dem12"), "dem5": t("dem5")}


def batches(data, bs, rng=None, crop12=None):
    """Mini-batches; with crop12, a random crop of crop12 x crop12 cells at 12 km and the aligned
    3x larger crop at 5 km (same crop for the whole batch)."""
    n = len(data["x12"])
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        b = {k: v[j] for k, v in data.items()}
        if crop12:
            N12 = b["x12"].shape[-1]
            i0, j0 = (int(x) for x in rng.integers(0, N12 - crop12 + 1, 2))
            for k in ("x12", "dem12"):
                b[k] = b[k][..., i0:i0 + crop12, j0:j0 + crop12]
            for k in ("y5", "dem5"):
                b[k] = b[k][..., 3 * i0:3 * (i0 + crop12), 3 * j0:3 * (j0 + crop12)]
        yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unet", "diffusion"], required=True)
    ap.add_argument("--bundle", default=str(ROOT / "bundle"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--crop12", type=int, default=32, help="random training crop (12 km cells); 0 = full")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)
    dev = device()
    print("compute:", describe(), flush=True)
    epochs = a.epochs or (20 if a.model == "unet" else 25)
    out = Path(a.out or ROOT / "models" / "downscale" / a.model)
    out.mkdir(parents=True, exist_ok=True)
    tr = load_split(a.bundle, "train", dev)
    va = load_split(a.bundle, "val", dev)
    mean = tr["y5"].mean((0, 2, 3)).numpy()
    std = tr["y5"].std((0, 2, 3)).numpy() + 1e-6
    norm = Normalizer(mean, std).to(dev)
    rng = np.random.default_rng(a.seed)
    to = lambda b: {k: v.to(dev) for k, v in b.items()}
    unet = UNetDownscaler(a.base).to(dev)
    if a.model == "diffusion":
        unet.load_state_dict(torch.load(ROOT / "models/downscale/unet/model.pt", map_location=dev))
        unet.eval()
        model = ResidualDiffusion(a.base).to(dev)
    else:
        model = unet
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    hist, best, t0 = [], None, time.time()
    for ep in range(epochs):
        model.train()
        tl, nb = 0.0, 0
        for b in batches(tr, a.batch, rng, a.crop12):
            b = to(b)
            opt.zero_grad()
            if a.model == "unet":
                pred = model(b["x12"], b["dem12"], b["dem5"], norm)
                loss, _ = downscale_loss(norm.n(pred), norm.n(b["y5"]))
            else:
                with torch.no_grad():
                    m = unet(b["x12"], b["dem12"], b["dem5"], norm)
                    inp, _ = cond_input(b["x12"], b["dem12"], b["dem5"], norm)
                r = norm.n(b["y5"]) - norm.n(m)
                loss = model.loss(r, model.cond(norm.n(m), inp))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            nb += 1
        model.eval()
        vl, vb = 0.0, 0
        with torch.no_grad():
            g = torch.Generator(device=dev).manual_seed(123)
            for b in batches(va, 32):
                b = to(b)
                if a.model == "unet":
                    pred = model(b["x12"], b["dem12"], b["dem5"], norm)
                    loss, _ = downscale_loss(norm.n(pred), norm.n(b["y5"]))
                else:
                    m = unet(b["x12"], b["dem12"], b["dem5"], norm)
                    inp, _ = cond_input(b["x12"], b["dem12"], b["dem5"], norm)
                    r = norm.n(b["y5"]) - norm.n(m)
                    torch.manual_seed(ep)                       # same noise draw per epoch
                    loss = model.loss(r, model.cond(norm.n(m), inp))
                vl += loss.item()
                vb += 1
        hist.append({"epoch": ep, "train_loss": tl / nb, "val_loss": vl / vb})
        if best is None or vl / vb < best[0]:
            best = (vl / vb, ep)
            torch.save(model.state_dict(), out / "model.pt")
        print(f"ep {ep:3d} train {tl / nb:.4f} val {vl / vb:.4f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out / "stats.npz", mean=mean, std=std)
    cfg = {"model": a.model, "epochs": epochs, "batch": a.batch, "lr": a.lr, "base": a.base,
           "crop12": a.crop12,
           "seed": a.seed, "best_epoch": best[1], "best_val_loss": best[0],
           "train_seconds": time.time() - t0, "compute": describe(),
           "n_train_patches": int(len(tr["x12"])), "synthetic_training_data": "true",
           "loss": "MSE + 0.1 spectral + 0.5 pinball(q=0.99, top-1% pixels)" if a.model == "unet"
           else "DDPM epsilon-MSE on normalised residual (T=500)"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(json.dumps({k: cfg[k] for k in ("model", "best_epoch", "best_val_loss", "train_seconds")}))


if __name__ == "__main__":
    main()
