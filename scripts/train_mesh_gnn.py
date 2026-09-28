"""Train the icosahedral mesh GNN (or the grid-CNN ablation) for per-cell hazard segmentation.

    python scripts/train_mesh_gnn.py --model mesh --level 6 [--epochs 10] [--name mesh_l6]
    python scripts/train_mesh_gnn.py --model cnn --name cnn

Train on the SYNTHETIC TRAIN cases, select the epoch on VAL (never TEST). Target: the member's
own injected-event mask (event_mask) in the channel of the case's hazard; the other two
hazard channels are 0. Loss = BCE (pos_weight 20) + soft Dice on the case's hazard channel.
Per epoch: every TRAIN case, `--members` random members, every `--lead-stride`-th lead (random
offset), pooled over all cases (float16) and SHUFFLED; batch 2. Streams one case / one member at a time (float32, del + gc). EFI inputs are
computed once per case from the whole ensemble and cached in data/graphs/efi/ (float16).
Writes models/mesh/{name}/{model.pt, config.json, log.json}; prints peak RSS.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.anomaly import Climatology  # noqa: E402
from pipeline.compute import describe as device_info  # noqa: E402
from pipeline.mesh_gnn import HAZ, N_IN, Graph, GridCNN, MeshGNN, Static, features, n_params  # noqa: E402
from pipeline.splits import TRAIN, VAL  # noqa: E402
from synth.grids import LAT12, LON12  # noqa: E402

SYN = ROOT / "data/synthetic"
EFI = ROOT / "data/graphs/efi"
VARS = ("t2m", "u10", "v10", "msl", "tp", "r850")
PEAK = [0.0]
POS_WEIGHT = 20.0          # event cells are ~2 % of the grid (5.0 collapsed heat/cold to 0 in a first run)


def rss():
    PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss / 1e6)
    return PEAK[0]


def case_efi(case, clim):
    p = EFI / f"{case}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    lab = json.loads((SYN / case / "labels.json").read_text())
    times = pd.to_datetime(lab["valid_times"])
    with xr.open_dataset(SYN / case / "fcst_12km.nc", decode_timedelta=False) as ds:
        out = np.zeros((len(times), 2, 333, 333), np.float32)
        from pipeline.efi import efi_gaussian
        for t, vt in enumerate(times):
            out[t, 0] = -efi_gaussian(ds.msl.isel(step=t).values.astype(np.float32), *clim.get("msl", vt))
            out[t, 1] = efi_gaussian(ds.t2m.isel(step=t).values.astype(np.float32), *clim.get("t2m", vt))
    EFI.mkdir(parents=True, exist_ok=True)
    np.save(p, out.astype(np.float16))
    return out


def member_fields(case, m):
    """One run of a case: member m (>= 0) or the truth run (m = -1), all leads, float32."""
    with xr.open_dataset(SYN / case / "fcst_12km.nc", decode_timedelta=False) as ds:
        if m < 0:
            f = {v: ds[f"{v}_truth"].values.astype(np.float32) for v in VARS if f"{v}_truth" in ds}
            mask = ds["event_mask_truth"].values.astype(bool)
        else:
            f = {v: ds[v].isel(number=m).values.astype(np.float32) for v in VARS if v in ds}
            mask = ds["event_mask"].isel(number=m).values.astype(bool)
    return f, mask


def samples(case, members, leads, clim, st, efi):
    lab = json.loads((SYN / case / "labels.json").read_text())
    times = pd.to_datetime(lab["valid_times"])
    hi = HAZ.index(lab["hazard"])
    for m in members:
        f, mask = member_fields(case, m)
        for t in leads:
            x = features(f, t, times[t], 6 * t, efi[t], clim, st)
            yield x.reshape(N_IN, -1).T, mask[t].ravel(), hi
        del f, mask
        gc.collect()


def loss_fn(logits, y, hi):
    """logits (B, N, 3), y (B, N) bool target of channel hi[b]."""
    tgt = torch.zeros_like(logits)
    for b, h in enumerate(hi):
        tgt[b, :, h] = y[b]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, tgt, pos_weight=torch.tensor(POS_WEIGHT))
    p = torch.sigmoid(torch.stack([logits[b, :, h] for b, h in enumerate(hi)]))
    yt = y.float()
    dice = 1 - (2 * (p * yt).sum(1) + 1) / (p.sum(1) + yt.sum(1) + 1)
    return bce + dice.mean()


def iou(p, y, thr=0.5):
    a = p >= thr
    u = (a | y).sum()
    return float((a & y).sum() / u) if u else np.nan


@torch.no_grad()
def validate(model, cases, clim, st, rng_seed=123):
    model.eval()
    rng = np.random.default_rng(rng_seed)
    out = {}
    for c in cases:
        efi = case_efi(c, clim)
        mem = sorted(rng.choice(20, 3, replace=False).tolist()) + [-1]
        ious = []
        for x, y, hi in samples(c, mem, range(0, 41, 4), clim, st, efi):
            p = torch.sigmoid(model(torch.tensor(x)[None]))[0, :, hi].numpy()
            if y.any() or (p >= 0.5).any():
                ious.append(iou(p, y))
        out[c] = float(np.nanmean(ious)) if ious else 0.0
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mesh", "cnn"], default="mesh")
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--lead-stride", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    name = a.name or (f"mesh_l{a.level}" if a.model == "mesh" else "cnn")
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(a.seed)
    clim, st = Climatology(), Static(LAT12, LON12)
    if a.model == "mesh":
        g = Graph(a.level, LAT12, LON12)
        model = MeshGNN(g, hidden=a.hidden)
    else:
        model = GridCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    out = ROOT / "models/mesh" / name
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], (-1.0, -1)
    t0 = time.time()
    print(f"{name}: {n_params(model)} params, {device_info()}", flush=True)
    for ep in range(a.epochs):
        losses = []
        pool = []                             # the epoch's samples from ALL cases, then shuffled
        for c in TRAIN:                       # (case-by-case order made the model forget heat/cold)
            efi = case_efi(c, clim)
            mem = sorted(rng.choice(20, a.members, replace=False).tolist())
            leads = range(int(rng.integers(a.lead_stride)), 41, a.lead_stride)
            for x, y, hi in samples(c, mem, leads, clim, st, efi):
                pool.append((x.astype(np.float16), np.packbits(y), hi))
            rss()
        order = rng.permutation(len(pool))
        for k in range(0, len(order) - 1, 2):
            bt = [pool[i] for i in order[k:k + 2]]
            X = torch.tensor(np.stack([b[0] for b in bt]).astype(np.float32))
            Y = torch.tensor(np.stack([np.unpackbits(b[1])[:X.shape[1]].astype(bool) for b in bt]))
            loss = loss_fn(model(X), Y, [b[2] for b in bt])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        del pool
        gc.collect()
        sched.step()
        v = validate(model, VAL, clim, st)
        vm = float(np.mean(list(v.values())))
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "val_iou": v, "val_iou_mean": vm,
                    "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(rss())})
        print(json.dumps(log[-1]), flush=True)
        if vm > best[0]:
            best = (vm, ep)
            torch.save(model.state_dict(), out / "model.pt")
    cfg = {"model": a.model, "level": a.level if a.model == "mesh" else None, "hidden": a.hidden,
           "params": n_params(model), "epochs": a.epochs, "members_per_case": a.members, "lead_stride": a.lead_stride,
           "lr": a.lr, "seed": a.seed, "best_epoch": best[1], "best_val_iou": best[0], "train_cases": TRAIN,
           "val_cases": VAL, "features": __import__("pipeline.mesh_gnn", fromlist=["FEATURES"]).FEATURES,
           "train_seconds": round(time.time() - t0, 1), "peak_rss_mb": round(PEAK[0]), "compute": device_info()}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"best epoch {best[1]} val IoU {best[0]:.3f}; peak RSS {PEAK[0]:.0f} MB; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
