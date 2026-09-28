# Compute (BRIEF2 Phase 0)

## Result of the check (2026-09-27)
| item | result |
|---|---|
| GPU | **none**: `nvidia-smi` is not present, so no CUDA device |
| `pip install torch --index-url https://download.pytorch.org/whl/cpu` | **OK**: torch 2.14.0+cpu installs, and Application Control did **not** block its DLLs this time |
| torch ops tested | matmul, Conv2d/ConvTranspose2d training steps, AdamW, autograd, `index_add` message passing and `torch.fft` all work |
| torch_geometric | 2.8.0.post1 installs and imports (pure Python). We still write message passing in plain PyTorch, so nothing depends on PyG. |
| scikit-learn | 1.9.0 installs and imports (same version as the system Python) |
| CPU | 12 logical cores, torch uses 10 threads, 15.6 GB RAM |

Measured on this machine (CPU, fixed seed):
- A training step of a small conv U-Net block on a batch of 8 × 4 × 192 × 192 takes **0.17–0.23 s**.
- A GNN message pass with 20,000 nodes, 120,000 edges and 64-d features (forward and backward)
  takes **0.08 s**.

## Which path is used
1. **Local CPU torch (default).** The GNN tracker (Phase 2) is small (thousands of object nodes),
   and the U-Net downscaler (Phase 3) trains on 48 → 144 px patches. Both are feasible on this CPU in
   minutes to tens of minutes. The later phases report the settings actually used.
2. **Colab / Kaggle GPU (optional, for scale-up).** This is mainly for the CorrDiff-style diffusion
   model if the CPU budget limits epochs or model size.
   - `python scripts/package_training_data.py` writes `training_bundle.zip`: **0.61 GB**, 3,900
     patch pairs from 13 SYNTHETIC cases, with whole-case splits from `pipeline/splits.py`, float16
     after fixed offsets. It took 1 min 55 s.
   - `notebooks/train_colab.ipynb` covers GPU check, code upload/clone, bundle upload, install,
     the same training commands, and checkpoint download.
   - The scripts pick the device through `pipeline/compute.py` (`cuda` if present, or the
     `SIH_DEVICE` env var), so they run unchanged.
3. **CPU-only without torch (fallback).** `pipeline/compute.py` sets `HAS_TORCH=False` when torch
   cannot be imported. The anomaly/EFI/tracker pipeline and the demo only need numpy, scipy and
   sklearn and cached model outputs, and `tests/test_compute.py` checks this by blocking the torch
   import. In this mode the model-based stages switch to sklearn or classical fallbacks:
   gradient-boosted link classifier instead of the GNN, and bicubic + DEM lapse-rate instead of the
   U-Net/diffusion. The demo labels which one produced each output.

## Why
torch now works locally, so the CPU is the reproducible reference path: seeds are fixed and
`torch.use_deterministic_algorithms(True, warn_only=True)` is set. The Colab path exists
because there is no GPU and diffusion training is the one piece that may need it. The no-torch path
exists because this machine's Application Control policy blocked newly downloaded binary wheels in
the previous session and may do so again after an update.

## Splits (fixed, whole cases, no leakage)
- train: cyc_01, cyc_02, heat_01, heat_02, cold_01, cold_02
- val: cyc_03, heat_03, cold_03 (model selection and threshold choice)
- test: cyc_04, heat_04, cold_04, amphan_replay (reported only, never tuned on)

The real test is Amphan ERA5 vs IBTrACS. The training set is small (6 cases). Later phases may add
more synthetic training cases (`gen_*`, train only) with the existing generator.

## BRIEF3: API and dashboard stack (2026-09-28)
- `pip install fastapi uvicorn httpx` installed fastapi 0.141.1 and pydantic-core 2.46.5. Application
  Control did **not** block pydantic-core, so the API uses **FastAPI + uvicorn**, not the Flask or
  stdlib fallback.
- Frontend: Node 24 / npm 11. React 18, Vite 5, TypeScript 5, MapLibre GL 4 and deck.gl 9 all
  installed.
  - npm's allow-scripts policy skipped esbuild's postinstall script, but the platform binary
    (`@esbuild/win32-x64`) was installed and works.
- End-to-end tests use Playwright with the **system Microsoft Edge** (`channel: "msedge"`), so no
  browser download is needed. WebGL runs through SwiftShader in headless mode.
- Basemap: Carto's free raster tiles now return "API key required", so the dashboard uses
  OpenStreetMap raster tiles instead (attribution shown).
