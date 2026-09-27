"""Render data/real/sources_log.json -> data/SOURCES.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log = json.loads((ROOT / "data/real/sources_log.json").read_text())
L = ["# Data sources", "",
     "Every real-data source attempted by `scripts/fetch_real.py` is listed here with its status "
     "(OK / FAILED / SKIPPED). Generated from `data/real/sources_log.json` by "
     "`scripts/write_sources.py`. Sizes are on-disk sizes of the files written (MB).", "",
     "| source | url | status | size (MB) | time range | note | logged (UTC) |",
     "|---|---|---|---|---|---|---|"]
for e in log:
    L.append(f"| {e['source']} | {e['url']} | {e['status']} | "
             f"{'' if e['size_mb'] is None else e['size_mb']} | {e['time_range']} | "
             f"{e['note'].replace('|', '/')} | {e['logged_at']} |")
L += ["", "## Synthetic data", "",
      "All synthetic data lives under `data/synthetic/` and every file carries the global attribute "
      "`synthetic = \"true\"`. Synthetic cases use the real ERA5 backgrounds listed above (with extremes "
      "soft-clipped), the real Copernicus DEM and real IBTrACS tracks. The analytic synthetic-climatology "
      "fallback in `synth/background.py` was **not** needed, because the ERA5 downloads succeeded. "
      "If it is ever used, the files record `background = synthetic_climatology`.", ""]
(ROOT / "data/SOURCES.md").write_text("\n".join(L), encoding="utf-8")
print(f"wrote data/SOURCES.md ({len(log)} entries)")
