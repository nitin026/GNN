"""Retune the synthetic fine-scale (k^-5/3) noise amplitude against the real ERA5 spectrum (BRIEF4 Phase 4).

    python scripts/retune_noise.py [--regenerate heat_04]

1. For t2m (heat_01) and u10 (amphan_replay), as in synth/validate.py: radial power spectra of the
   SYNTHETIC 5 km truth, of ERA5 at its native 0.25 deg, and of ERA5 interpolated to 5 km.
2. Reference at 25 km: ERA5's native spectrum is fitted (log-log) over the scales it resolves
   (100-400 km) and extrapolated to 25 km. (ERA5 interpolated to 5 km has almost no 25 km power,
   so matching it directly would remove all fine scales; the extrapolated ERA5 slope is the most
   defensible real reference available; there is no real km-scale product here.)
3. The synthetic field = background (ERA5 interpolated) + event + FINE[v] * unit noise. The unit
   noise's own power at 25 km is measured, and the new amplitude solves
       P_background(25) + A_new^2 P_unit(25) = P_ref(25).
4. --regenerate CASE writes a retuned copy to data/synthetic_retuned/CASE (the original is kept;
   models are NOT retrained on it) and compares its spectrum before/after.
Writes reports/noise_retune.json and reports/figures/noise_retune.png.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synth.generate import FINE  # noqa: E402
from synth.noise import radial_spectrum, spectral_noise  # noqa: E402
from synth.validate import read_packed  # noqa: E402

SYN = ROOT / "data/synthetic"
REAL = ROOT / "data/real"
CASES = {"t2m": ("heat_01", "heatwave"), "u10": ("amphan_replay", "amphan")}
STEP = 20


def at(k, p, km):
    return float(np.exp(np.interp(np.log(1 / km), np.log(k[1:]), np.log(p[1:]))))


def spectra(var, syn_root=SYN, cid=None):
    cid = cid or CASES[var][0]
    win = CASES[var][1]
    lab = json.loads((syn_root / cid / "labels.json").read_text())
    t = pd.Timestamp(lab["valid_times"][STEP])
    syn = read_packed(syn_root / cid / "truth_5km.nc", var, STEP)[250:762, 250:762]
    ks, ps = radial_spectrum(syn, 0.04 * 111.2)
    r25 = xr.open_dataset(REAL / "era5" / f"{win}_era5_0p25.nc")[var].sel(time=t, method="nearest").values
    kn, pn = radial_spectrum(r25[40:120, 40:120] if r25.shape[0] > 120 else r25, 0.25 * 111.2)
    g5 = xr.open_dataset(REAL / "era5" / f"{win}_era5_g5.nc")[var].sel(time=t, method="nearest").values
    kb, pb = radial_spectrum(g5[250:762, 250:762], 0.04 * 111.2)
    return (ks, ps), (kn, pn), (kb, pb)


def main():
    out, fig = {}, plt.figure(figsize=(12, 4.5))
    unit = spectral_noise((512, 512), np.random.default_rng(0))
    ku, pu = radial_spectrum(unit, 0.04 * 111.2)
    scale = {}
    for n, var in enumerate(("t2m", "u10")):
        (ks, ps), (kn, pn), (kb, pb) = spectra(var)
        sel = (kn > 1 / 400) & (kn < 1 / 100)
        b, a = np.polyfit(np.log(kn[sel]), np.log(pn[sel]), 1)
        p_ref25 = float(np.exp(a + b * np.log(1 / 25)))
        p_bg25, p_u25 = at(kb, pb, 25), at(ku, pu, 25)
        a_new = float(np.sqrt(max(p_ref25 - p_bg25, 0.0) / p_u25))
        scale[var] = a_new / FINE[var]
        out[var] = {"case": CASES[var][0], "era5_native_slope_100_400km": float(b), "p_ref_25km": p_ref25,
                    "p_syn_25km": at(ks, ps, 25), "p_era5interp_25km": p_bg25, "p_unit_noise_25km": p_u25,
                    "old_amplitude": FINE[var], "new_amplitude": a_new, "scale": scale[var],
                    "old_ratio_syn_over_ref_25km": at(ks, ps, 25) / p_ref25}
        ax = fig.add_subplot(1, 2, n + 1)
        ax.loglog(ks, ps, "k", label="SYNTHETIC before")
        ax.loglog(kn, pn, "C0", label="ERA5 0.25 deg (native)")
        ax.loglog(kb, pb, "C1", label="ERA5 interp. 5 km")
        kk = np.linspace(1 / 400, 1 / 10, 200)
        ax.loglog(kk, np.exp(a + b * np.log(kk)), "C0:", label="ERA5 slope 100-400 km, extrapolated")
        ax.axvline(1 / 25, color="grey", lw=0.6)
        ax.set_title(f"{var}: amplitude {FINE[var]:.3g} -> {a_new:.3g} (x{scale[var]:.2f})")
        ax.set_xlabel("wavenumber (1/km)")
        ax.legend(fontsize=7)
        out[var]["_ax"] = n
    for v in ("v10",):                     # same factor as u10 for the other wind component
        scale[v] = scale["u10"]
    regen = sys.argv[sys.argv.index("--regenerate") + 1] if "--regenerate" in sys.argv else None
    if regen:
        env = {**os.environ, "SIH_SYN_OUT": str(ROOT / "data/synthetic_retuned"), "SIH_FINE_SCALE": json.dumps(scale)}
        subprocess.run([sys.executable, "-m", "synth.generate", regen], cwd=ROOT, env=env, check=True)
        var = "t2m" if regen.startswith("heat") or regen.startswith("cold") else "u10"
        b0 = spectra(var, SYN, regen)[0]
        b1 = spectra(var, ROOT / "data/synthetic_retuned", regen)[0]
        out["regenerated"] = {"case": regen, "var": var, "p25_before": at(*b0, 25), "p25_after": at(*b1, 25),
                              "p50_before": at(*b0, 50), "p50_after": at(*b1, 50),
                              "note": "retuned copy in data/synthetic_retuned (not in git); models were not retrained"}
        ax = fig.axes[out[var]["_ax"]]
        ax.loglog(*b0, "k--", lw=0.8, label=f"{regen} before")
        ax.loglog(*b1, "C3", label=f"{regen} after retune")
        ax.legend(fontsize=7)
    for v in list(out):
        out[v].pop("_ax", None) if isinstance(out[v], dict) else None
    out["scale"] = scale
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/noise_retune.png", dpi=100)
    (ROOT / "reports/noise_retune.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
