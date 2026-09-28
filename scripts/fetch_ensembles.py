"""BRIEF4 Phase 1: REAL ensemble forecasts of Amphan (2020) from public WeatherBench 2 stores.

Direct HTTPS chunk reads (no login), India box only, streamed chunk by chunk:
  IFS ENS 1.5 deg (50 members; 6-hourly leads 0-240 h), inits 2020-05-08..05-20 00 UTC:
      MSLP for all inits; 10 m u/v for 3 inits
  GenCast 1.5 deg (AI ensemble, 12-hourly leads 12-240 h), same inits: MSLP
  IFS ENS 0.25 deg flagship: init 2020-05-16 00 UTC, MSLP, leads 0-144 h
Output: data/real/ensembles/{model}_{res}_{init}.nc (float32, CF), logged in sources_log.json.

    python scripts/fetch_ensembles.py
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import psutil
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_real import dir_mb, log  # noqa: E402

GCS = "https://storage.googleapis.com/weatherbench2/datasets"
STORES = {
    "ifs_ens_1p5": f"{GCS}/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr",
    "ifs_ens_0p25": f"{GCS}/ifs_ens/2018-2022-1440x721.zarr",
    "gencast_1p5": f"{GCS}/gencast/2020-240x121_equiangular_with_poles_conservative.zarr",
}
OUT = ROOT / "data" / "real" / "ensembles"
INITS = pd.date_range("2020-05-08T00", "2020-05-20T00", freq="24h")
UV_INITS = [pd.Timestamp("2020-05-14T00"), pd.Timestamp("2020-05-16T00"), pd.Timestamp("2020-05-18T00")]
VARS = {"mean_sea_level_pressure": "msl", "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10"}
PEAK = [0.0]


def get(url, tries=4):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=(20, 300))
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (k + 1))
    raise last


def arr(store, name, meta):
    za = meta[f"{name}/.zarray"]
    parts = []
    for c in range(-(-za["shape"][0] // za["chunks"][0])):
        parts.append(np.frombuffer(numcodecs.Blosc().decode(get(f"{store}/{name}/{c}")), za["dtype"]))
    return np.concatenate(parts)[:za["shape"][0]]


class Store:
    def __init__(self, key):
        self.url = STORES[key]
        self.meta = requests.get(self.url + "/.zmetadata", timeout=60).json()["metadata"]
        tunits = self.meta["time/.zattrs"]["units"]
        base = pd.Timestamp(tunits.split("since")[1].strip())
        self.time = base + pd.to_timedelta(arr(self.url, "time", self.meta), "h")
        self.lead = arr(self.url, "prediction_timedelta", self.meta)
        lu = self.meta["prediction_timedelta/.zattrs"].get("units", "hours")
        self.lead_h = self.lead / 3600.0 if "sec" in lu else self.lead.astype(float)
        latn = "latitude" if "latitude/.zarray" in self.meta else "lat"
        lonn = "longitude" if "longitude/.zarray" in self.meta else "lon"
        self.lat = arr(self.url, latn, self.meta)
        self.lon = arr(self.url, lonn, self.meta)

    def box(self):
        la = np.where((self.lat >= -0.01) & (self.lat <= 40.01))[0]
        lo = np.where((self.lon >= 59.99) & (self.lon <= 100.01))[0]
        return la, lo

    def fetch(self, var, init, max_lead_h):
        za = self.meta[f"{var}/.zarray"]
        dims = self.meta[f"{var}/.zattrs"]["_ARRAY_DIMENSIONS"]
        ti = int(np.where(self.time == init)[0][0])
        lead_idx = np.where(self.lead_h <= max_lead_h)[0]
        lc = za["chunks"][dims.index("prediction_timedelta")]
        chunks = sorted({int(i // lc) for i in lead_idx})
        la, lo = self.box()
        shape = za["chunks"]

        def one(c):
            raw = get(f"{self.url}/{var}/{ti}.0.{c}.0.0")
            a = np.frombuffer(numcodecs.Blosc().decode(raw), za["dtype"]).reshape(shape)[0]
            # a: (member, lead_in_chunk, d3, d4) with d3/d4 = lat/lon in either order
            if dims[3] in ("longitude", "lon"):
                a = a[:, :, lo][:, :, :, la].transpose(0, 1, 3, 2)
            else:
                a = a[:, :, la][:, :, :, lo]
            PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss / 1e6)
            return c, a.astype(np.float32)

        with ThreadPoolExecutor(2 if shape[-1] > 500 else 4) as ex:   # 0.25 deg chunks are ~200 MB
            got = dict(ex.map(one, chunks))
        full = np.concatenate([got[c] for c in chunks], axis=1)
        first = chunks[0] * lc
        sel = lead_idx - first
        n_valid = full.shape[1]
        sel = sel[sel < n_valid]
        return full[:, sel], self.lead_h[lead_idx][: len(sel)], self.lat[la], self.lon[lo]


def save(model, res, init, fields, leads, lat, lon, url):
    order = np.argsort(lat)
    ds = xr.Dataset({v: (("number", "step", "latitude", "longitude"), a[:, :, order, :],
                         {"units": {"msl": "Pa", "u10": "m s-1", "v10": "m s-1"}[v]}) for v, a in fields.items()},
                    coords={"number": np.arange(next(iter(fields.values())).shape[0]),
                            "step": leads.astype(int), "latitude": lat[order], "longitude": lon},
                    attrs={"Conventions": "CF-1.8", "synthetic": "false", "source": url,
                           "title": f"REAL {model} ensemble forecast, init {init}, India box",
                           "init_time": str(init), "step_units": "hours"})
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{model}_{res}_{init:%Y%m%d%H}.nc"
    ds.to_netcdf(p, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    return p


def main():
    t0 = time.time()
    jobs = [("ifs_ens", "1p5", i, ["mean_sea_level_pressure"] +
             (["10m_u_component_of_wind", "10m_v_component_of_wind"] if i in UV_INITS else []), 240)
            for i in INITS]
    jobs += [("gencast", "1p5", i, ["mean_sea_level_pressure"], 240) for i in INITS]
    jobs += [("ifs_ens", "0p25", pd.Timestamp("2020-05-16T00"), ["mean_sea_level_pressure"], 144)]
    stores = {}
    for model, res, init, variables, max_lead in jobs:
        key = f"{model}_{res}"
        p = OUT / f"{model}_{res}_{init:%Y%m%d%H}.nc"
        if p.exists():
            print("exists", p.name)
            continue
        try:
            st = stores.setdefault(key, Store(key))
            fields, leads = {}, None
            for v in variables:
                a, leads, lat, lon = st.fetch(v, init, max_lead)
                fields[VARS[v]] = a
            p = save(model, res, init, fields, leads, lat, lon, st.url)
            print(f"{p.name}: members {a.shape[0]}, leads {len(leads)}, {time.time() - t0:.0f}s, "
                  f"peak RSS {PEAK[0]:.0f} MB", flush=True)
        except Exception as e:  # noqa: BLE001
            log(f"{model} {res} init {init}", STORES[key], "FAILED", note=f"{type(e).__name__}: {e}")
    for key, st in stores.items():
        files = sorted(OUT.glob(f"{key}_*.nc"))
        log(f"REAL ensemble {key} (Amphan 2020)", st.url, "OK", dir_mb(*files),
            f"{len(files)} inits 2020-05-08..20 00 UTC", f"India box; direct chunk reads; peak RSS {PEAK[0]:.0f} MB")


if __name__ == "__main__":
    main()
