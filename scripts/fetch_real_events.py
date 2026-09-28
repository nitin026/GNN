"""BRIEF4 Phase 1: REAL evaluation events (ERA5 + IBTrACS + IMD), time-split and locked.

For each event: ERA5 surface (t2m, u10, v10, msl, tp 6 h), 6-hourly, and 850 hPa q/u/v daily at
12 UTC, India box, on the native 0.25 deg grid and G12 (no G5, to save disk). WB2 (to
2023-01-10) or ARCO-ERA5 (later) over HTTPS. Cyclone windows come from IBTrACS (first fix - 2 d
to last fix + 1 d); each best track is saved as CSV.
IMD 1 deg Tmax/Tmin (imdlib) for heat/cold truth, if the IMD server answers (2 attempts).
Split: events <= 2021 -> REAL-VAL, >= 2022 -> REAL-TEST, written once to data/real/SPLITS.json.

    python scripts/fetch_real_events.py [event ...]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask
import numcodecs
import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_real import (ARCO, ATTRS, BOX, GCS, PL850, SFC, WB2_ERA5, attempt, dir_mb, log,  # noqa: E402
                                open_store, packed_encoding)
from synth.grids import regrid  # noqa: E402

REAL = ROOT / "data" / "real"
EV = REAL / "events"
IBT_ALL = REAL / "ibtracs" / "ibtracs.NI.list.v04r01.csv"
WB2_END = pd.Timestamp("2023-01-10T18")
dask.config.set(scheduler="threads", num_workers=16)

CYCLONES = {  # event id -> (IBTrACS NAME, SEASON)
    "fani_2019": ("FANI", 2019), "vayu_2019": ("VAYU", 2019), "amphan_2020": ("AMPHAN", 2020),
    "nisarga_2020": ("NISARGA", 2020), "nivar_2020": ("NIVAR", 2020), "tauktae_2021": ("TAUKTAE", 2021),
    "yaas_2021": ("YAAS", 2021), "gulab_2021": ("GULAB:SHAHEEN-GU", 2021),
    "asani_2022": ("ASANI", 2022), "mocha_2023": ("MOCHA", 2023), "biparjoy_2023": ("BIPARJOY", 2023),
    "tej_2023": ("TEJ", 2023), "michaung_2023": ("MICHAUNG", 2023),
}
HEATCOLD = {  # event id -> (hazard, start, end)
    "heat_2019": ("heat_dome", "2019-05-25T00", "2019-06-15T18"),
    "cold_2019": ("cold_wave", "2019-12-15T00", "2020-01-10T18"),
    "heat_2022": ("heat_dome", "2022-04-15T00", "2022-05-10T18"),
}
EXISTING = {  # already fetched in Phase 2 of BRIEF.md (data/real/era5)
    "heat_2024": ("heat_dome", "heatwave"), "cold_2022": ("cold_wave", "coldwave"),
}


def split_of(year):
    return "REAL-VAL" if year <= 2021 else "REAL-TEST"


def ibtracs_track(name, season):
    df = pd.read_csv(IBT_ALL, skiprows=[1], low_memory=False, keep_default_na=False)
    g = df[(df.NAME == name) & (df.SEASON.astype(str) == str(season))]
    keep = ["SID", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES",
            "USA_RMW", "NEWDELHI_WIND", "NEWDELHI_PRES", "USA_SSHS", "DIST2LAND"]
    return g[[c for c in keep if c in g.columns]].reset_index(drop=True)


def arco_850(times):
    """850 hPa q/u/v at the given times from ARCO (direct 100 MB chunk reads)."""
    from scripts.fetch_real import _arco_chunk
    meta = requests.get(f"{GCS}/{ARCO}/.zmetadata", timeout=60).json()["metadata"]
    lev = np.frombuffer(numcodecs.Blosc().decode(requests.get(f"{GCS}/{ARCO}/level/0", timeout=60).content), "<i8")
    li = int(np.where(lev == 850)[0][0])
    tz = meta["time/.zarray"]
    tv = np.concatenate([np.frombuffer(numcodecs.Blosc().decode(
        requests.get(f"{GCS}/{ARCO}/time/{c}", timeout=120).content), tz["dtype"])
        for c in range(-(-tz["shape"][0] // tz["chunks"][0]))])[:tz["shape"][0]]
    base = pd.Timestamp(meta["time/.zattrs"]["units"].split("since")[1].strip())
    tidx = np.searchsorted(tv, ((times - base) / pd.Timedelta("1h")).astype(np.int64))
    out = {}
    for var, short in PL850.items():
        za = meta[f"{var}/.zarray"]
        with ThreadPoolExecutor(6) as ex:
            arrs = list(ex.map(lambda ti: _arco_chunk(var, int(ti), tuple(za["chunks"][1:]), za["dtype"], li), tidx))
        out[short] = (("time", "latitude", "longitude"), np.stack(arrs))
    return xr.Dataset(out, coords={"time": times, "latitude": np.arange(0, 40.01, 0.25),
                                   "longitude": np.arange(60, 100.01, 0.25)})


def fetch_era5(eid, start, end):
    t0 = time.time()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(start.normalize() + pd.Timedelta("12h"), end, freq="24h")
    if end <= WB2_END:
        ds = open_store(WB2_ERA5)
        sfc = ds[list(SFC) + ["total_precipitation_6hr"]].sel(time=slice(start, end), **BOX)
        out = attempt(lambda: sfc.rename({**SFC, "total_precipitation_6hr": "tp"}).load())
        out["tp"] = (out["tp"] * 1000.0).clip(min=0)
        pl = ds[list(PL850)].sel(time=days, level=850, **BOX).drop_vars("level")
        p850 = attempt(lambda: pl.rename(PL850).load())
        src = f"{GCS}/{WB2_ERA5}"
    else:
        ds = open_store(ARCO)
        inst = ds[list(SFC)].sel(time=pd.date_range(start, end, freq="6h"), **BOX)
        out = attempt(lambda: inst.rename(SFC).load())
        hourly = ds["total_precipitation"].sel(time=slice(start - pd.Timedelta("5h"), end), **BOX)
        hourly = attempt(lambda: hourly.load())
        out["tp"] = (hourly.rolling(time=6).sum().sel(time=out.time) * 1000.0).clip(min=0)
        p850 = attempt(lambda: arco_850(days))
        src = f"{GCS}/{ARCO}"
    out = out.sortby("latitude")
    p850 = p850.sortby("latitude")
    for d in (out, p850):
        for v in d.data_vars:
            d[v].attrs = {"units": ATTRS["units"][v], "long_name": ATTRS["long"][v]}
        d.attrs = {"Conventions": "CF-1.8", "synthetic": "false", "source": src, "event": eid}
    EV.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, d in ((f"{eid}_sfc", out), (f"{eid}_850daily", p850)):
        p0 = EV / f"{name}_0p25.nc"
        d.to_netcdf(p0, encoding={v: {"zlib": True, "dtype": "float32"} for v in d.data_vars})
        g = regrid(d, "g12")
        g.attrs = dict(d.attrs, grid="g12")
        p1 = EV / f"{name}_g12.nc"
        g.to_netcdf(p1, encoding=packed_encoding(g))
        paths += [p0, p1]
    log(f"ERA5 event {eid}", src, "OK", dir_mb(*paths), f"{start} .. {end}",
        f"sfc 6-hourly ({len(out.time)} steps) + 850 hPa daily 12 UTC ({len(days)}); 0.25 + G12; "
        f"{time.time() - t0:.0f}s")


def fetch_imd_temperature(years):
    import imdlib
    raw = REAL / "imd" / "raw"
    ok = []
    for var in ("tmax", "tmin"):
        for yr in years:
            p = REAL / "imd" / f"imd_{var}_{yr}_1deg.nc"
            if p.exists():
                ok.append(p)
                continue
            try:
                d = attempt(lambda: imdlib.get_data(var, yr, yr, fn_format="yearwise", file_dir=str(raw)))
                x = d.get_xarray().rename({"lat": "latitude", "lon": "longitude"})
                x = x[var].where(x[var] < 90).astype("float32").to_dataset(name=var)
                x.attrs = {"synthetic": "false", "source": "IMD Pune 1 deg gridded temperature via imdlib"}
                x.to_netcdf(p)
                ok.append(p)
            except Exception as e:  # noqa: BLE001
                log(f"IMD {var} {yr} 1 deg", "https://imdpune.gov.in (imdlib)", "FAILED",
                    note=f"{type(e).__name__}: {str(e)[:120]} (2 attempts); heat/cold truth falls back to "
                         "ERA5 Tmax/Tmin with IMD criteria")
                return ok
    log("IMD Tmax/Tmin 1 deg", "https://imdpune.gov.in (imdlib)", "OK", dir_mb(*ok), f"{years}", "")
    return ok


def write_splits():
    p = REAL / "SPLITS.json"
    if p.exists():
        print("SPLITS.json exists and is LOCKED; not rewritten")
        return json.loads(p.read_text())
    ev = {}
    for eid, (name, season) in CYCLONES.items():
        ev[eid] = {"hazard": "tropical_cyclone", "ibtracs_name": name, "season": season,
                   "split": split_of(season), "files": f"data/real/events/{eid}_sfc_*.nc"
                   if eid != "amphan_2020" else "data/real/era5/amphan_era5_*.nc"}
    for eid, (hz, s, e) in HEATCOLD.items():
        ev[eid] = {"hazard": hz, "start": s, "end": e, "split": split_of(pd.Timestamp(s).year),
                   "files": f"data/real/events/{eid}_sfc_*.nc"}
    for eid, (hz, stem) in EXISTING.items():
        yr = int(eid.split("_")[1])
        ev[eid] = {"hazard": hz, "split": split_of(yr), "files": f"data/real/era5/{stem}_era5_*.nc"}
    doc = {"locked": True, "created": pd.Timestamp.now(tz="UTC").isoformat(),
           "rule": "events up to 2021 -> REAL-VAL; events from 2022 -> REAL-TEST (split by time)",
           "note": "REAL-TEST is reserved for the final blind evaluation (BRIEF4 Phase 8). Nothing may be "
                   "tuned on it.",
           "events": ev,
           "REAL-VAL": sorted(k for k, v in ev.items() if v["split"] == "REAL-VAL"),
           "REAL-TEST": sorted(k for k, v in ev.items() if v["split"] == "REAL-TEST")}
    p.write_text(json.dumps(doc, indent=1))
    return doc


def main():
    want = sys.argv[1:]
    doc = write_splits()
    ib = REAL / "ibtracs" / "events"
    ib.mkdir(parents=True, exist_ok=True)
    for eid, (name, season) in CYCLONES.items():
        if want and eid not in want:
            continue
        tr = ibtracs_track(name, season)
        tr.to_csv(ib / f"{eid}.csv", index=False)
        if eid == "amphan_2020" or (EV / f"{eid}_sfc_g12.nc").exists():
            continue
        t0, t1 = pd.to_datetime(tr.ISO_TIME.iloc[0]), pd.to_datetime(tr.ISO_TIME.iloc[-1])
        start = (t0 - pd.Timedelta("2D")).normalize()
        end = (t1 + pd.Timedelta("1D")).normalize() + pd.Timedelta("18h")
        try:
            fetch_era5(eid, start, end)
        except Exception as e:  # noqa: BLE001
            log(f"ERA5 event {eid}", "WB2/ARCO", "FAILED", note=f"{type(e).__name__}: {e}")
    for eid, (hz, s, e) in HEATCOLD.items():
        if (want and eid not in want) or (EV / f"{eid}_sfc_g12.nc").exists():
            continue
        try:
            fetch_era5(eid, s, e)
        except Exception as ex:  # noqa: BLE001
            log(f"ERA5 event {eid}", "WB2/ARCO", "FAILED", note=f"{type(ex).__name__}: {ex}")
    if not want or "imd" in want:
        fetch_imd_temperature([2019, 2020, 2022, 2024])
    print(json.dumps({k: doc[k] for k in ("REAL-VAL", "REAL-TEST")}, indent=1))


if __name__ == "__main__":
    main()
