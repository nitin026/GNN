"""Phase 2: fetch real, no-login data for the India box and put it on the common grids.

Every attempt is appended to data/real/sources_log.json; scripts/write_sources.py turns
that log into data/SOURCES.md. A source that fails twice is logged FAILED and skipped.

Usage: python scripts/fetch_real.py [era5 clim ibtracs dem imd keyed]   (default: all)
"""
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synth.grids import regrid  # noqa: E402

REAL = ROOT / "data" / "real"
LOG = REAL / "sources_log.json"
dask.config.set(scheduler="threads", num_workers=24)

GCS = "https://storage.googleapis.com"
WB2_ERA5 = "weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
WB2_ERA5_BRIEF = "weatherbench2/datasets/era5/1959-2023_01_10-6h-1440x721_with_derived_variables.zarr"
WB2_CLIM = "weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
WB2_COARSE = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ARCO = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
IBTRACS = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
           "stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv")
DEM_BASE = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"

BOX = dict(latitude=slice(40, 0), longitude=slice(60, 100))  # ERA5 latitude is descending
SFC = {"2m_temperature": "t2m", "10m_u_component_of_wind": "u10",
       "10m_v_component_of_wind": "v10", "mean_sea_level_pressure": "msl"}
PL850 = {"specific_humidity": "q850", "u_component_of_wind": "u850",
         "v_component_of_wind": "v850"}
EVENTS = {
    # name: (start, end, store, with_850)
    "amphan": ("2020-05-10T00", "2020-05-25T18", "wb2", True),
    "coldwave": ("2022-12-20T00", "2023-01-20T18", "arco", False),
    "heatwave": ("2024-05-15T00", "2024-06-20T18", "arco", False),
}
ATTRS = {"units": {"t2m": "K", "u10": "m s-1", "v10": "m s-1", "msl": "Pa", "tp": "mm", "r850": "%",
                   "q850": "kg kg-1", "u850": "m s-1", "v850": "m s-1"},
         "long": {"t2m": "2 metre temperature", "u10": "10 metre U wind component",
                  "v10": "10 metre V wind component", "msl": "Mean sea level pressure",
                  "tp": "Total precipitation accumulated over the previous 6 h",
                  "q850": "Specific humidity at 850 hPa", "u850": "U wind at 850 hPa",
                  "v850": "V wind at 850 hPa",
                  "r850": "Relative humidity at 850 hPa (WB2 derived)"}}


# ----------------------------------------------------------------------------- logging
def log(source, url, status, size_mb=None, time_range="", note=""):
    entries = json.loads(LOG.read_text()) if LOG.exists() else []
    entries = [e for e in entries if not (e["source"] == source and e["status"] == status
                                         and e["url"] == url)]
    entries.append({"source": source, "url": url, "status": status,
                    "size_mb": None if size_mb is None else round(size_mb, 1),
                    "time_range": time_range, "note": note,
                    "logged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    LOG.write_text(json.dumps(entries, indent=1))
    print(f"[{status}] {source}: {note}")


def dir_mb(*paths):
    return sum(p.stat().st_size for q in paths for p in (Path(q).rglob("*") if Path(q).is_dir()
                                                        else [Path(q)]) if p.is_file()) / 1e6


def attempt(fn, tries=2):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - we log every failure
            last = e
            print(f"  attempt {i + 1} failed: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            time.sleep(5)
    raise last


def open_store(path):
    return xr.open_zarr(f"{GCS}/{path}", consolidated=True, zarr_format=2, chunks={})


def finish(ds, name, source):
    """CF attrs, write native 0.25 box + G12 + G5 NetCDF."""
    ds = ds.sortby("latitude")
    for v in ds.data_vars:
        ds[v].attrs = {"units": ATTRS["units"][v], "long_name": ATTRS["long"][v]}
    ds.latitude.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds.longitude.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    ds.attrs = {"Conventions": "CF-1.8", "title": f"ERA5 India box: {name}", "source": source,
                "synthetic": "false", "history": f"created {dt.date.today()} by scripts/fetch_real.py"}
    out = REAL / "era5"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for grid in ("0p25", "g12", "g5"):
        d = ds if grid == "0p25" else regrid(ds, grid)
        d.attrs = dict(ds.attrs, grid=grid)
        p = out / f"{name}_era5_{grid}.nc"
        d.to_netcdf(p, encoding=packed_encoding(d) if grid != "0p25" else
                    {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in d.data_vars})
        paths.append(p)
    return paths


PACK = {"t2m": (0.01, 273.15), "u10": (0.01, 0.0), "v10": (0.01, 0.0), "msl": (1.0, 100000.0),
        "tp": (0.01, 0.0), "q850": (1e-6, 0.0), "u850": (0.01, 0.0), "v850": (0.01, 0.0),
        "r850": (0.01, 0.0)}


def packed_encoding(d):
    """int16 packing for the (large) regridded files: 0.01 K, 0.01 m/s, 1 Pa, 0.01 mm, 1e-6 kg/kg."""
    return {v: {"zlib": True, "complevel": 4, "dtype": "int16", "scale_factor": PACK[v][0],
                "add_offset": PACK[v][1], "_FillValue": -32768} for v in d.data_vars}


def repack():
    for p in sorted((REAL / "era5").glob("*_g*.nc")):
        d = xr.open_dataset(p).load()
        d.close()
        before = p.stat().st_size / 1e6
        tmp = p.with_suffix(".tmp.nc")
        d.to_netcdf(tmp, encoding=packed_encoding(d))
        tmp.replace(p)
        print(f"repacked {p.name}: {before:.0f} -> {p.stat().st_size / 1e6:.0f} MB")


# ----------------------------------------------------------------------------- ERA5
def era5_event(name, start, end, store, with_850):
    t0 = time.time()
    if store == "wb2":
        ds = open_store(WB2_ERA5)
        sel = ds[list(SFC) + ["total_precipitation_6hr"]].sel(time=slice(start, end), **BOX)
        out = sel.rename({**SFC, "total_precipitation_6hr": "tp"}).load()
        out["tp"] = (out["tp"] * 1000.0).clip(min=0)  # m -> mm
        if with_850:
            pl = ds[list(PL850)].sel(time=slice(start, end), level=850, **BOX).drop_vars("level")
            out = out.merge(pl.rename(PL850).load())
        url = f"{GCS}/{WB2_ERA5}"
    else:
        ds = open_store(ARCO)
        inst = ds[list(SFC)].sel(time=pd.date_range(start, end, freq="6h"), **BOX)
        out = inst.rename(SFC).load()
        # 6-h accumulation ending at t = sum of the 6 hourly accumulations (t-5h .. t)
        hourly = ds["total_precipitation"].sel(
            time=slice(pd.Timestamp(start) - pd.Timedelta("5h"), end), **BOX).load()
        tp6 = hourly.rolling(time=6).sum().sel(time=out.time)
        out["tp"] = (tp6 * 1000.0).clip(min=0)
        url = f"{GCS}/{ARCO}"
    paths = finish(out, name, f"ERA5 ({'WeatherBench 2' if store == 'wb2' else 'ARCO-ERA5'})")
    log(f"ERA5 {name}", url, "OK", dir_mb(*paths), f"{start} .. {end} (6-hourly)",
        f"vars {sorted(out.data_vars)}; {len(out.time)} steps; {time.time() - t0:.0f}s; "
        f"files {[p.name for p in paths]}")


def era5():
    # The path given in the brief does not exist; record that and the gcsfs hang.
    log("ERA5 WB2 (path from brief)", f"gs://{WB2_ERA5_BRIEF}", "FAILED",
        note="HTTP 404 on .zmetadata: store name lacks 'wb13-'. Used "
             f"gs://{WB2_ERA5} instead (same product, 13 levels).")
    log("ERA5 WB2 via gcsfs token=anon", f"gs://{WB2_ERA5}", "FAILED",
        note="xr.open_zarr via gcsfs hung >180 s on Windows in 2 attempts; switched to the "
             "same public bucket over plain HTTPS (storage.googleapis.com) with fsspec.")
    for name, (s, e, store, w850) in EVENTS.items():
        src = f"ERA5 {name}"
        try:
            attempt(lambda: era5_event(name, s, e, store, w850))
        except Exception as ex:  # noqa: BLE001
            log(src, store, "FAILED", note=f"{type(ex).__name__}: {ex}")
    for name in ("coldwave", "heatwave"):
        log(f"ERA5 {name} 850 hPa q/u/v", f"{GCS}/{ARCO}", "SKIPPED",
            note="ARCO 3-D chunks hold all 37 levels of the full globe (~100 MB per step); "
                 "skipped to stay within the bandwidth/time budget. Surface fields only.")


def rh850():
    """Add 850-hPa relative humidity to the Amphan files (needed for the RH <= 100% rule)."""
    t0 = time.time()
    s, e = EVENTS["amphan"][:2]
    ds = open_store(WB2_ERA5)
    rh = ds["relative_humidity"].sel(time=slice(s, e), level=850, **BOX).drop_vars("level")
    rh = attempt(lambda: rh.load())
    if float(rh.max()) < 5:  # WB2 derived RH is a fraction -> percent
        rh = rh * 100.0
    rh = rh.sortby("latitude").rename("r850")
    rh.attrs = {"units": "%", "long_name": "Relative humidity at 850 hPa (WB2 derived)"}
    base = REAL / "era5" / "amphan_era5_0p25.nc"
    d = xr.open_dataset(base).load()
    d["r850"] = rh.astype("float32")
    d.close()
    d = d.drop_vars([v for v in d.data_vars if v not in ATTRS["units"] and v != "r850"])
    paths = finish(d, "amphan", "ERA5 (WeatherBench 2)")
    log("ERA5 amphan 850 hPa relative humidity", f"{GCS}/{WB2_ERA5}", "OK", None,
        f"{s} .. {e} (6-hourly)", f"r850 added to amphan files; {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- climatology
WINDOWS = {"amphan": ("2020-05-10", "2020-05-25"), "coldwave": ("2022-12-20", "2023-01-20"),
           "heatwave": ("2024-05-15", "2024-06-20")}
CLIM_VARS = {"2m_temperature": "t2m", "mean_sea_level_pressure": "msl",
             "total_precipitation_6hr": "tp"}


def needed_doys(pad=0):
    doys = set()
    for s, e in WINDOWS.values():
        for d in pd.date_range(pd.Timestamp(s) - pd.Timedelta(days=pad),
                               pd.Timestamp(e) + pd.Timedelta(days=pad + 10)):
            doys.add(d.dayofyear)  # +10 d covers the 240-h forecast horizon
    return sorted(doys)


def clim():
    t0 = time.time()
    doys = needed_doys()
    ds = open_store(WB2_CLIM)
    mean = ds[list(CLIM_VARS)].sel(dayofyear=doys, **BOX).rename(CLIM_VARS)
    mean = attempt(lambda: mean.load())
    mean["tp"] = (mean["tp"] * 1000).clip(min=0)
    # Std: WB2 hourly climatology has no std, so compute it from the 1.5 deg WB2 ERA5
    # (1990-2019, same years), per (hour, doy) with a +-15-day window, then interpolate.
    co = open_store(WB2_COARSE)[list(CLIM_VARS)].rename(CLIM_VARS)
    co = co.sel(time=slice("1990-01-01", "2019-12-31T18"),
                latitude=slice(-3, 43), longitude=slice(57, 103))
    months = co.time.dt.month.isin([4, 5, 6, 7, 11, 12, 1, 2])
    co = attempt(lambda: co.sel(time=months).load())
    co["tp"] = co["tp"] * 1000
    std = {}
    tdoy, thour = co.time.dt.dayofyear.values, co.time.dt.hour.values
    for v in CLIM_VARS.values():
        arr = co[v].transpose("time", "latitude", "longitude").values
        out = np.full((4, len(doys), arr.shape[1], arr.shape[2]), np.nan, np.float32)
        for hi, h in enumerate((0, 6, 12, 18)):
            for di, d in enumerate(doys):
                dd = np.abs(((tdoy - d + 183) % 366) - 183)
                m = (thour == h) & (dd <= 15)
                out[hi, di] = arr[m].std(axis=0)
        std[v] = (("hour", "dayofyear", "latitude", "longitude"), out)
    std = xr.Dataset(std, coords={"hour": [0, 6, 12, 18], "dayofyear": doys,
                                  "latitude": co.latitude.values, "longitude": co.longitude.values})
    std = std.sortby("latitude").interp(latitude=mean.latitude, longitude=mean.longitude)
    out = xr.Dataset({f"{v}_mean": mean[v] for v in CLIM_VARS.values()} |
                     {f"{v}_std": std[v].astype("float32") for v in CLIM_VARS.values()})
    out = out.sortby("latitude")
    out.attrs = {"Conventions": "CF-1.8", "synthetic": "false",
                 "title": "ERA5 1990-2019 climatology, India box, event day-of-year windows",
                 "mean_source": f"{GCS}/{WB2_CLIM}",
                 "std_source": f"{GCS}/{WB2_COARSE} (1.5 deg, +-15 d window, interpolated)"}
    d = REAL / "clim"
    d.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out.data_vars}
    out.to_netcdf(d / "era5_clim_0p25.nc", encoding=enc)
    g12 = regrid(out, "g12")
    g12.attrs = dict(out.attrs, grid="g12")
    g12.to_netcdf(d / "era5_clim_g12.nc", encoding=enc)
    log("ERA5 climatology (mean)", f"{GCS}/{WB2_CLIM}", "OK", dir_mb(d),
        "1990-2019, day-of-year " + f"{len(doys)} days", f"{time.time() - t0:.0f}s; vars t2m,msl,tp mean")
    log("ERA5 climatology (std)", f"{GCS}/{WB2_COARSE}", "OK", None, "1990-2019",
        "std per hour/doy from 1.5 deg ERA5 with a +-15-day window; interpolated to 0.25/G12. "
        "G5 climatology not written (anomalies are computed at 12 km).")


# ----------------------------------------------------------------------------- IBTrACS
def ibtracs():
    d = REAL / "ibtracs"
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "ibtracs.NI.list.v04r01.csv"

    def get():
        r = requests.get(IBTRACS, timeout=120)
        r.raise_for_status()
        raw.write_bytes(r.content)
    attempt(get)
    df = pd.read_csv(raw, skiprows=[1], low_memory=False, keep_default_na=False)
    am = df[(df.NAME == "AMPHAN") & (df.SEASON.astype(str) == "2020")].copy()
    keep = ["SID", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_WIND",
            "USA_PRES", "USA_RMW", "NEWDELHI_WIND", "NEWDELHI_PRES", "USA_SSHS", "DIST2LAND"]
    am = am[[c for c in keep if c in am.columns]]
    am.to_csv(d / "amphan_2020_ibtracs.csv", index=False)
    # compact NI track library (1990-2023) for the synthetic track sampler
    ni = df[pd.to_numeric(df.SEASON, errors="coerce").between(1990, 2023)]
    ni = ni[["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "USA_WIND", "WMO_WIND",
             "NEWDELHI_WIND"]]
    ni.to_csv(d / "ni_tracks_1990_2023.csv", index=False)
    log("IBTrACS v04r01 North Indian", IBTRACS, "OK", dir_mb(d),
        f"Amphan {am.ISO_TIME.iloc[0]} .. {am.ISO_TIME.iloc[-1]}",
        f"{len(am)} Amphan fixes; {ni.SID.nunique()} NI storms 1990-2023 kept for track sampling")


# ----------------------------------------------------------------------------- DEM
def dem():
    import rasterio
    from rasterio.enums import Resampling
    d = REAL / "dem"
    d.mkdir(parents=True, exist_ok=True)
    tiles_txt = requests.get(f"{DEM_BASE}/tileList.txt", timeout=60)
    tiles_txt.raise_for_status()
    have = set(tiles_txt.text.split())
    # Read each 1x1 deg tile at a decimated resolution (COG overviews) -> 0.01 deg mosaic
    res = 0.01
    ny, nx = int(40 / res), int(40 / res)
    mosaic = np.zeros((ny, nx), np.float32)  # sea / missing tiles = 0 m
    n_ok = 0
    from concurrent.futures import ThreadPoolExecutor

    def read(lat, lon):
        name = f"Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM"
        if name not in have:
            return lat, lon, None
        url = f"/vsicurl/{DEM_BASE}/{name}/{name}.tif"
        for _ in range(2):
            try:
                with rasterio.open(url) as src:
                    a = src.read(1, out_shape=(100, 100), resampling=Resampling.average)
                return lat, lon, a.astype(np.float32)
            except Exception:  # noqa: BLE001
                time.sleep(2)
        return lat, lon, "fail"

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    fails = 0
    cache = d / "raw" / "mosaic_0p01.npy"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        mosaic, n_ok = np.load(cache), -1
    with ThreadPoolExecutor(16) as ex:
        for lat, lon, a in ([] if n_ok == -1 else ex.map(
                lambda ll: read(*ll), [(la, lo) for la in range(0, 40) for lo in range(60, 100)])):
            if a is None:
                continue
            if isinstance(a, str):
                fails += 1
                continue
            i0, j0 = (lat - 0) * 100, (lon - 60) * 100
            mosaic[i0:i0 + 100, j0:j0 + 100] = np.flipud(a)  # tile rows run north->south
            n_ok += 1
    if n_ok == 0:
        raise RuntimeError("no DEM tiles read")
    np.save(cache, mosaic)
    lat = 0 + res * (np.arange(ny) + 0.5)
    lon = 60 + res * (np.arange(nx) + 0.5)
    da = xr.DataArray(np.maximum(mosaic, -50), dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": lon}, name="orog",
                      attrs={"units": "m", "long_name": "surface elevation (Copernicus GLO-90)"})
    from synth.grids import avgpool
    ds = xr.Dataset({"orog": da}, attrs={"Conventions": "CF-1.8", "synthetic": "false",
                                         "source": f"{DEM_BASE} (Copernicus DEM GLO-90)"})
    # 0.01 -> G5 (0.04 = 4x4 block mean) -> G12 (3x3 block mean of G5): conservative
    g5 = avgpool(mosaic[:3996, :3996], 4)  # G5 spans 0..39.96 deg
    g12 = avgpool(g5, 3)
    from synth.grids import LAT5, LON5, LAT12, LON12
    for name, arr, la, lo in (("g5", g5, LAT5, LON5), ("g12", g12, LAT12, LON12)):
        o = xr.Dataset({"orog": (("latitude", "longitude"), np.maximum(arr, 0).astype(np.float32),
                                 da.attrs)}, coords={"latitude": la, "longitude": lo},
                       attrs=dict(ds.attrs, grid=name))
        o.to_netcdf(d / f"dem_{name}.nc", encoding={"orog": {"zlib": True}})
    log("Copernicus DEM GLO-90", f"{DEM_BASE} (s3://copernicus-dem-90m)", "OK", dir_mb(d),
        "static", f"{'cached' if n_ok == -1 else n_ok} land tiles read via COG overviews at 0.01 deg; {fails} tile reads "
                  "failed (set to 0 m); block-averaged to G5 and G12")


# ----------------------------------------------------------------------------- IMD
def imd():
    import imdlib
    d = REAL / "imd"
    raw = d / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    got = []
    for yr in (2020, 2024):
        def get():
            data = imdlib.get_data("rain", yr, yr, fn_format="yearwise", file_dir=str(raw))
            ds = data.get_xarray()
            if "rain" not in ds or ds.sizes.get("time", 0) < 300:
                raise RuntimeError(f"unexpected IMD content {dict(ds.sizes)}")
            return ds
        try:
            ds = attempt(get)
        except Exception as ex:  # noqa: BLE001
            log(f"IMD 0.25 rainfall {yr}", "imdlib.get_data('rain')", "FAILED",
                note=f"{type(ex).__name__}: {ex}")
            continue
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
        rain = ds["rain"].where(ds["rain"] > -900).astype("float32")
        rain.attrs = {"units": "mm day-1", "long_name": "IMD gridded daily rainfall"}
        out = rain.to_dataset(name="rain")
        out.attrs = {"Conventions": "CF-1.8", "synthetic": "false",
                     "source": "IMD Pune 0.25 deg gridded rainfall via imdlib"}
        out.to_netcdf(d / f"imd_rain_{yr}_0p25.nc", encoding={"rain": {"zlib": True}})
        g12 = regrid(out, "g12")
        g12.attrs = dict(out.attrs, grid="g12")
        g12.to_netcdf(d / f"imd_rain_{yr}_g12.nc", encoding={"rain": {"zlib": True}})
        got.append(yr)
        log(f"IMD 0.25 rainfall {yr}", "https://imdpune.gov.in (via imdlib)", "OK",
            dir_mb(d / f"imd_rain_{yr}_0p25.nc", d / f"imd_rain_{yr}_g12.nc"),
            f"{yr}-01-01 .. {yr}-12-31 daily", "native 0.25 + G12 (bilinear; G5 not written)")


# ----------------------------------------------------------------------------- keyed sources
def keyed():
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for ln in p.read_text().splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                env[k.strip()] = v.strip()
    for src, key, url in (("TIGGE NEPS-G (origin dems)", "ECMWF_API_KEY",
                           "https://apps.ecmwf.int/datasets/data/tigge"),
                          ("GPM IMERG daily 0.1 deg", "EARTHDATA_TOKEN",
                           "https://gpm.nasa.gov/data/imerg"),
                          ("CDS ERA5-Land", "CDSAPI_KEY", "https://cds.climate.copernicus.eu")):
        if not env.get(key):
            log(src, url, "SKIPPED", note=f"SKIPPED - no key ({key} not in .env)")
        else:
            log(src, url, "SKIPPED", note="key present but downloader not implemented in this phase")


STEPS = {"era5": era5, "rh850": rh850, "repack": repack, "clim": clim, "ibtracs": ibtracs, "dem": dem, "imd": imd, "keyed": keyed}

if __name__ == "__main__":
    REAL.mkdir(parents=True, exist_ok=True)
    for step in (sys.argv[1:] or list(STEPS)):
        print(f"=== {step}")
        try:
            STEPS[step]()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            log(step, "", "FAILED", note=f"{type(ex).__name__}: {ex}")
