"""A small synthetic world for the Stage 3 tests: a Golden Dataset whose fields are simple functions of the cell
position, so every feature has a value that can be worked out by hand. Nothing here is real data."""

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from data_pipeline.alignment import GOLDEN_COLUMNS, GOLDEN_SCHEMA, imd_grid
from data_pipeline.ingestion.imd import imd_axes

RUNS = ["tigge_ecmwf_cf_2024070100", "tigge_ecmwf_cf_2024070200"]
GY, GX = 0.3, 0.2  # rain grows by 0.3 mm per cell northwards and 0.2 mm per cell eastwards
M, K = 0.05, 0.02  # u850 falls by M m/s per cell northwards; v850 grows by K*cos(lat) m/s per cell eastwards
C1, C2 = 40.0, 10.0  # orography grows by 40 m per cell northwards and 10 m per cell eastwards
N_SEA_COLS = 20  # columns 0..19 of the IMD grid are sea


def make_grid(cfg):
    """IMD grid table; valid cells = a box (lat 10-30, lon 70-90) with a 3 x 3 hole, like sea/outside-India cells."""
    grid = imd_grid(cfg)
    lat, lon = grid["latitude"].to_numpy(), grid["longitude"].to_numpy()
    valid = (lat >= 10) & (lat <= 30) & (lon >= 70) & (lon <= 90)
    valid &= ~((lat >= 20) & (lat <= 20.5) & (lon >= 80) & (lon <= 80.5))
    grid["is_valid"] = valid
    grid["valid_fraction"] = np.where(valid, 1.0, 0.0).astype("float32")
    return grid


def make_static_grid(cfg):
    """orog and lsm on the IMD grid (Stage 2 `align_static` output shape): linear terrain, sea in the west."""
    lat, lon = imd_axes(cfg.imd)
    ilat, ilon = np.meshgrid(np.arange(lat.size), np.arange(lon.size), indexing="ij")
    orog = (100.0 + C1 * ilat + C2 * ilon).astype("float32")
    lsm = np.where(ilon < N_SEA_COLS, 0.0, 1.0).astype("float32")
    return xr.Dataset(
        {"orog": (("lat", "lon"), orog), "lsm": (("lat", "lon"), lsm)}, coords={"lat": lat, "lon": lon}
    )


def make_golden(cfg, grid, runs=RUNS, leads=(1, 2, 3), obs_nan=lambda run_i, cell: False):
    """Golden Dataset rows (schema of Stage 2) for `runs` x `leads` x valid cells."""
    valid = grid.loc[grid["is_valid"]]
    cell = valid["cell_id"].to_numpy()
    ilat, ilon = cell // cfg.imd.n_lon, cell % cfg.imd.n_lon
    lat, lon = valid["latitude"].to_numpy().astype("float64"), valid["longitude"].to_numpy()
    frames = []
    for r_i, run in enumerate(runs):
        init = pd.Timestamp(run[-10:-2])
        for lead in leads:
            rain = 10.0 * lead + 3.0 * r_i + GY * ilat + GX * ilon
            u850 = 5.0 + 0.5 * lead - M * ilat
            v850 = K * ilon * np.cos(np.radians(lat))
            obs = rain * 1.1
            obs[[obs_nan(r_i, int(c)) for c in cell]] = np.nan
            imd_date = init.normalize() + pd.Timedelta(days=lead)
            n = len(cell)
            frames.append(
                pd.DataFrame(
                    {
                        "run_id": run,
                        "source": "ECMWF-TIGGE-control",
                        "initialization_time": pd.Timestamp(init, tz="UTC"),
                        "season": init.year,
                        "lead_day": lead,
                        "imd_date": imd_date,
                        "window_start_utc": pd.Timestamp(
                            init + pd.Timedelta(hours=24 * (lead - 1) + 3), tz="UTC"
                        ),
                        "window_end_utc": pd.Timestamp(init + pd.Timedelta(hours=24 * lead + 3), tz="UTC"),
                        "alignment_method": "C1",
                        "alignment_offset_hours": 0,
                        "imd_stamp": "end_date",
                        "unit": "mm",
                        "grid": "IMD_0.25",
                        "cell_id": cell,
                        "latitude": valid["latitude"].to_numpy(),
                        "longitude": lon,
                        "rain_mm": rain,
                        "obs_mm": obs,
                        "msl": 100000.0 - 10.0 * ilat,
                        "u850": u850,
                        "v850": v850,
                        "q850": 0.010 + 1e-5 * ilon,
                        "u200": u850 + 3.0,
                        "v200": v850 + 4.0,
                        "orog": 100.0 + C1 * ilat + C2 * ilon,
                        "lsm": 1.0,
                        "rain_clipped": False,
                        "obs_missing": np.isnan(obs),
                        "rain_regrid": "area_mean",
                    },
                    index=range(n),
                )
            )
    df = pd.concat(frames, ignore_index=True)[GOLDEN_COLUMNS]
    numeric = {
        f.name: f.type.to_pandas_dtype()
        for f in GOLDEN_SCHEMA
        if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)
    }
    return df.astype(numeric).sort_values(["run_id", "lead_day", "cell_id"]).reset_index(drop=True)


def make_climatology(cfg):
    cells = imd_grid(cfg)["cell_id"].to_numpy()
    return pd.DataFrame({"cell_id": cells, "clim_mean": np.float32(8.0), "clim_p95": np.float32(35.0)})


def make_tigge_static(shift=0.0):
    """Static fields on the TIGGE grid (N40 W55 S0 E100, 0.25 degree) as Stage 1 reads them; the IMD-grid values
    equal `make_static_grid` (same linear terrain, sea for the first 20 IMD columns)."""
    lat, lon = np.arange(0, 40.01, 0.25), np.arange(55, 100.01, 0.25)
    i_lat = (lat - 6.5) / 0.25
    i_lon = (lon - 66.5) / 0.25
    orog = 100.0 + C1 * i_lat[:, None] + C2 * i_lon[None, :] + shift
    lsm = np.where(i_lon[None, :] < N_SEA_COLS, 0.0, 1.0) * np.ones((lat.size, 1))
    return xr.Dataset(
        {"orog": (("lat", "lon"), orog.astype("float32")), "lsm": (("lat", "lon"), lsm.astype("float32"))},
        coords={"lat": lat, "lon": lon},
    )
