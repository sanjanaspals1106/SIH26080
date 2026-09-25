"""Multi-season data flow on synthetic files: hand-downloaded TIGGE layout, IMD as NetCDF, golden -> features
without districts, and the regime fields. Nothing is downloaded; the GRIB and IMD files are synthetic."""

import dataclasses
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import regime_engine.pipeline as rp
from data_pipeline.alignment import (
    atmosphere_fullgrid_windows,
    build_golden_season,
    get_valid_cells,
    read_golden,
)
from data_pipeline.features import build_stage3_season, read_features
from data_pipeline.ingestion import (
    download_tigge_month,
    download_tigge_static,
    load_config,
    read_atmosphere_month,
    read_month,
    read_static_from_grib,
    tigge_raw_paths,
)
from data_pipeline.ingestion.imd import read_imd_year
from data_pipeline.ingestion.synthetic import SyntheticClient, write_synthetic_imd_year


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Official-layout synthetic raw files, the same files renamed to the hand-downloaded layout, and IMD 2024
    kept only as a NetCDF file (no .grd), so every step below reads the formats a real download will bring."""
    cfg = load_config(data_dir=tmp_path_factory.mktemp("data"))
    cfg = dataclasses.replace(cfg, alignment=dataclasses.replace(cfg.alignment, static_date="2024-06-01"))
    client = SyntheticClient(cfg, max_dates=2)
    download_tigge_month(2024, 7, client=client, config=cfg)
    download_tigge_static(date(2024, 6, 1), config=cfg, client=client)

    legacy = tmp_path_factory.mktemp("legacy") / "2024"
    legacy.mkdir()
    shutil.copy(tigge_raw_paths("rain", "202407", cfg)[0], legacy / "tp_202407.grib")
    atm = tigge_raw_paths("atmosphere", "202407", cfg)
    sfc = next(p for p in atm if "sfc" in p.name)
    with open(legacy / "pressure_07.grib", "wb") as out:  # both pressure levels in one file, as downloaded by hand
        for p in atm:
            if p is not sfc:
                out.write(p.read_bytes())
    shutil.copy(sfc, legacy / "single_07.grib")

    write_synthetic_imd_year(cfg, 2024, missing_fraction=0.0, missing_cells_fraction=0.4)
    ds = read_imd_year(cfg.imd.year_path(2024), 2024, cfg)
    (cfg.data_dir / "imd").mkdir(parents=True, exist_ok=True)
    nc_desc = xr.Dataset({"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"), ds["rain"].values[:, ::-1, :])},
                         coords={"TIME": ds["time"].values, "LATITUDE": ds["lat"].values[::-1], "LONGITUDE": ds["lon"].values})
    nc_desc.to_netcdf(cfg.data_dir / "imd" / "RF25_ind2024_rfp25.nc")
    official_grd = cfg.imd.year_path(2024)
    grid = get_valid_cells(cfg, years=[2024])  # via the .grd, once; the mask is what golden needs
    official_grd.unlink()  # from here on IMD 2024 exists only as NetCDF
    return cfg, legacy, grid


def test_month_reader_accepts_both_layouts(world):
    cfg, legacy, _ = world
    tp_o, atm_o = read_month(2024, 7, cfg)
    tp_l, atm_l = read_month(2024, 7, cfg, tigge_dir=legacy.parent)
    xr.testing.assert_equal(tp_o, tp_l)
    xr.testing.assert_equal(atm_o, atm_l)
    xr.testing.assert_equal(atm_l, read_atmosphere_month(2024, 7, cfg, legacy.parent))


def test_static_fields_from_any_grib(world):
    cfg, *_ = world
    static = read_static_from_grib(tigge_raw_paths("static", "20240601", cfg)[0])
    assert set(static.data_vars) == {"orog", "lsm"} and static["orog"].dims == ("lat", "lon")
    assert (np.diff(static["lat"].values) > 0).all()


def test_golden_from_legacy_layout_and_netcdf_imd_equals_the_official_path(world):
    cfg, legacy, grid = world
    static = read_static_from_grib(tigge_raw_paths("static", "20240601", cfg)[0])
    assert not cfg.imd.year_path(2024).exists()  # IMD is read from the NetCDF file
    build_golden_season(2024, cfg, months=[7], static=static, tigge_dir=legacy.parent, grid=grid)
    legacy_df = read_golden(cfg, seasons=[2024]).copy()
    build_golden_season(2024, cfg, months=[7], grid=grid)  # official layout
    official_df = read_golden(cfg, seasons=[2024])
    pd.testing.assert_frame_equal(legacy_df, official_df)
    assert legacy_df["obs_mm"].notna().all() and len(legacy_df) == 2 * 3 * int(grid["is_valid"].sum())


def test_features_can_be_built_without_a_district_file(world):
    cfg, legacy, grid = world
    static = read_static_from_grib(tigge_raw_paths("static", "20240601", cfg)[0])
    build_golden_season(2024, cfg, months=[7], static=static, tigge_dir=legacy.parent, grid=grid)
    # climatology of "training seasons" read from IMD NetCDF (2024 stands in for a training year)
    out = build_stage3_season(2024, [2024], cfg, static=static, skip_districts=True)
    assert len(out["features"]) == 1 and out["forecasts"] == [] and out["history"] == []
    assert len(read_features(cfg, seasons=[2024])) == 2 * 3 * int(grid["is_valid"].sum())
    assert not (cfg.features.dir / "district_forecasts").exists()


def test_regime_fields_and_a_features_from_grib(world):
    cfg, legacy, grid = world
    path = rp.build_season_fields(2024, cfg, tigge_dir=legacy.parent, months=[7])
    fields = rp.read_season_fields(2024, cfg)
    assert path.name == "fields_2024.nc"
    assert dict(fields.sizes) == {"init_time": 2, "lead_day": 3, "lat": 129, "lon": 135}
    assert set(fields.data_vars) == set(rp.FIELD_VARS)
    assert np.isfinite(fields.to_array().values).all()  # the whole grid, ocean included
    # the same numbers as the golden atmosphere columns at the valid cells
    _, atm = read_month(2024, 7, cfg, legacy.parent)
    full = atmosphere_fullgrid_windows(atm, cfg)
    xr.testing.assert_allclose(fields, full, rtol=1e-6)
    golden = read_golden(cfg, seasons=[2024])
    one = golden[(golden["run_id"] == golden["run_id"].iloc[0]) & (golden["lead_day"] == 2)]
    got = full["msl"].isel(init_time=0, lead_day=1).values.ravel()[one["cell_id"].to_numpy()]
    np.testing.assert_allclose(got, one["msl"].to_numpy(), rtol=1e-5)
    a = rp.lead_a_table(fields, cfg)
    assert len(a) == 2 * 3 and a[rp.A_IDS].notna().all().all()
