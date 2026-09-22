"""Golden Dataset: schema, granularity, values, determinism, files (PRD 9.3, 9.7). Small in-memory world."""

import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from data_pipeline.alignment import (
    GOLDEN_COLUMNS,
    GOLDEN_SCHEMA,
    align_static,
    golden_rows,
    imd_grid,
    read_golden,
    validate_golden,
    write_golden,
)
from data_pipeline.alignment.lag import lag_test_season
from data_pipeline.ingestion import ValidationError
from data_pipeline.ingestion.imd import imd_axes

RATES = np.arange(1.0, 14.0)  # mm/h in each 6 h block of the forecast, as in the T3 test
ATM_STEPS = list(range(6, 73, 6))
TP_STEPS = list(range(0, 79, 6))
INITS = pd.to_datetime(["2024-07-01", "2024-07-02"])
LAT, LON = np.arange(0, 40.01, 0.25), np.arange(55, 100.01, 0.25)  # TIGGE 0.25 degree, N40 W55 S0 E100


def hourly_sum(lo, hi):
    return sum(RATES[h // 6] for h in range(lo, hi))


def cell_factor(lat, lon):
    return 1.0 + lat / 100.0 + lon / 1000.0


def make_tp():
    cum = np.concatenate([[0.0], np.cumsum(6 * RATES)])
    data = cum[None, :, None, None] * cell_factor(LAT[:, None], LON[None, :])[None, None]
    data = np.broadcast_to(data, (2, len(TP_STEPS), LAT.size, LON.size)).astype("float32").copy()
    data[1] *= 1.5  # the second start date rains 50% more
    return xr.DataArray(
        data,
        dims=("init_time", "lead_hours", "lat", "lon"),
        coords={"init_time": INITS, "lead_hours": TP_STEPS, "lat": LAT, "lon": LON},
        name="tp",
    )


def make_atm():
    """Each variable = step + lat/100 + 10*variable index, so window means are known by hand."""
    ds = {}
    for n, name in enumerate(["msl", "u850", "v850", "q850", "u200", "v200"]):
        val = (
            np.array(ATM_STEPS)[None, :, None, None]
            + LAT[None, None, :, None] / 100
            + 10 * n
            + 0 * LON[None, None, None, :]
        )
        ds[name] = (
            ("init_time", "lead_hours", "lat", "lon"),
            np.broadcast_to(val, (2, len(ATM_STEPS), LAT.size, LON.size)).astype("float32").copy(),
        )
    return xr.Dataset(ds, coords={"init_time": INITS, "lead_hours": ATM_STEPS, "lat": LAT, "lon": LON})


def make_static():
    return xr.Dataset(
        {
            "orog": (("lat", "lon"), (LAT[:, None] * 10 + LON[None, :]).astype("float32")),
            "lsm": (("lat", "lon"), (LON[None, :] / 100 + 0 * LAT[:, None]).astype("float32")),
        },
        coords={"lat": LAT, "lon": LON},
    )


def make_imd(cfg, until="2024-07-04"):
    """IMD field dated D: value = day of month + cell_id / 1e5. One cell is NaN on 2024-07-03."""
    lat, lon = imd_axes(cfg.imd)
    days = pd.date_range("2024-06-28", until)
    cell = np.arange(lat.size * lon.size).reshape(lat.size, lon.size)
    data = np.stack([d.day + cell / 1e5 for d in days]).astype("float32")
    data[list(days).index(pd.Timestamp("2024-07-03")), 10, 10] = np.nan
    return xr.DataArray(
        data, dims=("time", "lat", "lon"), coords={"time": days, "lat": lat, "lon": lon}, name="rain"
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from data_pipeline.ingestion import load_config

    cfg = load_config(data_dir=tmp_path_factory.mktemp("data"))
    grid = imd_grid(cfg)
    rng = np.random.default_rng(7)
    grid["is_valid"] = rng.random(len(grid)) < 0.3
    grid["valid_fraction"] = np.where(grid["is_valid"], 0.99, 0.5).astype("float32")
    grid.attrs["base_years"] = [1981, 2010]
    tp, atm = make_tp(), make_atm()
    static = align_static(make_static(), cfg)
    imd = make_imd(cfg)
    df = golden_rows(tp, atm, static, imd, grid, cfg)
    return cfg, grid, tp, atm, static, imd, df


# ---- schema and granularity ----------------------------------------------------------------


def test_schema_columns_and_types(world):
    cfg, grid, *_, df = world
    assert list(df.columns) == GOLDEN_COLUMNS
    table = pa.Table.from_pandas(
        df, schema=GOLDEN_SCHEMA, preserve_index=False
    )  # the typed schema accepts the frame
    assert table.schema.equals(GOLDEN_SCHEMA)
    for required in [
        "run_id",
        "lead_day",
        "cell_id",
        "season",
        "imd_date",
        "rain_mm",
        "obs_mm",
        "latitude",
        "longitude",
        "source",
        "initialization_time",
        "window_start_utc",
        "window_end_utc",
        "alignment_method",
        "alignment_offset_hours",
        "unit",
        "grid",
        "msl",
        "u850",
        "v850",
        "q850",
        "u200",
        "v200",
        "orog",
        "lsm",
    ]:  # PRD 9.3 and the data contract 9.7
        assert required in df.columns
    assert not {"p_active", "regime_confidence", "corrected_mean_mm", "q50_mm"} & set(
        df.columns
    )  # nothing from later stages


def test_row_granularity_and_no_duplicates(world):
    cfg, grid, *_, df = world
    n_valid = int(grid["is_valid"].sum())
    assert len(df) == 2 * 3 * n_valid  # 2 runs x 3 leads x valid cells
    assert not df.duplicated(["run_id", "lead_day", "cell_id"]).any()
    assert sorted(df["run_id"].unique()) == ["tigge_ecmwf_cf_2024070100", "tigge_ecmwf_cf_2024070200"]
    assert sorted(df["lead_day"].unique()) == [1, 2, 3]
    key = df[["run_id", "lead_day", "cell_id"]]
    assert key.equals(key.sort_values(["run_id", "lead_day", "cell_id"]))  # rows come out sorted


def test_only_valid_cells_are_present(world):
    _, grid, *_, df = world
    assert set(df["cell_id"]) == set(grid.loc[grid["is_valid"], "cell_id"])
    assert df["cell_id"].isin(grid.loc[~grid["is_valid"], "cell_id"]).sum() == 0


def test_t7_same_cell_id_same_coordinates_everywhere(world):
    cfg, grid, *_, df = world
    merged = df.drop_duplicates("cell_id").merge(grid, on="cell_id", suffixes=("", "_grid"))
    assert (merged["latitude"] == merged["latitude_grid"]).all() and (
        merged["longitude"] == merged["longitude_grid"]
    ).all()
    lat, lon = imd_axes(cfg.imd)
    row = df.iloc[0]
    assert (row["latitude"], row["longitude"]) == (lat[row["cell_id"] // 135], lon[row["cell_id"] % 135])


# ---- values --------------------------------------------------------------------------------


def row(df, run, lead, cell):
    return df[(df.run_id == run) & (df.lead_day == lead) & (df.cell_id == cell)].iloc[0]


def test_rain_windows_reach_the_golden_rows(world):
    cfg, grid, *_, df = world
    cell = int(grid.loc[grid["is_valid"], "cell_id"].iloc[3])
    lat, lon = imd_axes(cfg.imd)
    factor = cell_factor(lat[cell // 135], lon[cell % 135])
    for lead, (lo, hi) in {1: (3, 27), 2: (27, 51), 3: (51, 75)}.items():
        assert row(df, "tigge_ecmwf_cf_2024070100", lead, cell)["rain_mm"] == pytest.approx(
            hourly_sum(lo, hi) * factor, rel=1e-5
        )
        assert row(df, "tigge_ecmwf_cf_2024070200", lead, cell)["rain_mm"] == pytest.approx(
            1.5 * hourly_sum(lo, hi) * factor, rel=1e-5
        )
    assert row(df, "tigge_ecmwf_cf_2024070100", 1, cell)["rain_regrid"] == "area_mean"


def test_atmosphere_is_the_window_mean_on_the_imd_grid(world):
    cfg, grid, *_, df = world
    cell = int(grid.loc[grid["is_valid"], "cell_id"].iloc[3])
    lat = imd_axes(cfg.imd)[0][cell // 135]
    for lead, mean_step in {1: 15.0, 2: 39.0, 3: 63.0}.items():  # mean of steps 6..24, 30..48, 54..72
        r = row(df, "tigge_ecmwf_cf_2024070100", lead, cell)
        for n, name in enumerate(["msl", "u850", "v850", "q850", "u200", "v200"]):
            assert r[name] == pytest.approx(mean_step + lat / 100 + 10 * n, rel=1e-5)


def test_static_fields_are_on_the_imd_grid(world):
    cfg, grid, *_, df = world
    lat, lon = imd_axes(cfg.imd)
    r = df.iloc[100]
    assert r["orog"] == pytest.approx(r["latitude"] * 10 + r["longitude"], rel=1e-5)
    assert r["lsm"] == pytest.approx(r["longitude"] / 100, rel=1e-5)
    assert (
        df.groupby("cell_id")["orog"].nunique().max() == 1
    )  # static: one value per cell, every run and lead


def test_observed_rain_uses_the_imd_date_end_date_rule(world):
    cfg, grid, *_, df = world
    cell = int(grid.loc[grid["is_valid"], "cell_id"].iloc[3])
    r = row(df, "tigge_ecmwf_cf_2024070100", 1, cell)  # init 07-01, lead 1 -> IMD date 07-02
    assert r["imd_date"] == pd.Timestamp("2024-07-02") and r["obs_mm"] == pytest.approx(
        2 + cell / 1e5, rel=1e-6
    )
    assert row(df, "tigge_ecmwf_cf_2024070200", 1, cell)["imd_date"] == pd.Timestamp("2024-07-03")
    assert list(df.loc[df.run_id == "tigge_ecmwf_cf_2024070100", "imd_date"].unique()) == list(
        pd.to_datetime(["2024-07-02", "2024-07-03", "2024-07-04"])
    )


def test_missing_observations_are_flagged_not_dropped(world):
    cfg, grid, *_, df = world
    late = df[
        (df.run_id == "tigge_ecmwf_cf_2024070200") & (df.lead_day == 3)
    ]  # IMD date 07-05: not in the IMD data
    assert late["obs_mm"].isna().all() and late["obs_missing"].all()
    assert len(late) == int(grid["is_valid"].sum())  # the rows are kept
    nan_cell = 10 * 135 + 10  # NaN on 2024-07-03 (init 07-02, lead 1)
    if grid.loc[nan_cell, "is_valid"]:
        assert row(df, "tigge_ecmwf_cf_2024070200", 1, nan_cell)["obs_missing"]
    assert not df.loc[df.obs_mm.notna(), "obs_missing"].any()


def test_time_and_alignment_fields(world):
    *_, df = world
    r = df[(df.run_id == "tigge_ecmwf_cf_2024070100") & (df.lead_day == 2)].iloc[0]
    assert r["initialization_time"] == pd.Timestamp("2024-07-01 00:00", tz="UTC")
    assert r["window_start_utc"] == pd.Timestamp("2024-07-02 03:00", tz="UTC")  # 24(k-1)+3 = 27 h
    assert r["window_end_utc"] == pd.Timestamp("2024-07-03 03:00", tz="UTC")  # 24k+3 = 51 h
    assert (r["alignment_method"], r["alignment_offset_hours"], r["imd_stamp"]) == ("C1", 0, "end_date")
    assert (r["unit"], r["grid"], r["source"], r["season"]) == ("mm", "IMD_0.25", "ECMWF-TIGGE-control", 2024)


def test_start_date_stamp_shifts_the_imd_date_by_one_day(world):
    cfg, grid, tp, atm, static, imd, df = world
    alt = golden_rows(tp, atm, static, imd, grid, cfg, imd_stamp="start_date")
    a = alt[(alt.run_id == "tigge_ecmwf_cf_2024070100") & (alt.lead_day == 1)].iloc[0]
    assert a["imd_date"] == pd.Timestamp("2024-07-01") and a["imd_stamp"] == "start_date"
    assert alt["rain_mm"].equals(df["rain_mm"])  # the forecast side does not depend on imd_stamp


def test_start_times_missing_in_one_source_are_dropped(world):
    cfg, grid, tp, atm, static, imd, _ = world
    out = golden_rows(tp, atm.isel(init_time=[0]), static, imd, grid, cfg)
    assert list(out["run_id"].unique()) == ["tigge_ecmwf_cf_2024070100"]


# ---- checks --------------------------------------------------------------------------------


def test_validate_golden_rejects_bad_frames(world):
    *_, df = world
    validate_golden(df)
    with pytest.raises(ValidationError, match="columns differ"):
        validate_golden(df.drop(columns="msl"))
    with pytest.raises(ValidationError, match="duplicate"):
        validate_golden(pd.concat([df, df.iloc[:5]], ignore_index=True))
    with pytest.raises(ValidationError, match="rain_mm has 1 empty"):
        validate_golden(df.assign(rain_mm=df["rain_mm"].where(df.index != 0)))
    with pytest.raises(ValidationError, match="rain_mm outside"):
        validate_golden(df.assign(rain_mm=df["rain_mm"].where(df.index != 0, -1.0).astype("float32")))
    with pytest.raises(ValidationError, match="rain_mm outside"):
        validate_golden(df.assign(rain_mm=df["rain_mm"].where(df.index != 0, 1000.0).astype("float32")))


def test_forecast_outside_the_domain_is_not_silently_written(world):
    cfg, grid, tp, atm, static, imd, _ = world
    small = tp.sel(lat=slice(0, 30))  # IMD cells north of 30 N would be empty
    with pytest.raises(ValidationError, match="empty values"):
        golden_rows(small, atm, static, imd, grid, cfg)


# ---- determinism and files -----------------------------------------------------------------


def test_output_is_deterministic(world):
    cfg, grid, tp, atm, static, imd, df = world
    again = golden_rows(tp, atm, static, imd, grid, cfg)
    pd.testing.assert_frame_equal(df, again)


def test_written_files_are_stable_partitioned_and_read_back_identically(world, tmp_path):
    cfg, *_, df = world
    cfg2 = dataclasses.replace(
        cfg, alignment=dataclasses.replace(cfg.alignment, golden_dir=tmp_path / "golden")
    )
    paths = write_golden(df, cfg2)
    assert [p.relative_to(tmp_path / "golden").as_posix() for p in paths] == [
        "season_2024/golden_202407.parquet"
    ]
    digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    write_golden(df, cfg2)  # a second run replaces the month: same bytes, no duplicates
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == digest
    back = read_golden(cfg2)
    assert len(back) == len(df) and not back.duplicated(["run_id", "lead_day", "cell_id"]).any()
    pd.testing.assert_frame_equal(back, df, check_dtype=False)  # same values, NaNs included
    assert (
        back["rain_mm"].dtype == np.float32
        and back["lead_day"].dtype == np.int8
        and back["cell_id"].dtype == np.int32
    )
    assert list(read_golden(cfg2, seasons=[2024]).columns) == GOLDEN_COLUMNS
    with pytest.raises(Exception, match="No Golden Dataset files"):
        read_golden(cfg2, seasons=[1999])


def test_lag_test_on_a_written_season(world, tmp_path):
    """T4 on the Golden Dataset: an IMD field equal to the lead-1 window dated I + 1 gives shift 0 -> end_date."""
    cfg, grid, *_, df = world
    cfg2 = dataclasses.replace(
        cfg, alignment=dataclasses.replace(cfg.alignment, golden_dir=tmp_path / "golden")
    )
    write_golden(df, cfg2)
    from data_pipeline.alignment import save_valid_cells

    save_valid_cells(grid, cfg2)
    n_lat, n_lon = cfg.imd.n_lat, cfg.imd.n_lon
    days = pd.date_range("2024-06-28", "2024-07-06")
    rng = np.random.default_rng(3)
    imd = rng.gamma(0.6, 8.0, size=(len(days), n_lat, n_lon)).astype("float32")
    lead1 = df[df.lead_day == 1]
    for run, sub in lead1.groupby("run_id"):
        d = sub["imd_date"].iloc[0]
        field = np.zeros(n_lat * n_lon, dtype="float32")
        field[sub["cell_id"].to_numpy()] = sub["rain_mm"].to_numpy()
        imd[list(days).index(d)] = field.reshape(n_lat, n_lon)
    ds = xr.Dataset(
        {"rain": (("time", "lat", "lon"), imd)},
        coords={"time": days, "lat": imd_axes(cfg.imd)[0], "lon": imd_axes(cfg.imd)[1]},
    )
    res = lag_test_season(2024, cfg2, imd=ds)
    assert (
        res.best_shift == 0
        and res.imd_stamp == "end_date"
        and res.averages[0] == pytest.approx(1.0, abs=1e-4)
    )
