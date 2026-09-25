"""Integration: synthetic Golden Dataset -> features -> district weights -> district products -> district_history."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from tests.stage3_world import (
    RUNS,
    make_golden,
    make_grid,
    make_static_grid,
    make_tigge_static,
)

import data_pipeline.districts.weights as weights_module
import data_pipeline.features.climatology as clim_module
import data_pipeline.features.static as static_module
from data_pipeline.alignment import write_golden
from data_pipeline.features import (
    FEATURE_COLUMNS,
    build_features,
    build_stage3_season,
    compute_static_geography,
    get_climatology,
    read_district_forecasts,
    read_district_history,
    read_features,
)
from data_pipeline.ingestion import MissingInputError, ValidationError, load_config
from data_pipeline.ingestion.imd import read_imd_year
from data_pipeline.ingestion.synthetic import write_synthetic_imd_year


@pytest.fixture(scope="module")
def stage3(tmp_path_factory):
    cfg = load_config(data_dir=tmp_path_factory.mktemp("data"))
    grid = make_grid(cfg)
    golden = make_golden(cfg, grid, obs_nan=lambda run_i, cell: run_i == 1 and cell % 50 == 0)
    write_golden(golden, cfg)  # the Stage 2 writer: this is exactly what Stage 3 reads
    for year in (2019, 2020):  # "training seasons" for the climatology
        write_synthetic_imd_year(cfg, year, missing_fraction=0.02, seed=year)
    districts = gpd.GeoDataFrame(
        {"district_id": ["D001", "D002", "D003", "D004"], "name": ["A", "B", "C", "D"], "state": "S"},
        geometry=[
            box(74.875, 14.875, 78.875, 18.875),  # 16 x 16 cells, all valid
            box(69.875, 9.875, 70.375, 10.125),  # 2 cells (lat 10.0, lon 70.0 and 70.25)
            box(79.6, 19.6, 81.0, 21.0),  # straddles the hole (lat 20-20.5, lon 80-80.5) and the valid area
            box(74.875, 34.875, 75.375, 35.375),  # entirely outside the valid area
        ],
        crs="EPSG:4326",
    )
    paths = build_stage3_season(
        2024,
        [2019, 2020],
        cfg,
        holdout_seasons=[2024],
        static=make_tigge_static(),
        districts=districts,
        grid=grid,
    )
    return cfg, grid, golden, districts, paths


def test_files_are_written_per_season_and_month(stage3):
    cfg, *_, paths = stage3
    assert {k: [p.name for p in v] for k, v in paths.items()} == {
        "features": ["cell_features_202407.parquet"],
        "forecasts": ["district_forecasts_202407.parquet"],
        "history": ["district_history_202407.parquet"],
    }
    assert all(p.is_relative_to(cfg.data_dir / "features") for v in paths.values() for p in v)


def test_static_climatology_and_weights_were_cached(stage3):
    cfg, *_ = stage3
    for name in (
        "static/static_geography.parquet",
        "districts/cell_district_weights.parquet",
        "districts/district_summary.parquet",
    ):
        assert (cfg.features.dir / name).is_file()
    assert len(list((cfg.features.dir / "climatology").glob("climatology_*.parquet"))) == 1


def test_features_on_disk_equal_a_direct_build(stage3):
    cfg, grid, golden, *_ = stage3
    static = compute_static_geography(make_static_grid(cfg), cfg)
    clim = get_climatology([2019, 2020], cfg)
    on_disk = read_features(cfg)
    pd.testing.assert_frame_equal(on_disk, build_features(golden, static, clim, cfg), check_dtype=False)
    assert len(on_disk) == 2 * 3 * int(grid["is_valid"].sum())
    assert set(FEATURE_COLUMNS) <= set(on_disk.columns) and (on_disk["feature_set_version"] == "v1").all()


def test_climatology_feature_comes_from_the_training_years_only(stage3):
    cfg, *_ = stage3
    on_disk = read_features(cfg)
    cell = int(on_disk["cell_id"].iloc[0])
    values = []
    for year in (2019, 2020):
        rain = read_imd_year(cfg.imd.year_path(year), year, cfg)["rain"]
        jjas = rain.sel(time=rain["time"].dt.month.isin([6, 7, 8, 9])).values.reshape(-1, 129 * 135)[:, cell]
        values.append(jjas[np.isfinite(jjas)])
    values = np.concatenate(values)
    row = on_disk.loc[on_disk.cell_id == cell].iloc[0]
    assert row["clim_mean"] == pytest.approx(values.mean(), rel=1e-4) and row["clim_p95"] == pytest.approx(
        np.percentile(values, 95), rel=1e-4
    )
    assert not (on_disk["clim_mean"] == 8.0).any()  # not the constant of the unit-test table
    assert not cfg.imd.year_path(2024).exists()  # season 2024 (the one being featured) was never needed


def test_district_forecasts_cover_the_districts_that_have_valid_cells(stage3):
    cfg, grid, golden, districts, _ = stage3
    df = read_district_forecasts(cfg)
    assert sorted(df["district_id"].unique()) == [
        "D001",
        "D002",
        "D003",
    ]  # D004 has no valid cell: no rows, no guess
    assert len(df) == 2 * 3 * 3 and not df.duplicated(["run_id", "lead_day", "district_id"]).any()
    assert df["raw_mean_mm"].notna().all() and (df["n_effective_cells"] > 0).all()
    assert df["corrected_mean_mm"].isna().all() and df["heavy_prob_max_cell"].isna().all()


def test_district_means_agree_with_a_direct_weighted_sum(stage3):
    cfg, grid, golden, districts, _ = stage3
    df = read_district_forecasts(cfg)
    r = df[(df.run_id == RUNS[0]) & (df.lead_day == 1) & (df.district_id == "D002")].iloc[0]
    two = golden[
        (golden.run_id == RUNS[0])
        & (golden.lead_day == 1)
        & golden.cell_id.isin([14 * 135 + 14, 14 * 135 + 15])
    ]
    assert len(two) == 2
    assert r["raw_mean_mm"] == pytest.approx(
        two["rain_mm"].mean(), rel=1e-3
    )  # two ~equal cells: weights ~0.5 each
    assert bool(r["is_small"]) and r["n_effective_cells"] == pytest.approx(2.0, rel=0.01)
    big = df[(df.run_id == RUNS[0]) & (df.lead_day == 1) & (df.district_id == "D001")].iloc[0]
    assert not bool(big["is_small"]) and big["n_main_cells"] == 0  # 256 cells: none reaches w_min (PRD 14.2)


def test_district_c_uses_only_its_valid_cells(stage3):
    cfg, *_ = stage3
    s = pd.read_parquet(cfg.features.dir / "districts" / "district_summary.parquet").set_index("district_id")
    assert (
        0.0 < s.loc["D003", "valid_area_fraction"] < 1.0
    )  # the hole and the area outside are not valid cells
    w = pd.read_parquet(cfg.features.dir / "districts" / "cell_district_weights.parquet")
    assert w[w.district_id == "D003"]["area_weight"].sum() == pytest.approx(1.0)


def test_district_history_is_the_observed_subset(stage3):
    cfg, *_ = stage3
    fc, hist = read_district_forecasts(cfg), read_district_history(cfg)
    assert len(hist) == int(fc["observed_mean_mm"].notna().sum()) <= len(fc)
    assert not hist.duplicated(["run_id", "lead_day", "district_id"]).any()
    assert (
        hist["phase"].isna().all()
        and hist["lps_near"].isna().all()
        and hist["prediction_source"].isna().all()
    )


def test_a_second_run_is_identical_and_reuses_every_cache(stage3, monkeypatch):
    cfg, grid, golden, districts, _ = stage3
    before = (read_features(cfg), read_district_forecasts(cfg), read_district_history(cfg))
    for module, name in (
        (static_module, "compute_static_geography"),
        (clim_module, "compute_climatology"),
        (weights_module, "compute_district_weights"),
    ):
        monkeypatch.setattr(
            module, name, lambda *a, **k: pytest.fail("a cached static product was recomputed")
        )
    build_stage3_season(
        2024,
        [2019, 2020],
        cfg,
        holdout_seasons=[2024],
        static=make_tigge_static(),
        districts=districts,
        grid=grid,
    )
    for old, new in zip(
        before, (read_features(cfg), read_district_forecasts(cfg), read_district_history(cfg)), strict=True
    ):
        pd.testing.assert_frame_equal(old, new)  # deterministic, and no duplicate rows from re-running


def test_holdout_climatology_is_refused(stage3):
    cfg, grid, golden, districts, _ = stage3
    with pytest.raises(ValidationError, match="holdout season"):
        build_stage3_season(
            2024,
            [2019, 2020],
            cfg,
            holdout_seasons=[2020],
            static=make_tigge_static(),
            districts=districts,
            grid=grid,
        )


def test_missing_inputs_fail_clearly(stage3, tmp_path):
    cfg, grid, golden, districts, _ = stage3
    empty = load_config(data_dir=tmp_path / "nothing")
    with pytest.raises(MissingInputError, match="No Golden Dataset for season 2024"):
        build_stage3_season(2024, [2019], empty, districts=districts, grid=grid)
    write_golden(golden, empty)
    with pytest.raises(
        MissingInputError, match="No district file is configured"
    ):  # no silent substitute boundaries
        build_stage3_season(2024, [2019], empty, static=make_tigge_static(), grid=grid)
    assert (
        not (empty.features.dir / "climatology").exists() and not (empty.features.dir / "static").exists()
    )  # failed before any work
    with pytest.raises(MissingInputError, match="IMD file not found"):  # no real climatology without IMD data
        build_stage3_season(2024, [1999], empty, static=make_tigge_static(), districts=districts, grid=grid)
