"""The 27 features (PRD 9.4): names, definitions, units, boundaries, missing values, table schema."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from stage3_world import (
    GX,
    GY,
    RUNS,
    K,
    M,
    make_climatology,
    make_golden,
    make_grid,
    make_static_grid,
)

from data_pipeline.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    FEATURE_VERSION,
    TABLE_COLUMNS,
    build_features,
    compute_static_geography,
    feature_matrix,
)
from data_pipeline.features.grid import cell_size_m, earth_radius_m, masked_derivative
from data_pipeline.features.io import read_month_tables, write_month_tables
from data_pipeline.features.library import (
    gradient_size,
    neighbourhood_stats,
    relative_vorticity,
)
from data_pipeline.ingestion import ValidationError

PRD_FEATURES = [
    "rain_mm", "nbr_mean_3", "nbr_max_3", "nbr_mean_5", "nbr_max_5", "rain_grad", "rain_prev_lead", "rain_next_lead",
    "u850", "v850", "wspd850", "vort850", "q850", "msl", "shear_200_850",
    "elevation_m", "slope", "aspect_sin", "aspect_cos", "dist_coast_km",
    "clim_mean", "clim_p95",
    "doy_sin", "doy_cos", "lead_day", "latitude", "longitude",
]  # fmt: skip


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from data_pipeline.ingestion import load_config

    cfg = load_config(data_dir=tmp_path_factory.mktemp("data"))
    grid = make_grid(cfg)
    golden = make_golden(cfg, grid, obs_nan=lambda run_i, cell: run_i == 1 and cell % 50 == 0)
    static = compute_static_geography(make_static_grid(cfg), cfg)
    feats = build_features(golden, static, make_climatology(cfg), cfg)
    return cfg, grid, golden, static, feats


def cell_rows(feats, run, lead):
    return feats[(feats.run_id == run) & (feats.lead_day == lead)].set_index("cell_id")


# ---- names, schema, version ----------------------------------------------------------------


def test_the_27_feature_names_are_exact():
    assert FEATURE_COLUMNS == PRD_FEATURES and len(FEATURE_COLUMNS) == 27 == len(set(FEATURE_COLUMNS))


def test_feature_table_has_every_feature_the_keys_and_the_targets(world):
    *_, feats = world
    assert list(feats.columns) == TABLE_COLUMNS
    assert set(FEATURE_COLUMNS) <= set(feats.columns)  # lead_day is both a key and a feature
    assert list(feature_matrix(feats).columns) == FEATURE_COLUMNS
    assert {"obs_mm", "obs_ge_15_6", "obs_ge_64_5", "obs_ge_115_6"} <= set(feats.columns)
    assert not {"p_active", "lps_present", "regime_confidence", "upslope_flux"} & set(
        feats.columns
    )  # no regime features
    assert pa.Table.from_pandas(feats, schema=FEATURE_SCHEMA, preserve_index=False).schema.equals(
        FEATURE_SCHEMA
    )


def test_feature_version_is_stored_with_every_row(world):
    *_, feats = world
    assert FEATURE_VERSION == "v1" and (feats["feature_set_version"] == "v1").all()


def test_rows_match_golden_one_to_one(world):
    cfg, grid, golden, static, feats = world
    assert len(feats) == len(golden) and not feats.duplicated(["run_id", "lead_day", "cell_id"]).any()
    assert feats[["run_id", "lead_day", "cell_id"]].equals(golden[["run_id", "lead_day", "cell_id"]])


# ---- circulation ---------------------------------------------------------------------------


def test_wind_speed_and_shear(world):
    cfg, grid, golden, static, feats = world
    assert feats["wspd850"].to_numpy() == pytest.approx(np.hypot(golden["u850"], golden["v850"]), rel=1e-5)
    # the fixture has u200 = u850 + 3 and v200 = v850 + 4, so |wind200 - wind850| = 5 everywhere
    assert feats["shear_200_850"].to_numpy() == pytest.approx(5.0, rel=1e-5)
    assert feats["q850"].to_numpy() == pytest.approx(golden["q850"].to_numpy()) and feats[
        "msl"
    ].to_numpy() == pytest.approx(golden["msl"].to_numpy())


def test_vorticity_matches_the_finite_difference_formula(world):
    cfg, grid, golden, static, feats = world
    dy, _ = cell_size_m(np.array([0.0]), cfg)
    lat = golden["latitude"].to_numpy().astype("float64")
    expected = (K + M) / dy + golden["u850"].to_numpy() * np.tan(np.radians(lat)) / earth_radius_m(cfg)
    assert feats["vort850"].to_numpy() == pytest.approx(expected, rel=1e-3, abs=1e-9)
    assert 1e-6 < feats["vort850"].abs().max() < 1e-3  # a plausible size in s-1, not m-1 or degrees


def test_vorticity_of_solid_body_rotation_and_of_uniform_flow(cfg):
    n = 9
    valid = np.ones((1, n, n), dtype=bool)
    lat = np.linspace(10.0, 12.0, n)
    dy, dx = cell_size_m(lat, cfg)
    x = (np.arange(n)[None, :] * dx)[None]  # metres east of column 0 in each row: (1, rows, cols)
    y = (np.arange(n)[:, None] * dy)[None]  # metres north of row 0
    omega = 5e-5
    u, v = -omega * y * np.ones((1, n, n)), omega * x  # flat solid-body rotation: zeta = 2 * omega
    curvature = lambda uu: uu * np.tan(np.radians(lat))[:, None] / earth_radius_m(cfg)  # noqa: E731
    assert relative_vorticity(u, v, valid, lat, cfg) - curvature(u) == pytest.approx(2 * omega, rel=1e-6)
    uniform = np.full((1, n, n), 7.0)  # uniform flow has no shear, only the curvature term is left
    zeta = relative_vorticity(uniform, np.full((1, n, n), -3.0), valid, lat, cfg)
    assert zeta - curvature(uniform) == pytest.approx(0.0, abs=1e-15)


# ---- rain features -------------------------------------------------------------------------


def test_rain_mm_is_the_golden_rain(world):
    cfg, grid, golden, static, feats = world
    assert feats["rain_mm"].to_numpy() == pytest.approx(golden["rain_mm"].to_numpy(), rel=1e-6)


def test_neighbourhood_statistics_inside_the_valid_area(world):
    cfg, grid, golden, static, feats = world
    r = cell_rows(feats, RUNS[0], 1)
    cell = 100 * 0 + int(
        grid.loc[(grid.latitude == 15.0) & (grid.longitude == 80.0), "cell_id"].iloc[0]
    )  # interior
    rain = r.loc[cell, "rain_mm"]
    assert r.loc[cell, "nbr_mean_3"] == pytest.approx(
        rain, rel=1e-5
    )  # linear field: window mean = centre value
    assert r.loc[cell, "nbr_mean_5"] == pytest.approx(rain, rel=1e-5)
    assert r.loc[cell, "nbr_max_3"] == pytest.approx(
        rain + GY + GX, rel=1e-5
    )  # north-east corner of the window
    assert r.loc[cell, "nbr_max_5"] == pytest.approx(rain + 2 * (GY + GX), rel=1e-5)


def test_neighbourhood_uses_only_valid_cells_next_to_the_hole_and_the_edge(world):
    """Compare every cell against a slow, independent implementation."""
    cfg, grid, golden, static, feats = world
    r = cell_rows(feats, RUNS[1], 2)
    rain = r["rain_mm"].to_dict()
    n_lon = cfg.imd.n_lon
    for cell in list(r.index[::37]) + [
        int(grid.loc[(grid.latitude == 20.0) & (grid.longitude == 79.75), "cell_id"].iloc[0])
    ]:
        for size, half in ((3, 1), (5, 2)):
            vals = [
                rain[c]
                for di in range(-half, half + 1)
                for dj in range(-half, half + 1)
                if (c := (cell // n_lon + di) * n_lon + (cell % n_lon + dj)) in rain
            ]
            assert r.loc[cell, f"nbr_mean_{size}"] == pytest.approx(np.mean(vals), rel=1e-5)
            assert r.loc[cell, f"nbr_max_{size}"] == pytest.approx(np.max(vals), rel=1e-6)


def test_neighbourhood_helper_edge_cases():
    rain = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    valid = np.ones_like(rain, dtype=bool)
    mean, biggest = neighbourhood_stats(rain, valid, 3)
    assert mean[0, 1, 1] == 5.0 and biggest[0, 1, 1] == 9.0
    assert (
        mean[0, 0, 0] == pytest.approx((1 + 2 + 4 + 5) / 4) and biggest[0, 0, 0] == 5.0
    )  # corner: 4 cells only
    valid[0, 1, 1] = False  # a hole is not counted as dry: it is left out
    mean, _ = neighbourhood_stats(rain, valid, 3)
    assert mean[0, 0, 0] == pytest.approx((1 + 2 + 4) / 3)
    lone = np.zeros((1, 3, 3), dtype=bool)
    lone[0, 0, 0] = True
    m, b = neighbourhood_stats(rain, lone, 3)
    assert m[0, 0, 0] == 1.0 and b[0, 0, 0] == 1.0 and np.isnan(m[0, 2, 2])


def test_rain_gradient_uses_kilometres_on_the_grid(world):
    cfg, grid, golden, static, feats = world
    r = cell_rows(feats, RUNS[0], 1)
    dy_km = 27.75
    lat = grid.set_index("cell_id").loc[r.index, "latitude"].to_numpy().astype("float64")
    expected = np.hypot(GY / dy_km, GX / (dy_km * np.cos(np.radians(lat))))  # mm per km, valid at every cell
    assert r["rain_grad"].to_numpy() == pytest.approx(expected, rel=1e-4)


def test_gradient_helper_edges_and_isolated_cells():
    one_row = np.arange(4.0).reshape(1, 1, 4)
    assert np.isnan(
        gradient_size(one_row, np.ones_like(one_row, dtype=bool), 1.0, np.ones((1, 1)))
    ).all()  # no lat neighbours
    field = np.tile(np.arange(4.0), (3, 1))[None]  # 3 rows x 4 cols; ramp along lon
    valid = np.ones_like(field, dtype=bool)
    g = gradient_size(field, valid, 1.0, np.ones((3, 1)))
    assert g[0] == pytest.approx(np.ones((3, 4)))  # one-sided at the edges gives the same slope for a ramp
    valid[0, 1, 1] = False
    assert np.isnan(masked_derivative(field, valid, 1.0, -1)[0, 1, 1])
    only = np.zeros_like(valid)
    only[0, 1, 1] = True
    assert np.isnan(
        gradient_size(field, only, 1.0, np.ones((3, 1)))[0, 1, 1]
    )  # no valid neighbour: NaN, not 0


def test_lead_features_come_from_the_same_run_and_stop_at_the_boundaries(world):
    cfg, grid, golden, static, feats = world
    for run in RUNS:
        r1, r2, r3 = (cell_rows(feats, run, k) for k in (1, 2, 3))
        assert r2["rain_prev_lead"].to_numpy() == pytest.approx(r1["rain_mm"].to_numpy())
        assert r2["rain_next_lead"].to_numpy() == pytest.approx(r3["rain_mm"].to_numpy())
        assert r1["rain_prev_lead"].isna().all() and r3["rain_next_lead"].isna().all()  # PRD 9.4: left empty
        assert r1["rain_next_lead"].to_numpy() == pytest.approx(r2["rain_mm"].to_numpy())
        assert r3["rain_prev_lead"].to_numpy() == pytest.approx(r2["rain_mm"].to_numpy())
    # the two runs differ by 3 mm everywhere: a lead feature that leaked across runs would show it
    a, b = cell_rows(feats, RUNS[0], 2), cell_rows(feats, RUNS[1], 2)
    assert (b["rain_prev_lead"] - a["rain_prev_lead"]).to_numpy() == pytest.approx(3.0, rel=1e-5)


def test_a_missing_lead_gives_missing_lead_features_not_another_runs_value(world):
    cfg, grid, golden, static, feats = world
    gap = golden[~((golden.run_id == RUNS[0]) & (golden.lead_day == 2))]
    f = build_features(gap, static, make_climatology(cfg), cfg)
    assert f[(f.run_id == RUNS[0]) & (f.lead_day == 1)]["rain_next_lead"].isna().all()
    assert f[(f.run_id == RUNS[0]) & (f.lead_day == 3)]["rain_prev_lead"].isna().all()
    assert f[(f.run_id == RUNS[1]) & (f.lead_day == 1)]["rain_next_lead"].notna().all()


# ---- time, place, geography, climatology -----------------------------------------------------


def test_day_of_year_is_a_smooth_annual_cycle_of_the_start_date(world):
    cfg, grid, golden, static, feats = world
    r = cell_rows(feats, RUNS[0], 1).iloc[0]  # start 2024-07-01: day 183 of a leap year
    angle = 2 * np.pi * 183 / 365.25
    assert (r["doy_sin"], r["doy_cos"]) == (
        pytest.approx(np.sin(angle), abs=1e-6),
        pytest.approx(np.cos(angle), abs=1e-6),
    )
    assert np.hypot(feats["doy_sin"], feats["doy_cos"]).to_numpy() == pytest.approx(1.0, rel=1e-5)
    # no jump across New Year: day 366 and day 1 are neighbours on the circle
    a1, a366 = 2 * np.pi * 1 / 365.25, 2 * np.pi * 366 / 365.25
    assert np.hypot(np.cos(a1) - np.cos(a366), np.sin(a1) - np.sin(a366)) < 0.03
    assert cell_rows(feats, RUNS[1], 1).iloc[0]["doy_sin"] != r["doy_sin"]  # one day later


def test_time_and_place_features_and_geography_join(world):
    cfg, grid, golden, static, feats = world
    assert (feats["lead_day"] == golden["lead_day"]).all()
    assert feats["latitude"].to_numpy() == pytest.approx(golden["latitude"].to_numpy())
    assert feats["longitude"].to_numpy() == pytest.approx(golden["longitude"].to_numpy())
    joined = static.set_index("cell_id").loc[feats["cell_id"]]
    for c in ("elevation_m", "slope", "aspect_sin", "aspect_cos", "dist_coast_km"):
        assert feats[c].to_numpy() == pytest.approx(joined[c].to_numpy(), rel=1e-6)
    assert (feats["clim_mean"] == 8.0).all() and (feats["clim_p95"] == 35.0).all()


# ---- missing values ---------------------------------------------------------------------------


def test_missing_observations_give_null_targets_but_defined_features(world):
    cfg, grid, golden, static, feats = world
    miss = feats[feats["obs_mm"].isna()]
    assert len(miss) > 0 and (miss["run_id"] == RUNS[1]).all()
    assert miss[["obs_ge_15_6", "obs_ge_64_5", "obs_ge_115_6"]].isna().all().all()
    assert not feats[FEATURE_COLUMNS[:5] + ["u850", "wspd850"]].isna().any().any()
    ok = feats[feats["obs_mm"].notna()]
    assert (ok["obs_ge_15_6"].astype(bool) == (ok["obs_mm"] >= 15.6)).all()
    assert (ok["obs_ge_64_5"].astype(bool) == (ok["obs_mm"] >= 64.5)).all()


def test_missing_static_or_climatology_cells_are_reported(world):
    cfg, grid, golden, static, feats = world
    with pytest.raises(ValidationError, match="static geography table does not cover"):
        build_features(golden, static.iloc[:100], make_climatology(cfg), cfg)
    with pytest.raises(ValidationError, match="climatology table is missing columns"):
        build_features(golden, static, make_climatology(cfg).drop(columns="clim_p95"), cfg)
    with pytest.raises(ValidationError, match="missing columns"):
        build_features(golden.drop(columns="u200"), static, make_climatology(cfg), cfg)


def test_unknown_climatology_value_stays_null(world):
    cfg, grid, golden, static, feats = world
    clim = make_climatology(cfg)
    clim.loc[clim["cell_id"] == golden["cell_id"].iloc[0], "clim_mean"] = (
        np.nan
    )  # a cell with no historical data
    f = build_features(golden, static, clim, cfg)
    assert f["clim_mean"].isna().sum() == (golden["cell_id"] == golden["cell_id"].iloc[0]).sum()


# ---- determinism and files ---------------------------------------------------------------------


def test_features_are_deterministic_and_round_trip_through_parquet(world, tmp_path):
    cfg, grid, golden, static, feats = world
    pd.testing.assert_frame_equal(feats, build_features(golden, static, make_climatology(cfg), cfg))
    import dataclasses

    cfg2 = dataclasses.replace(cfg, features=dataclasses.replace(cfg.features, dir=tmp_path / "f"))
    month = feats["run_id"].str[-10:-4]
    paths = write_month_tables(
        feats, "cell_features", FEATURE_SCHEMA, ["run_id", "lead_day", "cell_id"], month, cfg2
    )
    assert [p.name for p in paths] == ["cell_features_202407.parquet"]
    back = read_month_tables("cell_features", FEATURE_SCHEMA, cfg2)
    pd.testing.assert_frame_equal(back, feats, check_dtype=False)
    assert back["rain_mm"].dtype == np.float32 and back["lead_day"].dtype == np.int8
    write_month_tables(feats, "cell_features", FEATURE_SCHEMA, ["run_id", "lead_day", "cell_id"], month, cfg2)
    assert len(read_month_tables("cell_features", FEATURE_SCHEMA, cfg2)) == len(
        feats
    )  # rewriting does not duplicate
