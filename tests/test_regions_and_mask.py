"""region_code from config/regions.yaml, one valid-cell mask for every season, and the M3 input frame."""

import numpy as np
import pandas as pd
import pytest
from tests.stage3_world import make_golden, make_grid

from data_pipeline.alignment import (
    build_golden_season,
    check_mask_consistency,
    save_valid_cells,
    write_golden,
)
from data_pipeline.features import assign_region_codes, load_region_rules, read_features
from data_pipeline.ingestion import ValidationError, load_config
from regime_engine import pipeline as rp
from regime_engine.contract import REGIME_14_FEATURES

# ---- region_code --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lat, lon, code",
    [
        (30.0, 75.0, "HIMALAYA"),  # rule 1 wins even where later rules also fit
        (28.0, 95.0, "HIMALAYA"),  # lat_gte is inclusive; the first matching region wins over NORTHEAST
        (27.75, 88.5, "NORTHEAST"),  # lon_gte inclusive
        (15.0, 75.0, "WEST_COAST"),
        (20.75, 76.75, "WEST_COAST"),  # lat_lt 21 exclusive of 21.0
        (21.0, 76.75, "NORTHWEST_WEST"),
        (25.0, 70.0, "NORTHWEST_WEST"),
        (10.0, 80.0, "SOUTH_EAST"),  # lon 77 is not < 77, lat < 18
        (17.75, 77.0, "SOUTH_EAST"),
        (18.0, 77.0, "CENTRAL_EAST"),  # lat_lt 18 exclusive: falls through to the default
        (22.0, 80.0, "CENTRAL_EAST"),
    ],
)
def test_region_codes_follow_the_first_matching_box(lat, lon, code):
    assert assign_region_codes([lat], [lon])[0] == code


def test_all_six_regions_appear_on_the_imd_grid():
    cfg = load_config()
    lat = np.repeat(6.5 + 0.25 * np.arange(129), 135)
    lon = np.tile(66.5 + 0.25 * np.arange(135), 129)
    codes = assign_region_codes(lat, lon)
    assert set(codes) == {r["region_code"] for r in load_region_rules()}
    assert len(codes) == 129 * 135  # every cell gets exactly one region


def test_bad_region_files_are_refused(tmp_path):
    bad = tmp_path / "r.yaml"
    bad.write_text("regions:\n  - {order: 1, region_code: A, rule: {lat_gte: 1}}\n")
    with pytest.raises(ValidationError, match="default"):
        load_region_rules(bad)
    bad.write_text("regions:\n  - {order: 1, region_code: A, rule: {lat_over: 1}}\n  - {order: 2, region_code: B, rule: {default: true}}\n")
    with pytest.raises(ValidationError, match="unknown rule keys"):
        load_region_rules(bad)


def test_read_features_can_add_region_code(m1_outputs):
    plain = read_features(m1_outputs.cfg, seasons=[2024])
    with_region = read_features(m1_outputs.cfg, seasons=[2024], with_region_code=True)
    assert "region_code" not in plain.columns  # the stored feature schema is unchanged
    assert list(with_region.columns) == [*plain.columns, "region_code"]
    assert with_region["region_code"].notna().all() and len(with_region) == len(plain)
    expected = assign_region_codes(plain["latitude"], plain["longitude"])
    assert (with_region["region_code"].to_numpy() == expected).all()


# ---- one mask for every season ------------------------------------------------------------------


@pytest.fixture()
def built(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data")
    grid = make_grid(cfg)
    write_golden(make_golden(cfg, grid), cfg)
    return cfg, grid


def test_consistent_mask_passes(built):
    cfg, grid = built
    grid.attrs["base_years"] = [1981, 2010]
    save_valid_cells(grid, cfg)
    report = check_mask_consistency(cfg)
    assert report["n_valid_cells"] == int(grid["is_valid"].sum()) and "season_2024" in report["seasons"]


def test_a_development_mask_is_reported_unless_allowed(built):
    cfg, grid = built
    grid.attrs["base_years"] = [2025, 2025]
    save_valid_cells(grid, cfg)
    with pytest.raises(ValidationError, match=r"base years \[2025, 2025\]"):
        check_mask_consistency(cfg)
    assert check_mask_consistency(cfg, strict_base_period=False)["n_valid_cells"] > 0


def test_a_season_built_on_another_mask_is_caught(built):
    cfg, grid = built
    other = grid.copy()
    first_invalid = int(np.flatnonzero(~other["is_valid"].to_numpy())[0])
    other.loc[first_invalid, "is_valid"] = True  # a mask with one more cell
    other.attrs["base_years"] = [1981, 2010]
    save_valid_cells(other, cfg)
    with pytest.raises(ValidationError, match=r"1 missing"):
        check_mask_consistency(cfg)


def test_golden_build_refuses_a_non_prd_mask_before_reading_anything(built):
    cfg, grid = built
    grid.attrs["base_years"] = [2025, 2025]
    save_valid_cells(grid, cfg)
    with pytest.raises(ValidationError, match="PRD base period 1981-2010"):
        build_golden_season(2021, cfg, months=[7], require_base_period=True)  # no raw files needed: it stops first


# ---- the M3 input frame ------------------------------------------------------------------------


def write_regime_tables(cfg, features, drop_last=False):
    d = rp.regime_dir(cfg)
    (d / "regime_features").mkdir(parents=True, exist_ok=True)
    keys = features[["run_id", "lead_day", "cell_id"]].copy()
    rng = np.random.default_rng(0)
    regime = keys.assign(season=np.int16(2024), **{c: rng.random(len(keys)).astype("float32") for c in REGIME_14_FEATURES},
                         regime_source="oof")
    if drop_last:
        regime = regime.iloc[:-1]
    regime.to_parquet(d / "regime_features" / "season_2024.parquet", index=False)
    dom = keys[["run_id", "lead_day"]].drop_duplicates().assign(ood_flag=False, regime_available=True)
    dom.to_parquet(d / "regime_domain.parquet", index=False)


def test_m3_frame_has_everything_the_models_need(m1_outputs):
    from ml.feature_contracts import B3_FEATURES

    cfg = m1_outputs.cfg
    write_regime_tables(cfg, read_features(cfg, seasons=[2024]))
    frame = rp.load_m3_frame(cfg, [2024])
    assert set(B3_FEATURES) <= set(frame.columns)  # 27 + 14
    assert {"region_code", "regime_source", "ood_flag", "regime_available", "obs_mm", "season", "run_id"} <= set(frame.columns)
    assert frame["regime_source"].eq("oof").all() and frame["region_code"].notna().all()
    from ml.orchestration import validate_m1_m2_inputs

    status = validate_m1_m2_inputs(frame, is_training=True, require_regime=True)  # the M3 input contract accepts it
    assert status["regime_available"] is True


def test_m3_frame_refuses_regime_rows_from_another_mask(m1_outputs):
    cfg = m1_outputs.cfg
    write_regime_tables(cfg, read_features(cfg, seasons=[2024]), drop_last=True)
    with pytest.raises(ValidationError, match="different valid-cell masks"):
        rp.load_m3_frame(cfg, [2024])
