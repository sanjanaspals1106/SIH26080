"""District products and district_history (PRD 14.4, 20.2): weighted means, wettest cell, area fractions, nulls."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from data_pipeline.districts import district_forecasts, district_history
from data_pipeline.districts.products import (
    FORECAST_COLUMNS,
    FORECAST_SCHEMA,
    HISTORY_COLUMNS,
    HISTORY_SCHEMA,
)
from data_pipeline.ingestion import ValidationError

RUN_A, RUN_B = "tigge_ecmwf_cf_2024070100", "tigge_ecmwf_cf_2024070200"

# District A: 4 cells, weights .4 .3 .2 .1 (all main). District B: 2 cells, .97 main and .03 not main.
# District C: 25 cells of weight 0.04 (below w_min = 0.05): no main cell at all.
WEIGHTS = pd.DataFrame(
    [
        ("A", 1, 0.4, True),
        ("A", 2, 0.3, True),
        ("A", 3, 0.2, True),
        ("A", 4, 0.1, True),
        ("B", 5, 0.97, True),
        ("B", 6, 0.03, False),
    ]
    + [("C", 100 + i, 0.04, False) for i in range(25)],
    columns=["district_id", "cell_id", "area_weight", "is_main"],
)
SUMMARY = pd.DataFrame(
    {
        "district_id": ["A", "B", "C"],
        "n_effective_cells": [1 / (0.16 + 0.09 + 0.04 + 0.01), 1 / (0.97**2 + 0.03**2), 25.0],
        "is_small": [True, True, False],
    }
)
RAIN = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0, 5: 5.0, 6: 500.0, **{100 + i: 1.0 for i in range(25)}}
OBS = {1: 8.0, 2: 16.0, 3: 24.0, 4: 100.0, 5: 4.0, 6: 300.0, **{100 + i: 2.0 for i in range(25)}}


def golden(run=RUN_A, lead=1, obs=OBS, rain=RAIN, offset=0):
    cells = sorted(rain)
    return pd.DataFrame(
        {
            "run_id": run,
            "lead_day": np.int8(lead),
            "cell_id": cells,
            "imd_date": pd.Timestamp("2024-07-01") + pd.Timedelta(days=lead),
            "season": np.int16(2024),
            "rain_mm": [rain[c] for c in cells],
            "obs_mm": [obs[c] for c in cells],
            "alignment_offset_hours": np.int8(offset),
            "imd_stamp": "end_date",
        }
    )


@pytest.fixture(scope="module")
def out(request):
    from data_pipeline.ingestion import load_config

    cfg = load_config()
    obs_gap = {**OBS, 2: np.nan}  # IMD missing in a main cell of A
    obs_gap_b = {**OBS, 6: np.nan}  # IMD missing only in the NON-main cell of B
    g = pd.concat(
        [
            golden(RUN_A, 1),
            golden(RUN_A, 2, obs={**OBS}),
            golden(RUN_B, 1, obs=obs_gap),
            golden(RUN_B, 2, obs=obs_gap_b),
        ],
        ignore_index=True,
    )
    return cfg, g, district_forecasts(g, WEIGHTS, SUMMARY, cfg)


def row(df, run, lead, district):
    return df[(df.run_id == run) & (df.lead_day == lead) & (df.district_id == district)].iloc[0]


def test_schema_key_and_granularity(out):
    cfg, g, df = out
    assert list(df.columns) == FORECAST_COLUMNS
    assert pa.Table.from_pandas(df, schema=FORECAST_SCHEMA, preserve_index=False).schema.equals(
        FORECAST_SCHEMA
    )
    assert len(df) == 2 * 2 * 3  # 2 runs x 2 leads x 3 districts
    assert not df.duplicated(["run_id", "lead_day", "district_id"]).any()
    key = df[["run_id", "lead_day", "district_id"]]
    assert key.equals(key.sort_values(["run_id", "lead_day", "district_id"]))
    r = row(df, RUN_A, 2, "A")
    assert (
        r["imd_date"] == pd.Timestamp("2024-07-03")
        and r["season"] == 2024
        and r["alignment_offset_hours"] == 0
        and r["imd_stamp"] == "end_date"
    )


def test_raw_and_observed_district_means_are_area_weighted(out):
    _, _, df = out
    a = row(df, RUN_A, 1, "A")
    assert a["raw_mean_mm"] == pytest.approx(
        0.4 * 10 + 0.3 * 20 + 0.2 * 30 + 0.1 * 40
    )  # 20.0, not the plain mean 25
    assert a["observed_mean_mm"] == pytest.approx(0.4 * 8 + 0.3 * 16 + 0.2 * 24 + 0.1 * 100)  # 22.8
    assert row(df, RUN_A, 1, "B")["raw_mean_mm"] == pytest.approx(
        0.97 * 5 + 0.03 * 500
    )  # the big non-main cell still counts in the mean
    assert row(df, RUN_A, 1, "C")["raw_mean_mm"] == pytest.approx(1.0)


def test_observed_max_cell_and_heavy_area_fraction(out):
    _, _, df = out
    a = row(df, RUN_A, 1, "A")
    assert a["observed_max_cell_mm"] == 100.0  # max over the main cells
    assert a["observed_heavy_area_fraction"] == pytest.approx(
        0.1
    )  # only the 100 mm cell (weight 0.1) is >= 64.5 mm
    b = row(df, RUN_A, 1, "B")
    assert b["observed_max_cell_mm"] == 4.0  # the 300 mm cell is not a main cell, so it is not in S
    assert b["observed_heavy_area_fraction"] == pytest.approx(
        0.03
    )  # but it does count in the area fraction (weight 0.03)


def test_wettest_cell_is_the_raw_wettest_main_cell(out):
    _, _, df = out
    a = row(df, RUN_A, 1, "A")
    assert (a["raw_wettest_cell_id"], a["raw_wettest_cell_mm"]) == (4, 40.0)
    b = row(df, RUN_A, 1, "B")
    assert (b["raw_wettest_cell_id"], b["raw_wettest_cell_mm"]) == (
        5,
        5.0,
    )  # cell 6 has 500 mm but is not main
    tie = golden(rain={**RAIN, 3: 40.0})  # tie between cells 3 and 4: lowest cell_id wins
    assert row(district_forecasts(tie, WEIGHTS, SUMMARY), RUN_A, 1, "A")["raw_wettest_cell_id"] == 3


def test_effective_cells_and_small_district_flag(out):
    _, _, df = out
    a, c = row(df, RUN_A, 1, "A"), row(df, RUN_A, 1, "C")
    assert a["n_effective_cells"] == pytest.approx(1 / 0.30) and bool(a["is_small"])
    assert c["n_effective_cells"] == pytest.approx(25.0) and not bool(c["is_small"])
    assert (a["n_main_cells"], row(df, RUN_A, 1, "B")["n_main_cells"], c["n_main_cells"]) == (4, 1, 0)


def test_district_without_a_main_cell_has_null_max_fields(out):
    _, _, df = out
    c = row(df, RUN_A, 1, "C")
    assert (
        np.isnan(c["observed_max_cell_mm"])
        and np.isnan(c["raw_wettest_cell_mm"])
        and pd.isna(c["raw_wettest_cell_id"])
    )
    assert c["raw_mean_mm"] == pytest.approx(1.0) and c["observed_mean_mm"] == pytest.approx(
        2.0
    )  # the means still exist


def test_missing_observations_give_null_observed_values_not_partial_ones(out):
    _, _, df = out
    a = row(df, RUN_B, 1, "A")  # IMD missing in main cell 2 of A
    assert (
        np.isnan(a["observed_mean_mm"])
        and np.isnan(a["observed_max_cell_mm"])
        and np.isnan(a["observed_heavy_area_fraction"])
    )
    assert a["raw_mean_mm"] == pytest.approx(20.0)  # the forecast side is unaffected
    b = row(df, RUN_B, 2, "B")  # IMD missing only in the non-main cell 6 of B
    assert (
        np.isnan(b["observed_mean_mm"]) and b["observed_max_cell_mm"] == 4.0
    )  # the mean needs every cell, S does not


def test_nothing_from_later_stages_is_invented(out):
    _, _, df = out
    for c in ["corrected_mean_mm", "wettest_cell_id", "wettest_cell_mean_mm", "wettest_cell_q10_mm", "wettest_cell_q50_mm",
              "wettest_cell_q90_mm", "heavy_prob_max_cell", "very_heavy_prob_max_cell", "heavy_area_fraction_expected",
              "very_heavy_area_fraction_expected"]:  # fmt: skip
        assert df[c].isna().all(), c
    assert not {"attention_level", "priority_rank", "product_type", "fallback_used"} & set(
        df.columns
    )  # M4 owns these


def test_weights_and_golden_must_use_the_same_valid_cells(out):
    cfg, g, _ = out
    with pytest.raises(ValidationError, match="same valid-cell mask"):
        district_forecasts(
            g[g["cell_id"] != 4], WEIGHTS, SUMMARY, cfg
        )  # golden lacks a cell that the weights include
    with pytest.raises(ValidationError, match="No golden cell"):
        district_forecasts(g.assign(cell_id=g["cell_id"] + 10_000), WEIGHTS, SUMMARY, cfg)
    with pytest.raises(ValidationError, match="missing columns"):
        district_forecasts(g.drop(columns="obs_mm"), WEIGHTS, SUMMARY, cfg)


def test_deterministic(out):
    cfg, g, df = out
    pd.testing.assert_frame_equal(
        df, district_forecasts(g.sample(frac=1.0, random_state=1), WEIGHTS, SUMMARY, cfg)
    )


# ---- district_history ---------------------------------------------------------------------------


def test_district_history_schema_and_content(out):
    _, _, df = out
    h = district_history(df)
    assert list(h.columns) == HISTORY_COLUMNS
    assert pa.Table.from_pandas(h, schema=HISTORY_SCHEMA, preserve_index=False).schema.equals(HISTORY_SCHEMA)
    assert not h.duplicated(["run_id", "lead_day", "district_id"]).any()
    assert (
        len(h) == int(df["observed_mean_mm"].notna().sum()) < len(df)
    )  # rows without an observation are not history
    r = row(h, RUN_A, 1, "A")
    assert (r["observed_mean_mm"], r["observed_max_cell_mm"], r["raw_mean_mm"]) == pytest.approx(
        (22.8, 100.0, 20.0)
    )
    assert r["imd_date"] == pd.Timestamp("2024-07-02") and r["season"] == 2024
    for c in ["corrected_mean_mm", "prediction_source", "phase", "lps_near"]:
        assert h[c].isna().all(), c  # owned by later stages: NULL, not invented
