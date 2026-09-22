"""Tests for M4 bias-table computation for F2 (PRD §15 F2 and §10.4).

Required tests:
1. median/q25/q75 correct on hand-computed values
2. development query excludes its own season
3. holdout query uses all development seasons
4. holdout rows never enter bias history
5. grouping separates lead_day
6. grouping separates phase
7. grouping separates lps_near
8. missing values excluded
9. n_dates <20 => few_past_cases=true
10. n_dates >=20 => false
11. empty history returns null stats
12. season_excluded stored correctly
"""

import math
import numpy as np
import pytest

from data_pipeline.districts.bias_table import (
    MIN_DATES_THRESHOLD,
    BiasRecord,
    BiasTable,
    compute_bias_statistics,
    compute_bias_table,
    compute_bias_table_entry,
)


def test_1_median_q25_q75_correct_hand_computed():
    """Test 1: median/q25/q75 match exact hand-computed values.
    
    Given observed - raw diffs: [1.0, 2.0, 3.0, 4.0, 5.0]
    median = 3.0, q25 = 2.0, q75 = 4.0.
    """
    history_rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 11.0, "raw_mean_mm": 10.0},  # diff=1.0
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 12.0, "raw_mean_mm": 10.0},  # diff=2.0
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 13.0, "raw_mean_mm": 10.0},  # diff=3.0
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 14.0, "raw_mean_mm": 10.0},  # diff=4.0
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 15.0, "raw_mean_mm": 10.0},  # diff=5.0
    ]

    record = compute_bias_table_entry(
        history_rows=history_rows,
        district_id="D01",
        lead_day=1,
        phase="active",
        lps_near=True,
        query_season=2018,
        evaluation_set="development",
    )

    assert record.n_dates == 5
    assert record.median_diff_mm == pytest.approx(3.0)
    assert record.q25_diff_mm == pytest.approx(2.0)
    assert record.q75_diff_mm == pytest.approx(4.0)


def test_2_development_query_excludes_its_own_season():
    """Test 2: A development query excludes its own season from history (PRD §15 F2, §10.4)."""
    history_rows = [
        # Season 2016 (development): diff = 2.0
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "normal", "lps_near": False, "observed_mean_mm": 12.0, "raw_mean_mm": 10.0},
        # Season 2017 (development): diff = 4.0
        {"district_id": "D01", "lead_day": 1, "season": 2017, "phase": "normal", "lps_near": False, "observed_mean_mm": 14.0, "raw_mean_mm": 10.0},
        # Season 2018 (query's own season!): diff = 100.0 (would strongly contaminate if included)
        {"district_id": "D01", "lead_day": 1, "season": 2018, "phase": "normal", "lps_near": False, "observed_mean_mm": 110.0, "raw_mean_mm": 10.0},
    ]

    record = compute_bias_table_entry(
        history_rows=history_rows,
        district_id="D01",
        lead_day=1,
        phase="normal",
        lps_near=False,
        query_season=2018,
        evaluation_set="development",
    )

    # Only 2016 and 2017 rows are counted (2 dates)
    assert record.n_dates == 2
    assert record.season_excluded == 2018
    # Median of [2.0, 4.0] is 3.0
    assert record.median_diff_mm == pytest.approx(3.0)


def test_3_holdout_query_uses_all_development_seasons():
    """Test 3: A holdout query uses all development seasons with no season excluded (season_excluded=None)."""
    history_rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "normal", "lps_near": False, "observed_mean_mm": 12.0, "raw_mean_mm": 10.0},  # 2.0
        {"district_id": "D01", "lead_day": 1, "season": 2017, "phase": "normal", "lps_near": False, "observed_mean_mm": 14.0, "raw_mean_mm": 10.0},  # 4.0
        {"district_id": "D01", "lead_day": 1, "season": 2018, "phase": "normal", "lps_near": False, "observed_mean_mm": 16.0, "raw_mean_mm": 10.0},  # 6.0
    ]

    record = compute_bias_table_entry(
        history_rows=history_rows,
        district_id="D01",
        lead_day=1,
        phase="normal",
        lps_near=False,
        query_season=2024,
        evaluation_set="holdout",
    )

    # All 3 development seasons are included
    assert record.n_dates == 3
    assert record.season_excluded is None
    # Median of [2.0, 4.0, 6.0] is 4.0
    assert record.median_diff_mm == pytest.approx(4.0)


def test_4_holdout_rows_never_enter_bias_history():
    """Test 4: Holdout rows are never included in the bias history, for dev or holdout queries."""
    history_rows = [
        # Development row
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "break", "lps_near": True, "observed_mean_mm": 15.0, "raw_mean_mm": 10.0, "evaluation_set": "development"},
        # Holdout rows (should be rejected)
        {"district_id": "D01", "lead_day": 1, "season": 2023, "phase": "break", "lps_near": True, "observed_mean_mm": 99.0, "raw_mean_mm": 10.0, "evaluation_set": "holdout"},
        {"district_id": "D01", "lead_day": 1, "season": 2024, "phase": "break", "lps_near": True, "observed_mean_mm": 99.0, "raw_mean_mm": 10.0, "prediction_source": "holdout"},
    ]

    # Query in development mode
    rec_dev = compute_bias_table_entry(
        history_rows=history_rows,
        district_id="D01",
        lead_day=1,
        phase="break",
        lps_near=True,
        query_season=2017,
        evaluation_set="development",
    )
    assert rec_dev.n_dates == 1
    assert rec_dev.median_diff_mm == pytest.approx(5.0)

    # Query in holdout mode
    rec_holdout = compute_bias_table_entry(
        history_rows=history_rows,
        district_id="D01",
        lead_day=1,
        phase="break",
        lps_near=True,
        query_season=2024,
        evaluation_set="holdout",
    )
    assert rec_holdout.n_dates == 1
    assert rec_holdout.median_diff_mm == pytest.approx(5.0)


def test_5_grouping_separates_lead_day():
    """Test 5: Grouping strictly separates lead_day."""
    history_rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 20.0, "raw_mean_mm": 10.0},  # diff=10.0
        {"district_id": "D01", "lead_day": 2, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 30.0, "raw_mean_mm": 10.0},  # diff=20.0
        {"district_id": "D01", "lead_day": 3, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 40.0, "raw_mean_mm": 10.0},  # diff=30.0
    ]

    rec_lead1 = compute_bias_table_entry(history_rows, "D01", 1, "active", True, query_season=2018)
    assert rec_lead1.n_dates == 1
    assert rec_lead1.median_diff_mm == pytest.approx(10.0)

    rec_lead2 = compute_bias_table_entry(history_rows, "D01", 2, "active", True, query_season=2018)
    assert rec_lead2.n_dates == 1
    assert rec_lead2.median_diff_mm == pytest.approx(20.0)


def test_6_grouping_separates_phase():
    """Test 6: Grouping strictly separates regime phase."""
    history_rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": False, "observed_mean_mm": 25.0, "raw_mean_mm": 10.0},  # diff=15
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "break", "lps_near": False, "observed_mean_mm": 5.0, "raw_mean_mm": 10.0},    # diff=-5
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "normal", "lps_near": False, "observed_mean_mm": 12.0, "raw_mean_mm": 10.0},  # diff=2
    ]

    rec_break = compute_bias_table_entry(history_rows, "D01", 1, "break", False, query_season=2018)
    assert rec_break.n_dates == 1
    assert rec_break.median_diff_mm == pytest.approx(-5.0)

    # Check case-insensitivity: "Active" matches "active"
    rec_active = compute_bias_table_entry(history_rows, "D01", 1, "Active", False, query_season=2018)
    assert rec_active.n_dates == 1
    assert rec_active.median_diff_mm == pytest.approx(15.0)


def test_7_grouping_separates_lps_near():
    """Test 7: Grouping strictly separates lps_near (True vs False)."""
    history_rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 30.0, "raw_mean_mm": 10.0},  # diff=20
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": False, "observed_mean_mm": 10.0, "raw_mean_mm": 10.0}, # diff=0
    ]

    rec_true = compute_bias_table_entry(history_rows, "D01", 1, "active", True, query_season=2018)
    assert rec_true.n_dates == 1
    assert rec_true.median_diff_mm == pytest.approx(20.0)

    rec_false = compute_bias_table_entry(history_rows, "D01", 1, "active", False, query_season=2018)
    assert rec_false.n_dates == 1
    assert rec_false.median_diff_mm == pytest.approx(0.0)


def test_8_missing_values_excluded():
    """Test 8: Rows with missing observed or raw rainfall are excluded."""
    history_rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 20.0, "raw_mean_mm": 10.0},
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": None, "raw_mean_mm": 10.0},
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": float("nan"), "raw_mean_mm": 10.0},
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 20.0, "raw_mean_mm": None},
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 20.0, "raw_mean_mm": np.nan},
    ]

    rec = compute_bias_table_entry(history_rows, "D01", 1, "active", True, query_season=2018)
    assert rec.n_dates == 1
    assert rec.median_diff_mm == pytest.approx(10.0)


def test_9_n_dates_below_20_exposes_few_past_cases():
    """Test 9: When n_dates < 20, few_past_cases is True and note is 'few past cases' (PRD §15 F2)."""
    # 14 dates like in PRD §18.5 mock example
    history_rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "normal", "lps_near": True, "observed_mean_mm": 15.0, "raw_mean_mm": 10.0}
        for _ in range(14)
    ]

    rec = compute_bias_table_entry(history_rows, "D01", 1, "normal", True, query_season=2018)
    assert rec.n_dates == 14
    assert rec.few_past_cases is True
    assert rec.note == "few past cases"


def test_10_n_dates_at_least_20_sets_few_past_cases_false():
    """Test 10: When n_dates >= 20, few_past_cases is False."""
    # Exactly 20 dates
    history_20 = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "normal", "lps_near": True, "observed_mean_mm": 15.0, "raw_mean_mm": 10.0}
        for _ in range(20)
    ]
    rec_20 = compute_bias_table_entry(history_20, "D01", 1, "normal", True, query_season=2018)
    assert rec_20.n_dates == 20
    assert rec_20.few_past_cases is False
    assert rec_20.note is None

    # 25 dates
    history_25 = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "normal", "lps_near": True, "observed_mean_mm": 15.0, "raw_mean_mm": 10.0}
        for _ in range(25)
    ]
    rec_25 = compute_bias_table_entry(history_25, "D01", 1, "normal", True, query_season=2018)
    assert rec_25.n_dates == 25
    assert rec_25.few_past_cases is False
    assert rec_25.note is None


def test_11_empty_history_returns_null_stats():
    """Test 11: No matching history returns n_dates=0 and statistics=None without inventing data."""
    rec = compute_bias_table_entry([], "D99", 1, "active", True, query_season=2018)
    assert rec.n_dates == 0
    assert rec.median_diff_mm is None
    assert rec.q25_diff_mm is None
    assert rec.q75_diff_mm is None
    assert rec.few_past_cases is True
    assert rec.note == "few past cases"


def test_12_season_excluded_stored_correctly():
    """Test 12: season_excluded stores query_season for development query and None for holdout query."""
    rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "normal", "lps_near": True, "observed_mean_mm": 15.0, "raw_mean_mm": 10.0}
    ]

    # Development query
    rec_dev = compute_bias_table_entry(rows, "D01", 1, "normal", True, query_season=2017, evaluation_set="development")
    assert rec_dev.season_excluded == 2017

    # Holdout query
    rec_holdout = compute_bias_table_entry(rows, "D01", 1, "normal", True, query_season=2024, evaluation_set="holdout")
    assert rec_holdout.season_excluded is None


def test_bias_table_class_and_full_table_build():
    """Test BiasTable class indexing, dictionary-style access, and compute_bias_table build."""
    rows = [
        {"district_id": "D01", "lead_day": 1, "season": 2016, "phase": "active", "lps_near": True, "observed_mean_mm": 20.0, "raw_mean_mm": 10.0},
        {"district_id": "D02", "lead_day": 2, "season": 2017, "phase": "break", "lps_near": False, "observed_mean_mm": 5.0, "raw_mean_mm": 15.0},
    ]

    table = BiasTable(rows)
    entry1 = table.get_entry("D01", 1, "active", True, query_season=2018)
    assert entry1["median_diff_mm"] == pytest.approx(10.0)
    assert entry1.get("n_dates") == 1

    d = entry1.to_dict()
    assert d["district_id"] == "D01"
    assert d["median_diff_mm"] == 10.0

    all_entries = compute_bias_table(rows, query_season=2018)
    assert len(all_entries) == 2
    assert ("D01", 1, "active", True) in all_entries
    assert ("D02", 2, "break", False) in all_entries
