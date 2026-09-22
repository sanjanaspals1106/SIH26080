"""Unit tests for F5 District Priority Table (PRD §15 F5) and RMSE sqrt regression check."""

import math
import pytest

from data_pipeline.districts.priority import (
    LEVEL_HIGH,
    LEVEL_NORMAL,
    LEVEL_UNAVAILABLE,
    LEVEL_WATCH,
    DistrictForecast,
    assign_district_priorities,
    determine_attention_level,
    load_priority_thresholds,
)
from verification.stats.bootstrap import RMSEComponents, _recompute_rmse


def test_high_at_exactly_0_50_heavy_prob():
    """Test 1: HIGH at exactly 0.50 heavy probability."""
    d = DistrictForecast(
        district_id="D1",
        district_name="Test District",
        state="Test State",
        corrected_mean_mm=10.0,
        wettest_cell_mean_mm=12.0,
        heavy_prob_max_cell=0.50,
        very_heavy_prob_max_cell=0.10,
    )
    assert determine_attention_level(d, is_64_5_available=True) == LEVEL_HIGH


def test_high_at_exactly_0_20_very_heavy_prob():
    """Test 2: HIGH at exactly 0.20 very-heavy probability."""
    d = DistrictForecast(
        district_id="D1",
        district_name="Test District",
        state="Test State",
        corrected_mean_mm=5.0,
        wettest_cell_mean_mm=10.0,
        heavy_prob_max_cell=0.15,
        very_heavy_prob_max_cell=0.20,
    )
    assert determine_attention_level(d, is_64_5_available=True) == LEVEL_HIGH


def test_watch_at_exactly_0_20_heavy_prob():
    """Test 3: WATCH at exactly 0.20 heavy probability (when not HIGH)."""
    d = DistrictForecast(
        district_id="D1",
        district_name="Test District",
        state="Test State",
        corrected_mean_mm=10.0,
        wettest_cell_mean_mm=12.0,
        heavy_prob_max_cell=0.20,
        very_heavy_prob_max_cell=0.10,
    )
    assert determine_attention_level(d, is_64_5_available=True) == LEVEL_WATCH


def test_watch_from_corrected_mean_ge_15_6():
    """Test 4: WATCH from corrected mean >= 15.6 mm (when not HIGH)."""
    d = DistrictForecast(
        district_id="D1",
        district_name="Test District",
        state="Test State",
        corrected_mean_mm=15.6,
        wettest_cell_mean_mm=20.0,
        heavy_prob_max_cell=0.10,
        very_heavy_prob_max_cell=0.05,
    )
    assert determine_attention_level(d, is_64_5_available=True) == LEVEL_WATCH


def test_normal_case():
    """Test 5: NORMAL case when neither HIGH nor WATCH criteria are met."""
    d = DistrictForecast(
        district_id="D1",
        district_name="Test District",
        state="Test State",
        corrected_mean_mm=12.0,
        wettest_cell_mean_mm=14.0,
        heavy_prob_max_cell=0.15,
        very_heavy_prob_max_cell=0.05,
    )
    assert determine_attention_level(d, is_64_5_available=True) == LEVEL_NORMAL


def test_fallback_overrides_to_unavailable():
    """Test 6: Fallback district is always UNAVAILABLE, even with high probabilities."""
    d = DistrictForecast(
        district_id="D1",
        district_name="Fallback District",
        state="Test State",
        corrected_mean_mm=100.0,
        wettest_cell_mean_mm=150.0,
        heavy_prob_max_cell=0.99,
        very_heavy_prob_max_cell=0.95,
        fallback_used=True,
    )
    assert determine_attention_level(d, is_64_5_available=True) == LEVEL_UNAVAILABLE
    assert determine_attention_level(d, is_64_5_available=False) == LEVEL_UNAVAILABLE


def test_fallback_15_6_rule_boundaries():
    """Test 7: When 64.5 model is unavailable, boundaries 0.80 (HIGH) and 0.50 (WATCH) are used."""
    # Boundary 0.80 -> HIGH
    d_high = DistrictForecast(
        district_id="D1",
        district_name="D1",
        state="S",
        corrected_mean_mm=5.0,
        wettest_cell_mean_mm=5.0,
        moderate_prob_max_cell=0.80,
    )
    assert determine_attention_level(d_high, is_64_5_available=False) == LEVEL_HIGH

    # Just below 0.80 (0.79) -> WATCH
    d_watch1 = DistrictForecast(
        district_id="D2",
        district_name="D2",
        state="S",
        corrected_mean_mm=5.0,
        wettest_cell_mean_mm=5.0,
        moderate_prob_max_cell=0.79,
    )
    assert determine_attention_level(d_watch1, is_64_5_available=False) == LEVEL_WATCH

    # Exactly 0.50 -> WATCH
    d_watch2 = DistrictForecast(
        district_id="D3",
        district_name="D3",
        state="S",
        corrected_mean_mm=5.0,
        wettest_cell_mean_mm=5.0,
        moderate_prob_max_cell=0.50,
    )
    assert determine_attention_level(d_watch2, is_64_5_available=False) == LEVEL_WATCH

    # Just below 0.50 (0.49) -> NORMAL
    d_norm = DistrictForecast(
        district_id="D4",
        district_name="D4",
        state="S",
        corrected_mean_mm=5.0,
        wettest_cell_mean_mm=5.0,
        moderate_prob_max_cell=0.49,
    )
    assert determine_attention_level(d_norm, is_64_5_available=False) == LEVEL_NORMAL


def test_full_multi_key_sorting():
    """Test 8: Sorting by level, heavy_prob, heavy_area, wettest_cell, and district_name."""
    districts = [
        # D_norm: NORMAL
        DistrictForecast(district_id="1", district_name="Alpha", state="S", corrected_mean_mm=5, wettest_cell_mean_mm=5, heavy_prob_max_cell=0.05),
        # D_unavail: UNAVAILABLE
        DistrictForecast(district_id="2", district_name="Beta", state="S", corrected_mean_mm=50, wettest_cell_mean_mm=50, fallback_used=True),
        # D_watch: WATCH
        DistrictForecast(district_id="3", district_name="Gamma", state="S", corrected_mean_mm=20, wettest_cell_mean_mm=25, heavy_prob_max_cell=0.10),
        # D_high1: HIGH (heavy_prob = 0.6)
        DistrictForecast(district_id="4", district_name="Delta", state="S", corrected_mean_mm=30, wettest_cell_mean_mm=40, heavy_prob_max_cell=0.60),
        # D_high2_b: HIGH (heavy_prob = 0.5, area = 0.4, wettest = 35, name = 'Zeta')
        DistrictForecast(district_id="5", district_name="Zeta", state="S", corrected_mean_mm=30, wettest_cell_mean_mm=35, heavy_prob_max_cell=0.50, heavy_area_fraction_expected=0.4),
        # D_high2_a: HIGH (heavy_prob = 0.5, area = 0.4, wettest = 35, name = 'Epsilon') -> comes before Zeta by name
        DistrictForecast(district_id="6", district_name="Epsilon", state="S", corrected_mean_mm=30, wettest_cell_mean_mm=35, heavy_prob_max_cell=0.50, heavy_area_fraction_expected=0.4),
        # D_high3: HIGH (heavy_prob = 0.5, area = 0.6, wettest = 30) -> comes before Epsilon/Zeta due to higher area
        DistrictForecast(district_id="7", district_name="Theta", state="S", corrected_mean_mm=30, wettest_cell_mean_mm=30, heavy_prob_max_cell=0.50, heavy_area_fraction_expected=0.6),
    ]

    sorted_d = assign_district_priorities(districts, is_64_5_available=True)

    expected_order = [
        "Delta",    # HIGH, prob=0.60
        "Theta",    # HIGH, prob=0.50, area=0.60
        "Epsilon",  # HIGH, prob=0.50, area=0.40, name='Epsilon' < 'Zeta'
        "Zeta",     # HIGH, prob=0.50, area=0.40, name='Zeta'
        "Gamma",    # WATCH
        "Alpha",    # NORMAL
        "Beta",     # UNAVAILABLE
    ]

    actual_order = [d.district_name for d in sorted_d]
    assert actual_order == expected_order


def test_priority_ranks_assigned_correctly():
    """Test 9: Priority ranks are assigned starting at 1 and strictly increasing."""
    districts = [
        DistrictForecast(district_id="D1", district_name="D1", state="S", corrected_mean_mm=5, wettest_cell_mean_mm=5, heavy_prob_max_cell=0.6),
        DistrictForecast(district_id="D2", district_name="D2", state="S", corrected_mean_mm=5, wettest_cell_mean_mm=5, heavy_prob_max_cell=0.2),
        DistrictForecast(district_id="D3", district_name="D3", state="S", corrected_mean_mm=5, wettest_cell_mean_mm=5, heavy_prob_max_cell=0.05),
    ]
    sorted_d = assign_district_priorities(districts)

    ranks = [d.priority_rank for d in sorted_d]
    assert ranks == [1, 2, 3]


def test_rmse_sqrt_regression_hand_check():
    """Test 10: Hand-computed check verifying RMSE uses sqrt(SSE / n) where SSE=18, n=2 => RMSE=3.0."""
    comp = RMSEComponents(n_samples=2, sum_squared_errors=18.0)
    rmse = _recompute_rmse([comp])
    assert rmse is not None
    assert math.isclose(rmse, 3.0)
    assert not math.isclose(rmse, 9.0)  # Verify RMSE != MSE
