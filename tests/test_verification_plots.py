"""Unit tests for PRD §16.9 standard verification plot and data preparation.

Required tests:
1. success ratio = 1-FAR
2. FSS neighbourhood order correct
3. km mapping correct
4. reliability bins remain ordered
5. raw-rain-size bin order correct
6. improvement mean computed correctly
7. season order deterministic
8. B0->B3 row order fixed
9. undefined metrics preserved
10. development/holdout mixing rejected
11. matplotlib helper returns Figure without mutating data
"""

import copy
import matplotlib
import matplotlib.pyplot as plt
import pytest

# Ensure non-interactive backend for automated testing
matplotlib.use("Agg")

from verification.components import VerificationComponent
from verification.plots.data import (
    B0ToB3Row,
    CellImprovementPoint,
    FSSNeighbourhoodPoint,
    PerformanceDiagramPoint,
    RawRainBiasPoint,
    ReliabilityBinPoint,
    SeasonImprovementDot,
    prepare_average_improvement_map_data,
    prepare_b0_to_b3_table_data,
    prepare_fss_neighbourhood_data,
    prepare_per_season_improvement_data,
    prepare_performance_diagram_data,
    prepare_raw_rain_bias_data,
    prepare_reliability_diagram_data,
)
from verification.plots.matplotlib import (
    plot_average_improvement_map,
    plot_b0_to_b3_comparison,
    plot_fss_vs_neighbourhood,
    plot_per_season_improvement_dots,
    plot_performance_diagram,
    plot_raw_rain_bias,
    plot_reliability_diagram,
)
from verification.reliability import ReliabilityBinSummary
from verification.results import MetricRecord, VerificationResult


def test_1_success_ratio_is_one_minus_far():
    """Test 1: Performance diagram success ratio derives as 1 - FAR and keeps CSI/frequency_bias."""
    pt = prepare_performance_diagram_data(
        pod=0.85,
        far=0.25,
        csi=0.68,
        frequency_bias=1.10,
        label="B3 ML",
    )
    assert pt.pod == pytest.approx(0.85)
    assert pt.far == pytest.approx(0.25)
    assert pt.success_ratio == pytest.approx(0.75)  # 1 - 0.25 = 0.75
    assert pt.csi == pytest.approx(0.68)
    assert pt.frequency_bias == pytest.approx(1.10)
    assert pt.label == "B3 ML"

    # Undefined FAR preserves None
    pt_undef = prepare_performance_diagram_data(pod=0.85, far=None)
    assert pt_undef.success_ratio is None
    assert pt_undef.far is None


def test_2_fss_neighbourhood_order_correct():
    """Test 2: FSS vs neighbourhood size returns points ordered by neighbourhood_cells ascending."""
    scrambled = [
        {"neighbourhood_cells": 5, "fss": 0.65},
        {"neighbourhood_cells": 1, "fss": 0.40},
        {"neighbourhood_cells": 9, "fss": 0.80},
        {"neighbourhood_cells": 3, "fss": 0.55},
    ]
    prepared = prepare_fss_neighbourhood_data(scrambled)

    cells = [p.neighbourhood_cells for p in prepared]
    assert cells == [1, 3, 5, 9]


def test_3_km_mapping_correct():
    """Test 3: Approximate km uses exact PRD §16.8 mapping: 1->28, 3->84, 5->140, 9->250 km."""
    items = [
        {"neighbourhood_cells": 1, "fss": 0.40},
        {"neighbourhood_cells": 3, "fss": 0.55},
        {"neighbourhood_cells": 5, "fss": 0.65},
        {"neighbourhood_cells": 9, "fss": 0.80},
    ]
    prepared = prepare_fss_neighbourhood_data(items, useful_skill=0.55)

    kms = [p.approximate_km for p in prepared]
    assert kms == [28, 84, 140, 250]
    for p in prepared:
        assert p.useful_skill_line == pytest.approx(0.55)


def test_4_reliability_bins_remain_ordered():
    """Test 4: Reliability diagram bins remain sorted by bin_lower ascending."""
    scrambled_summaries = [
        ReliabilityBinSummary(
            bin_lower=0.8,
            bin_upper=0.9,
            n=20,
            sum_predicted_probability=17.0,
            n_observed_events=18,
            mean_predicted_probability=0.85,
            observed_frequency=0.90,
        ),
        ReliabilityBinSummary(
            bin_lower=0.1,
            bin_upper=0.2,
            n=40,
            sum_predicted_probability=6.0,
            n_observed_events=5,
            mean_predicted_probability=0.15,
            observed_frequency=0.125,
        ),
        ReliabilityBinSummary(
            bin_lower=0.0,
            bin_upper=0.1,
            n=100,
            sum_predicted_probability=4.0,
            n_observed_events=2,
            mean_predicted_probability=0.04,
            observed_frequency=0.02,
        ),
    ]

    prepared = prepare_reliability_diagram_data(scrambled_summaries)
    lowers = [p.bin_lower for p in prepared]
    assert lowers == [0.0, 0.1, 0.8]


def test_5_raw_rain_size_bin_order_correct():
    """Test 5: Raw-rain-size categories are returned in PRD canonical order."""
    scrambled = [
        {"group_type": "raw_rain_size", "group_value": "64_5_and_above", "bias": 5.2, "n_samples": 40},
        {"group_type": "raw_rain_size", "group_value": "below_1", "bias": -0.1, "n_samples": 1200},
        {"group_type": "raw_rain_size", "group_value": "15_6_to_64_5", "bias": 1.4, "n_samples": 150},
        {"group_type": "raw_rain_size", "group_value": "1_to_15_6", "bias": 0.3, "n_samples": 500},
    ]

    prepared = prepare_raw_rain_bias_data(scrambled)
    names = [p.bin_name for p in prepared]
    assert names == ["below_1", "1_to_15_6", "15_6_to_64_5", "64_5_and_above"]
    assert prepared[0].bias == pytest.approx(-0.1)
    assert prepared[1].bias == pytest.approx(0.3)
    assert prepared[2].bias == pytest.approx(1.4)
    assert prepared[3].bias == pytest.approx(5.2)


def test_6_improvement_mean_computed_correctly():
    """Test 6: Mean improvement is computed correctly and lat/lon are not invented."""
    # Cell 101: improvements [2.0, 4.0] -> mean = 3.0, n_dates = 2
    # Cell 102: improvements [-1.0, 1.0, 3.0] -> mean = 1.0, n_dates = 3
    data = {
        101: [2.0, 4.0],
        102: [-1.0, 1.0, 3.0],
    }
    lat_lons = {
        101: (18.5, 73.8),
        # 102 has no supplied lat/lon
    }

    prepared = prepare_average_improvement_map_data(data, lat_lon_map=lat_lons)
    assert len(prepared) == 2

    p101 = [p for p in prepared if p.cell_id == 101][0]
    assert p101.mean_improvement_mm == pytest.approx(3.0)
    assert p101.n_dates == 2
    assert p101.latitude == pytest.approx(18.5)
    assert p101.longitude == pytest.approx(73.8)

    p102 = [p for p in prepared if p.cell_id == 102][0]
    assert p102.mean_improvement_mm == pytest.approx(1.0)
    assert p102.n_dates == 3
    # Lat/Lon must remain None (not fabricated)
    assert p102.latitude is None
    assert p102.longitude is None


def test_7_season_order_deterministic():
    """Test 7: Per-season improvement dots are returned in deterministic season order with correct direction."""
    season_a = {2021: 8.5, 2018: 10.0, 2020: 9.0}
    season_b = {2021: 9.5, 2018: 12.0, 2020: 8.5}

    prepared = prepare_per_season_improvement_data(
        season_metrics_a=season_a,
        season_metrics_b=season_b,
        system_comparison="B3 vs B2",
        metric_name="RMSE",
        metric_direction="lower",
    )

    seasons = [d.season for d in prepared]
    assert seasons == [2018, 2020, 2021]

    # 2018: 10.0 - 12.0 = -2.0 (lower is better -> system_a favored)
    d2018 = prepared[0]
    assert d2018.metric_difference == pytest.approx(-2.0)
    assert d2018.favored_system == "system_a"
    assert d2018.metric_direction == "lower"

    # 2020: 9.0 - 8.5 = +0.5 (lower is better -> system_b favored)
    d2020 = prepared[1]
    assert d2020.metric_difference == pytest.approx(0.5)
    assert d2020.favored_system == "system_b"


def test_8_b0_to_b3_row_order_fixed():
    """Test 8: B0 -> B3 table rows follow fixed sequence: B0 raw_nwp -> B1 quantile_mapping -> B2 global_ml -> B3 regime_aware_ml."""
    scrambled = [
        {"forecast_type": "regime_aware_ml", "evaluation_set": "development", "rmse": 4.1},
        {"forecast_type": "raw_nwp", "evaluation_set": "development", "rmse": 6.2},
        {"forecast_type": "global_ml", "evaluation_set": "development", "rmse": 4.8},
        {"forecast_type": "quantile_mapping", "evaluation_set": "development", "rmse": 5.5},
    ]

    rows = prepare_b0_to_b3_table_data(scrambled, metric_name="rmse")
    assert len(rows) == 4

    expected_progression = [
        ("B0", "raw_nwp"),
        ("B1", "quantile_mapping"),
        ("B2", "global_ml"),
        ("B3", "regime_aware_ml"),
    ]
    for row, (exp_id, exp_type) in zip(rows, expected_progression):
        assert row.system_id == exp_id
        assert row.forecast_type == exp_type


def test_9_undefined_metrics_preserved():
    """Test 9: Undefined metric values in B0->B3 rows stay None and preserve reasons."""
    records = [
        {"forecast_type": "raw_nwp", "evaluation_set": "development", "rmse": 6.2},
        {"forecast_type": "quantile_mapping", "evaluation_set": "development", "rmse": None, "undefined_reason": "NO_VALID_SAMPLES"},
        # B2 and B3 omitted entirely
    ]

    rows = prepare_b0_to_b3_table_data(records, metric_name="rmse")

    # B1 has undefined metric
    assert rows[1].metric_value is None
    assert rows[1].undefined_reason == "NO_VALID_SAMPLES"

    # B2 missing
    assert rows[2].metric_value is None
    assert rows[2].undefined_reason == "SYSTEM_NOT_PROVIDED"


def test_10_development_and_holdout_mixing_rejected():
    """Test 10: Attempting to mix development and holdout evaluation sets raises ValueError."""
    mixed_records = [
        VerificationComponent(forecast_type="raw_nwp", evaluation_set="development", n_samples=100),
        VerificationComponent(forecast_type="regime_aware_ml", evaluation_set="holdout", n_samples=100),
    ]

    with pytest.raises(ValueError, match="Cannot mix development and holdout"):
        prepare_b0_to_b3_table_data(mixed_records, metric_name="rmse")


def test_11_matplotlib_helper_returns_figure_without_mutating_data():
    """Test 11: Matplotlib rendering wrappers return Figure instances without modifying input data."""
    pt = prepare_performance_diagram_data(pod=0.80, far=0.20, csi=0.67, frequency_bias=1.0)
    pt_orig = copy.deepcopy(pt)

    fig_perf = plot_performance_diagram(pt)
    assert isinstance(fig_perf, plt.Figure)
    assert pt == pt_orig

    fss_pts = prepare_fss_neighbourhood_data([
        {"neighbourhood_cells": 1, "fss": 0.40},
        {"neighbourhood_cells": 5, "fss": 0.65},
    ])
    fss_orig = copy.deepcopy(fss_pts)
    fig_fss = plot_fss_vs_neighbourhood(fss_pts)
    assert isinstance(fig_fss, plt.Figure)
    assert fss_pts == fss_orig

    bias_pts = prepare_raw_rain_bias_data([
        {"group_type": "raw_rain_size", "group_value": "below_1", "bias": -0.2, "n_samples": 10},
    ])
    bias_orig = copy.deepcopy(bias_pts)
    fig_bias = plot_raw_rain_bias(bias_pts)
    assert isinstance(fig_bias, plt.Figure)
    assert bias_pts == bias_orig

    plt.close("all")
