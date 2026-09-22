"""Unit tests for verification.compute engine (PRD §16.1–§16.5, §16.10).

Required tests:
1. hand-computed continuous sums
2. hand-computed a,b,c,d
3. missing observations excluded
4. invalid cells excluded
5. two forecast systems using common mask get same n_samples
6. Brier components correct
7. pinball/coverage components correct
8. FSS component produced correctly
9. evaluation_set preserved
10. n_events only carried when explicitly supplied
11. no n_events inference from a+c
12. multiple neighbourhoods produce separate compatible records
"""

import math
import numpy as np
import pytest

from verification.compute import (
    VerificationComputeResult,
    compute,
    compute_verification_components,
)


def test_1_hand_computed_continuous_sums():
    """Test 1: Verify hand-computed continuous sums (n, sum_error, sum_abs_error, sum_squared_error)."""
    forecast = [10.0, 20.0, 30.0]
    observed = [8.0, 25.0, 30.0]
    # diffs = [2.0, -5.0, 0.0]
    # sum_error = 2 - 5 + 0 = -3.0
    # sum_abs_error = 2 + 5 + 0 = 7.0
    # sum_squared_error = 4 + 25 + 0 = 29.0

    result = compute(
        forecast=forecast,
        observed=observed,
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        thresholds=None,
        neighbourhood_sizes=None,
    )

    cont = result.get_continuous()
    assert cont is not None
    assert cont.n == 3
    assert cont.n_samples == 3
    assert cont.sum_error == pytest.approx(-3.0)
    assert cont.sum_abs_error == pytest.approx(7.0)
    assert cont.sum_squared_error == pytest.approx(29.0)


def test_2_hand_computed_a_b_c_d():
    """Test 2: Verify hand-computed contingency table counts a, b, c, d."""
    forecast = [10.0, 20.0, 5.0, 2.0]
    observed = [15.0, 5.0, 18.0, 0.0]
    threshold = 10.0
    # cell 0: F=10 >= 10, O=15 >= 10 -> hit (a)
    # cell 1: F=20 >= 10, O=5 < 10   -> false alarm (b)
    # cell 2: F=5 < 10,   O=18 >= 10 -> miss (c)
    # cell 3: F=2 < 10,   O=0 < 10   -> correct negative (d)

    result = compute(
        forecast=forecast,
        observed=observed,
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        thresholds=[threshold],
        neighbourhood_sizes=None,
    )

    cat = result.get_contingency(threshold_mm=10.0)
    assert cat is not None
    assert cat.a == 1
    assert cat.b == 1
    assert cat.c == 1
    assert cat.d == 1
    assert cat.n_samples == 4


def test_3_missing_observations_excluded():
    """Test 3: Missing observations (NaNs) are excluded from evaluation."""
    forecast = [10.0, 20.0, 30.0]
    observed = [10.0, np.nan, 30.0]

    result = compute(
        forecast=forecast,
        observed=observed,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[15.6],
        neighbourhood_sizes=None,
    )

    cont = result.get_continuous()
    assert cont is not None
    assert cont.n == 2
    assert cont.n_samples == 2

    cat = result.get_contingency(threshold_mm=15.6)
    assert cat is not None
    assert cat.n_samples == 2


def test_4_invalid_cells_excluded():
    """Test 4: Sea / invalid cells in valid_mask are excluded from evaluation."""
    forecast = [10.0, 20.0, 30.0]
    observed = [10.0, 20.0, 30.0]
    valid_mask = [True, False, True]  # cell 1 is sea

    result = compute(
        forecast=forecast,
        observed=observed,
        valid_mask=valid_mask,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[15.6],
        neighbourhood_sizes=None,
    )

    cont = result.get_continuous()
    assert cont is not None
    assert cont.n == 2
    assert cont.n_samples == 2


def test_5_two_forecast_systems_using_common_mask_get_same_n_samples():
    """Test 5: Two forecast systems compared on a common mask evaluate exact same n_samples."""
    f_raw = [10.0, 15.0, 20.0, 25.0, 30.0]
    f_ai = [12.0, 14.0, 22.0, 24.0, 29.0]
    obs = [11.0, 16.0, 19.0, 26.0, 31.0]
    common_mask = [True, True, True, False, False]  # 3 valid samples

    res_raw = compute(
        forecast=f_raw,
        observed=obs,
        common_mask=common_mask,
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="raw_nwp",
        evaluation_set="holdout",
        thresholds=[15.6],
        neighbourhood_sizes=None,
    )
    res_ai = compute(
        forecast=f_ai,
        observed=obs,
        common_mask=common_mask,
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
        thresholds=[15.6],
        neighbourhood_sizes=None,
    )

    cont_raw = res_raw.get_continuous()
    cont_ai = res_ai.get_continuous()
    assert cont_raw.n_samples == 3
    assert cont_ai.n_samples == 3
    assert cont_raw.n_samples == cont_ai.n_samples


def test_6_brier_components_correct():
    """Test 6: Brier terms and sample counts are correctly computed for probabilities."""
    forecast = [12.0, 5.0]
    observed = [15.0, 5.0]
    threshold = 10.0
    # Outcomes: obs >= 10 -> [1.0, 0.0]
    probabilities = {10.0: [0.8, 0.2]}
    # Brier terms = (0.8 - 1.0)^2 + (0.2 - 0.0)^2 = 0.04 + 0.04 = 0.08
    # brier_n = 2

    result = compute(
        forecast=forecast,
        observed=observed,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[10.0],
        probabilities=probabilities,
        neighbourhood_sizes=None,
    )

    cat = result.get_contingency(threshold_mm=10.0)
    assert cat is not None
    assert cat.brier_n == 2
    assert cat.sum_brier_terms == pytest.approx(0.08)


def test_7_pinball_coverage_components_correct():
    """Test 7: Range pinball losses and coverage counts are correctly computed."""
    forecast = [18.0, 6.0]
    observed = [20.0, 5.0]
    quantiles = {
        "q10": [10.0, 2.0],
        "q50": [18.0, 6.0],
        "q90": [30.0, 10.0],
    }
    # Cell 0: obs=20 in [10, 30] -> inside
    # Cell 1: obs=5 in [2, 10]   -> inside
    # coverage_count = 2, range_n = 2

    result = compute(
        forecast=forecast,
        observed=observed,
        imd_date="2024-07-15",
        lead_day=1,
        quantiles=quantiles,
        thresholds=None,
        neighbourhood_sizes=None,
    )

    cont = result.get_continuous()
    assert cont is not None
    assert cont.range_n == 2
    assert cont.coverage_count == 2
    assert cont.pinball_q50_sum > 0.0


def test_8_fss_component_produced_correctly():
    """Test 8: FSS component is produced with valid additive sums from spatial fields."""
    f_grid = np.array([
        [10.0, 20.0, 0.0],
        [50.0, 70.0, 80.0],
        [0.0, 10.0, 15.0],
    ])
    o_grid = np.array([
        [12.0, 18.0, 0.0],
        [60.0, 65.0, 75.0],
        [0.0, 5.0, 10.0],
    ])

    result = compute(
        forecast=f_grid,
        observed=o_grid,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[64.5],
        neighbourhood_sizes=[1, 3],
    )

    fss_1 = result.get_fss(threshold_mm=64.5, neighbourhood_cells=1)
    fss_3 = result.get_fss(threshold_mm=64.5, neighbourhood_cells=3)
    assert fss_1 is not None
    assert fss_3 is not None
    assert fss_1.neighbourhood_cells == 1
    assert fss_3.neighbourhood_cells == 3
    assert fss_1.n_samples == 9
    assert fss_3.n_samples == 9


def test_9_evaluation_set_preserved():
    """Test 9: evaluation_set is preserved exactly across all output records."""
    f = [10.0, 20.0]
    o = [12.0, 18.0]

    res_dev = compute(
        forecast=f,
        observed=o,
        imd_date="2024-07-15",
        lead_day=1,
        evaluation_set="development",
        thresholds=[15.6],
        neighbourhood_sizes=None,
    )
    for c in res_dev:
        assert c.evaluation_set == "development"

    res_holdout = compute(
        forecast=f,
        observed=o,
        imd_date="2024-07-15",
        lead_day=1,
        evaluation_set="holdout",
        thresholds=[15.6],
        neighbourhood_sizes=None,
    )
    for c in res_holdout:
        assert c.evaluation_set == "holdout"


def test_10_n_events_only_carried_when_explicitly_supplied():
    """Test 10: n_events is carried when explicitly supplied, and None for thresholds without it."""
    f = [20.0, 70.0]
    o = [25.0, 80.0]

    # Supply events only for threshold 64.5
    result = compute(
        forecast=f,
        observed=o,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[15.6, 64.5],
        neighbourhood_sizes=None,
        n_events={64.5: 42},
    )

    c_64 = result.get_contingency(threshold_mm=64.5)
    c_15 = result.get_contingency(threshold_mm=15.6)

    assert c_64 is not None
    assert c_64.n_events == 42

    assert c_15 is not None
    assert c_15.n_events is None


def test_11_no_n_events_inference_from_a_plus_c():
    """Test 11: When n_events is not supplied, it MUST remain None despite hits/misses a+c > 0."""
    f = [20.0, 10.0]
    o = [25.0, 30.0]
    threshold = 15.6
    # cell 0: F=20 >= 15.6, O=25 >= 15.6 -> a = 1
    # cell 1: F=10 < 15.6,  O=30 >= 15.6 -> c = 1
    # a + c = 2

    result = compute(
        forecast=f,
        observed=o,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[threshold],
        neighbourhood_sizes=None,
        n_events=None,
    )

    cat = result.get_contingency(threshold_mm=15.6)
    assert cat is not None
    assert cat.a == 1
    assert cat.c == 1
    # PRD §10.7 / §13.2: n_events comes strictly from spatiotemporal events, NEVER from a+c
    assert cat.n_events is None


def test_12_multiple_neighbourhoods_produce_separate_compatible_records():
    """Test 12: Multiple neighbourhood sizes produce separate compatible FSS component records."""
    f_grid = np.zeros((4, 4))
    o_grid = np.zeros((4, 4))
    f_grid[1, 1] = 70.0
    o_grid[1, 1] = 80.0

    result = compute(
        forecast=f_grid,
        observed=o_grid,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[64.5],
        neighbourhood_sizes=[1, 3, 5],
    )

    fss_records = [c for c in result if c.neighbourhood_cells is not None]
    assert len(fss_records) == 3

    sizes = {c.neighbourhood_cells for c in fss_records}
    assert sizes == {1, 3, 5}

    for c in fss_records:
        assert c.threshold_mm == 64.5
        assert c.lead_day == 1
        assert c.forecast_type == "regime_aware_ml"
