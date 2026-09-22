"""Unit tests for range coverage checks per PRD §13.7."""

import math
import numpy as np
import pytest

from probability.coverage import check_range_coverage, load_coverage_parameters


def test_exact_80_percent_coverage():
    """Test 1: Exact 80% coverage matches target (0.80) and is not flagged."""
    # 10 samples: 8 inside, 2 outside
    leads = [1] * 10
    q10 = [10.0] * 10
    q50 = [20.0] * 10
    q90 = [30.0] * 10
    # 8 samples inside [10, 30], 2 outside (<10 and >30)
    obs = [15.0, 20.0, 25.0, 10.0, 30.0, 12.0, 18.0, 22.0, 5.0, 35.0]

    results = check_range_coverage(leads, obs, q10, q50, q90)
    assert len(results) == 1
    res = results[0]

    assert res.lead_day == 1
    assert math.isclose(res.coverage, 0.80)
    assert res.n_samples == 10
    assert res.target == 0.80
    assert res.tolerance == 0.10
    assert res.needs_display_warning is False

    # Wording data helper check
    wording = res.get_wording_data()
    assert wording["measured_coverage_pct"] == 80.0
    assert wording["needs_display_warning"] is False


def test_more_than_10_points_low_flagged():
    """Test 2: Coverage more than 10 percentage points low (<70%) is flagged."""
    # 6 inside out of 10 = 60% coverage (abs(0.60 - 0.80) = 0.20 > 0.10)
    leads = [1] * 10
    q10 = [10.0] * 10
    q50 = [20.0] * 10
    q90 = [30.0] * 10
    obs = [15.0, 20.0, 25.0, 12.0, 18.0, 22.0, 2.0, 5.0, 40.0, 50.0]

    results = check_range_coverage(leads, obs, q10, q50, q90)
    res = results[0]
    assert math.isclose(res.coverage, 0.60)
    assert res.needs_display_warning is True


def test_more_than_10_points_high_flagged():
    """Test 3: Coverage more than 10 percentage points high (>90%) is flagged."""
    # 10 inside out of 10 = 100% coverage (abs(1.0 - 0.80) = 0.20 > 0.10)
    leads = [1] * 10
    q10 = [10.0] * 10
    q50 = [20.0] * 10
    q90 = [30.0] * 10
    obs = [15.0, 20.0, 25.0, 12.0, 18.0, 22.0, 11.0, 29.0, 14.0, 26.0]

    results = check_range_coverage(leads, obs, q10, q50, q90)
    res = results[0]
    assert math.isclose(res.coverage, 1.00)
    assert res.needs_display_warning is True


def test_exact_70_and_90_boundary_not_flagged():
    """Test 4: Boundaries at exactly 70% and 90% (tolerance +/- 0.10) are NOT flagged."""
    # 7 inside out of 10 = exactly 70% coverage
    leads = [1] * 10
    q10 = [10.0] * 10
    q50 = [20.0] * 10
    q90 = [30.0] * 10
    obs_70 = [15.0, 20.0, 25.0, 12.0, 18.0, 22.0, 28.0, 5.0, 35.0, 40.0]

    res_70 = check_range_coverage(leads, obs_70, q10, q50, q90)[0]
    assert math.isclose(res_70.coverage, 0.70)
    assert res_70.needs_display_warning is False

    # 9 inside out of 10 = exactly 90% coverage
    obs_90 = [15.0, 20.0, 25.0, 12.0, 18.0, 22.0, 28.0, 11.0, 29.0, 50.0]

    res_90 = check_range_coverage(leads, obs_90, q10, q50, q90)[0]
    assert math.isclose(res_90.coverage, 0.90)
    assert res_90.needs_display_warning is False


def test_crossed_quantiles_are_defensively_sorted():
    """Test 5: Crossed quantiles are defensively sorted so q10 <= q50 <= q90."""
    leads = [1]
    # Inverted / crossed quantiles: q10=40.0, q50=20.0, q90=10.0
    q10 = [40.0]
    q50 = [20.0]
    q90 = [10.0]
    # Observed value is 25.0
    # After sorting: q10=10.0, q50=20.0, q90=40.0 -> 25.0 is inside [10.0, 40.0]
    obs = [25.0]

    results = check_range_coverage(leads, obs, q10, q50, q90)
    res = results[0]
    assert math.isclose(res.coverage, 1.0)


def test_missing_rows_excluded():
    """Test 6: Rows with missing observation or missing quantiles are excluded."""
    leads = [1, 1, 1, 1]
    obs = [20.0, np.nan, 20.0, 20.0]
    q10 = [10.0, 10.0, np.nan, 10.0]
    q50 = [20.0, 20.0, 20.0, np.nan]
    q90 = [30.0, 30.0, 30.0, 30.0]

    # Only row 0 has all valid values
    results = check_range_coverage(leads, obs, q10, q50, q90)
    assert len(results) == 1
    res = results[0]
    assert res.n_samples == 1
    assert math.isclose(res.coverage, 1.0)


def test_leads_calculated_independently():
    """Test 7: Leads are evaluated completely independently."""
    leads = [1] * 10 + [2] * 10
    q10 = [10.0] * 20
    q50 = [20.0] * 20
    q90 = [30.0] * 20

    # Lead 1 has 8 inside out of 10 (80% coverage -> unflagged)
    obs_lead1 = [15.0, 20.0, 25.0, 10.0, 30.0, 12.0, 18.0, 22.0, 5.0, 35.0]
    # Lead 2 has 5 inside out of 10 (50% coverage -> flagged)
    obs_lead2 = [15.0, 20.0, 25.0, 10.0, 30.0, 2.0, 4.0, 6.0, 8.0, 45.0]

    obs = obs_lead1 + obs_lead2

    results = check_range_coverage(leads, obs, q10, q50, q90)
    assert len(results) == 2

    res_l1 = next(r for r in results if r.lead_day == 1)
    res_l2 = next(r for r in results if r.lead_day == 2)

    assert math.isclose(res_l1.coverage, 0.80)
    assert res_l1.needs_display_warning is False

    assert math.isclose(res_l2.coverage, 0.50)
    assert res_l2.needs_display_warning is True


def test_holdout_input_rejected():
    """Test 8: Holdout input is strictly rejected per PRD §10.4, §10.6 L4."""
    leads = [1, 2]
    obs = [10.0, 20.0]
    q10 = [5.0, 10.0]
    q50 = [10.0, 20.0]
    q90 = [20.0, 30.0]

    # String 'holdout'
    with pytest.raises(ValueError, match="Holdout data cannot be used"):
        check_range_coverage(leads, obs, q10, q50, q90, evaluation_set="holdout")

    # Sequence containing 'holdout'
    with pytest.raises(ValueError, match="Holdout data detected"):
        check_range_coverage(
            leads, obs, q10, q50, q90, evaluation_set=["development", "holdout"]
        )

    # Non-OOF prediction_source
    with pytest.raises(ValueError, match="prediction_source must be 'oof'"):
        check_range_coverage(leads, obs, q10, q50, q90, prediction_source="train")
