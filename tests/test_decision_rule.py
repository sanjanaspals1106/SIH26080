"""Unit tests for PRD §10.8 decision rule and bootstrap RMSE square root verification."""

import math
from datetime import date, timedelta
import pytest

from verification.decision import (
    STATUS_B2_BETTER,
    STATUS_B3_BETTER,
    STATUS_NO_DIFF,
    classify_paired_metric,
    evaluate_regime_benefit_decision,
)
from verification.stats.bootstrap import (
    RMSEComponents,
    _recompute_rmse,
    paired_block_bootstrap,
)


def test_p1_and_p2_favor_b3_none_worse_demonstrates_benefit():
    """Test 1: P1 and P2 significantly favor B3, and no metric favors B2 => demonstrated_benefit = True."""
    intervals = {
        "P1": (-0.80, -0.15),   # RMSE lower is better -> strictly negative -> B3_BETTER
        "P2": (0.03, 0.09),     # ETS higher is better -> strictly positive -> B3_BETTER
        "P3": (-0.02, 0.04),    # FSS crosses 0 -> NO_SIGNIFICANT_DIFFERENCE
        "P4": (-0.01, 0.03),    # BSS crosses 0 -> NO_SIGNIFICANT_DIFFERENCE
    }
    decision = evaluate_regime_benefit_decision(intervals)

    assert decision.demonstrated_benefit is True
    assert decision.per_metric_status["P1"] == STATUS_B3_BETTER
    assert decision.per_metric_status["P2"] == STATUS_B3_BETTER
    assert decision.per_metric_status["P3"] == STATUS_NO_DIFF
    assert decision.per_metric_status["P4"] == STATUS_NO_DIFF
    assert decision.supporting_metrics == ["P1", "P2"]
    assert decision.worse_metrics == []
    assert decision.failed_conditions == []


def test_p1_not_significant_fails():
    """Test 2: P1 not significantly favoring B3 => demonstrated_benefit = False."""
    intervals = {
        "P1": (-0.20, 0.05),    # RMSE crosses 0 -> NO_SIGNIFICANT_DIFFERENCE
        "P2": (0.02, 0.08),     # B3_BETTER
        "P3": (0.01, 0.06),     # B3_BETTER
        "P4": (0.02, 0.05),     # B3_BETTER
    }
    decision = evaluate_regime_benefit_decision(intervals)

    assert decision.demonstrated_benefit is False
    assert decision.per_metric_status["P1"] == STATUS_NO_DIFF
    assert "P1_RMSE_DOES_NOT_FAVOR_B3" in decision.failed_conditions


def test_p1_better_but_p2_p4_all_nonsignificant_fails():
    """Test 3: P1 favors B3, but none of P2, P3, P4 favors B3 => demonstrated_benefit = False."""
    intervals = {
        "P1": (-0.60, -0.10),   # B3_BETTER
        "P2": (-0.02, 0.02),    # NO_SIGNIFICANT_DIFFERENCE
        "P3": (-0.03, 0.01),    # NO_SIGNIFICANT_DIFFERENCE
        "P4": (-0.04, 0.02),    # NO_SIGNIFICANT_DIFFERENCE
    }
    decision = evaluate_regime_benefit_decision(intervals)

    assert decision.demonstrated_benefit is False
    assert "NO_CATEGORICAL_OR_PROBABILITY_METRIC_FAVORS_B3" in decision.failed_conditions


def test_required_wins_but_one_favors_b2_fails():
    """Test 4: P1 and P2 favor B3, but P3 significantly favors B2 => demonstrated_benefit = False."""
    intervals = {
        "P1": (-0.50, -0.10),   # B3_BETTER
        "P2": (0.02, 0.07),     # B3_BETTER
        "P3": (-0.08, -0.01),   # FSS strictly negative -> B2_BETTER (B3 is significantly worse)
        "P4": (-0.02, 0.03),    # NO_SIGNIFICANT_DIFFERENCE
    }
    decision = evaluate_regime_benefit_decision(intervals)

    assert decision.demonstrated_benefit is False
    assert decision.worse_metrics == ["P3"]
    assert "PRIMARY_METRIC_SIGNIFICANTLY_FAVORS_B2" in decision.failed_conditions


def test_ci_touching_zero_is_nonsignificant():
    """Test 5: CI touching 0 (boundary at 0) is strictly classified as NO_SIGNIFICANT_DIFFERENCE."""
    # For higher is better (ETS)
    assert classify_paired_metric("P2", 0.0, 0.10) == STATUS_NO_DIFF
    assert classify_paired_metric("P2", -0.10, 0.0) == STATUS_NO_DIFF

    # For lower is better (RMSE)
    assert classify_paired_metric("P1", -0.20, 0.0) == STATUS_NO_DIFF
    assert classify_paired_metric("P1", 0.0, 0.20) == STATUS_NO_DIFF


def test_correct_direction_for_rmse():
    """Test 6: P1 RMSE uses LOWER is better."""
    # Strictly negative (ci_high < 0) => B3 has lower RMSE => B3_BETTER
    assert classify_paired_metric("P1", -0.40, -0.05) == STATUS_B3_BETTER

    # Strictly positive (ci_low > 0) => B2 has lower RMSE => B2_BETTER
    assert classify_paired_metric("P1", 0.05, 0.40) == STATUS_B2_BETTER


def test_correct_direction_for_ets_fss_bss():
    """Test 7: P2 (ETS), P3 (FSS), P4 (BSS) use HIGHER is better."""
    for metric_name in ["P2", "P3", "P4", "ETS", "FSS", "BSS"]:
        # Strictly positive (ci_low > 0) => B3 has higher score => B3_BETTER
        assert classify_paired_metric(metric_name, 0.02, 0.10) == STATUS_B3_BETTER

        # Strictly negative (ci_high < 0) => B2 has higher score => B2_BETTER
        assert classify_paired_metric(metric_name, -0.10, -0.02) == STATUS_B2_BETTER


def test_bootstrap_rmse_hand_check_uses_square_root():
    """Test 8: Explicit hand-check verifying bootstrap RMSE recomputation uses sqrt(SSE / n) and MSE != RMSE.

    Example:
      sum_squared_errors = 1800.0, n_samples = 200
      Mean Squared Error (MSE) = 1800.0 / 200 = 9.0
      Root Mean Squared Error (RMSE) = sqrt(9.0) = 3.0
      Verify RMSE != MSE.
    """
    comp = RMSEComponents(n_samples=200, sum_squared_errors=1800.0)
    mse = comp.sum_squared_errors / comp.n_samples
    rmse = _recompute_rmse([comp])

    assert math.isclose(mse, 9.0)
    assert rmse is not None
    assert math.isclose(rmse, 3.0)
    assert not math.isclose(rmse, mse)

    # Verify end-to-end via paired_block_bootstrap
    start_date = date(2023, 6, 1)
    dates = [start_date + timedelta(days=i) for i in range(10)]
    comps_a = [RMSEComponents(n_samples=20, sum_squared_errors=180.0) for _ in dates]
    comps_b = [RMSEComponents(n_samples=20, sum_squared_errors=720.0) for _ in dates]
    # For A: total SSE = 1800, total n = 200 -> RMSE = sqrt(9.0) = 3.0 (MSE = 9.0)
    # For B: total SSE = 7200, total n = 200 -> RMSE = sqrt(36.0) = 6.0 (MSE = 36.0)
    # Paired diff = 3.0 - 6.0 = -3.0

    res = paired_block_bootstrap(
        dates=dates,
        components_a=comps_a,
        components_b=comps_b,
        metric="rmse",
        block_days=2,
        n_resamples=50,
        random_seed=42,
    )

    assert math.isclose(res.metric_a, 3.0)
    assert math.isclose(res.metric_b, 6.0)
    assert math.isclose(res.paired_difference, -3.0)
    # Verify neither point estimate is equal to MSE
    assert not math.isclose(res.metric_a, 9.0)
    assert not math.isclose(res.metric_b, 36.0)
