"""Tests for Verification Component & Aggregation Layer (PRD §16.4, §16.5, §16.10).

Required tests:
1. two daily continuous components aggregate to correct RMSE
2. Bias and MAE aggregate correctly
3. contingency counts sum before computing metrics
4. FSS components sum before computing FSS
5. Brier components aggregate correctly
6. coverage counts aggregate correctly
7. pinball sums aggregate correctly
8. incompatible thresholds rejected
9. incompatible evaluation sets rejected
10. undefined denominator returns None/reason
11. n_samples aggregated correctly
12. n_events aggregated correctly
"""

import math
import pytest

from verification.aggregation import (
    aggregate_components,
    compute_bias_from_components,
    compute_brier_from_components,
    compute_contingency_metrics_from_components,
    compute_coverage_from_components,
    compute_csi_from_components,
    compute_far_from_components,
    compute_fss_from_components_record,
    compute_mae_from_components,
    compute_pinball_from_components,
    compute_pod_from_components,
    compute_rmse_from_components,
    group_and_aggregate,
)
from verification.components import VerificationComponent


def test_1_continuous_components_aggregate_to_correct_rmse():
    """Test 1: Two daily continuous components sum sum_squared_error and n before sqrt."""
    # Day 1: n=2, sum_squared_error=8.0 (daily RMSE = sqrt(4) = 2.0)
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        n=2,
        sum_squared_error=8.0,
    )
    # Day 2: n=2, sum_squared_error=10.0 (daily RMSE = sqrt(5) ≈ 2.236)
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        n=2,
        sum_squared_error=10.0,
    )

    agg = aggregate_components([c1, c2])
    res = compute_rmse_from_components(agg)

    # Correct RMSE = sqrt((8 + 10) / (2 + 2)) = sqrt(18 / 4) = sqrt(4.5) ≈ 2.12132
    # Simple average of daily RMSE would be (2.0 + 2.236068)/2 ≈ 2.118034 (WRONG)
    expected_rmse = math.sqrt(4.5)
    assert res.value == pytest.approx(expected_rmse)
    assert res.value != pytest.approx((2.0 + math.sqrt(5.0)) / 2.0)


def test_2_bias_and_mae_aggregate_correctly():
    """Test 2: Bias and MAE sum errors and divide by total n across dates."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        n=2,
        sum_error=4.0,
        sum_abs_error=6.0,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        n=3,
        sum_error=-1.0,
        sum_abs_error=9.0,
    )

    agg = aggregate_components([c1, c2])

    bias = compute_bias_from_components(agg)
    mae = compute_mae_from_components(agg)

    # Bias = (4.0 + -1.0) / 5 = 3.0 / 5 = 0.6
    assert bias.value == pytest.approx(0.6)
    # MAE = (6.0 + 9.0) / 5 = 15.0 / 5 = 3.0
    assert mae.value == pytest.approx(3.0)


def test_3_contingency_counts_sum_before_computing_metrics():
    """Test 3: Contingency counts (a, b, c, d) sum before computing POD, FAR, CSI."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        a=5,
        b=2,
        c=1,
        d=10,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        a=3,
        b=4,
        c=2,
        d=15,
    )

    agg = aggregate_components([c1, c2])

    # Aggregated counts: a=8, b=6, c=3, d=25
    assert agg.a == 8
    assert agg.b == 6
    assert agg.c == 3
    assert agg.d == 25

    pod = compute_pod_from_components(agg)
    far = compute_far_from_components(agg)
    csi = compute_csi_from_components(agg)

    # POD = 8 / (8 + 3) = 8 / 11
    assert pod.value == pytest.approx(8.0 / 11.0)
    # FAR = 6 / (8 + 6) = 6 / 14 = 3 / 7
    assert far.value == pytest.approx(6.0 / 14.0)
    # CSI = 8 / (8 + 6 + 3) = 8 / 17
    assert csi.value == pytest.approx(8.0 / 17.0)


def test_4_fss_components_sum_before_computing_fss():
    """Test 4: FSS components sum numerators and denominators across dates; never average daily FSS."""
    # Day 1: num=2, f_sq=5, o_sq=5 -> daily FSS = 1 - 2/10 = 0.80
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        neighbourhood_cells=3,
        fss_numerator_sum=2.0,
        fss_forecast_fraction_sq_sum=5.0,
        fss_observed_fraction_sq_sum=5.0,
    )
    # Day 2: num=4, f_sq=10, o_sq=10 -> daily FSS = 1 - 4/20 = 0.80
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        neighbourhood_cells=3,
        fss_numerator_sum=4.0,
        fss_forecast_fraction_sq_sum=10.0,
        fss_observed_fraction_sq_sum=10.0,
    )
    # Day 3: num=1, f_sq=1, o_sq=1 -> daily FSS = 1 - 1/2 = 0.50
    c3 = VerificationComponent(
        imd_date="2024-07-03",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        neighbourhood_cells=3,
        fss_numerator_sum=1.0,
        fss_forecast_fraction_sq_sum=1.0,
        fss_observed_fraction_sq_sum=1.0,
    )

    agg = aggregate_components([c1, c2, c3])
    res = compute_fss_from_components_record(agg)

    # Aggregated sums: num = 7.0, f_sq = 16.0, o_sq = 16.0
    # Correct FSS = 1 - 7.0 / (16.0 + 16.0) = 1 - 7/32 = 25/32 = 0.78125
    # Daily average = (0.8 + 0.8 + 0.5) / 3 = 0.70 (WRONG)
    assert res.value == pytest.approx(0.78125)
    assert res.value != pytest.approx(0.70)


def test_5_brier_components_aggregate_correctly():
    """Test 5: Brier terms and sample counts aggregate correctly."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        brier_n=10,
        sum_brier_terms=1.5,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        brier_n=20,
        sum_brier_terms=4.5,
    )

    agg = aggregate_components([c1, c2])
    res = compute_brier_from_components(agg)

    # BS = (1.5 + 4.5) / (10 + 20) = 6.0 / 30 = 0.20
    assert res.value == pytest.approx(0.20)


def test_6_coverage_counts_aggregate_correctly():
    """Test 6: Coverage hits and range_n aggregate correctly."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        range_n=50,
        coverage_count=40,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        range_n=50,
        coverage_count=42,
    )

    agg = aggregate_components([c1, c2])
    res = compute_coverage_from_components(agg)

    # Coverage = (40 + 42) / (50 + 50) = 82 / 100 = 0.82
    assert res.value == pytest.approx(0.82)


def test_7_pinball_sums_aggregate_correctly():
    """Test 7: Pinball sums across q10, q50, q90 aggregate correctly."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        range_n=10,
        pinball_q10_sum=2.0,
        pinball_q50_sum=5.0,
        pinball_q90_sum=3.0,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        range_n=10,
        pinball_q10_sum=4.0,
        pinball_q50_sum=7.0,
        pinball_q90_sum=5.0,
    )

    agg = aggregate_components([c1, c2])
    pinball = compute_pinball_from_components(agg)

    # q10 = (2 + 4) / 20 = 0.30
    # q50 = (5 + 7) / 20 = 0.60
    # q90 = (3 + 5) / 20 = 0.40
    assert pinball["q10"].value == pytest.approx(0.30)
    assert pinball["q50"].value == pytest.approx(0.60)
    assert pinball["q90"].value == pytest.approx(0.40)


def test_8_incompatible_thresholds_rejected():
    """Test 8: Aggregation across different thresholds raises ValueError."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
    )

    with pytest.raises(ValueError, match="Incompatible threshold_mm"):
        aggregate_components([c1, c2])

    with pytest.raises(ValueError, match="Incompatible threshold_mm"):
        c1 + c2


def test_9_incompatible_evaluation_sets_rejected():
    """Test 9: Aggregation across development and holdout evaluation sets raises ValueError."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
    )

    with pytest.raises(ValueError, match="Incompatible evaluation_set"):
        aggregate_components([c1, c2])


def test_10_undefined_denominator_returns_none_and_reason():
    """Test 10: Zero denominators return MetricResult(None, reason) rather than 0 or error."""
    # Zero samples for continuous
    c_empty = VerificationComponent(n=0)
    res_rmse = compute_rmse_from_components(c_empty)
    assert res_rmse.value is None
    assert res_rmse.undefined_reason == "NO_VALID_SAMPLES"

    # Zero observed events for contingency
    c_no_obs = VerificationComponent(a=0, b=5, c=0, d=20)
    res_pod = compute_pod_from_components(c_no_obs)
    assert res_pod.value is None
    assert res_pod.undefined_reason == "NO_OBSERVED_EVENTS"

    # Zero forecast events for contingency
    c_no_fc = VerificationComponent(a=0, b=0, c=5, d=20)
    res_far = compute_far_from_components(c_no_fc)
    assert res_far.value is None
    assert res_far.undefined_reason == "NO_FORECAST_EVENTS"

    # Zero FSS denominator
    c_fss_zero = VerificationComponent(
        fss_numerator_sum=0.0,
        fss_forecast_fraction_sq_sum=0.0,
        fss_observed_fraction_sq_sum=0.0,
    )
    res_fss = compute_fss_from_components_record(c_fss_zero)
    assert res_fss.value is None
    assert res_fss.undefined_reason == "ZERO_DENOMINATOR"


def test_11_n_samples_aggregated_correctly():
    """Test 11: n_samples is accumulated accurately across components."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        n_samples=100,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        n_samples=150,
    )

    agg = aggregate_components([c1, c2])
    assert agg.n_samples == 250


def test_12_n_events_aggregated_correctly():
    """Test 12: n_events is accumulated accurately when supplied."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        n_events=12,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        n_events=18,
    )

    agg = aggregate_components([c1, c2])
    assert agg.n_events == 30


def test_regression_no_n_events_inference_from_contingency():
    """Regression test: component with a=3, c=2 and no n_events must keep n_events=None."""
    comp = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        a=3,
        b=1,
        c=2,
        d=10,
    )
    assert comp.n_events is None
    assert comp.n_samples == 16


def test_group_and_aggregate():
    """Test grouping by metadata and aggregating each group."""
    c1 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=1,
        forecast_type="raw_nwp",
        evaluation_set="development",
        n=10,
        sum_error=20.0,
    )
    c2 = VerificationComponent(
        imd_date="2024-07-02",
        lead_day=1,
        forecast_type="raw_nwp",
        evaluation_set="development",
        n=10,
        sum_error=10.0,
    )
    c3 = VerificationComponent(
        imd_date="2024-07-01",
        lead_day=2,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        n=5,
        sum_error=5.0,
    )

    grouped = group_and_aggregate([c1, c2, c3], group_keys=["lead_day", "forecast_type"])
    assert len(grouped) == 2

    key_raw = (1, "raw_nwp")
    assert key_raw in grouped
    assert grouped[key_raw].n == 20
    assert grouped[key_raw].sum_error == 30.0

    key_ml = (2, "regime_aware_ml")
    assert key_ml in grouped
    assert grouped[key_ml].n == 5
    assert grouped[key_ml].sum_error == 5.0
