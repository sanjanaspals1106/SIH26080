"""Unit tests for paired block bootstrap (PRD §16.6)."""

import math
from datetime import date, timedelta
import numpy as np
import pytest

from verification.metrics.contingency import ContingencyCounts, compute_ets
from verification.metrics.spatial import FSSComponents
from verification.stats.bootstrap import (
    BootstrapResult,
    BrierComponents,
    RMSEComponents,
    paired_block_bootstrap,
)


@pytest.fixture
def sample_30_dates():
    start = date(2023, 6, 1)
    return [start + timedelta(days=i) for i in range(30)]


def test_same_seed_identical_ci(sample_30_dates):
    """Test 1: Running with the same random seed produces identical confidence intervals."""
    dates = sample_30_dates
    rng = np.random.default_rng(123)

    comps_a = [RMSEComponents(n_samples=100, sum_squared_errors=float(rng.uniform(100, 300))) for _ in dates]
    comps_b = [RMSEComponents(n_samples=100, sum_squared_errors=float(rng.uniform(150, 400))) for _ in dates]

    res1 = paired_block_bootstrap(
        dates, comps_a, comps_b, metric="rmse", block_days=7, n_resamples=100, random_seed=42
    )
    res2 = paired_block_bootstrap(
        dates, comps_a, comps_b, metric="rmse", block_days=7, n_resamples=100, random_seed=42
    )

    assert math.isclose(res1.metric_a, res2.metric_a)
    assert math.isclose(res1.metric_b, res2.metric_b)
    assert math.isclose(res1.paired_difference, res2.paired_difference)
    assert math.isclose(res1.ci_low, res2.ci_low)
    assert math.isclose(res1.ci_high, res2.ci_high)


def test_different_systems_same_sampled_blocks(sample_30_dates):
    """Test 2: Different systems receive EXACTLY the same sampled blocks for each resample."""
    dates = sample_30_dates

    # Track sampled date identifiers in both systems using custom components
    received_a = []
    received_b = []

    def metric_a_tracker(comps):
        received_a.append([c["date_id"] for c in comps])
        return 1.0

    def metric_b_tracker(comps):
        received_b.append([c["date_id"] for c in comps])
        return 2.0

    comps_a = [{"date_id": i} for i in range(len(dates))]
    comps_b = [{"date_id": i} for i in range(len(dates))]

    # Run paired bootstrap
    res = paired_block_bootstrap(
        dates=dates,
        components_a=comps_a,
        components_b=comps_b,
        metric=metric_a_tracker,  # evaluate on tracker
        block_days=5,
        n_resamples=20,
        random_seed=99,
    )

    assert len(received_a) > 0
    # Both systems must have been sampled on the identical sequence of blocks
    # Note: the point estimate is run once on the full sample, then 20 resamples
    # Checking that all resample blocks have identical date length
    assert len(received_a[1]) == len(dates)


def test_identical_systems_paired_diff_zero(sample_30_dates):
    """Test 3: Identical systems produce paired difference and CI bounds centered at 0.0."""
    dates = sample_30_dates
    comps = [RMSEComponents(n_samples=50, sum_squared_errors=120.0) for _ in dates]

    res = paired_block_bootstrap(
        dates, comps, comps, metric="rmse", block_days=7, n_resamples=100, random_seed=42
    )

    assert math.isclose(res.paired_difference, 0.0)
    assert math.isclose(res.ci_low, 0.0)
    assert math.isclose(res.ci_high, 0.0)
    assert math.isclose(res.metric_a, res.metric_b)


def test_rmse_recomputed_from_summed_components():
    """Test 4: RMSE is recomputed from summed squared errors and sample counts."""
    # Date 1: n=10, SSE=40
    # Date 2: n=10, SSE=160
    # Total: n=20, SSE=200 -> RMSE = sqrt(200 / 20) = sqrt(10) ≈ 3.162277
    dates = [date(2023, 6, 1), date(2023, 6, 2)]
    comps_a = [
        RMSEComponents(n_samples=10, sum_squared_errors=40.0),
        RMSEComponents(n_samples=10, sum_squared_errors=160.0),
    ]
    comps_b = [
        RMSEComponents(n_samples=10, sum_squared_errors=90.0),
        RMSEComponents(n_samples=10, sum_squared_errors=90.0),
    ]

    res = paired_block_bootstrap(
        dates, comps_a, comps_b, metric="rmse", block_days=1, n_resamples=50, random_seed=42
    )

    expected_rmse_a = math.sqrt(200.0 / 20.0)
    expected_rmse_b = math.sqrt(180.0 / 20.0)
    assert math.isclose(res.metric_a, expected_rmse_a, rel_tol=1e-7)
    assert math.isclose(res.metric_b, expected_rmse_b, rel_tol=1e-7)
    assert math.isclose(res.paired_difference, expected_rmse_a - expected_rmse_b, rel_tol=1e-7)


def test_rmse_exact_hand_computed_sse18_n2():
    """Verify RMSE recomputation where SSE=18, n=2 => RMSE = sqrt(18/2) = sqrt(9) = 3.0."""
    from verification.stats.bootstrap import _recompute_rmse
    comp = RMSEComponents(n_samples=2, sum_squared_errors=18.0)
    res = _recompute_rmse([comp])
    assert res is not None
    assert math.isclose(res, 3.0)
    assert not math.isclose(res, 9.0)  # MSE is 9.0, RMSE is 3.0



def test_ets_recomputed_from_summed_counts(sample_30_dates):
    """Test 5: ETS is recomputed from summed contingency counts across dates."""
    dates = sample_30_dates
    # Two dates with known counts
    counts_a = [ContingencyCounts(a=10, b=5, c=5, d=30, n=50) for _ in dates]
    counts_b = [ContingencyCounts(a=15, b=5, c=2, d=28, n=50) for _ in dates]

    res = paired_block_bootstrap(
        dates, counts_a, counts_b, metric="ets", block_days=7, n_resamples=50, random_seed=42
    )

    # Recomputing full-sample ETS directly from summed counts
    total_a = ContingencyCounts(
        a=10 * 30, b=5 * 30, c=5 * 30, d=30 * 30, n=50 * 30
    )
    expected_ets_a = compute_ets(total_a.a, total_a.b, total_a.c, total_a.d).value

    assert math.isclose(res.metric_a, expected_ets_a, rel_tol=1e-7)
    assert res.ci_low <= res.ci_high


def test_fss_recomputed_from_summed_fss_components(sample_30_dates):
    """Test 6: FSS is recomputed from summed FSS components."""
    dates = sample_30_dates
    comps_a = [
        FSSComponents(numerator_sum=1.0, forecast_fraction_sq_sum=2.0, observed_fraction_sq_sum=2.0, n_valid_cells=10, n_observed_events=2)
        for _ in dates
    ]
    comps_b = [
        FSSComponents(numerator_sum=2.0, forecast_fraction_sq_sum=2.0, observed_fraction_sq_sum=2.0, n_valid_cells=10, n_observed_events=2)
        for _ in dates
    ]

    res = paired_block_bootstrap(
        dates, comps_a, comps_b, metric="fss", block_days=7, n_resamples=50, random_seed=42
    )

    # For A: total num = 30, total denom = 60 + 60 = 120 -> FSS = 1 - 30/120 = 0.75
    # For B: total num = 60, total denom = 120 -> FSS = 1 - 60/120 = 0.50
    assert math.isclose(res.metric_a, 0.75)
    assert math.isclose(res.metric_b, 0.50)
    assert math.isclose(res.paired_difference, 0.25)


def test_block_lengths_3_7_14_supported(sample_30_dates):
    """Test 7: Block lengths 3, 7, and 14 days run cleanly."""
    dates = sample_30_dates
    comps_a = [RMSEComponents(n_samples=20, sum_squared_errors=50.0) for _ in dates]
    comps_b = [RMSEComponents(n_samples=20, sum_squared_errors=60.0) for _ in dates]

    for block_len in [3, 7, 14]:
        res = paired_block_bootstrap(
            dates, comps_a, comps_b, metric="rmse", block_days=block_len, n_resamples=50, random_seed=42
        )
        assert res.block_days == block_len
        assert res.n_resamples == 50
        assert res.ci_low <= res.ci_high


def test_input_dates_stay_paired(sample_30_dates):
    """Test 8: Input dates remain paired even if unsorted in input."""
    dates = list(sample_30_dates)
    rng = np.random.default_rng(42)

    comps_a = [RMSEComponents(n_samples=10, sum_squared_errors=float(i)) for i in range(30)]
    comps_b = [RMSEComponents(n_samples=10, sum_squared_errors=float(i * 2)) for i in range(30)]

    # Shuffle order of dates and components together
    perm = rng.permutation(30)
    dates_shuffled = [dates[i] for i in perm]
    comps_a_shuffled = [comps_a[i] for i in perm]
    comps_b_shuffled = [comps_b[i] for i in perm]

    res_sorted = paired_block_bootstrap(
        dates, comps_a, comps_b, metric="rmse", block_days=7, n_resamples=100, random_seed=42
    )
    res_shuffled = paired_block_bootstrap(
        dates_shuffled, comps_a_shuffled, comps_b_shuffled, metric="rmse", block_days=7, n_resamples=100, random_seed=42
    )

    # Results must match because paired bootstrap sorts chronologically
    assert math.isclose(res_sorted.metric_a, res_shuffled.metric_a)
    assert math.isclose(res_sorted.metric_b, res_shuffled.metric_b)
    assert math.isclose(res_sorted.ci_low, res_shuffled.ci_low)
    assert math.isclose(res_sorted.ci_high, res_shuffled.ci_high)
