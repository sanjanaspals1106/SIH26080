"""Unit tests and hand-computed checks for verification scalar metrics (PRD §16, Appendix A)."""

import math
import numpy as np
import pytest

from verification.metrics import (
    ContingencyCounts,
    MetricResult,
    compute_bias,
    compute_brier_score,
    compute_brier_skill_score,
    compute_contingency_counts,
    compute_contingency_metrics,
    compute_continuous_metrics,
    compute_coverage,
    compute_csi,
    compute_ets,
    compute_far,
    compute_frequency_bias,
    compute_mae,
    compute_pinball_loss,
    compute_pod,
    compute_rmse,
)


# ---------------------------------------------------------------------------
# Continuous metrics tests
# ---------------------------------------------------------------------------

def test_continuous_hand_computed():
    """F = [10.0, 20.0, 30.0], O = [12.0, 18.0, 34.0].
    Errors: -2, +2, -4
    Bias: (-2 + 2 - 4) / 3 = -4/3
    MAE:  (2 + 2 + 4) / 3  = 8/3
    RMSE: sqrt((4 + 4 + 16) / 3) = sqrt(8)
    """
    f = [10.0, 20.0, 30.0]
    o = [12.0, 18.0, 34.0]

    bias = compute_bias(f, o)
    mae = compute_mae(f, o)
    rmse = compute_rmse(f, o)

    assert bias.is_defined()
    assert math.isclose(bias.value, -4.0 / 3.0, rel_tol=1e-7)
    assert bias.undefined_reason is None

    assert mae.is_defined()
    assert math.isclose(mae.value, 8.0 / 3.0, rel_tol=1e-7)

    assert rmse.is_defined()
    assert math.isclose(rmse.value, math.sqrt(8.0), rel_tol=1e-7)


def test_continuous_with_nans():
    """NaNs in forecast or observed must be safely excluded."""
    f = [10.0, np.nan, 30.0, 40.0]
    o = [12.0, 18.0, 34.0, np.nan]
    # Valid indices: 0 and 2
    # Errors: -2.0, -4.0
    # Mean error = -3.0
    # MAE = 3.0
    # RMSE = sqrt((4 + 16)/2) = sqrt(10)

    res = compute_continuous_metrics(f, o)
    assert math.isclose(res["bias"].value, -3.0)
    assert math.isclose(res["mae"].value, 3.0)
    assert math.isclose(res["rmse"].value, math.sqrt(10.0))


def test_continuous_all_nan_or_empty():
    """Empty arrays or all-NaN inputs must return None with NO_VALID_SAMPLES."""
    res_empty = compute_bias([], [])
    assert res_empty.value is None
    assert res_empty.undefined_reason == "NO_VALID_SAMPLES"
    assert not res_empty.is_defined()

    res_nan = compute_rmse([np.nan, np.nan], [1.0, 2.0])
    assert res_nan.value is None
    assert res_nan.undefined_reason == "NO_VALID_SAMPLES"


# ---------------------------------------------------------------------------
# Contingency metrics tests
# ---------------------------------------------------------------------------

def test_contingency_counts_from_arrays():
    """Counts a, b, c, d at threshold t = 15.6 mm."""
    f = [10.0, 20.0, 15.6, 5.0, np.nan]
    o = [5.0, 25.0, 12.0, 16.0, 20.0]
    # Valid pairs:
    # 0: F=10.0 (<15.6), O=5.0 (<15.6)   -> d (correct neg)
    # 1: F=20.0 (>=15.6), O=25.0 (>=15.6) -> a (hit)
    # 2: F=15.6 (>=15.6), O=12.0 (<15.6)  -> b (false alarm)
    # 3: F=5.0 (<15.6), O=16.0 (>=15.6)  -> c (miss)

    counts = compute_contingency_counts(f, o, threshold=15.6)
    assert counts.a == 1
    assert counts.b == 1
    assert counts.c == 1
    assert counts.d == 1
    assert counts.n == 4


def test_contingency_hand_computed():
    """Hand-computed 2x2 table:
    a = 40, b = 10, c = 20, d = 30, n = 100
    POD = 40 / 60 = 2/3
    FAR = 10 / 50 = 0.2
    CSI = 40 / 70 = 4/7
    Frequency bias = 50 / 60 = 5/6
    a_ref = (50 * 60) / 100 = 30
    ETS = (40 - 30) / (70 - 30) = 10 / 40 = 0.25
    """
    counts = ContingencyCounts(a=40, b=10, c=20, d=30, n=100)
    metrics = compute_contingency_metrics(counts)

    assert math.isclose(metrics["pod"].value, 2.0 / 3.0, rel_tol=1e-7)
    assert math.isclose(metrics["far"].value, 0.2, rel_tol=1e-7)
    assert math.isclose(metrics["csi"].value, 4.0 / 7.0, rel_tol=1e-7)
    assert math.isclose(metrics["frequency_bias"].value, 5.0 / 6.0, rel_tol=1e-7)
    assert math.isclose(metrics["ets"].value, 0.25, rel_tol=1e-7)

    # All defined
    for k, m in metrics.items():
        assert m.is_defined()
        assert m.undefined_reason is None


def test_contingency_edge_cases_never_zero():
    """PRD §16.7: If a+c=0 or a+b=0, affected metrics are undefined. Never store 0."""
    # Case 1: No observed events (a=0, c=0)
    pod = compute_pod(a=0, c=0)
    assert pod.value is None
    assert pod.value != 0
    assert pod.undefined_reason == "NO_OBSERVED_EVENTS"

    freq_bias = compute_frequency_bias(a=0, b=5, c=0)
    assert freq_bias.value is None
    assert freq_bias.undefined_reason == "NO_OBSERVED_EVENTS"

    ets_no_obs = compute_ets(a=0, b=5, c=0, d=10)
    assert ets_no_obs.value is None
    assert ets_no_obs.undefined_reason == "NO_OBSERVED_EVENTS"

    # Case 2: No forecast events (a=0, b=0)
    far = compute_far(a=0, b=0)
    assert far.value is None
    assert far.value != 0
    assert far.undefined_reason == "NO_FORECAST_EVENTS"

    ets_no_fcst = compute_ets(a=0, b=0, c=5, d=10)
    assert ets_no_fcst.value is None
    assert ets_no_fcst.undefined_reason == "NO_FORECAST_EVENTS"

    # Case 3: No events at all (a=0, b=0, c=0)
    csi = compute_csi(a=0, b=0, c=0)
    assert csi.value is None
    assert csi.value != 0
    assert csi.undefined_reason == "NO_EVENTS"

    # Case 4: No samples (n=0)
    ets_empty = compute_ets(a=0, b=0, c=0, d=0)
    assert ets_empty.value is None
    assert ets_empty.undefined_reason == "NO_SAMPLES"


# ---------------------------------------------------------------------------
# Probabilistic and range metrics tests
# ---------------------------------------------------------------------------

def test_brier_score_hand_computed():
    """p = [0.8, 0.2, 0.6, 0.1], y = [1, 0, 1, 0].
    p - y = [-0.2, 0.2, -0.4, 0.1]
    (p - y)^2 = [0.04, 0.04, 0.16, 0.01]
    sum = 0.25 -> mean = 0.0625
    """
    p = [0.8, 0.2, 0.6, 0.1]
    y = [1, 0, 1, 0]

    bs = compute_brier_score(p, y)
    assert bs.is_defined()
    assert math.isclose(bs.value, 0.0625, rel_tol=1e-7)


def test_brier_skill_score():
    """BSS = 1 - (BS / BS_ref).
    BS = 0.0625, BS_ref = 0.25 -> 1 - 0.25 = 0.75
    """
    bss = compute_brier_skill_score(0.0625, 0.25)
    assert bss.is_defined()
    assert math.isclose(bss.value, 0.75, rel_tol=1e-7)

    # Reference BS is 0
    bss_zero_ref = compute_brier_skill_score(0.05, 0.0)
    assert bss_zero_ref.value is None
    assert bss_zero_ref.undefined_reason == "ZERO_REFERENCE_BRIER_SCORE"


def test_pinball_loss_hand_computed():
    """q = [10.0, 20.0], y = [12.0, 16.0].
    alpha = 0.5:
      item 1: y - q = +2.0 -> 0.5 * 2.0 = 1.0
      item 2: y - q = -4.0 -> (-0.5) * (-4.0) = 2.0
      loss = (1.0 + 2.0) / 2 = 1.5

    alpha = 0.1:
      item 1: y - q = +2.0 -> 0.1 * 2.0 = 0.2
      item 2: y - q = -4.0 -> (-0.9) * (-4.0) = 3.6
      loss = (0.2 + 3.6) / 2 = 1.9
    """
    q = [10.0, 20.0]
    y = [12.0, 16.0]

    loss_50 = compute_pinball_loss(q, y, alpha=0.5)
    assert loss_50.is_defined()
    assert math.isclose(loss_50.value, 1.5, rel_tol=1e-7)

    loss_10 = compute_pinball_loss(q, y, alpha=0.1)
    assert loss_10.is_defined()
    assert math.isclose(loss_10.value, 1.9, rel_tol=1e-7)


def test_coverage_hand_computed():
    """q_low = [5, 10, 15, 20], q_high = [25, 30, 35, 40], y = [10, 35, 20, 50].
    item 1: 5 <= 10 <= 25 -> True
    item 2: 10 <= 35 <= 30 -> False
    item 3: 15 <= 20 <= 35 -> True
    item 4: 20 <= 50 <= 40 -> False
    Coverage = 2 / 4 = 0.50
    """
    ql = [5.0, 10.0, 15.0, 20.0]
    qh = [25.0, 30.0, 35.0, 40.0]
    y = [10.0, 35.0, 20.0, 50.0]

    cov = compute_coverage(ql, qh, y)
    assert cov.is_defined()
    assert math.isclose(cov.value, 0.50, rel_tol=1e-7)


def test_metric_result_unpacking():
    """Verify MetricResult can be unpacked as (val, reason)."""
    res = compute_pod(a=40, c=20)
    val, reason = res
    assert math.isclose(val, 2.0 / 3.0)
    assert reason is None
