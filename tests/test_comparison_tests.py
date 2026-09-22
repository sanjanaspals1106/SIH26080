"""Unit tests for remaining statistical checks from PRD §16.6.

Tests:
1. season consistency lower-is-better
2. season consistency higher-is-better
3. Wilcoxon blocked when <6 seasons
4. Wilcoxon runs with >=6 seasons
5. missing season pairs excluded
6. DM identical series handled safely
7. DM detects clearly shifted synthetic losses
8. DM uses paired dates only
9. DM excludes missing values
10. deterministic HAC lag selection
11. too-few-date DM returns reason
12. no t-test/ordinary independent-sample logic used
"""

import numpy as np
import pytest

from verification.stats.comparison_tests import (
    DieboldMarianoResult,
    SeasonConsistencyResult,
    WilcoxonResult,
    compute_diebold_mariano,
    compute_season_consistency,
    compute_season_wilcoxon,
    get_dm_lag,
)


def test_1_season_consistency_lower_is_better():
    """Test 1: Verify season consistency counting when lower metric is better (e.g. RMSE)."""
    # System A vs System B per season (RMSE values)
    # Season 0: 3.0 vs 3.5 -> A better
    # Season 1: 4.5 vs 4.0 -> B better
    # Season 2: 2.0 vs 2.5 -> A better
    # Season 3: 6.0 vs 6.0 -> Tie
    # Season 4: 5.0 vs 5.2 -> A better
    a = [3.0, 4.5, 2.0, 6.0, 5.0]
    b = [3.5, 4.0, 2.5, 6.0, 5.2]

    res = compute_season_consistency(a, b, direction="lower")

    assert res.n_seasons == 5
    assert res.n_seasons_a_better == 3
    assert res.n_seasons_b_better == 1
    assert res.n_ties == 1
    assert res.direction == "lower"


def test_2_season_consistency_higher_is_better():
    """Test 2: Verify season consistency counting when higher metric is better (e.g. ETS/FSS)."""
    # Season 0: 0.45 vs 0.40 -> A better
    # Season 1: 0.50 vs 0.55 -> B better
    # Season 2: 0.30 vs 0.30 -> Tie
    # Season 3: 0.40 vs 0.35 -> A better
    a = [0.45, 0.50, 0.30, 0.40]
    b = [0.40, 0.55, 0.30, 0.35]

    res = compute_season_consistency(a, b, direction="higher")

    assert res.n_seasons == 4
    assert res.n_seasons_a_better == 2
    assert res.n_seasons_b_better == 1
    assert res.n_ties == 1
    assert res.direction == "higher"

    # Reject invalid direction
    with pytest.raises(ValueError, match="Invalid metric direction"):
        compute_season_consistency(a, b, direction="auto")


def test_3_wilcoxon_blocked_when_fewer_than_6_seasons():
    """Test 3: Wilcoxon test is blocked and returns INSUFFICIENT_SEASONS when < 6 seasons."""
    a = [2.0, 3.0, 2.5, 4.0, 3.5]
    b = [2.2, 3.1, 2.4, 4.2, 3.6]

    res = compute_season_wilcoxon(a, b)

    assert res.statistic is None
    assert res.p_value is None
    assert res.n_seasons == 5
    assert res.reason == "INSUFFICIENT_SEASONS"


def test_4_wilcoxon_runs_with_ge_6_seasons():
    """Test 4: Wilcoxon test runs successfully when >= 6 seasons are available."""
    # 8 seasons
    a = [2.0, 2.5, 1.8, 3.0, 2.2, 1.9, 2.8, 2.1]
    b = [2.5, 2.9, 2.2, 3.5, 2.6, 2.3, 3.1, 2.4]

    res = compute_season_wilcoxon(a, b)

    assert res.statistic is not None
    assert res.p_value is not None
    assert 0.0 <= res.p_value <= 1.0
    assert res.n_seasons == 8
    assert res.reason is None


def test_5_missing_season_pairs_excluded():
    """Test 5: Seasons with missing values (None or NaN) in either system are cleanly dropped."""
    a = {
        2015: 1.0,
        2016: None,      # missing in A
        2017: 2.0,
        2018: 3.0,
        2019: 4.0,
        2020: 5.0,
        2021: 6.0,
        2022: float("nan"),  # NaN in A
    }
    b = {
        2015: 1.1,
        2016: 2.0,
        2017: 2.1,
        2018: 3.1,
        2019: 4.1,
        2020: 5.1,
        2021: None,      # missing in B
        2022: 6.1,
    }

    # Only 5 seasons are complete: 2015, 2017, 2018, 2019, 2020
    res_cons = compute_season_consistency(a, b, direction="lower")
    assert res_cons.n_seasons == 5

    res_wilc = compute_season_wilcoxon(a, b)
    assert res_wilc.n_seasons == 5
    assert res_wilc.reason == "INSUFFICIENT_SEASONS"
    assert res_wilc.statistic is None


def test_6_dm_identical_series_handled_safely():
    """Test 6: Identical daily losses return neutral result without ZeroDivisionError or crash."""
    loss_a = [15.0] * 25
    loss_b = [15.0] * 25

    res = compute_diebold_mariano(loss_a, loss_b)

    assert res.dm_statistic == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)
    assert res.mean_loss_difference == pytest.approx(0.0)
    assert res.n_dates == 25
    assert res.reason == "IDENTICAL_SERIES"


def test_7_dm_detects_clearly_shifted_synthetic_losses():
    """Test 7: Diebold-Mariano test correctly detects significant loss difference between systems."""
    rng = np.random.RandomState(42)
    t = 60
    loss_a = 30.0 + rng.normal(0, 1.0, t)
    loss_b = 10.0 + rng.normal(0, 1.0, t)

    res = compute_diebold_mariano(loss_a, loss_b)

    assert res.dm_statistic is not None
    assert res.dm_statistic > 10.0
    assert res.p_value < 1e-6
    assert res.mean_loss_difference == pytest.approx(20.0, abs=0.5)
    assert res.reason is None


def test_8_dm_uses_paired_dates_only():
    """Test 8: Only mutually present dates are paired and evaluated in DM test."""
    dates_a = [f"2024-07-{d:02d}" for d in range(1, 16)]  # days 1 to 15
    dates_b = [f"2024-07-{d:02d}" for d in range(5, 21)]  # days 5 to 20

    dict_a = {d: 10.0 for d in dates_a}
    dict_b = {d: 8.0 for d in dates_b}

    # Common dates are days 5 to 15 (11 dates)
    res = compute_diebold_mariano(dict_a, dict_b)

    assert res.n_dates == 11
    assert res.mean_loss_difference == pytest.approx(2.0)


def test_9_dm_excludes_missing_values():
    """Test 9: Dates containing None or NaN in either loss series are excluded."""
    loss_a = [10.0] * 20
    loss_b = [8.0] * 20

    loss_a[2] = None
    loss_a[5] = float("nan")
    loss_b[8] = None
    loss_b[10] = float("nan")

    # 4 dates have missing values -> 16 valid paired dates
    res = compute_diebold_mariano(loss_a, loss_b)

    assert res.n_dates == 16
    assert res.mean_loss_difference == pytest.approx(2.0)


def test_10_deterministic_hac_lag_selection():
    """Test 10: Deterministic lag selection rules for HAC variance."""
    # 1. Explicit caller lag
    assert get_dm_lag(n_dates=100, lag=7) == 7

    # 2. Lead-day rule: h-step ahead forecast has (lead_day - 1) autocorrelation lag
    assert get_dm_lag(n_dates=100, lead_day=3) == 2
    assert get_dm_lag(n_dates=100, lead_day=1) == 4  # lead_day=1 falls back to sample size rule

    # 3. Sample-size based rule: floor(4 * (T / 100)^(2/9))
    # For T = 100: 4 * 1.0 = 4
    assert get_dm_lag(n_dates=100) == 4
    # For T = 20: floor(4 * (0.2)^0.222) = floor(4 * 0.6988) = 2
    assert get_dm_lag(n_dates=20) == 2


def test_11_too_few_date_dm_returns_reason():
    """Test 11: Too few dates (< 10) returns None statistics and INSUFFICIENT_DATES reason."""
    loss_a = [10.0, 12.0, 9.0, 11.0, 10.5]
    loss_b = [8.0, 9.0, 7.5, 9.5, 8.5]

    res = compute_diebold_mariano(loss_a, loss_b)

    assert res.dm_statistic is None
    assert res.p_value is None
    assert res.mean_loss_difference is None
    assert res.n_dates == 5
    assert res.reason == "INSUFFICIENT_DATES"


def test_12_no_t_test_ordinary_independent_sample_logic_used():
    """Test 12: Verify that DM accounts for autocorrelation via HAC variance rather than naive independent t-test."""
    rng = np.random.RandomState(42)
    t = 120
    # Construct AR(1) loss differentials with strong positive autocorrelation
    innovations = rng.normal(0, 1.0, t)
    diff = np.zeros(t)
    for i in range(1, t):
        diff[i] = 0.75 * diff[i - 1] + innovations[i]
    diff += 0.3  # Add true mean difference

    loss_b = np.ones(t) * 10.0
    loss_a = loss_b + diff

    # Compute naive sample standard error (as in ordinary independent t-test)
    sample_var = float(np.var(diff, ddof=1))
    naive_se = np.sqrt(sample_var / t)
    naive_t = float(np.mean(diff) / naive_se)

    # Compute DM with HAC
    res = compute_diebold_mariano(loss_a, loss_b)

    assert res.dm_statistic is not None
    # Because autocorrelation is positive, HAC long-run variance is strictly greater than naive variance,
    # making the DM statistic smaller (less artificially overconfident) than the naive t-statistic.
    assert abs(res.dm_statistic) < abs(naive_t)
