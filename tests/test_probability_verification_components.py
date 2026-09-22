"""Unit tests for probability verification components (PRD §13.5, §16.2, §16.4).

Required tests:
1. hand_computed_model_brier_sum
2. hand_computed_reference_brier_sum
3. aggregated_bss_correct
4. zero_reference_bs_undefined
5. reliability_bin_assignment_at_0_0
6. reliability_bin_assignment_at_1_0
7. boundary_probabilities_assigned_exactly_once
8. mean_predicted_probability_correct
9. observed_frequency_correct
10. empty_bin_represented_safely
11. missing_reference_causes_paired_exclusion
12. parquet_roundtrip_preserves_new_fields
13. existing_verification_tests_remain_green
"""

from pathlib import Path
import numpy as np
import pytest

from verification.aggregation import (
    aggregate_components,
    compute_brier_from_components,
    compute_brier_skill_score_from_components,
    compute_bss_from_components,
)
from verification.components import VerificationComponent
from verification.compute import (
    VerificationComputeResult,
    compute,
    compute_verification_components,
)
from verification.io import load_components, save_components
from verification.reliability import (
    ReliabilityBinSummary,
    assign_probability_bins,
    compute_reliability_bin_components,
    get_reliability_bins_from_config,
    summarize_reliability_bins,
)


def test_1_hand_computed_model_brier_sum():
    """Test 1: Hand-computed model Brier sum and sample counts."""
    # Forecast probabilities for >=10.0 mm: [0.8, 0.2]
    # Observed: [15.0, 5.0] -> outcomes: [1.0, 0.0]
    # Brier terms: (0.8 - 1.0)^2 + (0.2 - 0.0)^2 = 0.04 + 0.04 = 0.08
    # brier_n = 2
    f = [12.0, 4.0]
    obs = [15.0, 5.0]
    probs = {10.0: [0.8, 0.2]}

    res = compute(
        forecast=f,
        observed=obs,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[10.0],
        probabilities=probs,
        neighbourhood_sizes=None,
    )

    cat = res.get_contingency(threshold_mm=10.0)
    assert cat is not None
    assert cat.brier_n == 2
    assert cat.sum_brier_terms == pytest.approx(0.08)

    # Metric recomputation
    bs = compute_brier_from_components(cat)
    assert bs.value == pytest.approx(0.04)
    assert bs.undefined_reason is None


def test_2_hand_computed_reference_brier_sum():
    """Test 2: Hand-computed reference Brier sum when caller supplies climatology reference."""
    # Model probabilities: [0.8, 0.2]
    # Reference probabilities: [0.5, 0.5]
    # Observed: [15.0, 5.0] -> outcomes: [1.0, 0.0]
    # Reference Brier terms: (0.5 - 1.0)^2 + (0.5 - 0.0)^2 = 0.25 + 0.25 = 0.50
    # brier_n = 2
    f = [12.0, 4.0]
    obs = [15.0, 5.0]
    probs = {10.0: [0.8, 0.2]}
    probs_ref = {10.0: [0.5, 0.5]}

    res = compute(
        forecast=f,
        observed=obs,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[10.0],
        probabilities=probs,
        probability_reference=probs_ref,
        neighbourhood_sizes=None,
    )

    cat = res.get_contingency(threshold_mm=10.0)
    assert cat is not None
    assert cat.brier_n == 2
    assert cat.sum_brier_terms == pytest.approx(0.08)
    assert cat.sum_reference_brier_terms == pytest.approx(0.50)


def test_3_aggregated_bss_correct():
    """Test 3: BSS after aggregation: BSS = 1 - BS / BS_reference."""
    # Comp 1: n=2, sum_brier=0.08, sum_ref_brier=0.50
    c1 = VerificationComponent(
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=10.0,
        brier_n=2,
        sum_brier_terms=0.08,
        sum_reference_brier_terms=0.50,
    )
    # Comp 2: n=2, sum_brier=0.12, sum_ref_brier=0.30
    c2 = VerificationComponent(
        imd_date="2024-07-16",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=10.0,
        brier_n=2,
        sum_brier_terms=0.12,
        sum_reference_brier_terms=0.30,
    )

    aggregated = aggregate_components([c1, c2])
    assert aggregated.brier_n == 4
    assert aggregated.sum_brier_terms == pytest.approx(0.20)
    assert aggregated.sum_reference_brier_terms == pytest.approx(0.80)

    # BS = 0.20 / 4 = 0.05
    # BS_ref = 0.80 / 4 = 0.20
    # BSS = 1 - 0.05 / 0.20 = 1 - 0.25 = 0.75
    bss_res = compute_brier_skill_score_from_components(aggregated)
    assert bss_res.value == pytest.approx(0.75)
    assert bss_res.undefined_reason is None

    # Verify alias
    assert compute_bss_from_components(aggregated).value == pytest.approx(0.75)


def test_4_zero_reference_bs_undefined():
    """Test 4: Zero reference Brier score yields undefined reason ZERO_REFERENCE_BRIER_SCORE, not numeric 0."""
    comp = VerificationComponent(
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=10.0,
        brier_n=5,
        sum_brier_terms=0.15,
        sum_reference_brier_terms=0.0,  # Zero reference error
    )

    bss = compute_brier_skill_score_from_components(comp)
    assert bss.value is None
    assert bss.undefined_reason == "ZERO_REFERENCE_BRIER_SCORE"
    assert not bss.is_defined()


def test_5_reliability_bin_assignment_at_0_0():
    """Test 5: Probability 0.0 is assigned to [0.0, 0.1) and only that bin."""
    probs = [0.0]
    masks = assign_probability_bins(probs)
    assert len(masks) == 10

    # First bin [0.0, 0.1) contains 0.0
    assert bool(masks[0][0]) is True

    # No other bin contains 0.0
    for i in range(1, 10):
        assert bool(masks[i][0]) is False


def test_6_reliability_bin_assignment_at_1_0():
    """Test 6: Probability 1.0 is assigned to [0.9, 1.0] and only that bin."""
    probs = [1.0]
    masks = assign_probability_bins(probs)
    assert len(masks) == 10

    # Last bin [0.9, 1.0] contains 1.0
    assert bool(masks[9][0]) is True

    # No earlier bin contains 1.0
    for i in range(0, 9):
        assert bool(masks[i][0]) is False


def test_7_boundary_probabilities_assigned_exactly_once():
    """Test 7: Boundary values are assigned to exactly one bin (no overlap, no gaps)."""
    boundaries = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    masks = assign_probability_bins(boundaries)

    stacked = np.column_stack(masks)  # shape (11, 10)
    # Every probability must belong to exactly one bin
    bin_counts = np.sum(stacked, axis=1)
    np.testing.assert_array_equal(bin_counts, np.ones(len(boundaries), dtype=int))

    # Test half-open interval rule [lower, upper) for interior boundaries:
    # 0.1 must belong to bin 1 [0.1, 0.2), not bin 0 [0.0, 0.1)
    assert bool(masks[1][1]) is True  # 0.1 in bin 1
    assert bool(masks[0][1]) is False  # 0.1 not in bin 0

    # 0.5 must belong to bin 5 [0.5, 0.6), not bin 4 [0.4, 0.5)
    assert bool(masks[5][5]) is True
    assert bool(masks[4][5]) is False

    # 0.9 must belong to bin 9 [0.9, 1.0], not bin 8 [0.8, 0.9)
    assert bool(masks[9][9]) is True
    assert bool(masks[8][9]) is False


def test_8_mean_predicted_probability_correct():
    """Test 8: Aggregated mean predicted probability = sum_p / n."""
    # In bin [0.1, 0.2), probabilities are 0.12 and 0.18 -> sum = 0.30, n = 2 -> mean = 0.15
    probs = [0.12, 0.18]
    obs = [20.0, 5.0]

    comps = compute_reliability_bin_components(
        predicted_probabilities=probs,
        observed_grid_mm=obs,
        threshold_mm=15.6,
        lead_day=1,
    )
    summary = summarize_reliability_bins(comps)

    # Bin 1 is [0.1, 0.2)
    bin_1 = summary[1]
    assert bin_1.bin_lower == pytest.approx(0.1)
    assert bin_1.bin_upper == pytest.approx(0.2)
    assert bin_1.n == 2
    assert bin_1.sum_predicted_probability == pytest.approx(0.30)
    assert bin_1.mean_predicted_probability == pytest.approx(0.15)


def test_9_observed_frequency_correct():
    """Test 9: Aggregated observed frequency = n_observed_events / n."""
    # In bin [0.1, 0.2): probs [0.12, 0.18], obs [20.0, 5.0], threshold 15.6
    # 20.0 >= 15.6 -> event (1)
    # 5.0 < 15.6 -> non-event (0)
    # n_events = 1, n = 2 -> observed frequency = 0.50
    probs = [0.12, 0.18]
    obs = [20.0, 5.0]

    comps = compute_reliability_bin_components(
        predicted_probabilities=probs,
        observed_grid_mm=obs,
        threshold_mm=15.6,
        lead_day=1,
    )
    summary = summarize_reliability_bins(comps)

    bin_1 = summary[1]
    assert bin_1.n_observed_events == 1
    assert bin_1.observed_frequency == pytest.approx(0.50)


def test_10_empty_bin_represented_safely():
    """Test 10: Empty bin produces None for mean_predicted_probability and observed_frequency (no invented 0.0)."""
    probs = [0.85]
    obs = [20.0]

    comps = compute_reliability_bin_components(
        predicted_probabilities=probs,
        observed_grid_mm=obs,
        threshold_mm=15.6,
    )
    summary = summarize_reliability_bins(comps)

    # Bin 0 [0.0, 0.1) has no samples
    bin_0 = summary[0]
    assert bin_0.n == 0
    assert bin_0.is_empty is True
    assert bin_0.mean_predicted_probability is None
    assert bin_0.observed_frequency is None

    d = bin_0.to_dict()
    assert d["mean_predicted_probability"] is None
    assert d["observed_frequency"] is None
    assert d["is_empty"] is True


def test_11_missing_reference_causes_paired_exclusion():
    """Test 11: Missing reference or model probability excludes sample from both model and reference Brier sums."""
    # 4 grid points:
    # Cell 0: model=0.8, ref=0.4, obs=20.0 (event=1) -> valid paired
    # Cell 1: model=0.2, ref=0.3, obs=5.0  (event=0) -> valid paired
    # Cell 2: model=0.7, ref=NaN, obs=25.0 (event=1) -> ref is NaN -> EXCLUDE
    # Cell 3: model=NaN, ref=0.5, obs=10.0 (event=0) -> model is NaN -> EXCLUDE
    f = [10.0, 10.0, 10.0, 10.0]
    obs = [20.0, 5.0, 25.0, 10.0]
    probs = {15.6: [0.8, 0.2, 0.7, np.nan]}
    probs_ref = {15.6: [0.4, 0.3, np.nan, 0.5]}

    res = compute(
        forecast=f,
        observed=obs,
        imd_date="2024-07-15",
        lead_day=1,
        thresholds=[15.6],
        probabilities=probs,
        probability_reference=probs_ref,
        neighbourhood_sizes=None,
    )

    cat = res.get_contingency(threshold_mm=15.6)
    assert cat is not None
    # Only 2 valid paired samples
    assert cat.brier_n == 2

    # Model Brier sum on cells 0 and 1 only: (0.8 - 1.0)^2 + (0.2 - 0.0)^2 = 0.04 + 0.04 = 0.08
    assert cat.sum_brier_terms == pytest.approx(0.08)

    # Reference Brier sum on cells 0 and 1 only: (0.4 - 1.0)^2 + (0.3 - 0.0)^2 = 0.36 + 0.09 = 0.45
    assert cat.sum_reference_brier_terms == pytest.approx(0.45)


def test_12_parquet_roundtrip_preserves_new_fields(tmp_path):
    """Test 12: Parquet roundtrip preserves bin_lower, bin_upper, sum_reference_brier_terms, and reliability counts."""
    # 1. Component with reference Brier
    c_prob = VerificationComponent(
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        brier_n=50,
        sum_brier_terms=4.25,
        sum_reference_brier_terms=7.80,
    )

    # 2. Reliability bin component
    c_bin = VerificationComponent(
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        bin_lower=0.1,
        bin_upper=0.2,
        n=15,
        sum_predicted_probability=2.25,
        n_observed_events=3,
    )

    # 3. Continuous component with None bin fields
    c_cont = VerificationComponent(
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=None,
        n=50,
        sum_error=2.5,
        sum_squared_error=10.0,
    )

    save_components([c_prob, c_bin, c_cont], target=tmp_path)
    loaded = load_components(target=tmp_path)

    assert len(loaded) == 3

    # Verify probability record
    prob_loaded = [c for c in loaded if c.threshold_mm == 15.6 and c.bin_lower is None][0]
    assert prob_loaded.brier_n == 50
    assert prob_loaded.sum_brier_terms == pytest.approx(4.25)
    assert prob_loaded.sum_reference_brier_terms == pytest.approx(7.80)
    assert prob_loaded.bin_lower is None
    assert prob_loaded.bin_upper is None

    # Verify reliability bin record
    bin_loaded = [c for c in loaded if c.bin_lower is not None][0]
    assert bin_loaded.bin_lower == pytest.approx(0.1)
    assert bin_loaded.bin_upper == pytest.approx(0.2)
    assert bin_loaded.n == 15
    assert bin_loaded.sum_predicted_probability == pytest.approx(2.25)
    assert bin_loaded.n_observed_events == 3

    # Verify continuous record has None for bin and threshold
    cont_loaded = [c for c in loaded if c.threshold_mm is None][0]
    assert cont_loaded.bin_lower is None
    assert cont_loaded.bin_upper is None
    assert cont_loaded.sum_reference_brier_terms == pytest.approx(0.0)


def test_13_existing_verification_tests_remain_green():
    """Test 13: Full end-to-end integration: compute produces contingency, FSS, and reliability components correctly."""
    f_grid = np.array([
        [10.0, 20.0],
        [70.0, 80.0],
    ])
    obs_grid = np.array([
        [12.0, 18.0],
        [65.0, 85.0],
    ])
    p_64 = np.array([
        [0.05, 0.15],
        [0.75, 0.95],
    ])
    p_64_ref = np.array([
        [0.10, 0.10],
        [0.50, 0.50],
    ])

    result = compute(
        forecast=f_grid,
        observed=obs_grid,
        imd_date="2024-07-15",
        lead_day=1,
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        thresholds=[64.5],
        neighbourhood_sizes=[1],
        probabilities={64.5: p_64},
        probability_reference={64.5: p_64_ref},
    )

    # 1. Main contingency record
    cat = result.get_contingency(threshold_mm=64.5)
    assert cat is not None
    assert cat.threshold_mm == 64.5
    assert cat.bin_lower is None
    assert cat.brier_n == 4
    assert cat.sum_reference_brier_terms > 0.0

    # 2. Spatial FSS record
    fss = result.get_fss(threshold_mm=64.5, neighbourhood_cells=1)
    assert fss is not None
    assert fss.neighbourhood_cells == 1

    # 3. Reliability bin records
    rel_bins = result.get_reliability_bins(threshold_mm=64.5)
    assert len(rel_bins) == 10

    # 4. Reliability summary
    summary = result.get_reliability_summary(threshold_mm=64.5)
    assert len(summary) == 10
    total_samples = sum(s.n for s in summary)
    assert total_samples == 4
