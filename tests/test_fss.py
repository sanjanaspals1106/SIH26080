"""Unit tests for Fractions Skill Score (FSS) per PRD §16.8 and Appendix A."""

import math
import numpy as np
import pytest

from verification.metrics import (
    FSSComponents,
    MetricResult,
    aggregate_fss,
    compute_fss,
    compute_fss_components,
    compute_fss_from_components,
    compute_observed_event_fraction,
    compute_useful_skill,
)


def test_identical_nonzero_fields():
    """Test 1: Identical nonzero forecast and observation fields produce FSS = 1.0."""
    grid = np.array([
        [0.0, 15.0, 70.0, 0.0],
        [5.0, 80.0, 120.0, 10.0],
        [0.0, 65.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    # Threshold 64.5 mm
    for n in [1, 3, 5]:
        res = compute_fss(grid, grid, threshold=64.5, neighbourhood_size=n)
        assert res.is_defined()
        assert math.isclose(res.value, 1.0, rel_tol=1e-7)
        assert res.undefined_reason is None


def test_small_shifted_field_hand_calculated():
    """Test 2: Small 2x2 grid with 1-cell shift matches hand calculation at n=1 and n=3.

    Grid 2x2:
      F = [[15.0,  0.0], [0.0, 0.0]]  (event at (0, 0))
      O = [[ 0.0, 15.0], [0.0, 0.0]]  (event at (0, 1))
    Threshold = 10.0 mm.

    For n = 1:
      Pf = [[1, 0], [0, 0]], Po = [[0, 1], [0, 0]]
      num_sum = (1-0)^2 + (0-1)^2 = 2.0
      denom = (1^2) + (1^2) = 2.0
      FSS = 1 - (2.0 / 2.0) = 0.0

    For n = 3:
      Window 3x3 covers all 4 cells for every cell in the grid.
      m_filt has 4 valid cells for each cell.
      Pf has 1 event in window -> Pf = 1/4 = 0.25 at all 4 cells.
      Po has 1 event in window -> Po = 1/4 = 0.25 at all 4 cells.
      num_sum = Σ(0.25 - 0.25)^2 = 0.0
      denom = 4*(0.25^2) + 4*(0.25^2) = 4*0.0625 + 4*0.0625 = 0.5
      FSS = 1 - 0.0 / 0.5 = 1.0
    """
    f = np.array([[15.0, 0.0], [0.0, 0.0]])
    o = np.array([[0.0, 15.0], [0.0, 0.0]])

    # n = 1 -> FSS = 0.0
    res_n1 = compute_fss(f, o, threshold=10.0, neighbourhood_size=1)
    assert res_n1.is_defined()
    assert math.isclose(res_n1.value, 0.0, abs_tol=1e-7)

    # n = 3 -> FSS = 1.0
    res_n3 = compute_fss(f, o, threshold=10.0, neighbourhood_size=3)
    assert res_n3.is_defined()
    assert math.isclose(res_n3.value, 1.0, rel_tol=1e-7)


def test_invalid_sea_cells_excluded():
    """Test 3: Invalid/sea cells are excluded from both window fraction and grid sum.

    Sea and missing cells must NOT count as dry cells (PRD §16.8).
    """
    f = np.array([
        [15.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
    ])
    o = np.array([
        [15.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
    ])

    # Valid mask where only (0, 0) is land, others are sea
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True

    # At (0, 0), it is a hit, and only valid cell in window is (0, 0)
    # If sea cells counted as dry, Pf would be diluted to 1/4 or 1/9.
    # Because sea is excluded, Pf = 1.0 / 1.0 = 1.0, Po = 1.0 / 1.0 = 1.0.
    res = compute_fss(f, o, threshold=10.0, neighbourhood_size=3, valid_mask=mask)
    assert res.is_defined()
    assert math.isclose(res.value, 1.0, rel_tol=1e-7)

    comps = compute_fss_components(f, o, threshold=10.0, neighbourhood_size=3, valid_mask=mask)
    assert comps.n_valid_cells == 1
    assert comps.n_observed_events == 1
    assert math.isclose(comps.forecast_fraction_sq_sum, 1.0)
    assert math.isclose(comps.observed_fraction_sq_sum, 1.0)
    assert math.isclose(comps.numerator_sum, 0.0)


def test_missing_cells_excluded():
    """Test 4: Missing cells (NaNs in forecast or observed) are excluded."""
    f = np.array([
        [15.0, np.nan],
        [ 0.0,  0.0  ],
    ])
    o = np.array([
        [15.0,  0.0  ],
        [np.nan, 0.0 ],
    ])
    # Valid cells: only (0, 0) and (1, 1). (0, 1) and (1, 0) are missing.
    comps = compute_fss_components(f, o, threshold=10.0, neighbourhood_size=1)
    assert comps.n_valid_cells == 2
    assert comps.n_observed_events == 1

    # At (0, 0): hit (F=15>=10, O=15>=10). At (1, 1): both 0. FSS=1.0 on valid points.
    fss = compute_fss_from_components(comps)
    assert fss.is_defined()
    assert math.isclose(fss.value, 1.0)


def test_zero_denominator_undefined_not_zero():
    """Test 5: Zero events in forecast and observation results in undefined metric, NEVER 0."""
    f = np.array([[0.0, 5.0], [2.0, 1.0]])
    o = np.array([[0.0, 1.0], [0.0, 0.0]])
    threshold = 15.6  # No cell exceeds 15.6 mm

    res = compute_fss(f, o, threshold=threshold, neighbourhood_size=3)
    assert res.value is None
    assert res.value != 0.0
    assert res.undefined_reason == "NO_EVENTS_AT_THRESHOLD"

    # All invalid / no valid cells
    empty_mask = np.zeros((2, 2), dtype=bool)
    res_empty = compute_fss(f, o, threshold=threshold, neighbourhood_size=3, valid_mask=empty_mask)
    assert res_empty.value is None
    assert res_empty.undefined_reason == "NO_VALID_CELLS"


def test_combining_two_dates_via_components():
    """Test 6: Aggregating components across dates gives correct component-summed FSS.

    PRD §16.4: FSS is always added up as (sum of numerators) / (sum of denominators),
    NEVER as an average of daily FSS.
    """
    # Date 1: perfect forecast
    f1 = np.array([[20.0, 0.0], [0.0, 0.0]])
    o1 = np.array([[20.0, 0.0], [0.0, 0.0]])
    c1 = compute_fss_components(f1, o1, threshold=10.0, neighbourhood_size=1)
    fss1 = compute_fss_from_components(c1).value
    assert math.isclose(fss1, 1.0)

    # Date 2: completely mismatched forecast with large event count
    f2 = np.array([[20.0, 20.0], [0.0, 0.0]])
    o2 = np.array([[ 0.0,  0.0], [20.0, 20.0]])
    c2 = compute_fss_components(f2, o2, threshold=10.0, neighbourhood_size=1)
    fss2 = compute_fss_from_components(c2).value
    # c2: num_sum = 4, denom = 2 + 2 = 4 -> FSS = 1 - 4/4 = 0.0
    assert math.isclose(fss2, 0.0)

    # Unweighted average of daily FSS would be (1.0 + 0.0) / 2 = 0.5
    daily_average = (fss1 + fss2) / 2.0
    assert math.isclose(daily_average, 0.5)

    # Aggregated FSS via component summation:
    # c1: num_sum = 0, denom = 1 + 1 = 2
    # c2: num_sum = 4, denom = 2 + 2 = 4
    # total: num_sum = 4, denom = 2 + 4 = 6
    # True aggregated FSS = 1 - (4 / 6) = 1/3 ≈ 0.333333333
    agg_res = aggregate_fss([c1, c2])
    assert agg_res.is_defined()
    assert math.isclose(agg_res.value, 1.0 / 3.0, rel_tol=1e-7)
    # Verify it is strictly different from daily average!
    assert not math.isclose(agg_res.value, daily_average)


def test_neighbourhood_size_1():
    """Test 7: Neighbourhood size 1 works correctly and matches Dice coefficient."""
    f = np.array([
        [20.0, 20.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0, 20.0],
    ])
    o = np.array([
        [20.0,  0.0, 20.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0, 20.0],
    ])
    # Threshold 10.0:
    # Hits (a) = 2 at (0, 0) and (2, 2)
    # False alarms (b) = 1 at (0, 1)
    # Misses (c) = 1 at (0, 2)
    # For n=1: FSS = 2*a / (2*a + b + c) = 4 / (4 + 1 + 1) = 4/6 = 2/3
    res = compute_fss(f, o, threshold=10.0, neighbourhood_size=1)
    assert res.is_defined()
    assert math.isclose(res.value, 2.0 / 3.0, rel_tol=1e-7)


def test_supported_neighbourhoods_and_useful_skill():
    """Test 8: Verify neighbourhoods 1, 3, 5, 9 run and useful skill line calculation."""
    f = np.zeros((15, 15))
    o = np.zeros((15, 15))
    f[5:10, 5:10] = 50.0
    o[6:11, 6:11] = 50.0

    for n in [1, 3, 5, 9]:
        res = compute_fss(f, o, threshold=25.0, neighbourhood_size=n)
        assert res.is_defined()
        assert 0.0 <= res.value <= 1.0

    # Useful skill calculation: f0 = 25 / 225 = 1/9
    comps = compute_fss_components(f, o, threshold=25.0, neighbourhood_size=5)
    f0 = compute_observed_event_fraction(comps)
    assert f0.is_defined()
    assert math.isclose(f0.value, 25.0 / 225.0)

    # useful skill = 0.5 + f0 / 2 = 0.5 + (1/18) = 10/18 = 5/9
    useful = compute_useful_skill(f0)
    assert useful.is_defined()
    assert math.isclose(useful.value, 0.5 + (25.0 / 225.0) / 2.0)


def test_invalid_neighbourhood_even_raises():
    """Neighbourhood size must be odd."""
    f = np.ones((4, 4))
    o = np.ones((4, 4))
    with pytest.raises(ValueError, match="odd positive integer"):
        compute_fss(f, o, threshold=0.5, neighbourhood_size=4)
