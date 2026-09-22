"""Tests for F1 Raw vs AI-Corrected improvement logic (PRD §15 F1).

Required tests:
1. corrected closer cell -> positive improvement
2. raw closer cell -> negative improvement
3. exact tie -> zero improvement
4. missing observation excluded
5. missing corrected value excluded
6. cell summary counts match
7. district summary counts match
8. district ties not counted as corrected wins
9. mixed run IDs rejected
10. mixed lead days rejected
"""

import math
import numpy as np
import pytest

from verification.improvement import (
    CLOSER_CORRECTED,
    CLOSER_RAW,
    CLOSER_TIE,
    CellImprovement,
    CellImprovementSummary,
    DistrictImprovement,
    DistrictImprovementSummary,
    ImprovementSummary,
    compute_cell_improvements,
    compute_cell_summary,
    compute_district_improvements,
    compute_district_summary,
    compute_grid_improvements,
    compute_improvement_summary,
    evaluate_cell_improvement,
    evaluate_district_improvement,
    format_f1_summary_line,
)


def test_1_corrected_closer_cell_positive_improvement():
    """Test 1: Corrected closer cell produces positive improvement and 'corrected' label."""
    # raw_mm = 10.0, corrected = 18.0, obs = 20.0
    # raw error = |10 - 20| = 10.0
    # corrected error = |18 - 20| = 2.0
    # improvement = 10.0 - 2.0 = 8.0 > 0
    res = evaluate_cell_improvement(raw_mm=10.0, corrected_mean_mm=18.0, observed_mm=20.0, cell_id="C1")
    assert res is not None
    assert res.cell_id == "C1"
    assert res.improvement_mm == pytest.approx(8.0)
    assert res.improvement_mm > 0
    assert res.closer_system == CLOSER_CORRECTED


def test_2_raw_closer_cell_negative_improvement():
    """Test 2: Raw closer cell produces negative improvement and 'raw' label."""
    # raw_mm = 19.0, corrected = 10.0, obs = 20.0
    # raw error = |19 - 20| = 1.0
    # corrected error = |10 - 20| = 10.0
    # improvement = 1.0 - 10.0 = -9.0 < 0
    res = evaluate_cell_improvement(raw_mm=19.0, corrected_mean_mm=10.0, observed_mm=20.0, cell_id="C2")
    assert res is not None
    assert res.cell_id == "C2"
    assert res.improvement_mm == pytest.approx(-9.0)
    assert res.improvement_mm < 0
    assert res.closer_system == CLOSER_RAW


def test_3_exact_tie_zero_improvement():
    """Test 3: Exact tie produces zero improvement and 'tie' label."""
    # raw_mm = 15.0, corrected = 25.0, obs = 20.0
    # raw error = |15 - 20| = 5.0
    # corrected error = |25 - 20| = 5.0
    # improvement = 5.0 - 5.0 = 0.0
    res = evaluate_cell_improvement(raw_mm=15.0, corrected_mean_mm=25.0, observed_mm=20.0, cell_id="C3")
    assert res is not None
    assert res.cell_id == "C3"
    assert res.improvement_mm == 0.0
    assert res.closer_system == CLOSER_TIE


def test_4_missing_observation_excluded():
    """Test 4: Rows with missing observations are excluded from both cell and district comparisons."""
    cells = [
        {"cell_id": 1, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0},
        {"cell_id": 2, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": None},
        {"cell_id": 3, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": float("nan")},
        {"cell_id": 4, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": "NaN"},
    ]
    cell_results = compute_cell_improvements(cells)
    assert len(cell_results) == 1
    assert cell_results[0].cell_id == 1

    districts = [
        {"district_id": "D1", "raw_mean_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mean_mm": 15.0},
        {"district_id": "D2", "raw_mean_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mean_mm": None},
        {"district_id": "D3", "raw_mean_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mean_mm": np.nan},
    ]
    dist_results = compute_district_improvements(districts)
    assert len(dist_results) == 1
    assert dist_results[0].district_id == "D1"


def test_5_missing_corrected_value_excluded():
    """Test 5: Missing corrected forecast values or raw-only/fallback without corrected are excluded."""
    cells = [
        {"cell_id": 1, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0},
        {"cell_id": 2, "raw_mm": 10.0, "corrected_mean_mm": None, "observed_mm": 15.0},
        {"cell_id": 3, "raw_mm": 10.0, "corrected_mean_mm": float("nan"), "observed_mm": 15.0},
        {"cell_id": 4, "raw_mm": 10.0, "observed_mm": 15.0, "product_type": "raw_only"},
        {"cell_id": 5, "raw_mm": 10.0, "observed_mm": 15.0, "fallback_used": True, "corrected_mean_mm": None},
        {"cell_id": 6, "raw_mm": 10.0, "corrected_mean_mm": 10.0, "observed_mm": 15.0, "product_type": "raw_nwp"},
        {"cell_id": 7, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0, "has_corrected": False},
    ]
    cell_results = compute_cell_improvements(cells)
    assert len(cell_results) == 1
    assert cell_results[0].cell_id == 1


def test_6_cell_summary_counts_match():
    """Test 6: Cell summary counts accurately tally corrected closer, raw closer, ties, and compared."""
    cells = [
        # Corrected wins (3)
        {"cell_id": 1, "raw_mm": 10.0, "corrected_mean_mm": 14.0, "observed_mm": 15.0},  # raw_err=5, corr_err=1 -> corr
        {"cell_id": 2, "raw_mm": 0.0, "corrected_mean_mm": 10.0, "observed_mm": 12.0},   # raw_err=12, corr_err=2 -> corr
        {"cell_id": 3, "raw_mm": 20.0, "corrected_mean_mm": 22.0, "observed_mm": 25.0},  # raw_err=5, corr_err=3 -> corr
        # Raw wins (2)
        {"cell_id": 4, "raw_mm": 14.0, "corrected_mean_mm": 10.0, "observed_mm": 15.0},  # raw_err=1, corr_err=5 -> raw
        {"cell_id": 5, "raw_mm": 5.0, "corrected_mean_mm": 20.0, "observed_mm": 6.0},    # raw_err=1, corr_err=14 -> raw
        # Ties (2)
        {"cell_id": 6, "raw_mm": 10.0, "corrected_mean_mm": 20.0, "observed_mm": 15.0},  # raw_err=5, corr_err=5 -> tie
        {"cell_id": 7, "raw_mm": 30.0, "corrected_mean_mm": 30.0, "observed_mm": 30.0},  # raw_err=0, corr_err=0 -> tie
        # Invalid / missing (excluded)
        {"cell_id": 8, "raw_mm": None, "corrected_mean_mm": 10.0, "observed_mm": 10.0},
        {"cell_id": 9, "raw_mm": 10.0, "corrected_mean_mm": None, "observed_mm": 10.0},
        {"cell_id": 10, "raw_mm": 10.0, "corrected_mean_mm": 10.0, "observed_mm": None},
    ]

    summary = compute_cell_summary(cells)
    assert summary.corrected_closer_cells == 3
    assert summary.raw_closer_cells == 2
    assert summary.tied_cells == 2
    assert summary.compared_cells == 7


def test_7_district_summary_counts_match():
    """Test 7: District summary counts accurately tally district comparisons."""
    districts = [
        # Corrected wins (4)
        {"district_id": "D1", "raw_mean_mm": 5.0, "corrected_mean_mm": 9.0, "observed_mean_mm": 10.0},
        {"district_id": "D2", "raw_mean_mm": 20.0, "corrected_mean_mm": 15.0, "observed_mean_mm": 14.0},
        {"district_id": "D3", "raw_mean_mm": 0.0, "corrected_mean_mm": 4.0, "observed_mean_mm": 5.0},
        {"district_id": "D4", "raw_mean_mm": 12.0, "corrected_mean_mm": 8.0, "observed_mean_mm": 8.5},
        # Raw wins (2)
        {"district_id": "D5", "raw_mean_mm": 10.0, "corrected_mean_mm": 15.0, "observed_mean_mm": 9.5},
        {"district_id": "D6", "raw_mean_mm": 2.0, "corrected_mean_mm": 8.0, "observed_mean_mm": 1.0},
        # Tie (1)
        {"district_id": "D7", "raw_mean_mm": 6.0, "corrected_mean_mm": 10.0, "observed_mean_mm": 8.0},
        # Missing (excluded)
        {"district_id": "D8", "raw_mean_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mean_mm": None},
    ]

    summary = compute_district_summary(districts)
    assert summary.corrected_closer_districts == 4
    assert summary.raw_closer_districts == 2
    assert summary.tied_districts == 1
    assert summary.compared_districts == 7


def test_8_district_ties_not_counted_as_corrected_wins():
    """Test 8: District ties must NOT be counted as corrected wins."""
    districts = [
        # Corrected win: 1
        {"district_id": "D1", "raw_mean_mm": 10.0, "corrected_mean_mm": 14.0, "observed_mean_mm": 15.0},
        # Raw win: 1
        {"district_id": "D2", "raw_mean_mm": 14.0, "corrected_mean_mm": 10.0, "observed_mean_mm": 15.0},
        # Ties: 3
        {"district_id": "D3", "raw_mean_mm": 10.0, "corrected_mean_mm": 20.0, "observed_mean_mm": 15.0},
        {"district_id": "D4", "raw_mean_mm": 0.0, "corrected_mean_mm": 0.0, "observed_mean_mm": 0.0},
        {"district_id": "D5", "raw_mean_mm": 8.0, "corrected_mean_mm": 12.0, "observed_mean_mm": 10.0},
    ]

    summary = compute_district_summary(districts)
    assert summary.compared_districts == 5
    assert summary.corrected_closer_districts == 1  # Strictly 1, NOT 4!
    assert summary.raw_closer_districts == 1
    assert summary.tied_districts == 3


def test_9_mixed_run_ids_rejected():
    """Test 9: Passing records with mixed run IDs fails fast with ValueError."""
    # In cells alone
    mixed_cells = [
        {"cell_id": 1, "run_id": "run_2024071500", "lead_day": 1, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0},
        {"cell_id": 2, "run_id": "run_2024071600", "lead_day": 1, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0},
    ]
    with pytest.raises(ValueError, match="Mixed run IDs rejected"):
        compute_cell_improvements(mixed_cells)

    with pytest.raises(ValueError, match="Mixed run IDs rejected"):
        compute_improvement_summary(cells=mixed_cells)

    # Across expected run_id parameter and record run_id
    single_cell = [
        {"cell_id": 1, "run_id": "run_A", "lead_day": 1, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0},
    ]
    with pytest.raises(ValueError, match="Mixed run IDs rejected"):
        compute_improvement_summary(run_id="run_B", cells=single_cell)

    # Between cells and districts
    cells_ok = [{"cell_id": 1, "run_id": "run_A", "lead_day": 1, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0}]
    districts_other = [{"district_id": "D1", "run_id": "run_B", "lead_day": 1, "raw_mean_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mean_mm": 15.0}]
    with pytest.raises(ValueError, match="Mixed run IDs rejected"):
        compute_improvement_summary(cells=cells_ok, districts=districts_other)


def test_10_mixed_lead_days_rejected():
    """Test 10: Passing records with mixed lead days fails fast with ValueError."""
    # In cells alone
    mixed_cells = [
        {"cell_id": 1, "run_id": "run_1", "lead_day": 1, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0},
        {"cell_id": 2, "run_id": "run_1", "lead_day": 2, "raw_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mm": 15.0},
    ]
    with pytest.raises(ValueError, match="Mixed lead days rejected"):
        compute_cell_improvements(mixed_cells)

    with pytest.raises(ValueError, match="Mixed lead days rejected"):
        compute_improvement_summary(cells=mixed_cells)

    # In districts alone
    mixed_districts = [
        {"district_id": "D1", "lead_day": 1, "raw_mean_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mean_mm": 15.0},
        {"district_id": "D2", "lead_day": 3, "raw_mean_mm": 10.0, "corrected_mean_mm": 12.0, "observed_mean_mm": 15.0},
    ]
    with pytest.raises(ValueError, match="Mixed lead days rejected"):
        compute_district_improvements(mixed_districts)

    # Across parameter lead_day and record lead_day
    with pytest.raises(ValueError, match="Mixed lead days rejected"):
        compute_improvement_summary(lead_day=1, districts=mixed_districts)


def test_improvement_summary_combined_and_formatting():
    """Test full compute_improvement_summary and PRD §15 F1 summary sentence formatting."""
    cells = [
        {"cell_id": 1, "run_id": "run_1", "lead_day": 1, "raw_mm": 10.0, "corrected_mean_mm": 14.0, "observed_mm": 15.0},
        {"cell_id": 2, "run_id": "run_1", "lead_day": 1, "raw_mm": 15.0, "corrected_mean_mm": 10.0, "observed_mm": 15.0},
    ]
    districts = [
        {"district_id": "D1", "run_id": "run_1", "lead_day": 1, "raw_mean_mm": 10.0, "corrected_mean_mm": 14.0, "observed_mean_mm": 15.0},
        {"district_id": "D2", "run_id": "run_1", "lead_day": 1, "raw_mean_mm": 10.0, "corrected_mean_mm": 20.0, "observed_mean_mm": 15.0},
    ]

    summary = compute_improvement_summary(run_id="run_1", lead_day=1, cells=cells, districts=districts)
    assert summary.run_id == "run_1"
    assert summary.lead_day == 1
    assert summary.corrected_closer_cells == 1
    assert summary.raw_closer_cells == 1
    assert summary.tied_cells == 0
    assert summary.compared_cells == 2

    assert summary.corrected_closer_districts == 1
    assert summary.raw_closer_districts == 0
    assert summary.tied_districts == 1
    assert summary.compared_districts == 2

    # Check dict-like and to_dict access
    assert summary["corrected_closer_cells"] == 1
    d = summary.to_dict()
    assert d["compared_districts"] == 2

    # Check summary line sentence
    expected_line = "Corrected is closer to the observation in 1 of 2 cells and 1 of 2 districts. One day only, not evidence."
    assert summary.summary_line == expected_line
    assert format_f1_summary_line(1, 2, 1, 2) == expected_line


def test_compute_grid_improvements():
    """Test 2D spatial grid improvement computation."""
    raw = np.array([[10.0, 15.0], [20.0, np.nan]])
    corr = np.array([[14.0, 10.0], [20.0, 15.0]])
    obs = np.array([[15.0, 15.0], [10.0, 12.0]])

    # (0,0): raw_err=|10-15|=5, corr_err=|14-15|=1 -> diff=4 (corrected)
    # (0,1): raw_err=|15-15|=0, corr_err=|10-15|=5 -> diff=-5 (raw)
    # (1,0): raw_err=|20-10|=10, corr_err=|20-10|=10 -> diff=0 (tie)
    # (1,1): raw has NaN -> invalid
    imp_grid, closer_grid, summary = compute_grid_improvements(raw, corr, obs)

    assert imp_grid[0, 0] == pytest.approx(4.0)
    assert closer_grid[0, 0] == CLOSER_CORRECTED

    assert imp_grid[0, 1] == pytest.approx(-5.0)
    assert closer_grid[0, 1] == CLOSER_RAW

    assert imp_grid[1, 0] == 0.0
    assert closer_grid[1, 0] == CLOSER_TIE

    assert np.isnan(imp_grid[1, 1])
    assert closer_grid[1, 1] is None

    assert summary.corrected_closer_cells == 1
    assert summary.raw_closer_cells == 1
    assert summary.tied_cells == 1
    assert summary.compared_cells == 3
