"""Unit tests for M4 event counting and model availability decisions (PRD §10.7, §13.2)."""

from datetime import date
from pathlib import Path
import numpy as np
import pytest

from scripts.build_event_counts import build_event_counts
from verification.events import (
    DailyObservation,
    EventCountRecord,
    ModelAvailabilityDecision,
    count_events_in_grid_series,
    determine_model_availability,
    load_rain_thresholds,
)


def test_one_connected_blob():
    """Test 1: One 8-connected blob on a single day counts as 1 event."""
    grid = np.zeros((5, 5))
    grid[1:3, 1:3] = 20.0  # 2x2 square (4 cells)

    obs = [DailyObservation(season=2023, lead_day=1, date="2023-07-01", observed_rain=grid)]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 1
    assert records[0].n_events == 1
    assert records[0].n_cell_days == 4


def test_two_disconnected_blobs_same_day():
    """Test 2: Two disconnected blobs on the same day count as 2 events."""
    grid = np.zeros((6, 6))
    grid[0, 0] = 25.0
    grid[5, 5] = 30.0

    obs = [DailyObservation(season=2023, lead_day=1, date="2023-07-01", observed_rain=grid)]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 1
    assert records[0].n_events == 2
    assert records[0].n_cell_days == 2


def test_same_blob_continuing_next_day():
    """Test 3: Same blob continuing onto next consecutive day merges into 1 event."""
    grid1 = np.zeros((5, 5))
    grid1[2, 2] = 20.0
    grid1[2, 3] = 20.0

    grid2 = np.zeros((5, 5))
    grid2[2, 3] = 25.0  # Shared cell (2, 3)
    grid2[2, 4] = 25.0

    obs = [
        DailyObservation(season=2023, lead_day=1, date="2023-07-01", observed_rain=grid1),
        DailyObservation(season=2023, lead_day=1, date="2023-07-02", observed_rain=grid2),
    ]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 1
    assert records[0].n_events == 1
    assert records[0].n_cell_days == 4


def test_consecutive_day_groups_with_no_shared_cell():
    """Test 4: Consecutive-day groups with no shared cell remain separate events."""
    grid1 = np.zeros((5, 5))
    grid1[0, 0] = 20.0

    grid2 = np.zeros((5, 5))
    grid2[4, 4] = 20.0  # Disjoint from (0, 0)

    obs = [
        DailyObservation(season=2023, lead_day=1, date="2023-07-01", observed_rain=grid1),
        DailyObservation(season=2023, lead_day=1, date="2023-07-02", observed_rain=grid2),
    ]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 1
    assert records[0].n_events == 2
    assert records[0].n_cell_days == 2


def test_diagonal_touching_counts_as_connected():
    """Test 5: Diagonally touching cells count as 1 event (8-connectivity per PRD §10.7)."""
    grid = np.zeros((4, 4))
    grid[1, 1] = 20.0
    grid[2, 2] = 20.0  # Diagonally adjacent

    obs = [DailyObservation(season=2023, lead_day=1, date="2023-07-01", observed_rain=grid)]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 1
    assert records[0].n_events == 1
    assert records[0].n_cell_days == 2


def test_missing_date_prevents_merging():
    """Test 6: A missing date/gap prevents merging groups with shared cells."""
    grid1 = np.zeros((5, 5))
    grid1[2, 2] = 20.0

    grid3 = np.zeros((5, 5))
    grid3[2, 2] = 20.0  # Same cell, but date gap (Day 1 -> Day 3)

    obs = [
        DailyObservation(season=2023, lead_day=1, date="2023-07-01", observed_rain=grid1),
        DailyObservation(season=2023, lead_day=1, date="2023-07-03", observed_rain=grid3),
    ]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 1
    assert records[0].n_events == 2  # Not merged across missing date
    assert records[0].n_cell_days == 2


def test_invalid_cells_excluded():
    """Test 7: Missing/invalid cells are excluded from event groups and cell-day counts."""
    grid = np.zeros((4, 4))
    grid[1, 1] = 30.0
    grid[1, 2] = 30.0
    grid[2, 2] = np.nan  # Missing value

    mask = np.ones((4, 4), dtype=bool)
    mask[1, 1] = False  # Cell (1, 1) is invalid/sea

    obs = [
        DailyObservation(
            season=2023,
            lead_day=1,
            date="2023-07-01",
            observed_rain=grid,
            valid_mask=mask,
        )
    ]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 1
    assert records[0].n_events == 1  # Only valid cell (1, 2)
    assert records[0].n_cell_days == 1


def test_leads_never_merge():
    """Test 8: Leads are kept strictly separate; events never merge across leads."""
    grid = np.zeros((4, 4))
    grid[2, 2] = 25.0

    obs = [
        DailyObservation(season=2023, lead_day=1, date="2023-07-01", observed_rain=grid),
        DailyObservation(season=2023, lead_day=2, date="2023-07-01", observed_rain=grid),
    ]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 2
    r_lead1 = next(r for r in records if r.lead_day == 1)
    r_lead2 = next(r for r in records if r.lead_day == 2)
    assert r_lead1.n_events == 1
    assert r_lead2.n_events == 1


def test_seasons_never_merge():
    """Test 9: Seasons are kept strictly separate; events never merge across season boundaries."""
    grid = np.zeros((4, 4))
    grid[2, 2] = 25.0

    obs = [
        DailyObservation(season=2022, lead_day=1, date="2022-09-30", observed_rain=grid),
        DailyObservation(season=2023, lead_day=1, date="2023-06-01", observed_rain=grid),
    ]
    records = count_events_in_grid_series(obs, thresholds=[15.6])

    assert len(records) == 2
    r_2022 = next(r for r in records if r.season == 2022)
    r_2023 = next(r for r in records if r.season == 2023)
    assert r_2022.n_events == 1
    assert r_2023.n_events == 1


def test_model_availability_boundary():
    """Test 10: PRD §13.2 model availability decisions at the 29/30 event boundary."""
    # Case A: 64.5 mm has 29 events (< 30)
    dev_a = {15.6: 50, 64.5: 29, 115.6: 10}
    dec_a = determine_model_availability(dev_a, min_events=30)
    assert dec_a.p_15_6_available is True
    assert dec_a.p_15_6_mode == "STANDALONE"
    assert dec_a.p_64_5_available is False
    assert dec_a.p_64_5_mode == "UNAVAILABLE"
    assert dec_a.p_115_6_available is False
    assert dec_a.p_115_6_mode == "UNAVAILABLE"
    assert dec_a.hotspot_threshold_mm == 15.6

    # Case B: 64.5 mm has 30 events, 115.6 mm has 29 events (< 30)
    dev_b = {15.6: 50, 64.5: 30, 115.6: 29}
    dec_b = determine_model_availability(dev_b, min_events=30)
    assert dec_b.p_64_5_available is True
    assert dec_b.p_64_5_mode == "STANDALONE"
    assert dec_b.p_115_6_available is True
    assert dec_b.p_115_6_mode == "CHAINED"
    assert dec_b.hotspot_threshold_mm == 64.5

    # Case C: Both have >= 30 events
    dev_c = {15.6: 50, 64.5: 30, 115.6: 30}
    dec_c = determine_model_availability(dev_c, min_events=30)
    assert dec_c.p_64_5_mode == "STANDALONE"
    assert dec_c.p_115_6_mode == "STANDALONE"
    assert dec_c.hotspot_threshold_mm == 64.5


def test_build_event_counts_generates_docs_file(tmp_path):
    """Verify build_event_counts outputs markdown file without inventing data."""
    out_file = tmp_path / "event-counts.md"
    generated_path = build_event_counts(output_path=out_file)
    assert generated_path.is_file()
    content = generated_path.read_text(encoding="utf-8")
    assert "Observed Rainfall Event Counts and Model Availability" in content
    assert "PRD §13.2" in content
