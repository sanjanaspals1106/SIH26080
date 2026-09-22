"""Tests for M4 Spatial Hotspots computation for Feature F6 (PRD §15 F6).

Required tests:
1. one hotspot cell
2. two side-touching cells => one hotspot
3. diagonal touching => one hotspot
4. disconnected cells => separate hotspots
5. probability below 0.50 excluded
6. corrected mean below 15.6 excluded
7. fallback cell excluded
8. 15.6 fallback mode works
9. max probability / max corrected mean correct
10. centroid correct
11. district IDs deduplicated
12. threshold_mm correct
13. outline contains actual cell union, not convex hull
14. deterministic hotspot IDs
"""

import math
import pytest

from spatial.hotspots import (
    Hotspot,
    build_hotspot_outline,
    compute_hotspots,
    group_hotspot_cells_8conn,
    is_hotspot_cell,
    load_hotspot_thresholds,
)


def test_1_one_hotspot_cell():
    """Test 1: Single valid hotspot cell generates exactly one hotspot."""
    cells = [
        {
            "cell_id": 1,
            "lat": 18.0,
            "lon": 73.0,
            "p_ge_64_5": 0.60,
            "corrected_mean_mm": 25.0,
            "district_ids": ["D1"],
        }
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 1
    h = hotspots[0]
    assert h.hotspot_id == 1
    assert h.n_cells == 1
    assert h.max_probability == pytest.approx(0.60)
    assert h.max_corrected_mean_mm == pytest.approx(25.0)
    assert h.centroid_lat == pytest.approx(18.0)
    assert h.centroid_lon == pytest.approx(73.0)
    assert h.threshold_mm == 64.5
    assert h.district_ids == ["D1"]
    assert h.outline["type"] == "Polygon"
    ring = h.outline["coordinates"][0]
    assert len(ring) == 5
    assert ring[0] == ring[-1]


def test_2_two_side_touching_cells_one_hotspot():
    """Test 2: Two side-touching cells form a single unified hotspot."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.60, "corrected_mean_mm": 25.0},
        {"cell_id": 2, "lat": 18.0, "lon": 73.25, "p_ge_64_5": 0.65, "corrected_mean_mm": 30.0},
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 1
    h = hotspots[0]
    assert h.n_cells == 2
    assert h.outline["type"] == "Polygon"


def test_3_diagonal_touching_one_hotspot():
    """Test 3: Diagonal touching cells belong to the same hotspot (8-connected rule)."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.55, "corrected_mean_mm": 20.0},
        {"cell_id": 2, "lat": 18.25, "lon": 73.25, "p_ge_64_5": 0.70, "corrected_mean_mm": 35.0},
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 1
    assert hotspots[0].n_cells == 2


def test_4_disconnected_cells_separate_hotspots():
    """Test 4: Disconnected cells form separate distinct hotspots."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0},
        {"cell_id": 2, "lat": 25.0, "lon": 85.0, "p_ge_64_5": 0.80, "corrected_mean_mm": 40.0},
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 2
    assert hotspots[0].hotspot_id == 1
    assert hotspots[1].hotspot_id == 2
    assert hotspots[0].n_cells == 1
    assert hotspots[1].n_cells == 1


def test_5_probability_below_half_excluded():
    """Test 5: Probability strictly below 0.50 is excluded."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.49, "corrected_mean_mm": 40.0}
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 0


def test_6_corrected_mean_below_threshold_excluded():
    """Test 6: Corrected mean strictly below 15.6 mm is excluded."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.85, "corrected_mean_mm": 15.5}
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 0


def test_7_fallback_cell_excluded():
    """Test 7: Fallback cells are NEVER hotspot candidates (PRD §15 F6)."""
    cells = [
        {
            "cell_id": 1,
            "lat": 18.0,
            "lon": 73.0,
            "p_ge_64_5": 0.90,
            "corrected_mean_mm": 50.0,
            "fallback_used": True,
        }
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 0


def test_8_15_6_fallback_mode_works():
    """Test 8: Fallback mode uses P(>=15.6) and sets threshold_mm=15.6."""
    cells = [
        {
            "cell_id": 1,
            "lat": 18.0,
            "lon": 73.0,
            "p_ge_64_5": None,
            "p_ge_15_6": 0.62,
            "corrected_mean_mm": 22.0,
        }
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=False)
    assert len(hotspots) == 1
    h = hotspots[0]
    assert h.threshold_mm == 15.6
    assert h.max_probability == pytest.approx(0.62)


def test_9_max_probability_max_corrected_mean():
    """Test 9: Max probability and max corrected mean are correctly found across cluster cells."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.52, "corrected_mean_mm": 55.0},
        {"cell_id": 2, "lat": 18.0, "lon": 73.25, "p_ge_64_5": 0.88, "corrected_mean_mm": 20.0},
        {"cell_id": 3, "lat": 18.25, "lon": 73.0, "p_ge_64_5": 0.65, "corrected_mean_mm": 35.0},
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 1
    h = hotspots[0]
    assert h.max_probability == pytest.approx(0.88)
    assert h.max_corrected_mean_mm == pytest.approx(55.0)


def test_10_centroid_correct():
    """Test 10: Centroid lat/lon is the mean of member cell centers."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0},
        {"cell_id": 2, "lat": 18.0, "lon": 73.25, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0},
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 1
    h = hotspots[0]
    assert h.centroid_lat == pytest.approx(18.0)
    assert h.centroid_lon == pytest.approx(73.125)
    assert h.centroid == {"lat": 18.0, "lon": 73.125}


def test_11_district_ids_deduplicated():
    """Test 11: District IDs from all member cells are collected and deduplicated."""
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0, "district_ids": ["D01", "D02"]},
        {"cell_id": 2, "lat": 18.0, "lon": 73.25, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0, "district_ids": ["D02", "D03"]},
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 1
    assert hotspots[0].district_ids == ["D01", "D02", "D03"]


def test_12_threshold_mm_correct():
    """Test 12: threshold_mm correctly reflects 64.5 for normal and 15.6 for fallback."""
    cell = {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.60, "p_ge_15_6": 0.70, "corrected_mean_mm": 20.0}

    h_norm = compute_hotspots([cell], is_64_5_available=True)
    assert h_norm[0].threshold_mm == 64.5

    h_fb = compute_hotspots([cell], is_64_5_available=False)
    assert h_fb[0].threshold_mm == 15.6


def test_13_outline_contains_actual_cell_union_not_convex_hull():
    """Test 13: Outline geometry is the actual union of member cell squares, NOT a convex hull."""
    # L-shaped cluster of 3 cells of size 0.25°:
    # Cell 1: (18.0, 73.0)
    # Cell 2: (18.25, 73.0)
    # Cell 3: (18.0, 73.25)
    # Missing 4th corner: (18.25, 73.25)
    cells = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0},
        {"cell_id": 2, "lat": 18.25, "lon": 73.0, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0},
        {"cell_id": 3, "lat": 18.0, "lon": 73.25, "p_ge_64_5": 0.60, "corrected_mean_mm": 20.0},
    ]
    hotspots = compute_hotspots(cells, is_64_5_available=True)
    assert len(hotspots) == 1
    outline = hotspots[0].outline
    assert outline["type"] == "Polygon"

    ring = outline["coordinates"][0]
    # Check that reflex vertex (73.125, 18.125) is part of the polygon boundary
    # In convex hull, the reflex corner would be cut off!
    reflex_corner = (pytest.approx(73.125), pytest.approx(18.125))
    found_reflex = any(
        math.isclose(pt[0], 73.125, abs_tol=1e-5) and math.isclose(pt[1], 18.125, abs_tol=1e-5)
        for pt in ring
    )
    assert found_reflex, "Outline must contain the inner reflex corner (73.125, 18.125); convex hull would skip it."

    # Compute polygon area via Shoelace formula
    poly_area = 0.5 * abs(sum(ring[j][0] * ring[j + 1][1] - ring[j + 1][0] * ring[j][1] for j in range(len(ring) - 1)))
    expected_area = 3 * (0.25 ** 2)  # 3 cells * 0.0625 = 0.1875
    assert poly_area == pytest.approx(expected_area, abs=1e-6)

    # Convex hull area would be 3.5 * (0.25 ** 2) = 0.21875 > 0.1875
    assert poly_area < 0.21


def test_14_deterministic_hotspot_ids():
    """Test 14: Hotspot IDs are deterministic regardless of input cell ordering."""
    cluster_high_prob = [
        {"cell_id": 1, "lat": 18.0, "lon": 73.0, "p_ge_64_5": 0.90, "corrected_mean_mm": 40.0}
    ]
    cluster_low_prob = [
        {"cell_id": 2, "lat": 25.0, "lon": 85.0, "p_ge_64_5": 0.55, "corrected_mean_mm": 20.0}
    ]

    # Order A: high prob first
    hotspots_a = compute_hotspots(cluster_high_prob + cluster_low_prob)
    assert hotspots_a[0].hotspot_id == 1
    assert hotspots_a[0].max_probability == pytest.approx(0.90)
    assert hotspots_a[1].hotspot_id == 2
    assert hotspots_a[1].max_probability == pytest.approx(0.55)

    # Order B: low prob first
    hotspots_b = compute_hotspots(cluster_low_prob + cluster_high_prob)
    assert hotspots_b[0].hotspot_id == 1
    assert hotspots_b[0].max_probability == pytest.approx(0.90)
    assert hotspots_b[1].hotspot_id == 2
    assert hotspots_b[1].max_probability == pytest.approx(0.55)
