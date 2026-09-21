"""District-cell area weights (PRD 14.2) on a small synthetic grid and synthetic polygons."""

import dataclasses

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import data_pipeline.districts.weights as weights_module
from data_pipeline.alignment import imd_grid
from data_pipeline.districts import compute_district_weights, get_district_weights
from data_pipeline.districts.weights import cell_squares, check_weights
from data_pipeline.ingestion import MissingInputError, ValidationError
from data_pipeline.ingestion.imd import imd_axes


def districts(*specs):
    """specs: (district_id, geometry). Cells of the IMD grid are 0.25 degree squares centred on 6.5 + 0.25 k."""
    return gpd.GeoDataFrame(
        {"district_id": [s[0] for s in specs], "name": [f"District {s[0]}" for s in specs], "state": "S"},
        geometry=[s[1] for s in specs],
        crs="EPSG:4326",
    )


def grid_with_valid(cfg, valid_fn=lambda lat, lon: True):
    g = imd_grid(cfg)
    g["is_valid"] = [bool(valid_fn(la, lo)) for la, lo in zip(g["latitude"], g["longitude"], strict=True)]
    g["valid_fraction"] = 1.0
    return g


def cid(cfg, lat, lon):
    la, lo = imd_axes(cfg.imd)
    return int(np.argmin(abs(la - lat))) * cfg.imd.n_lon + int(np.argmin(abs(lo - lon)))


# a district that is exactly 2 x 2 cells: cell centres lat 10.0, 10.25 and lon 70.0, 70.25
FOUR = box(69.875, 9.875, 70.375, 10.375)


def test_four_cell_district_has_four_weights_that_add_up_to_one(cfg):
    w, s = compute_district_weights(districts(("D1", FOUR)), grid_with_valid(cfg), cfg)
    assert sorted(w["cell_id"]) == sorted(cid(cfg, la, lo) for la in (10.0, 10.25) for lo in (70.0, 70.25))
    assert w["area_weight"].sum() == pytest.approx(1.0, abs=1e-12)
    assert w["area_weight"].to_numpy() == pytest.approx(
        0.25, rel=0.01
    )  # nearly equal (cells differ slightly in area)
    assert w["is_main"].all()
    assert s.loc[0, "n_cells"] == 4 and s.loc[0, "n_main_cells"] == 4
    assert s.loc[0, "n_effective_cells"] == pytest.approx(4.0, rel=0.01) and s.loc[
        0, "valid_area_fraction"
    ] == pytest.approx(1.0)


def test_intersection_areas_are_real_areas_in_km2(cfg):
    w, s = compute_district_weights(districts(("D1", FOUR)), grid_with_valid(cfg), cfg)
    # 0.5 x 0.5 degree at ~10 N: about 55.4 km x 54.6 km
    assert s.loc[0, "district_area_km2"] == pytest.approx(55.4 * 54.7, rel=0.01)
    assert w["overlap_km2"].sum() == pytest.approx(s.loc[0, "district_area_km2"], rel=1e-9)
    assert (w["overlap_km2"] > 700).all() and (w["overlap_km2"] < 800).all()


def test_partial_overlap_gives_proportional_weights(cfg):
    # covers cell A (lon 69.875-70.125) fully and the western half of cell B (lon 70.125-70.375), one row
    poly = box(69.875, 9.875, 70.25, 10.125)
    w, _ = compute_district_weights(districts(("D1", poly)), grid_with_valid(cfg), cfg)
    by = w.set_index("cell_id")["area_weight"]
    a, b = by[cid(cfg, 10.0, 70.0)], by[cid(cfg, 10.0, 70.25)]
    assert (a, b) == (pytest.approx(2 / 3, rel=1e-9), pytest.approx(1 / 3, rel=1e-9))


def test_areas_use_the_equal_area_projection_not_degrees(cfg):
    """A one-cell-wide strip from 6.375 to 38.625 N: in degrees every cell would weigh the same (1/129)."""
    strip = box(69.875, 6.375, 70.125, 38.625)
    w, _ = compute_district_weights(districts(("D1", strip)), grid_with_valid(cfg), cfg)
    by = w.set_index("cell_id")["area_weight"]
    south, north = by[cid(cfg, 6.5, 70.0)], by[cid(cfg, 38.5, 70.0)]
    assert south / north == pytest.approx(
        np.cos(np.radians(6.5)) / np.cos(np.radians(38.5)), rel=0.01
    )  # equal-area
    assert south / north > 1.2  # far from the 1.0 that degree areas would give
    assert len(w) == 129


def test_masked_cells_are_dropped_and_weights_renormalised(cfg):
    invalid = cid(cfg, 10.25, 70.25)
    grid = grid_with_valid(cfg)
    grid.loc[grid["cell_id"] == invalid, "is_valid"] = False
    full, _ = compute_district_weights(districts(("D1", FOUR)), grid_with_valid(cfg), cfg)
    w, s = compute_district_weights(districts(("D1", FOUR)), grid, cfg)
    assert invalid not in set(w["cell_id"]) and len(w) == 3
    assert w["area_weight"].sum() == pytest.approx(1.0, abs=1e-12)  # renormalised
    kept = full[full["cell_id"] != invalid].set_index("cell_id")
    assert w.set_index("cell_id")["area_weight"].to_numpy() == pytest.approx(
        (kept["overlap_km2"] / kept["overlap_km2"].sum()).loc[w["cell_id"]].to_numpy()
    )
    assert s.loc[0, "valid_area_fraction"] == pytest.approx(
        0.75, rel=0.01
    )  # a quarter of the district is not on valid cells
    assert s.loc[0, "n_effective_cells"] == pytest.approx(3.0, rel=0.01)


def test_is_main_uses_the_w_min_threshold(cfg):
    # cell A fully, plus a 4% sliver of cell B: B's renormalised weight is ~3.8% < 0.05
    poly = box(69.875, 9.875, 70.135, 10.125)
    w, s = compute_district_weights(districts(("D1", poly)), grid_with_valid(cfg), cfg)
    by = w.set_index("cell_id")
    assert by.loc[cid(cfg, 10.0, 70.0), "is_main"] and not by.loc[cid(cfg, 10.0, 70.25), "is_main"]
    assert 0.03 < by.loc[cid(cfg, 10.0, 70.25), "area_weight"] < 0.05
    assert s.loc[0, "n_main_cells"] == 1
    at_threshold = dataclasses.replace(cfg, features=dataclasses.replace(cfg.features, w_min=0.03))
    assert compute_district_weights(districts(("D1", poly)), grid_with_valid(cfg), at_threshold)[0][
        "is_main"
    ].all()


def test_small_and_large_districts_and_effective_cells(cfg):
    two = box(69.875, 9.875, 70.375, 10.125)  # 2 cells
    big = box(74.875, 14.875, 78.875, 18.875)  # 16 x 16 = 256 cells
    w, s = compute_district_weights(districts(("A", two), ("B", big)), grid_with_valid(cfg), cfg)
    s = s.set_index("district_id")
    assert s.loc["A", "n_effective_cells"] == pytest.approx(2.0, rel=0.01) and bool(s.loc["A", "is_small"])
    assert s.loc["B", "n_effective_cells"] == pytest.approx(256.0, rel=0.02) and not bool(
        s.loc["B", "is_small"]
    )
    assert (
        s.loc["B", "n_main_cells"] == 0
    )  # 256 cells of weight ~0.004: no cell reaches w_min = 0.05 (PRD 14.2)
    assert bool(w[w.district_id == "A"]["is_main"].all())


def test_district_without_valid_cells_is_left_out_with_a_warning(cfg, caplog):
    grid = grid_with_valid(cfg, lambda la, lo: not (69.8 < lo < 70.4))
    with caplog.at_level("WARNING"):
        w, s = compute_district_weights(
            districts(("D1", FOUR), ("D2", box(74.875, 14.875, 75.375, 15.375))), grid, cfg
        )
    assert list(s["district_id"]) == ["D2"] and set(w["district_id"]) == {"D2"}
    assert s.attrs["districts_without_cells"] == ["D1"] and "no valid cell" in caplog.text


def test_district_crossing_a_cell_edge_in_several_pieces_and_two_districts_per_cell(cfg):
    a, b = box(69.875, 9.875, 70.0, 10.125), box(70.0, 9.875, 70.125, 10.125)  # two districts share one cell
    w, _ = compute_district_weights(districts(("A", a), ("B", b)), grid_with_valid(cfg), cfg)
    assert len(w) == 2 and set(w["cell_id"]) == {cid(cfg, 10.0, 70.0)}
    assert (
        w["area_weight"] == 1.0
    ).all()  # each district lies wholly in that cell; the cell is shared, weights are per district
    assert w["overlap_km2"].nunique() <= 2


def test_cell_squares_are_quarter_degree_squares(cfg):
    sq = cell_squares(imd_grid(cfg).iloc[:2], cfg)
    assert sq.crs.to_epsg() == 4326 and sq.geometry.iloc[0].bounds == pytest.approx(
        (66.375, 6.375, 66.625, 6.625)
    )


def test_weights_are_deterministic_and_sorted(cfg):
    ds = districts(("B", box(74.875, 14.875, 75.375, 15.375)), ("A", FOUR))
    w1, s1 = compute_district_weights(ds, grid_with_valid(cfg), cfg)
    w2, s2 = compute_district_weights(ds.iloc[::-1].reset_index(drop=True), grid_with_valid(cfg), cfg)
    pd.testing.assert_frame_equal(w1, w2)
    pd.testing.assert_frame_equal(s1, s2)
    assert list(w1["district_id"]) == sorted(w1["district_id"])


def test_check_weights_catches_bad_tables():
    good = pd.DataFrame({"district_id": ["A", "A"], "cell_id": [1, 2], "area_weight": [0.5, 0.5]})
    check_weights(good)
    with pytest.raises(ValidationError, match="add up"):
        check_weights(good.assign(area_weight=[0.5, 0.4]))
    with pytest.raises(ValidationError, match="Duplicate"):
        check_weights(pd.concat([good, good.iloc[:1]]))


# ---- caching and failures -----------------------------------------------------------------------


def test_weights_are_cached_and_reused(cfg, monkeypatch):
    ds, grid = districts(("A", FOUR)), grid_with_valid(cfg)
    w1, s1 = get_district_weights(ds, grid, cfg)
    monkeypatch.setattr(weights_module, "compute_district_weights", lambda *a, **k: pytest.fail("recomputed"))
    w2, s2 = get_district_weights(ds, grid, cfg)
    pd.testing.assert_frame_equal(w1, w2)
    pd.testing.assert_frame_equal(s1, s2, check_dtype=False)
    assert s2.attrs["districts_without_cells"] == []


def test_cache_is_recomputed_when_districts_or_valid_cells_change(cfg):
    grid = grid_with_valid(cfg)
    w1, _ = get_district_weights(districts(("A", FOUR)), grid, cfg)
    changed = grid.copy()
    changed.loc[changed["cell_id"] == cid(cfg, 10.0, 70.0), "is_valid"] = False
    w2, _ = get_district_weights(districts(("A", FOUR)), changed, cfg)
    assert len(w1) == 4 and len(w2) == 3
    w3, _ = get_district_weights(
        districts(("A", box(69.875, 9.875, 70.125, 10.125))), changed, cfg
    )  # different geometry
    assert len(w3) == 0  # its only cell is invalid in `changed`: the district has no weights


def test_no_district_file_configured_fails_clearly(cfg):
    with pytest.raises(MissingInputError, match="No district file is configured"):
        get_district_weights(
            None, grid_with_valid(cfg), cfg
        )  # falls back to the Stage 1 loader: no silent substitute
