"""IMD grid / cell_id (T7) and the valid-cell mask: at least 95% non-missing JJAS days (PRD 8.5)."""

import numpy as np
import pytest

import data_pipeline.alignment.cells as cells_module
from data_pipeline.alignment import (
    compute_valid_cells,
    get_valid_cells,
    imd_grid,
    load_valid_cells,
)
from data_pipeline.alignment.cells import valid_fraction_from_counts
from data_pipeline.ingestion import MissingInputError, ValidationError
from data_pipeline.ingestion.imd import imd_axes

JJAS_DAYS = 30 + 31 + 31 + 30  # 122


def test_t7_grid_identity(cfg):
    g = imd_grid(cfg)
    lat, lon = imd_axes(cfg.imd)
    assert len(g) == 129 * 135 and g["cell_id"].is_unique
    assert list(g["cell_id"]) == list(range(len(g)))
    first, last = g.iloc[0], g.iloc[-1]
    assert (first.latitude, first.longitude) == (6.5, 66.5) and (last.latitude, last.longitude) == (
        38.5,
        100.0,
    )
    # cell_id = i_lat * 135 + i_lon, and lat/lon are exactly the IMD axes (float32 holds 0.25 steps exactly)
    row = g[g.cell_id == 5 * 135 + 7].iloc[0]
    assert (row.latitude, row.longitude) == (lat[5], lon[7])
    assert np.array_equal(g["latitude"].to_numpy(), np.repeat(lat, 135).astype("float32"))


def test_95_percent_threshold_is_inclusive():
    n_days = 30 * JJAS_DAYS  # the PRD base period: 3660 days
    frac, ok = valid_fraction_from_counts(np.array([3477, 3476, 3660, 0]), n_days, 0.95)
    assert list(ok) == [True, False, True, False]  # 3477 / 3660 is exactly 95.0%
    assert frac[2] == 1.0
    _, ok = valid_fraction_from_counts(np.array([95, 94]), 100, 0.95)
    assert list(ok) == [True, False]


def test_no_days_is_an_error():
    with pytest.raises(ValidationError, match="No JJAS days"):
        valid_fraction_from_counts(np.array([0]), 0, 0.95)


def write_year(cfg, year, missing):
    """Write an IMD year file with all rain = 1 and -999 at the given (lat_i, lon_i, day-of-year list)."""
    n_days = 366 if year % 4 == 0 else 365
    data = np.ones((n_days, cfg.imd.n_lat, cfg.imd.n_lon), dtype="float32")
    for lat_i, lon_i, days in missing:
        data[days, lat_i, lon_i] = cfg.imd.missing_value
    path = cfg.imd.year_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)


@pytest.fixture
def two_years(cfg):
    """2021 (365 d) and 2022 (365 d): JJAS days are doy 151..272 (0-based 151..272 -> 122 days) in both."""
    jjas = list(range(151, 151 + JJAS_DAYS))  # 1 June ... 30 Sept in a non-leap year (0-based doy 151..272)
    cell = lambda i, days: (i, 0, days)  # noqa: E731
    for year in (2021, 2022):
        write_year(
            cfg,
            year,
            [
                cell(0, jjas[:6] if year == 2021 else jjas[:6]),  # A: 12 of 244 JJAS days missing -> 95.08%
                cell(1, jjas[:7] if year == 2021 else jjas[:6]),  # B: 13 missing -> 94.67%
                cell(2, jjas),  # C: every JJAS day missing
                cell(3, list(range(0, 151))),  # D: missing only Jan-May -> still fully valid
            ],
        )
    return cfg


def test_valid_cells_from_two_synthetic_years(two_years):
    cfg = two_years
    g = compute_valid_cells(cfg, years=[2021, 2022])
    assert g.attrs["jjas_days"] == 2 * JJAS_DAYS and g.attrs["base_years"] == [2021, 2022]
    by_lon0 = g[g.longitude == 66.5].set_index("latitude")
    lat = imd_axes(cfg.imd)[0]
    a, b, c, d, other = (by_lon0.loc[lat[i]] for i in range(5))
    assert (a.is_valid, b.is_valid, c.is_valid, d.is_valid, other.is_valid) == (
        True,
        False,
        False,
        True,
        True,
    )
    assert a.valid_fraction == pytest.approx(232 / 244, abs=1e-6) and b.valid_fraction == pytest.approx(
        231 / 244, abs=1e-6
    )
    assert c.valid_fraction == 0.0 and d.valid_fraction == 1.0
    assert g["is_valid"].sum() == len(g) - 2  # only B and C are invalid


def test_mask_is_saved_and_reused_instead_of_recomputed(two_years, monkeypatch):
    cfg = two_years
    first = get_valid_cells(cfg, years=[2021, 2022])
    calls = []
    monkeypatch.setattr(
        cells_module, "read_imd_year", lambda *a, **k: calls.append(a) or pytest.fail("recomputed")
    )
    again = get_valid_cells(cfg, years=[2021, 2022])  # must come from the saved file
    assert not calls
    assert first[["cell_id", "is_valid"]].equals(again[["cell_id", "is_valid"]])
    assert load_valid_cells(cfg).attrs["jjas_days"] == str(2 * JJAS_DAYS)  # provenance travels with the mask


def test_missing_mask_and_missing_imd_years_are_reported(cfg):
    with pytest.raises(MissingInputError, match="scripts/build_golden.py mask"):
        load_valid_cells(cfg)
    with pytest.raises(MissingInputError, match="IMD file not found"):
        compute_valid_cells(cfg, years=[1981])  # nothing is invented when the base-period data is absent
