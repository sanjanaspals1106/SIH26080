"""Climatology features clim_mean / clim_p95: values, missing data, no leakage, caching (PRD 9.4, 9.5, L4)."""

import numpy as np
import pytest

import data_pipeline.features.climatology as clim_module
from data_pipeline.features import (
    assert_no_holdout,
    compute_climatology,
    get_climatology,
)
from data_pipeline.features.climatology import climatology_path
from data_pipeline.ingestion import MissingInputError, ValidationError

JJAS = slice(151, 151 + 122)  # 0-based day of year of 1 June .. 30 Sept in a non-leap year


def write_year(cfg, year, rng, dead_cell=(3, 3), gappy_cell=(4, 4)):
    n_days = 366 if year % 4 == 0 else 365
    data = rng.gamma(0.7, 9.0, size=(n_days, cfg.imd.n_lat, cfg.imd.n_lon)).astype("float32")
    data[:, dead_cell[0], dead_cell[1]] = cfg.imd.missing_value  # never observed
    data[JJAS.start : JJAS.start + 40, gappy_cell[0], gappy_cell[1]] = (
        cfg.imd.missing_value
    )  # 40 missing JJAS days
    data[:100, 5, 5] = 9999.0  # huge values outside JJAS: must not count
    path = cfg.imd.year_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)
    return data


@pytest.fixture
def years(cfg):
    rng = np.random.default_rng(11)
    return cfg, {y: write_year(cfg, y, rng) for y in (2017, 2018, 2019)}  # all non-leap


def jjas_values(raw, cell):
    """JJAS values of a cell over the given years with -999 removed, as a plain array."""
    vals = np.concatenate([d[JJAS, cell[0], cell[1]] for d in raw])
    return vals[vals != -999]


def test_mean_and_p95_of_jjas_observed_rain(years):
    cfg, raw = years
    clim = compute_climatology([2017, 2018], cfg)
    cid = 60 * 135 + 70
    vals = jjas_values([raw[2017], raw[2018]], (60, 70))
    assert clim.loc[clim.cell_id == cid, "clim_mean"].item() == pytest.approx(vals.mean(), rel=1e-5)
    assert clim.loc[clim.cell_id == cid, "clim_p95"].item() == pytest.approx(
        np.percentile(vals, 95), rel=1e-5
    )
    assert len(clim) == 129 * 135 and clim.attrs["seasons"] == [2017, 2018]
    assert clim["clim_mean"].dtype == np.float32


def test_only_jjas_days_count(years):
    cfg, raw = years
    clim = compute_climatology([2017], cfg)
    row = clim.loc[clim.cell_id == 5 * 135 + 5].iloc[
        0
    ]  # 9999 mm on the first 100 days of the year (not JJAS)
    assert row["clim_p95"] < 500 and row["clim_mean"] == pytest.approx(
        jjas_values([raw[2017]], (5, 5)).mean(), rel=1e-5
    )


def test_missing_days_are_ignored_and_never_observed_cells_are_null(years):
    cfg, raw = years
    clim = compute_climatology([2017, 2018, 2019], cfg)
    gappy = clim.loc[clim.cell_id == 4 * 135 + 4].iloc[0]
    vals = jjas_values(raw.values(), (4, 4))
    assert len(vals) == 3 * (122 - 40)  # 40 of 122 days missing each year
    assert gappy["clim_mean"] == pytest.approx(vals.mean(), rel=1e-5)  # a -999 is not averaged in
    assert gappy["clim_mean"] > 0
    dead = clim.loc[clim.cell_id == 3 * 135 + 3].iloc[0]
    assert np.isnan(dead["clim_mean"]) and np.isnan(dead["clim_p95"])  # no data: NaN, not 0 and not -999


def test_no_leakage_only_the_named_seasons_are_read(years):
    cfg, raw = years
    before = compute_climatology([2017], cfg)
    cfg.imd.year_path(2018).unlink()  # a "holdout" season that is not even on disk
    cfg.imd.year_path(2019).write_bytes(b"not a real file")  # or is corrupt: it must never be opened
    after = compute_climatology([2017], cfg)
    assert before[["clim_mean", "clim_p95"]].equals(after[["clim_mean", "clim_p95"]])
    with pytest.raises(ValidationError):
        compute_climatology(
            [2017, 2019], cfg
        )  # naming the bad season does fail: a season is read only when named


def test_holdout_seasons_are_rejected(years):
    cfg, _ = years
    clim = compute_climatology([2017, 2018], cfg)
    assert_no_holdout(clim, [2019, 2020])  # fine
    with pytest.raises(ValidationError, match=r"holdout season\(s\) \[2018\]"):
        assert_no_holdout(clim, [2018])
    clim.attrs.pop("seasons")
    with pytest.raises(ValidationError, match="does not record"):
        assert_no_holdout(clim, [2019])


def test_seasons_are_required_and_missing_years_are_reported(cfg):
    with pytest.raises(ValidationError, match="at least one"):
        compute_climatology([], cfg)
    with pytest.raises(MissingInputError, match="IMD file not found"):
        compute_climatology([1999], cfg)  # nothing is invented for years that are not on disk


def test_climatology_is_cached_per_season_set(years, monkeypatch):
    cfg, _ = years
    a = get_climatology([2017, 2018], cfg)
    assert climatology_path([2018, 2017], cfg).is_file()  # order does not matter
    monkeypatch.setattr(clim_module, "read_imd_year", lambda *x, **k: pytest.fail("recomputed"))
    b = get_climatology([2018, 2017, 2017], cfg)
    assert a[["cell_id", "clim_mean", "clim_p95"]].equals(
        b[["cell_id", "clim_mean", "clim_p95"]]
    ) and b.attrs["seasons"] == [2017, 2018]
    assert climatology_path([2017], cfg) != climatology_path(
        [2017, 2018], cfg
    )  # a different season set is a different file
