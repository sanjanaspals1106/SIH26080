"""IMD ingestion: binary parsing, -999 -> NaN, grid validation, download (mocked), bad input."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.ingestion import (
    DownloadError,
    MissingInputError,
    ValidationError,
    download_imd,
    load_config,
    read_imd,
    validate_imd,
)
from data_pipeline.ingestion.imd import read_imd_year
from data_pipeline.ingestion.synthetic import write_synthetic_imd_year


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    cfg = load_config(data_dir=tmp_path_factory.mktemp("data"))
    write_synthetic_imd_year(cfg, 2023)
    write_synthetic_imd_year(cfg, 2024, seed=1)  # leap year
    return cfg


@pytest.fixture(scope="module")
def imd(world):
    return read_imd([2023], world)


def test_read_imd_shape_and_coordinates(imd):
    assert imd["rain"].dims == ("time", "lat", "lon")
    assert imd["rain"].shape == (365, 129, 135)
    assert imd["lat"].values[0] == 6.5 and imd["lat"].values[-1] == 38.5
    assert imd["lon"].values[0] == 66.5 and imd["lon"].values[-1] == 100.0
    assert (np.diff(imd["lat"].values) > 0).all() and (np.diff(imd["lon"].values) > 0).all()
    assert imd["time"].values[0] == np.datetime64("2023-01-01")
    assert imd["time"].values[-1] == np.datetime64("2023-12-31")
    assert imd["rain"].attrs["units"] == "mm/day"


def test_minus_999_becomes_nan_and_other_values_are_untouched(world, imd):
    raw = np.fromfile(world.imd.year_path(2023), dtype="float32").reshape(365, 129, 135)
    assert (raw == -999).sum() > 0  # the fixture really has missing values
    rain = imd["rain"].values
    assert np.array_equal(np.isnan(rain), raw == -999)
    assert not (rain == -999).any()
    assert np.array_equal(rain[~np.isnan(rain)], raw[raw != -999])


def test_leap_year_and_multiple_years(world):
    ds = read_imd([2024, 2023], world)  # order does not matter
    assert ds.sizes["time"] == 365 + 366
    assert pd.DatetimeIndex(ds["time"].values).is_monotonic_increasing
    assert ds["time"].sel(time="2024-02-29").size == 1


def test_orientation_row_is_latitude(world):
    """A value written at (day 0, lat index 5, lon index 7) must come out at lat 6.5+5*0.25, lon 66.5+7*0.25."""
    raw = np.fromfile(world.imd.year_path(2023), dtype="float32").reshape(365, 129, 135)
    raw[0, 5, 7] = 123.0
    path = world.imd.raw_dir / "rain" / "2022.grd"
    raw[:365].tofile(path)
    ds = read_imd_year(path, 2022, world)
    assert ds["rain"].sel(time="2022-01-01", lat=7.75, lon=68.25).item() == 123.0


# ---- malformed / missing input -------------------------------------------------------------


def test_missing_file(cfg):
    with pytest.raises(MissingInputError, match="download_imd"):
        read_imd([2019], cfg)


def test_truncated_file_names_the_expected_size(cfg):
    path = cfg.imd.year_path(2020)
    path.parent.mkdir(parents=True)
    np.zeros(1000, dtype="float32").tofile(path)
    with pytest.raises(ValidationError, match="366 days.*truncated"):
        read_imd([2020], cfg)


def test_file_for_the_wrong_year_is_caught_by_size(world, tmp_path):
    with pytest.raises(ValidationError, match="bytes"):
        read_imd_year(world.imd.year_path(2023), 2024, world)  # 365-day file read as a leap year


# ---- validation ----------------------------------------------------------------------------


def test_valid_dataset_passes(world, imd):
    validate_imd(imd, world)


def test_wrong_grid(world, imd):
    with pytest.raises(ValidationError, match="grid is 100 x 135"):
        validate_imd(imd.isel(lat=slice(0, 100)), world)


def test_shifted_axes(world, imd):
    with pytest.raises(ValidationError, match="axes differ"):
        validate_imd(imd.assign_coords(lon=imd["lon"] + 1.0), world)


def test_missing_variable_and_coordinates(world, imd):
    with pytest.raises(ValidationError, match="'rain' is missing"):
        validate_imd(imd.rename(rain="precip"), world)
    with pytest.raises(ValidationError, match="missing coordinates"):
        validate_imd(imd.drop_vars("time"), world)


def test_empty_data(world, imd):
    with pytest.raises(ValidationError, match="empty"):
        validate_imd(imd.where(False), world)


def test_unconverted_missing_value(world, imd):
    bad = imd.copy(deep=True)
    bad["rain"][0, 0, 0] = -999
    with pytest.raises(ValidationError, match="not converted to NaN"):
        validate_imd(bad, world)


def test_other_negative_values(world, imd):
    bad = imd.copy(deep=True)
    bad["rain"][0, 0, 0] = -9999
    with pytest.raises(ValidationError, match="negative"):
        validate_imd(bad, world)


def test_duplicate_dates(world, imd):
    dup = imd.isel(time=[0, 0, 1])
    with pytest.raises(ValidationError, match="duplicate"):
        validate_imd(dup, world)


# ---- download (network mocked) -------------------------------------------------------------


def test_download_skips_years_on_disk(cfg):
    calls = []

    def fake_fetch(year, c):
        calls.append(year)
        p = c.year_path(year)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    assert [p.name for p in download_imd([2001, 2000, 2001], cfg, fetch=fake_fetch)] == [
        "2000.grd",
        "2001.grd",
    ]
    assert calls == [2000, 2001]
    download_imd([2000, 2001], cfg, fetch=fake_fetch)
    assert calls == [2000, 2001]  # nothing fetched twice
    download_imd([2000], cfg, fetch=fake_fetch, force=True)
    assert calls == [2000, 2001, 2000]


def test_download_that_writes_nothing_is_an_error(cfg):
    with pytest.raises(DownloadError, match="no file"):
        download_imd([2000], cfg, fetch=lambda year, c: None)


def test_download_exception_is_wrapped(cfg):
    def boom(year, c):
        raise ConnectionError("no route to host")

    with pytest.raises(DownloadError, match="no route to host"):
        download_imd([2000], cfg, fetch=boom)
