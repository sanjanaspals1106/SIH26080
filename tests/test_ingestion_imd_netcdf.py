"""IMD adapter: the same year read from a `.grd` file or from an IMD NetCDF file gives the same Dataset."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_pipeline.ingestion import DownloadError, MissingInputError, ValidationError, download_imd, load_config, read_imd
from data_pipeline.ingestion.imd import imd_axes, read_imd_year, resolve_imd_path
from data_pipeline.ingestion.synthetic import write_synthetic_imd_year


def write_nc(cfg, path, year, *, names=("TIME", "LATITUDE", "LONGITUDE", "RAINFALL"), descending=True, seed=0,
             n_days=None, missing=-999.0):
    """An IMD-style NetCDF year (uppercase names, latitude descending, -999 for missing)."""
    days = n_days or (366 if year % 4 == 0 else 365)
    lat, lon = imd_axes(cfg.imd)
    data = np.random.default_rng(seed).gamma(0.5, 8.0, size=(days, lat.size, lon.size)).astype("float32")
    data[:, :5, :5] = missing
    if descending:
        lat, data = lat[::-1], data[:, ::-1, :]
    t, la, lo, v = names
    ds = xr.Dataset({v: ((t, la, lo), data)}, coords={t: pd.date_range(f"{year}-01-01", periods=days), la: lat, lo: lon})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return data


@pytest.fixture()
def cfg(tmp_path):
    return load_config(data_dir=tmp_path / "data")


def test_netcdf_year_is_read_like_a_grd_year(cfg):
    raw = write_nc(cfg, cfg.data_dir / "imd" / "RF25_ind2023_rfp25.nc", 2023)  # IMD's own naming, not in raw/
    # (write_nc returns the array as stored in the file: latitude descending)
    ds = read_imd_year(cfg.imd.year_path(2023), 2023, cfg)
    assert ds["rain"].dims == ("time", "lat", "lon") and ds["rain"].shape == (365, 129, 135)
    assert ds["lat"].values[0] == 6.5 and ds["lat"].values[-1] == 38.5  # descending file -> ascending
    assert ds["time"].values[0] == np.datetime64("2023-01-01") and ds["time"].values[-1] == np.datetime64("2023-12-31")
    assert np.isnan(ds["rain"].values[:, :5, :5]).all()  # the -999 block (lat 6.5-7.5, lon 66.5-67.5)
    assert not (ds["rain"].values == -999).any()
    assert np.array_equal(ds["rain"].values[:, 100, 100], raw[:, ::-1, :][:, 100, 100])  # file order was descending
    assert ds["rain"].attrs["units"] == "mm/day"
    assert ds["rain"].dtype == "float32"
    assert read_imd([2023], cfg)["rain"].shape == (365, 129, 135)  # the multi-year reader takes it too


def test_other_variable_and_dimension_names(cfg):
    write_nc(cfg, cfg.imd.raw_dir / "rain" / "2022.nc", 2022, names=("time", "lat", "lon", "rf"), descending=False)
    ds = read_imd_year(cfg.imd.year_path(2022), 2022, cfg)
    assert ds["rain"].shape == (365, 129, 135)


def test_grd_wins_over_netcdf_and_is_untouched(cfg):
    write_nc(cfg, cfg.data_dir / "imd" / "RF25_ind2023_rfp25.nc", 2023)
    grd = write_synthetic_imd_year(cfg, 2023, seed=5)
    assert resolve_imd_path(2023, cfg.imd) == grd
    raw = np.fromfile(grd, dtype="float32").reshape(365, 129, 135)
    ds = read_imd_year(grd, 2023, cfg)
    assert np.array_equal(ds["rain"].values[~np.isnan(ds["rain"].values)], raw[raw != -999])


def test_netcdf_problems_are_reported_not_guessed(cfg):
    write_nc(cfg, cfg.data_dir / "imd" / "RF25_ind2021_rfp25.nc", 2021, n_days=300)
    with pytest.raises(ValidationError, match="300 days"):
        read_imd_year(cfg.imd.year_path(2021), 2021, cfg)
    # a grid that is not the IMD grid
    lat, lon = np.arange(10.0, 20.0, 0.25), np.arange(70.0, 80.0, 0.25)
    ds = xr.Dataset({"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"), np.zeros((365, lat.size, lon.size), "float32"))},
                    coords={"TIME": pd.date_range("2020-01-01", periods=365), "LATITUDE": lat, "LONGITUDE": lon})
    ds = ds.assign_coords(TIME=pd.date_range("2019-01-01", periods=365))
    (cfg.data_dir / "imd").mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cfg.data_dir / "imd" / "RF25_ind2019_rfp25.nc")
    with pytest.raises(ValidationError, match="expected 129 x 135"):
        read_imd_year(cfg.imd.year_path(2019), 2019, cfg)


def test_two_netcdf_files_for_one_year_are_an_error(cfg):
    write_nc(cfg, cfg.data_dir / "imd" / "RF25_ind2023_rfp25.nc", 2023)
    write_nc(cfg, cfg.imd.raw_dir / "rain" / "2023.nc", 2023, seed=1)
    with pytest.raises(ValidationError, match="More than one"):
        resolve_imd_path(2023, cfg.imd)


def test_year_not_on_disk_is_a_missing_input(cfg):
    assert resolve_imd_path(1999, cfg.imd) is None
    with pytest.raises(MissingInputError):
        read_imd_year(cfg.imd.year_path(1999), 1999, cfg)


def test_download_is_skipped_when_a_netcdf_year_is_already_there(cfg):
    path = cfg.data_dir / "imd" / "RF25_ind2023_rfp25.nc"
    write_nc(cfg, path, 2023)

    def fetch(year, c):
        raise DownloadError("must not be called")

    assert download_imd([2023], cfg, fetch=fetch) == [path.resolve()]
