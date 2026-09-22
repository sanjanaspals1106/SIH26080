"""Spatial alignment onto the IMD grid (PRD 8.4): rain area-mean / containing cell, bilinear atmosphere."""

import numpy as np
import pytest
import xarray as xr

from data_pipeline.alignment import build_grid_map, regrid_bilinear, regrid_rain
from data_pipeline.ingestion import ValidationError
from data_pipeline.ingestion.imd import imd_axes

TGT_LAT = np.array([10.0, 10.25, 10.5])
TGT_LON = np.array([70.0, 70.25, 70.5])


def field(lat, lon, values, **kw):
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, **kw)


def brute_area_mean(src_lat, src_lon, values, tgt_lat, tgt_lon, half=0.125):
    """Independent implementation: for every target cell, mean of the source points inside [c-h, c+h)."""
    out = np.full((tgt_lat.size, tgt_lon.size), np.nan)
    for i, la in enumerate(tgt_lat):
        for j, lo in enumerate(tgt_lon):
            in_lat = (src_lat >= la - half - 1e-9) & (src_lat < la + half - 1e-9)
            in_lon = (src_lon >= lo - half - 1e-9) & (src_lon < lo + half - 1e-9)
            if in_lat.any() and in_lon.any():
                out[i, j] = values[np.ix_(in_lat, in_lon)].mean()
    return out


# ---- rain: finer source --------------------------------------------------------------------


def test_finer_source_is_averaged_inside_each_cell():
    src_lat = 9.875 + 0.125 * np.arange(7)  # 9.875 ... 10.625
    src_lon = 69.875 + 0.125 * np.arange(7)
    values = np.random.default_rng(1).random((7, 7)) * 20
    gm = build_grid_map(src_lat, src_lon, TGT_LAT, TGT_LON)
    out = regrid_rain(field(src_lat, src_lon, values), gm)
    assert gm.rain_method == "area_mean"
    assert out.values == pytest.approx(brute_area_mean(src_lat, src_lon, values, TGT_LAT, TGT_LON), abs=1e-5)
    # cell (10.0, 70.0) is the mean of the 2 x 2 points {9.875, 10.0} x {69.875, 70.0}
    assert out.values[0, 0] == pytest.approx(values[:2, :2].mean(), abs=1e-5)


def test_equal_spacing_aligned_grids_keep_every_value():
    values = np.random.default_rng(2).random((3, 3))
    gm = build_grid_map(TGT_LAT, TGT_LON, TGT_LAT, TGT_LON)
    assert regrid_rain(field(TGT_LAT, TGT_LON, values), gm).values == pytest.approx(values, abs=1e-6)


def test_area_mean_keeps_the_total_rain():
    src_lat, src_lon = 9.875 + 0.125 * np.arange(6), 69.875 + 0.125 * np.arange(6)  # exactly 3 x 3 cells
    values = np.random.default_rng(3).random((6, 6)) * 50
    out = regrid_rain(field(src_lat, src_lon, values), build_grid_map(src_lat, src_lon, TGT_LAT, TGT_LON))
    assert out.values.mean() == pytest.approx(values.mean(), rel=1e-5)


# ---- rain: coarser source ------------------------------------------------------------------


def test_coarser_source_gives_the_value_of_the_containing_cell_not_an_interpolation():
    src_lat, src_lon = np.array([9.5, 10.0, 10.5, 11.0]), np.array([69.5, 70.0, 70.5, 71.0])
    values = np.arange(16, dtype=float).reshape(4, 4) * 10  # 0, 10, 20 ... 150
    gm = build_grid_map(src_lat, src_lon, TGT_LAT, TGT_LON)
    out = regrid_rain(field(src_lat, src_lon, values), gm).values
    assert gm.rain_method == "containing_cell"
    # Source cells are [c-0.25, c+0.25). Target 10.0 -> 10.0; 10.25 (on the edge) -> 10.5; 10.5 -> 10.5. Same in lon.
    lat_idx, lon_idx = [1, 2, 2], [1, 2, 2]
    assert out == pytest.approx(values[np.ix_(lat_idx, lon_idx)])
    assert set(np.round(out.ravel(), 6)) <= set(values.ravel())  # only existing source values, nothing new


def test_target_outside_the_source_domain_is_nan():
    src = np.array([10.0, 10.5])
    far_lon = np.array([130.0, 130.5])  # nowhere near TGT_LON
    out = regrid_rain(field(src, far_lon, np.ones((2, 2))), build_grid_map(src, far_lon, TGT_LAT, TGT_LON))
    assert np.isnan(out.values).all()


# ---- atmosphere: bilinear ------------------------------------------------------------------


@pytest.mark.parametrize("step", [0.5, 0.125])
def test_bilinear_reproduces_a_linear_field_exactly(step):
    src_lat = np.arange(9.5, 11.01, step)
    src_lon = np.arange(69.5, 71.01, step)
    f = lambda la, lo: 2.0 * la + 3.0 * lo + 1.0  # noqa: E731
    da = field(src_lat, src_lon, f(src_lat[:, None], src_lon[None, :]))
    out = regrid_bilinear(da, build_grid_map(src_lat, src_lon, TGT_LAT, TGT_LON))
    assert out.values == pytest.approx(f(TGT_LAT[:, None], TGT_LON[None, :]), abs=1e-4)


def test_bilinear_differs_from_containing_cell_where_it_should():
    src = np.array([9.5, 10.5])  # 1 degree spacing
    da = field(src, src + 60.0, np.array([[0.0, 0.0], [10.0, 10.0]]))  # rises 10 per degree in lat
    gm = build_grid_map(src, src + 60.0, np.array([10.0, 10.25]), np.array([70.0, 70.25]))
    assert regrid_bilinear(da, gm).values[0, 0] == pytest.approx(5.0)
    assert regrid_rain(da, gm).values[0, 0] in (0.0, 10.0)


def test_bilinear_is_nan_outside_the_domain_and_where_source_is_nan():
    src = np.array([10.0, 10.25, 10.5, 10.75])
    vals = np.ones((4, 4))
    vals[1, 1] = np.nan
    tgt_lat, tgt_lon = np.array([10.125, 10.375, 10.625, 10.875]), np.array([70.125, 70.375, 70.625])
    out = regrid_bilinear(field(src, src + 60, vals), build_grid_map(src, src + 60, tgt_lat, tgt_lon)).values
    assert np.isnan(out[:2, :2]).all()  # the four cells around the NaN source point
    assert not np.isnan(out[2, 2]) and not np.isnan(out[0, 2])
    assert np.isnan(out[3]).all()  # lat 10.875 is outside the source domain (max 10.75)


# ---- coordinates, names, units -------------------------------------------------------------


def test_dims_coords_names_and_units_are_preserved():
    src_lat, src_lon = np.arange(9.5, 11.01, 0.25), np.arange(69.5, 71.01, 0.25)
    da = xr.DataArray(
        np.ones((2, 3, src_lat.size, src_lon.size)),
        dims=("init_time", "lead_day", "lat", "lon"),
        coords={"init_time": [0, 1], "lead_day": [1, 2, 3], "lat": src_lat, "lon": src_lon},
        name="u850",
        attrs={"units": "m s**-1"},
    )
    gm = build_grid_map(src_lat, src_lon, TGT_LAT, TGT_LON)
    for out in (regrid_bilinear(da, gm), regrid_rain(da, gm)):
        assert out.dims == da.dims and out.name == "u850" and out.attrs["units"] == "m s**-1"
        assert list(out["lat"].values) == list(TGT_LAT) and list(out["lon"].values) == list(TGT_LON)
        assert list(out["lead_day"].values) == [1, 2, 3]


def test_real_tigge_domain_onto_the_imd_grid_is_identity(cfg):
    """TIGGE 0.25 degree, N40 W55 S0 E100 -> IMD grid: every IMD cell centre is a TIGGE point."""
    src_lat, src_lon = np.arange(0, 40.01, 0.25), np.arange(55, 100.01, 0.25)
    lat, lon = imd_axes(cfg.imd)
    values = src_lat[:, None] * 1000 + src_lon[None, :]  # value encodes its own coordinates
    gm = build_grid_map(src_lat, src_lon, lat, lon)
    assert gm.rain_method == "area_mean"
    for regrid in (regrid_rain, regrid_bilinear):
        out = regrid(field(src_lat, src_lon, values), gm)
        assert out.shape == (129, 135) and not np.isnan(out.values).any()
        assert out.values == pytest.approx(lat[:, None] * 1000 + lon[None, :], rel=1e-6)


def test_irregular_source_axis_is_rejected():
    with pytest.raises(ValidationError, match="regular"):
        build_grid_map(np.array([0.0, 0.25, 0.6]), np.array([0.0, 0.25]), TGT_LAT, TGT_LON)


def test_field_from_another_grid_is_rejected():
    gm = build_grid_map(np.array([10.0, 10.5]), np.array([70.0, 70.5]), TGT_LAT, TGT_LON)
    with pytest.raises(ValidationError, match="does not match"):
        regrid_rain(field(np.array([10.0, 10.5, 11.0]), np.array([70.0, 70.5]), np.ones((3, 2))), gm)
