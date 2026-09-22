"""Static geography (slope, aspect, distance to coast) and the metric-grid convention (PRD 8.6)."""

import numpy as np
import pytest
import xarray as xr
from stage3_world import C1, C2, N_SEA_COLS, make_static_grid

import data_pipeline.features.static as static_module
from data_pipeline.features import compute_static_geography, get_static_geography
from data_pipeline.features.grid import cell_size_m, earth_radius_m, masked_derivative
from data_pipeline.features.static import distance_to_coast_km
from data_pipeline.ingestion import MissingInputError, ValidationError
from data_pipeline.ingestion.imd import imd_axes


@pytest.fixture(scope="module")
def table():
    from data_pipeline.ingestion import load_config

    cfg = load_config()
    return cfg, compute_static_geography(make_static_grid(cfg), cfg)


def test_metric_convention_is_27_75_km_per_quarter_degree(cfg):
    lat, _ = imd_axes(cfg.imd)
    dy, dx = cell_size_m(lat, cfg)
    assert dy == pytest.approx(27_750.0)  # metres north-south, everywhere
    assert dx[:, 0] == pytest.approx(27_750.0 * np.cos(np.radians(lat)))  # shrinks with latitude, in metres
    assert dx[0, 0] > dx[-1, 0]
    assert earth_radius_m(cfg) == pytest.approx(
        27_750.0 / np.radians(0.25)
    )  # ~6.36e6 m: consistent with the km/cell rule


def test_static_table_has_every_grid_cell_once(table):
    cfg, t = table
    assert list(t.columns) == ["cell_id", "elevation_m", "slope", "aspect_sin", "aspect_cos", "dist_coast_km"]
    assert len(t) == 129 * 135 and t["cell_id"].is_unique
    assert t["elevation_m"].iloc[0] == pytest.approx(100.0)  # cell 0 (south-west corner) = orog there


def test_slope_is_metres_per_metre_from_central_differences(table):
    cfg, t = table
    lat, _ = imd_axes(cfg.imd)
    dy, dx = cell_size_m(lat, cfg)
    ilat = t["cell_id"].to_numpy() // 135
    gy, gx = C1 / dy, C2 / dx[ilat, 0]  # terrain rises C1 m per cell north and C2 m per cell east
    assert t["slope"].to_numpy() == pytest.approx(
        np.hypot(gy, gx), rel=1e-4
    )  # also right on the grid edge (one-sided)
    assert 0.001 < t["slope"].max() < 0.01  # a few mm per m: a gentle slope, not degrees or km


def test_aspect_is_the_downhill_bearing(table):
    cfg, t = table
    lat, _ = imd_axes(cfg.imd)
    dy, dx = cell_size_m(lat, cfg)
    ilat = t["cell_id"].to_numpy() // 135
    gy, gx = C1 / dy, C2 / dx[ilat, 0]
    # terrain rises to the north-east, so downhill is south-west: negative east (sin) and negative north (cos) parts
    assert t["aspect_sin"].to_numpy() == pytest.approx(-gx / np.hypot(gy, gx), rel=1e-4)
    assert t["aspect_cos"].to_numpy() == pytest.approx(-gy / np.hypot(gy, gx), rel=1e-4)
    assert (t["aspect_sin"] < 0).all() and (t["aspect_cos"] < 0).all()
    assert np.hypot(t["aspect_sin"], t["aspect_cos"]).to_numpy() == pytest.approx(1.0, rel=1e-5)


def test_slope_facing_each_compass_direction(cfg):
    lat, lon = imd_axes(cfg.imd)
    ilat, ilon = np.meshgrid(np.arange(lat.size), np.arange(lon.size), indexing="ij")
    lsm = np.ones_like(ilat, dtype="float32")

    def aspect(orog):
        ds = xr.Dataset(
            {"orog": (("lat", "lon"), orog.astype("float32")), "lsm": (("lat", "lon"), lsm)},
            coords={"lat": lat, "lon": lon},
        )
        row = compute_static_geography(ds, cfg).iloc[64 * 135 + 60]
        return round(float(row.aspect_sin), 3), round(float(row.aspect_cos), 3)

    assert aspect(100.0 * ilat) == (0.0, -1.0)  # rises to the north: downhill is south (bearing 180)
    assert aspect(-100.0 * ilat) == (0.0, 1.0)  # downhill is north (bearing 0)
    assert aspect(100.0 * ilon) == (-1.0, 0.0)  # rises to the east: downhill is west (bearing 270)
    assert aspect(-100.0 * ilon) == (1.0, 0.0)  # downhill is east (bearing 90)
    assert aspect(np.full_like(ilat, 500.0, dtype=float)) == (0.0, 0.0)  # flat: no direction
    flat = compute_static_geography(
        xr.Dataset(
            {"orog": (("lat", "lon"), np.full(ilat.shape, 500.0, "float32")), "lsm": (("lat", "lon"), lsm)},
            coords={"lat": lat, "lon": lon},
        ),
        cfg,
    )
    assert (flat["slope"] == 0).all()


def test_distance_to_coast_uses_kilometres_not_degrees(table):
    cfg, t = table
    lat, _ = imd_axes(cfg.imd)
    t = t.assign(ilat=t["cell_id"] // 135, ilon=t["cell_id"] % 135)
    sea = t[t["ilon"] < N_SEA_COLS]
    assert (sea["dist_coast_km"] == 0).all()  # sea cells
    land = t[t["ilon"] >= N_SEA_COLS]
    # the nearest sea cell is in the same row, column 19: (ilon - 19) cells east-west, 27.75 * cos(lat) km each
    expected = (land["ilon"] - (N_SEA_COLS - 1)) * 27.75 * np.cos(np.radians(lat[land["ilat"]]))
    assert land["dist_coast_km"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-5)
    # the same number of cells is fewer km far north: a distance in degrees would not show this
    far = t[t["ilon"] == 39].sort_values("ilat")
    assert far["dist_coast_km"].iloc[0] > far["dist_coast_km"].iloc[-1]


def test_distance_to_coast_diagonal_and_no_sea(cfg):
    lat, lon = imd_axes(cfg.imd)
    lsm = np.ones((lat.size, lon.size))
    lsm[10, 10] = 0.0  # a single sea cell
    d = distance_to_coast_km(lsm, lat, cfg)
    assert d[10, 10] == 0
    assert d[13, 10] == pytest.approx(3 * 27.75)  # three rows north: pure north-south, 27.75 km per row
    east_km = 27.75 * np.cos(np.radians(lat[10]))
    assert d[10, 14] == pytest.approx(4 * east_km)
    assert d[13, 14] == pytest.approx(
        np.hypot(3 * 27.75, 4 * 27.75 * np.cos(np.radians(lat[13])))
    )  # uses the land cell's latitude
    assert np.isnan(distance_to_coast_km(np.ones((lat.size, lon.size)), lat, cfg)).all()  # no sea at all
    assert (distance_to_coast_km(np.zeros((lat.size, lon.size)), lat, cfg) == 0).all()  # only sea


def test_lsm_threshold_is_half(cfg):
    lat, _ = imd_axes(cfg.imd)
    lsm = np.ones((lat.size, 135))
    lsm[5, 5], lsm[7, 5] = 0.49, 0.5  # 0.49 is sea, 0.5 is land
    d = distance_to_coast_km(lsm, lat, cfg)
    assert d[5, 5] == 0 and d[7, 5] > 0


def test_bad_static_input_is_rejected(cfg):
    grid = make_static_grid(cfg)
    with pytest.raises(ValidationError, match="missing 'lsm'"):
        compute_static_geography(grid.drop_vars("lsm"), cfg)
    with pytest.raises(ValidationError, match="expected the IMD grid"):
        compute_static_geography(grid.isel(lat=slice(0, 50)), cfg)
    with pytest.raises(ValidationError, match="NaN"):
        compute_static_geography(grid.where(grid["orog"] < 1000), cfg)


def test_masked_derivative_rules():
    f = np.arange(5.0).reshape(1, 1, 5) ** 2  # 0 1 4 9 16
    v = np.ones_like(f, dtype=bool)
    d = masked_derivative(f, v, 1.0, -1)[0, 0]
    assert d == pytest.approx([1, 2, 4, 6, 7])  # one-sided at the ends, central inside
    v[0, 0, 2] = False
    d = masked_derivative(f, v, 1.0, -1)[0, 0]
    assert (
        np.isnan(d[2]) and d[1] == 1 and d[3] == 7
    )  # next to an invalid cell: one-sided from the valid side
    with pytest.raises(ValueError):
        masked_derivative(f, v, 1.0, 0)


# ---- caching -----------------------------------------------------------------------------------


def tigge_static(cfg, shift=0.0):
    """Static fields on the TIGGE grid (N40 W55 S0 E100, 0.25 degree), as read by Stage 1."""
    lat, lon = np.arange(0, 40.01, 0.25), np.arange(55, 100.01, 0.25)
    orog = 100.0 + 40.0 * lat[:, None] * 4 + 10.0 * (lon[None, :] - 55) * 4 + shift
    lsm = (lon[None, :] >= 55 + 0.25 * N_SEA_COLS).astype("float32") * np.ones((lat.size, 1), "float32")
    return xr.Dataset(
        {"orog": (("lat", "lon"), orog.astype("float32")), "lsm": (("lat", "lon"), lsm)},
        coords={"lat": lat, "lon": lon},
    )


def test_static_geography_is_cached_and_reused(cfg, monkeypatch):
    first = get_static_geography(cfg, static=tigge_static(cfg))
    assert static_module.static_path(cfg).is_file()
    monkeypatch.setattr(static_module, "compute_static_geography", lambda *a, **k: pytest.fail("recomputed"))
    again = get_static_geography(cfg, static=tigge_static(cfg))  # identical inputs: read from the cache
    assert first.equals(again)
    assert get_static_geography(cfg).equals(first)  # no inputs at all: the cache is used as is


def test_static_geography_cache_is_invalidated_when_the_inputs_change(cfg):
    a = get_static_geography(cfg, static=tigge_static(cfg))
    b = get_static_geography(cfg, static=tigge_static(cfg, shift=500.0))
    assert not a["elevation_m"].equals(b["elevation_m"])
    assert b["elevation_m"].iloc[0] == pytest.approx(a["elevation_m"].iloc[0] + 500.0)


def test_no_static_data_and_no_cache_is_a_clear_error(cfg):
    with pytest.raises(MissingInputError, match="static_date"):
        get_static_geography(cfg)
