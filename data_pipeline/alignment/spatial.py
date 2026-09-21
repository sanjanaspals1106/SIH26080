"""Spatial alignment onto the IMD 0.25 degree grid (PRD 8.4). Owner: M1.

Every regridding here is separable (latitude and longitude are treated one after the other), so each
one is a pair of small weight matrices `W_lat (n_target_lat x n_source_lat)` and `W_lon`. The matrices
are built once per source grid (`build_grid_map`) and reused for every variable and every date.

* **Rain** (`regrid_rain`): if the source grid is finer than (or equal to) the IMD grid, average all
  source points inside each IMD cell; if it is coarser, take the source cell that contains the IMD cell
  centre. Never interpolates, so area-average rain stays about right (PRD 8.4).
  A cell is the half-open box `[centre - 0.125, centre + 0.125)` in each direction, so a point exactly on
  a cell edge belongs to the cell above/east of it.
* **Atmosphere and static fields** (`regrid_bilinear`): bilinear interpolation.

Target points outside the source domain, or whose source points include a NaN, become NaN.
Latitude, longitude, dimension order, names and attributes (units) are preserved.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import xarray as xr

from data_pipeline.ingestion.errors import ValidationError

log = logging.getLogger(__name__)

_EPS = 1e-9


class GridMap(NamedTuple):
    """Weight matrices from one source grid to the target grid."""

    tgt_lat: np.ndarray
    tgt_lon: np.ndarray
    rain_lat: np.ndarray
    rain_lon: np.ndarray
    lin_lat: np.ndarray
    lin_lon: np.ndarray
    rain_method: str  # "area_mean", "containing_cell" or "mixed"


def _step(axis: np.ndarray, name: str) -> float:
    if axis.size < 2:
        raise ValidationError(f"Source {name} axis has fewer than 2 points; cannot regrid.")
    d = np.diff(axis)
    if (d <= 0).any() or not np.allclose(d, d[0], atol=1e-6):
        raise ValidationError(f"Source {name} axis must be regular and ascending to regrid.")
    return float(d[0])


def _rain_weights(src: np.ndarray, tgt: np.ndarray, tgt_step: float, name: str) -> tuple[np.ndarray, str]:
    step = _step(src, name)
    w = np.zeros((tgt.size, src.size))
    if step <= tgt_step + 1e-6:
        # finer or equal: each source point belongs to one target cell; average within the cell
        idx = np.floor((src - tgt[0]) / tgt_step + 0.5 + _EPS).astype(int)
        for j, i in enumerate(idx):
            if 0 <= i < tgt.size:
                w[i, j] = 1.0
        counts = w.sum(axis=1, keepdims=True)
        np.divide(w, counts, out=w, where=counts > 0)
        return w, "area_mean"
    # coarser: the source cell containing each target centre
    idx = np.floor((tgt - src[0]) / step + 0.5 + _EPS).astype(int)
    for i, j in enumerate(idx):
        if 0 <= j < src.size:
            w[i, j] = 1.0
    return w, "containing_cell"


def _linear_weights(src: np.ndarray, tgt: np.ndarray, name: str) -> np.ndarray:
    _step(src, name)
    w = np.zeros((tgt.size, src.size))
    for i, t in enumerate(tgt):
        if t < src[0] - 1e-6 or t > src[-1] + 1e-6:
            continue  # outside the source domain: row stays 0 -> NaN
        j = int(np.clip(np.searchsorted(src, t, side="right") - 1, 0, src.size - 2))
        f = float(np.clip((t - src[j]) / (src[j + 1] - src[j]), 0.0, 1.0))
        w[i, j], w[i, j + 1] = 1 - f, f
    return w


def build_grid_map(
    src_lat: np.ndarray, src_lon: np.ndarray, tgt_lat: np.ndarray, tgt_lon: np.ndarray
) -> GridMap:
    """Weights from a source grid to the target (IMD) grid. All axes ascending and regular."""
    src_lat, src_lon = np.asarray(src_lat, float), np.asarray(src_lon, float)
    tgt_lat, tgt_lon = np.asarray(tgt_lat, float), np.asarray(tgt_lon, float)
    tgt_step_lat, tgt_step_lon = _step(tgt_lat, "target lat"), _step(tgt_lon, "target lon")
    rlat, mlat = _rain_weights(src_lat, tgt_lat, tgt_step_lat, "lat")
    rlon, mlon = _rain_weights(src_lon, tgt_lon, tgt_step_lon, "lon")
    method = mlat if mlat == mlon else "mixed"
    return GridMap(
        tgt_lat,
        tgt_lon,
        rlat,
        rlon,
        _linear_weights(src_lat, tgt_lat, "lat"),
        _linear_weights(src_lon, tgt_lon, "lon"),
        method,
    )


def _apply(da: xr.DataArray, w_lat: np.ndarray, w_lon: np.ndarray, gm: GridMap) -> xr.DataArray:
    if da.dims[-2:] != ("lat", "lon"):
        da = da.transpose(..., "lat", "lon")
    if da.sizes["lat"] != w_lat.shape[1] or da.sizes["lon"] != w_lon.shape[1]:
        raise ValidationError("Field grid does not match the grid the GridMap was built for.")
    data = da.values.astype("float64")
    nan = np.isnan(data)
    covered = (w_lat.sum(1)[:, None] * w_lon.sum(1)[None, :]) > 1 - 1e-9  # (tgt_lat, tgt_lon)
    out = np.einsum("il,...lk,jk->...ij", w_lat, np.where(nan, 0.0, data), w_lon, optimize=True)
    if nan.any():  # any NaN among the contributing source points -> NaN
        ok = np.einsum("il,...lk,jk->...ij", w_lat, (~nan).astype("float64"), w_lon, optimize=True)
        out = np.where(ok > 1 - 1e-6, out, np.nan)
    else:
        out = np.where(covered, out, np.nan)
    coords = {d: da.coords[d] for d in da.dims[:-2] if d in da.coords}
    coords.update(lat=gm.tgt_lat, lon=gm.tgt_lon)
    return xr.DataArray(out.astype("float32"), dims=da.dims, coords=coords, name=da.name, attrs=da.attrs)


def regrid_rain(da: xr.DataArray, gm: GridMap) -> xr.DataArray:
    """Rain: area mean (finer source) or containing cell (coarser source). No interpolation."""
    return _apply(da, gm.rain_lat, gm.rain_lon, gm)


def regrid_bilinear(da: xr.DataArray, gm: GridMap) -> xr.DataArray:
    """Atmosphere and static fields: bilinear interpolation."""
    return _apply(da, gm.lin_lat, gm.lin_lon, gm)


def regrid_dataset_bilinear(ds: xr.Dataset, gm: GridMap) -> xr.Dataset:
    return xr.Dataset({v: regrid_bilinear(ds[v], gm) for v in ds.data_vars}, attrs=ds.attrs)
