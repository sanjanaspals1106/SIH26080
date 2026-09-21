"""Metric grid helpers shared by the Stage 3 features (PRD 8.4, 8.6, 9.4). Owner: M1.

**Metric convention.** All distances and finite differences on the IMD grid use the PRD's local metric
(8.6, Appendix B): one 0.25 degree cell is `27.75 km` north-south and `27.75 * cos(latitude) km` east-west,
with the cell's own latitude. So a cell is `dy` metres tall and `dx = dy * cos(lat)` metres wide. Nothing is
ever computed in raw degrees. (District *areas* are the one exception: they use the equal-area EPSG:6933
projection, see `data_pipeline/districts/weights.py`.)

Finite differences (`masked_derivative`) use the central difference when both neighbours along the axis
exist, a one-sided difference when only one does (edges of the grid, coast of the valid area), and NaN when
none does.
"""

from __future__ import annotations

import numpy as np

from data_pipeline.ingestion.config import IngestionConfig


def cell_size_m(lat: np.ndarray, config: IngestionConfig) -> tuple[float, np.ndarray]:
    """(dy, dx) in metres: dy is one scalar, dx has shape (n_lat, 1) (one value per latitude row)."""
    step = config.imd.grid_spacing_deg
    dy = config.features.km_per_quarter_degree_lat * 1000.0 * (step / 0.25)
    return dy, (dy * np.cos(np.radians(np.asarray(lat, float))))[:, None]


def earth_radius_m(config: IngestionConfig) -> float:
    """Radius implied by the PRD's 27.75 km per 0.25 degree of latitude (used for the curvature term)."""
    return config.features.km_per_quarter_degree_lat * 1000.0 / np.radians(0.25)


def masked_derivative(field: np.ndarray, valid: np.ndarray, spacing, axis: int) -> np.ndarray:
    """d(field)/d(distance) along `axis` (-2 = latitude rows, -1 = longitude columns) of a (..., lat, lon) array.

    `spacing` is a scalar or an array broadcastable to `field` (metres, or km if you want per-km). Cells that
    are not `valid`, and cells with no valid neighbour along the axis, give NaN.
    """
    if axis not in (-2, -1):
        raise ValueError("axis must be -2 (lat) or -1 (lon)")
    f = np.where(valid, field, np.nan).astype("float64")
    prev, nxt = np.full_like(f, np.nan), np.full_like(f, np.nan)
    if axis == -2:
        prev[..., 1:, :], nxt[..., :-1, :] = f[..., :-1, :], f[..., 1:, :]
    else:
        prev[..., :, 1:], nxt[..., :, :-1] = f[..., :, :-1], f[..., :, 1:]
    has_p, has_n = ~np.isnan(prev), ~np.isnan(nxt)
    with np.errstate(invalid="ignore"):
        diff = np.where(
            has_p & has_n, (nxt - prev) / 2.0, np.where(has_n, nxt - f, np.where(has_p, f - prev, np.nan))
        )
    return np.where(valid, diff / spacing, np.nan)
