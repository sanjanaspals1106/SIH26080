"""Raw indices computation for Layer C: coast and mountains.

PRD Section 11.5 & Appendix B:
- grad_h = gradient of elevation_m
- upslope_flux = q850 * max(0, wind · grad_h)
- onshore_flux = q850 * max(0, wind · coast_normal) * exp(-dist_coast_km / 100)
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_terrain_gradients(
    elevation_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spatial gradient of terrain height (metres per metre).

    Returns:
        (dh_dx, dh_dy)
    """
    # Grid spacing in meters: 1 deg lat ~ 111,000 m
    dy = np.gradient(lats) * 111000.0
    dx = np.gradient(lons) * 111000.0 * np.cos(np.radians(lats[:, None]))

    dh_dy = np.gradient(elevation_grid, axis=0) / dy[:, None]
    dh_dx = np.gradient(elevation_grid, axis=1) / dx

    return dh_dx, dh_dy


def compute_raw_fluxes(
    u850: np.ndarray,
    v850: np.ndarray,
    q850: np.ndarray,
    grad_h_x: np.ndarray,
    grad_h_y: np.ndarray,
    coast_normal_x: np.ndarray,
    coast_normal_y: np.ndarray,
    dist_coast_km: np.ndarray,
    coast_decay_scale_km: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute upslope_flux and onshore_flux for arrays/cells.

    Args:
        u850: Zonal wind (m/s)
        v850: Meridional wind (m/s)
        q850: Specific humidity at 850 hPa (kg/kg or g/kg)
        grad_h_x: d(elevation)/dx (m/m)
        grad_h_y: d(elevation)/dy (m/m)
        coast_normal_x: Zonal component of unit vector pointing sea->land
        coast_normal_y: Meridional component of unit vector pointing sea->land
        dist_coast_km: Distance to coast in km
        coast_decay_scale_km: Exponential decay distance (default 100 km)

    Returns:
        (upslope_flux, onshore_flux)
    """
    # Wind dot terrain gradient
    wind_dot_grad = u850 * grad_h_x + v850 * grad_h_y
    upslope_component = np.maximum(0.0, wind_dot_grad)
    upslope_flux = q850 * upslope_component

    # Wind dot coast normal
    wind_dot_coast = u850 * coast_normal_x + v850 * coast_normal_y
    onshore_component = np.maximum(0.0, wind_dot_coast)
    decay = np.exp(-np.maximum(0.0, dist_coast_km) / coast_decay_scale_km)
    onshore_flux = q850 * onshore_component * decay

    return np.maximum(0.0, upslope_flux), np.maximum(0.0, onshore_flux)
