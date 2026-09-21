"""Geographical calculations for Low-Pressure System (LPS) detection and cell features.

Implements great-circle distances, azimuth/bearing, and annular ring masks.
"""

from typing import Tuple, Union
import numpy as np


EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute great-circle distance between two points or point grids in kilometers."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def compute_bearing_and_components(
    lat_from: Union[float, np.ndarray],
    lon_from: Union[float, np.ndarray],
    lat_to: Union[float, np.ndarray],
    lon_to: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute initial bearing (degrees 0-360) and its (sin, cos) components from (from) to (to).

    Returns:
        (bearing_deg, bearing_sin, bearing_cos)
    """
    phi1 = np.radians(lat_from)
    phi2 = np.radians(lat_to)
    dlambda = np.radians(lon_to - lon_from)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)

    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0

    return bearing_deg, np.sin(bearing_rad), np.cos(bearing_rad)
