"""Compute per-cell Low-Pressure System features.

PRD Section 11.4 & Appendix B:
- lps_present: 1 if at least one centre found, else 0
- distance_to_lps_km: distance to nearest centre (3000 km if none)
- bearing_deg, bearing_sin, bearing_cos: direction to nearest centre (0, 0, 0 if none)
- lps_strength: strength_percentile of nearest centre (0 if none)
- lps_influence: lps_strength * exp(-(distance / 600)^2) (0 if none)
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from regime_engine.lps.geo_utils import haversine_distance_km, compute_bearing_and_components


def compute_lps_cell_features(
    cell_df: pd.DataFrame,
    centers: List[Dict[str, Any]],
    no_lps_distance_km: float = 3000.0,
    lps_influence_scale_km: float = 600.0,
) -> pd.DataFrame:
    """Compute cell-level LPS features for a given list of detected LPS centers.

    Args:
        cell_df: DataFrame containing at least ['cell_id', 'latitude', 'longitude'].
        centers: List of detected LPS centers [{'lat', 'lon', 'zeta_max', 'strength_percentile', ...}].
        no_lps_distance_km: Distance assigned when no LPS is detected (default 3000.0 km).
        lps_influence_scale_km: Distance decay scale for influence (default 600.0 km).

    Returns:
        pd.DataFrame with ['cell_id', 'lps_present', 'distance_to_lps_km',
                           'bearing_deg', 'bearing_sin', 'bearing_cos',
                           'lps_strength', 'lps_influence'].
    """
    n_cells = len(cell_df)
    cell_lats = cell_df["latitude"].to_numpy(dtype=float)
    cell_lons = cell_df["longitude"].to_numpy(dtype=float)
    cell_ids = cell_df["cell_id"].to_numpy()

    if not centers:
        return pd.DataFrame({
            "cell_id": cell_ids,
            "lps_present": np.zeros(n_cells, dtype=int),
            "distance_to_lps_km": np.full(n_cells, no_lps_distance_km, dtype=float),
            "bearing_deg": np.zeros(n_cells, dtype=float),
            "bearing_sin": np.zeros(n_cells, dtype=float),
            "bearing_cos": np.zeros(n_cells, dtype=float),
            "lps_strength": np.zeros(n_cells, dtype=float),
            "lps_influence": np.zeros(n_cells, dtype=float),
        })

    # When centres exist, find nearest centre for each cell
    best_distances = np.full(n_cells, np.inf, dtype=float)
    best_center_idx = np.zeros(n_cells, dtype=int)

    for c_idx, c in enumerate(centers):
        dists = haversine_distance_km(cell_lats, cell_lons, c["lat"], c["lon"])
        closer = dists < best_distances
        best_distances[closer] = dists[closer]
        best_center_idx[closer] = c_idx

    # Compute bearings and components to the nearest center
    nearest_lats = np.array([centers[i]["lat"] for i in best_center_idx])
    nearest_lons = np.array([centers[i]["lon"] for i in best_center_idx])
    nearest_strengths = np.array([centers[i].get("strength_percentile", 0.5) for i in best_center_idx])

    bearings, sin_b, cos_b = compute_bearing_and_components(cell_lats, cell_lons, nearest_lats, nearest_lons)

    # lps_influence = lps_strength * exp(-(dist / 600)^2)
    influence = nearest_strengths * np.exp(-((best_distances / lps_influence_scale_km) ** 2))

    return pd.DataFrame({
        "cell_id": cell_ids,
        "lps_present": np.ones(n_cells, dtype=int),
        "distance_to_lps_km": best_distances,
        "bearing_deg": bearings,
        "bearing_sin": sin_b,
        "bearing_cos": cos_b,
        "lps_strength": nearest_strengths,
        "lps_influence": influence,
    })
