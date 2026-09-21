"""Feature extraction for Layer A monsoon phase classification.

PRD Section 11.3 & Appendix B:
Features A1-A6 from forecast fields:
- A1: Peninsular westerly wind: mean u850 over 10-20°N, 65-85°E
- A2: Core-zone vorticity: mean vort850 over 18-28°N, 68-88°E
- A3: Trough position: Latitude of lowest value of zonal-mean msl over 75-90°E (searched 15-32°N)
- A4: Core-zone pressure anomaly: Mean msl over 18-28°N, 68-88°E minus training-season mean
- A5: Moisture: Mean q850 over 10-25°N, 70-90°E
- A6: Wind shear: Mean of (u850 - u200) over 5-20°N, 60-100°E
- Season position: doy_sin, doy_cos

Each of A1-A6 is used for the lead itself and for the two neighbouring leads of the same run,
repeating the nearest available lead when at boundaries (giving 18 + 2 = 20 features).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_box_mean(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> float:
    """Compute spatial mean over a latitude-longitude bounding box."""
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)

    sub = field[np.ix_(lat_mask, lon_mask)]
    valid = sub[~np.isnan(sub)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def compute_trough_position(
    msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    zonal_lon_range: Tuple[float, float] = (75.0, 90.0),
    search_lat_range: Tuple[float, float] = (15.0, 32.0),
) -> float:
    """Find the latitude of the lowest zonal-mean MSLP between search_lat_range.

    Feature A3 (PRD 11.3).
    """
    lon_mask = (lons >= zonal_lon_range[0]) & (lons <= zonal_lon_range[1])
    lat_mask = (lats >= search_lat_range[0]) & (lats <= search_lat_range[1])

    sub_lats = lats[lat_mask]
    sub_msl = msl[np.ix_(lat_mask, lon_mask)]

    # Compute zonal mean across longitude for each latitude
    zonal_mean = np.nanmean(sub_msl, axis=1)
    if len(zonal_mean) == 0 or np.all(np.isnan(zonal_mean)):
        return 22.0  # Safe default within search range

    min_idx = np.nanargmin(zonal_mean)
    return float(sub_lats[min_idx])


def compute_season_position(doy: int) -> Tuple[float, float]:
    """Compute sinusoidal day of year features doy_sin and doy_cos."""
    rad = 2.0 * np.pi * (doy / 365.25)
    return float(np.sin(rad)), float(np.cos(rad))


def extract_single_lead_a_features(
    fields: Dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    training_core_msl_mean: float = 1005.0,
) -> Dict[str, float]:
    """Extract features A1-A6 for a single forecast lead window.

    Expected fields:
        'u850': 2D array (lat, lon)
        'v850': 2D array (lat, lon)
        'vort850': 2D array (lat, lon) or None (computed from u/v if not present)
        'u200': 2D array (lat, lon)
        'q850': 2D array (lat, lon)
        'msl': 2D array (lat, lon) in hPa or Pa (if Pa, converted to hPa)

    Args:
        fields: Mapping of field name to 2D numpy array on (lats, lons) grid.
        lats: 1D array of latitudes (monotonic).
        lons: 1D array of longitudes (monotonic).
        training_core_msl_mean: Mean MSLP over core zone from training seasons.

    Returns:
        Dict with keys 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'.
    """
    msl = fields["msl"].copy()
    # Normalize Pa to hPa if values are in thousands
    if np.nanmean(msl) > 2000.0:
        msl = msl / 100.0

    # A1: Peninsular westerly wind: mean u850 over 10-20°N, 65-85°E
    a1 = compute_box_mean(fields["u850"], lats, lons, (10.0, 20.0), (65.0, 85.0))

    # A2: Core-zone vorticity: mean vort850 over 18-28°N, 68-88°E
    if "vort850" in fields and fields["vort850"] is not None:
        vort850 = fields["vort850"]
    else:
        # Approximate relative vorticity dv/dx - du/dy
        # 1 deg lat ~ 111,000 m
        dy = np.gradient(lats) * 111000.0
        dx = np.gradient(lons) * 111000.0 * np.cos(np.radians(lats[:, None]))
        du_dy = np.gradient(fields["u850"], axis=0) / dy[:, None]
        dv_dx = np.gradient(fields["v850"], axis=1) / dx
        vort850 = dv_dx - du_dy
    a2 = compute_box_mean(vort850, lats, lons, (18.0, 28.0), (68.0, 88.0))

    # A3: Trough position: Latitude of lowest zonal-mean msl over 75-90°E (15-32°N)
    a3 = compute_trough_position(msl, lats, lons, (75.0, 90.0), (15.0, 32.0))

    # A4: Core-zone pressure anomaly: mean msl over core zone minus training mean
    core_msl = compute_box_mean(msl, lats, lons, (18.0, 28.0), (68.0, 88.0))
    a4 = core_msl - training_core_msl_mean

    # A5: Moisture: mean q850 over 10-25°N, 70-90°E
    # q850 typically kg/kg; if g/kg, standard units preserved
    a5 = compute_box_mean(fields["q850"], lats, lons, (10.0, 25.0), (70.0, 90.0))

    # A6: Wind shear: mean of (u850 - u200) over 5-20°N, 60-100°E
    shear_field = fields["u850"] - fields["u200"]
    a6 = compute_box_mean(shear_field, lats, lons, (5.0, 20.0), (60.0, 100.0))

    return {
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "A5": a5,
        "A6": a6,
    }


def build_phase_feature_vector(
    lead_features: Dict[int, Dict[str, float]],
    target_lead: int,
    doy: int,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble the 20-feature input vector for the phase model at target_lead.

    Uses A1-A6 for (prev_lead, curr_lead, next_lead) with edge boundary repetition,
    plus doy_sin, doy_cos.

    Args:
        lead_features: Dict mapping lead_day (e.g. 1, 2, 3) to {'A1'..'A6': float}.
        target_lead: Target lead day (1, 2, or 3).
        doy: Day of year (1..366).

    Returns:
        (feature_vector, feature_names) where len == 20.
    """
    # Determine neighbouring leads
    if target_lead == 1:
        leads = [1, 1, 2]
    elif target_lead == 2:
        leads = [1, 2, 3]
    elif target_lead == 3:
        leads = [2, 3, 3]
    else:
        leads = [target_lead, target_lead, target_lead]

    prefixes = ["prev", "curr", "next"]
    vec = []
    names = []

    for prefix, lk in zip(prefixes, leads):
        feat_dict = lead_features.get(lk, lead_features.get(target_lead, {}))
        for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            val = feat_dict.get(aid, 0.0)
            vec.append(val)
            names.append(f"{aid}_{prefix}")

    doy_sin, doy_cos = compute_season_position(doy)
    vec.extend([doy_sin, doy_cos])
    names.extend(["doy_sin", "doy_cos"])

    return np.array(vec, dtype=float), names
