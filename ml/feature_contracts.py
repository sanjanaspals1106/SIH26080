"""Canonical M3 Feature Contracts (PRD §9.4, §10.6, §11.7, §12.1).

Defines the official, canonical feature lists in exact PRD order:
- BASE_FEATURES (27 features for B2)
- REGIME_FEATURES (14 features from Regime Engine)
- B2_FEATURES (identical to BASE_FEATURES, len=27)
- B3_FEATURES (BASE_FEATURES + REGIME_FEATURES, len=41)

Rules:
- Cell ID must never be a model feature (PRD L7).
- Observed rain (obs_mm) must never be a feature (PRD L1).
- NaN is allowed in feature values (e.g. rain_prev_lead at lead 1, rain_next_lead at lead 3).
- Imputation of rain_prev_lead / rain_next_lead is strictly forbidden.
"""

from typing import List, Sequence, Union
import pandas as pd

# Exactly 27 Base Features in PRD §9.4 Order
BASE_FEATURES: List[str] = [
    # Rain group (8)
    "rain_mm",
    "nbr_mean_3",
    "nbr_max_3",
    "nbr_mean_5",
    "nbr_max_5",
    "rain_grad",
    "rain_prev_lead",
    "rain_next_lead",
    # Circulation and moisture group (7)
    "u850",
    "v850",
    "wspd850",
    "vort850",
    "q850",
    "msl",
    "shear_200_850",
    # Geography group (5)
    "elevation_m",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "dist_coast_km",
    # Climatology group (2)
    "clim_mean",
    "clim_p95",
    # Time and place group (5)
    "doy_sin",
    "doy_cos",
    "lead_day",
    "latitude",
    "longitude",
]

# Exactly 14 Regime Features in PRD §11.7 Order
REGIME_FEATURES: List[str] = [
    # Layer A - Monsoon Phase (4)
    "p_active",
    "p_normal",
    "p_break",
    "regime_confidence",
    # Layer B - Low-Pressure Systems (6)
    "lps_present",
    "distance_to_lps_km",
    "bearing_sin",
    "bearing_cos",
    "lps_strength",
    "lps_influence",
    # Layer C - Local Context (4)
    "upslope_flux",
    "onshore_flux",
    "orographic_influence",
    "coastal_influence",
]

# Model Feature Sets
B2_FEATURES: List[str] = list(BASE_FEATURES)
B3_FEATURES: List[str] = list(BASE_FEATURES) + list(REGIME_FEATURES)

# Forbidden feature identifiers (PRD §10.6 L1, L7)
FORBIDDEN_FEATURES: List[str] = [
    "cell_id",
    "obs_mm",
    "observed_rain",
    "obs_ge_15_6",
    "obs_ge_64_5",
    "obs_ge_115_6",
]

# Structural Contract Invariants
assert len(BASE_FEATURES) == 27, f"Expected 27 base features, got {len(BASE_FEATURES)}"
assert len(REGIME_FEATURES) == 14, f"Expected 14 regime features, got {len(REGIME_FEATURES)}"
assert len(B2_FEATURES) == 27, f"Expected 27 B2 features, got {len(B2_FEATURES)}"
assert len(B3_FEATURES) == 41, f"Expected 41 B3 features, got {len(B3_FEATURES)}"
assert len(set(B2_FEATURES)) == 27, "Duplicate feature found in B2_FEATURES"
assert len(set(B3_FEATURES)) == 41, "Duplicate feature found in B3_FEATURES"
assert "cell_id" not in B3_FEATURES, "cell_id must never be in feature contract (L7)"
assert "obs_mm" not in B3_FEATURES, "obs_mm must never be in feature contract (L1)"


def get_feature_list(model_type: str) -> List[str]:
    """Return the canonical feature list for a given model type.

    Args:
        model_type: 'B2' or 'B3' (case-insensitive).

    Returns:
        List of feature column names in exact PRD order.
    """
    m = model_type.upper()
    if m == "B2":
        return list(B2_FEATURES)
    elif m == "B3":
        return list(B3_FEATURES)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'B2' or 'B3'.")


def validate_feature_columns(
    df: pd.DataFrame,
    model_type: str = "B2",
    custom_features: Sequence[str] = None,
) -> List[str]:
    """Validate that DataFrame contains all required feature columns.

    Rules:
    - All required features must be present in df.columns.
    - Forbidden columns ('cell_id', 'obs_mm') must never be in the feature list.
    - NaN values are allowed (e.g. rain_prev_lead at lead 1) and will NOT raise error.
    - Does NOT impute missing values.

    Args:
        df: DataFrame to validate.
        model_type: 'B2' or 'B3'.
        custom_features: Optional explicit feature list to validate instead of model_type default.

    Returns:
        The validated feature list in exact canonical order.

    Raises:
        ValueError: If any required feature is missing, or a forbidden column is in features.
    """
    target_features = list(custom_features) if custom_features is not None else get_feature_list(model_type)

    # Check for forbidden features
    for forbidden in FORBIDDEN_FEATURES:
        if forbidden in target_features:
            raise ValueError(
                f"Protocol violation (PRD §10.6): '{forbidden}' is forbidden as a model input feature."
            )

    # Check for missing required columns in df
    missing = [col for col in target_features if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required feature columns for {model_type}: {missing}"
        )

    return target_features
