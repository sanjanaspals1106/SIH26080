"""Regime output contracts and 14-feature assembler for B3.

PRD Section 11.6 & 11.7:
- Domain level regime contract (§11.6)
- Cell level regime contract (§11.6)
- Extraction of the 14 regime features for model B3 (§11.7)
- Enforcement of rule: training code must stop with error if regime_source == 'final'.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

REGIME_14_FEATURES = [
    "p_active",
    "p_normal",
    "p_break",
    "regime_confidence",
    "lps_present",
    "distance_to_lps_km",
    "bearing_sin",
    "bearing_cos",
    "lps_strength",
    "lps_influence",
    "upslope_flux",
    "onshore_flux",
    "orographic_influence",
    "coastal_influence",
]


def validate_training_regime_source(regime_source: str) -> None:
    """Enforce PRD 11.6 rule: training rows must never use regime_source == 'final'."""
    if regime_source == "final":
        raise ValueError(
            "Forbidden leakage (PRD 11.6): Training code cannot use regime_source='final'. "
            "Must use out-of-fold ('oof') regime predictions for all training data."
        )


def build_domain_regime_contract(
    run_id: str,
    lead_day: int,
    p_active: float,
    p_normal: float,
    p_break: float,
    regime_confidence: float,
    confidence_band: str,
    regime_source: str,
    lps_detected: bool,
    lps_centres: List[Dict[str, Any]],
    lps_settings: str = "tuned",
    ood_flag: bool = False,
    regime_available: bool = True,
) -> Dict[str, Any]:
    """Construct Domain-level regime output dictionary conforming to PRD 11.6."""
    return {
        "run_id": str(run_id),
        "lead_day": int(lead_day),
        "phase": {
            "active_probability": round(float(p_active), 4),
            "normal_probability": round(float(p_normal), 4),
            "break_probability": round(float(p_break), 4),
            "regime_confidence": round(float(regime_confidence), 4),
            "confidence_band": str(confidence_band),
            "regime_source": str(regime_source),
        },
        "lps": {
            "detected": bool(lps_detected),
            "centres": [
                {
                    "lat": round(float(c["lat"]), 2),
                    "lon": round(float(c["lon"]), 2),
                    "zeta_max": float(c["zeta_max"]),
                    "mslp_min_hpa": round(float(c["mslp_min_hpa"]), 1),
                    "strength_percentile": round(float(c.get("strength_percentile", 0.0)), 2),
                }
                for c in lps_centres
            ],
            "settings": str(lps_settings),
        },
        "quality": {
            "ood_flag": bool(ood_flag),
            "regime_available": bool(regime_available),
        },
    }


def build_cell_regime_contract(
    cell_id: int,
    lps_present: bool,
    distance_to_lps_km: float,
    bearing_to_lps_deg: float,
    lps_influence: float,
    orographic_influence: float,
    coastal_influence: float,
    orographic_favorable: bool,
    coastal_favorable: bool,
) -> Dict[str, Any]:
    """Construct Cell-level regime output dictionary conforming to PRD 11.6."""
    return {
        "cell_id": int(cell_id),
        "lps_present": bool(lps_present),
        "distance_to_lps_km": round(float(distance_to_lps_km), 1),
        "bearing_to_lps_deg": round(float(bearing_to_lps_deg), 1),
        "lps_influence": round(float(lps_influence), 4),
        "orographic_influence": round(float(orographic_influence), 4),
        "coastal_influence": round(float(coastal_influence), 4),
        "orographic_favorable": bool(orographic_favorable),
        "coastal_favorable": bool(coastal_favorable),
    }


def assemble_14_regime_features(
    domain_regime: Dict[str, Any],
    cell_regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine domain-level phase info with cell-level LPS and local context into 14 features for B3.

    Args:
        domain_regime: Domain regime dict (from build_domain_regime_contract).
        cell_regime_df: DataFrame containing per-cell outputs:
            ['cell_id', 'lps_present', 'distance_to_lps_km', 'bearing_sin', 'bearing_cos',
             'lps_strength', 'lps_influence', 'upslope_flux', 'onshore_flux',
             'orographic_influence', 'coastal_influence']

    Returns:
        pd.DataFrame containing 'cell_id' plus the exact 14 columns of REGIME_14_FEATURES.
    """
    phase_info = domain_regime["phase"]
    p_act = phase_info["active_probability"]
    p_norm = phase_info["normal_probability"]
    p_brk = phase_info["break_probability"]
    r_conf = phase_info["regime_confidence"]

    df = cell_regime_df.copy()
    n_cells = len(df)

    df["p_active"] = np.full(n_cells, p_act, dtype=float)
    df["p_normal"] = np.full(n_cells, p_norm, dtype=float)
    df["p_break"] = np.full(n_cells, p_brk, dtype=float)
    df["regime_confidence"] = np.full(n_cells, r_conf, dtype=float)

    # Ensure all 14 columns exist
    for col in REGIME_14_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    output_cols = ["cell_id"] + REGIME_14_FEATURES
    return df[output_cols]
