"""Inference, Extrapolation Detection, and Fallback Selection (PRD §12.3, §17.1-§17.2).

Rules:
- corrected_mean_mm is the model's expected rain.
- Extrapolation flag: if a cell's rain_mm is above the 99.9th percentile
  of the training seasons (PRD §12.3), set extrapolation_flag = True.
- Trees cannot extrapolate beyond their training range.
- Fallback ladder (PRD §17.1):
    1. Normal -> serve B3, product_type="regime_aware_ml", fallback_reason=None
    2. Regime features unavailable -> serve B2, product_type="global_ml", fallback_reason="REGIME_UNAVAILABLE"
    3. ML model files unavailable -> serve B1, product_type="quantile_mapping", fallback_reason="ML_UNAVAILABLE"
    4. Run-level ood_flag == True -> serve B1, product_type="quantile_mapping", fallback_reason="OOD_INPUT"
    5. Cell extrapolation_flag == True -> serve RAW for THAT CELL, product_type="raw_nwp", fallback_reason="EXTRAPOLATION"
    6. Validation / check failure -> serve RAW, product_type="raw_nwp", fallback_reason="VALIDATION_FAILED"
    7. No correction exists -> serve RAW, product_type="raw_nwp", fallback_reason="NO_CORRECTION"
- Precedence: Cell-level EXTRAPOLATION overrides the selected corrected product for that cell.
- Fallback outputs rule (PRD §17.2): In ANY fallback product, range and probabilities must be null/empty.
"""

from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import pandas as pd

from ml.feature_contracts import B2_FEATURES, B3_FEATURES, validate_feature_columns


def compute_extrapolation_threshold(
    training_df: pd.DataFrame,
    rain_col: str = "rain_mm",
    percentile: float = 99.9,
) -> float:
    """Compute the 99.9th percentile of forecast rain_mm from training seasons only (PRD §12.3).

    Requirements:
    - Fitted strictly from training seasons only.
    - Never use validation or holdout data to fit the threshold.
    """
    if rain_col not in training_df.columns:
        raise ValueError(f"Missing required column '{rain_col}' to compute extrapolation threshold.")

    vals = training_df[rain_col].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        raise ValueError(f"No valid non-null values found in '{rain_col}'.")

    return float(np.percentile(vals, percentile))


def check_extrapolation(
    rain_mm: Union[np.ndarray, pd.Series],
    p99_9_threshold: float,
) -> np.ndarray:
    """Flag cells where forecast rain exceeds 99.9th percentile of training seasons (PRD §12.3).

    Strict inequality:
    - Values <= threshold are NOT flagged.
    - Values > threshold ARE flagged.
    """
    vals = np.asarray(rain_mm, dtype=float)
    return vals > float(p99_9_threshold)


def select_fallback_product(
    has_b3: bool = True,
    has_b2: bool = True,
    has_b1: bool = True,
    regime_available: bool = True,
    ml_available: bool = True,
    ood_flag: bool = False,
    validation_failed: bool = False,
    no_correction: bool = False,
    extrapolation_flag: bool = False,
) -> Tuple[str, Optional[str]]:
    """Determine product_type and fallback_reason per PRD §17.1 ladder.

    Precedence order:
    1. no_correction -> ("raw_nwp", "NO_CORRECTION")
    2. validation_failed -> ("raw_nwp", "VALIDATION_FAILED")
    3. cell extrapolation_flag -> ("raw_nwp", "EXTRAPOLATION")
    4. not ml_available -> ("quantile_mapping", "ML_UNAVAILABLE")
    5. ood_flag -> ("quantile_mapping", "OOD_INPUT")
    6. not regime_available -> ("global_ml", "REGIME_UNAVAILABLE")
    7. normal -> ("regime_aware_ml", None)

    Returns:
        Tuple of (product_type, fallback_reason)
    """
    if no_correction or (not has_b1 and not has_b2 and not has_b3):
        return "raw_nwp", "NO_CORRECTION"
    if validation_failed:
        return "raw_nwp", "VALIDATION_FAILED"
    if extrapolation_flag:
        return "raw_nwp", "EXTRAPOLATION"
    if not ml_available or (not has_b3 and not has_b2):
        return "quantile_mapping", "ML_UNAVAILABLE"
    if ood_flag:
        return "quantile_mapping", "OOD_INPUT"
    if not regime_available or not has_b3:
        return "global_ml", "REGIME_UNAVAILABLE"
    return "regime_aware_ml", None


def predict_b2(model: Any, df: pd.DataFrame) -> pd.Series:
    """Predict corrected rainfall using fitted B2 model.

    Requirements:
    - Input: DataFrame containing all 27 B2 features.
    - Preserves input row/index order.
    - XGBoost handles NaNs directly (no imputation).
    - Output: pd.Series of corrected_mean_mm with non-negative values (>= 0).
    """
    validate_feature_columns(df, model_type="B2")
    X = df[B2_FEATURES]
    raw_preds = model.predict(X)
    # Clip tiny numerical negatives from Tweedie regression to 0
    corrected = np.maximum(raw_preds, 0.0)
    return pd.Series(corrected, index=df.index, name="corrected_mean_mm")


def predict_b3(model: Any, df: pd.DataFrame) -> pd.Series:
    """Predict corrected rainfall using fitted B3 regime-aware model (PRD §12).

    Requirements:
    - Input: DataFrame containing all 41 B3 features.
    - For inference, regime_source may be 'final' or 'oof'.
    - Preserves input row/index order.
    - XGBoost handles NaNs directly (no imputation).
    - Output: pd.Series of corrected_mean_mm with non-negative values (>= 0).
    """
    validate_feature_columns(df, model_type="B3")
    X = df[B3_FEATURES]
    raw_preds = model.predict(X)
    # Clip tiny numerical negatives from Tweedie regression to 0
    corrected = np.maximum(raw_preds, 0.0)
    return pd.Series(corrected, index=df.index, name="corrected_mean_mm")


def apply_rainfall_correction(
    features_df: pd.DataFrame,
    model: Any,
    p99_9_threshold: float,
    model_type: str = "B3",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform model inference and apply cell-level extrapolation fallbacks (PRD §12.3, §17.1).

    Args:
        features_df: DataFrame containing features and raw rain_mm.
        model: Fitted B2 or B3 model.
        p99_9_threshold: Stored training-season 99.9th percentile.
        model_type: 'B3' (default) or 'B2'.

    Returns:
        Tuple of (corrected_mean_mm, product_type, fallback_reason, extrapolation_flag)
    """
    if "rain_mm" not in features_df.columns:
        raise ValueError("Missing required 'rain_mm' column for inference and extrapolation detection.")

    raw_mm = features_df["rain_mm"].to_numpy(dtype=float)
    extrap_flags = check_extrapolation(raw_mm, p99_9_threshold)

    if model_type == "B3":
        raw_preds = predict_b3(model, features_df).to_numpy()
        normal_product = "regime_aware_ml"
    else:
        raw_preds = predict_b2(model, features_df).to_numpy()
        normal_product = "global_ml"

    n_rows = len(features_df)
    corrected_mean = np.array(raw_preds, dtype=float, copy=True)
    product_types = np.array([normal_product] * n_rows, dtype=object)
    fallback_reasons = np.array([None] * n_rows, dtype=object)

    for i in range(n_rows):
        if extrap_flags[i]:
            corrected_mean[i] = raw_mm[i]
            product_types[i] = "raw_nwp"
            fallback_reasons[i] = "EXTRAPOLATION"

    return corrected_mean, product_types, fallback_reasons, extrap_flags
