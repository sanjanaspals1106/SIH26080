"""Probability Calibration Integration Seam (PRD §10.4, §13.3).

Responsibilities:
1. M3 NEVER fits probability calibrators (calibration is strictly owned and fitted by M4 on OOF data).
2. Apply external pre-fitted M4 calibrators to uncalibrated probability predictions before serving.
3. Clip calibrated outputs to [0.0, 1.0].
4. Re-enforce monotonic ordering: P(>= 115.6) <= P(>= 64.5) <= P(>= 15.6).
5. Preserve NaN for unavailable thresholds.
6. Refuse to silently label uncalibrated probabilities as calibrated in final/holdout serving.
"""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

from probability.classifiers import enforce_probability_monotonicity


def apply_single_calibrator(calibrator: Any, raw_p: np.ndarray) -> np.ndarray:
    """Apply an external pre-fitted M4 calibrator to a 1D array of raw probabilities.

    Guarantees:
    - M3 NEVER calls fit() or modifies the calibrator.
    - Preserves NaN values (calibrator is applied only to finite, non-null values).
    - Supports scikit-learn CalibratedClassifierCV / LogisticRegression (predict_proba),
      IsotonicRegression (predict / transform), or callable objects.
    - Output values are strictly clipped to [0.0, 1.0].
    """
    raw_arr = np.asarray(raw_p, dtype=float)
    if raw_arr.size == 0:
        return raw_arr.copy()

    valid_mask = ~np.isnan(raw_arr)
    if not np.any(valid_mask):
        return raw_arr.copy()

    valid_inputs = raw_arr[valid_mask]

    # Evaluate calibrator
    if hasattr(calibrator, "predict_proba"):
        try:
            res = calibrator.predict_proba(valid_inputs.reshape(-1, 1))
        except Exception:
            res = calibrator.predict_proba(valid_inputs)
        if hasattr(res, "ndim") and res.ndim == 2 and res.shape[1] >= 2:
            out_vals = res[:, 1]
        else:
            out_vals = np.asarray(res).ravel()
    elif hasattr(calibrator, "predict"):
        try:
            out_vals = calibrator.predict(valid_inputs.reshape(-1, 1))
        except Exception:
            out_vals = calibrator.predict(valid_inputs)
    elif hasattr(calibrator, "transform"):
        try:
            out_vals = calibrator.transform(valid_inputs.reshape(-1, 1))
        except Exception:
            out_vals = calibrator.transform(valid_inputs)
    elif callable(calibrator):
        out_vals = calibrator(valid_inputs)
    else:
        raise TypeError(
            f"Unsupported calibrator object of type {type(calibrator)}. "
            "Expected scikit-learn compatible model or callable."
        )

    out_arr = np.clip(np.asarray(out_vals, dtype=float).ravel(), 0.0, 1.0)

    calibrated = np.array(raw_arr, dtype=float, copy=True)
    calibrated[valid_mask] = out_arr
    return calibrated


def normalize_calibrator_key(key: Any) -> Optional[float]:
    """Map flexible dictionary keys (15.6, '15.6', 'p_ge_15_6', 'uncalibrated_p_ge_15_6') to float threshold."""
    if isinstance(key, (int, float)):
        val = float(key)
        if any(np.isclose(val, t) for t in [15.6, 64.5, 115.6]):
            return val

    s = str(key).strip().lower()
    if "15" in s:
        return 15.6
    elif "64" in s:
        return 64.5
    elif "115" in s:
        return 115.6
    return None


def apply_probability_calibrators(
    raw_prob_df: pd.DataFrame,
    calibrators: Optional[Dict[Any, Any]] = None,
    allow_uncalibrated: bool = False,
) -> pd.DataFrame:
    """Transform uncalibrated probabilities using external pre-fitted M4 calibrators (PRD §13.3).

    Args:
        raw_prob_df: DataFrame containing uncalibrated probability columns
                     (either 'p_ge_*' or 'uncalibrated_p_ge_*').
        calibrators: Optional dictionary mapping threshold (15.6, 64.5, 115.6)
                     to pre-fitted M4 calibrator objects.
        allow_uncalibrated: If True, permits serving raw uncalibrated probabilities
                            (for explicit development/uncalibrated mode).
                            If False and calibrators is None/empty, raises ValueError.

    Returns:
        pd.DataFrame with columns ['p_ge_15_6', 'p_ge_64_5', 'p_ge_115_6'],
        monotonically ordered and clipped in [0.0, 1.0].
    """
    has_calibrators = calibrators is not None and len(calibrators) > 0

    if not has_calibrators:
        if not allow_uncalibrated:
            raise ValueError(
                "Final/holdout serving requires pre-fitted M4 probability calibrators. "
                "Cannot silently expose uncalibrated probabilities as calibrated. "
                "Supply 'calibrators' dict or set allow_uncalibrated=True for explicit development mode."
            )
        # In explicit development uncalibrated mode, return formatted raw probabilities
        out_df = pd.DataFrame(index=raw_prob_df.index)
        for t in [15.6, 64.5, 115.6]:
            col = f"p_ge_{str(t).replace('.', '_')}"
            raw_col = f"uncalibrated_{col}"
            if col in raw_prob_df.columns:
                out_df[col] = raw_prob_df[col].values
            elif raw_col in raw_prob_df.columns:
                out_df[col] = raw_prob_df[raw_col].values
            else:
                out_df[col] = np.nan
        return out_df

    # Normalize calibrator dictionary
    norm_calibrators: Dict[float, Any] = {}
    for k, v in calibrators.items():
        thresh = normalize_calibrator_key(k)
        if thresh is not None and v is not None:
            norm_calibrators[thresh] = v

    n_rows = len(raw_prob_df)
    calibrated_cols: Dict[float, np.ndarray] = {}

    for t in [15.6, 64.5, 115.6]:
        col = f"p_ge_{str(t).replace('.', '_')}"
        raw_col = f"uncalibrated_{col}"

        # Locate raw column
        source_col = None
        if col in raw_prob_df.columns:
            source_col = col
        elif raw_col in raw_prob_df.columns:
            source_col = raw_col

        if source_col is not None and not raw_prob_df[source_col].isna().all():
            raw_vals = raw_prob_df[source_col].to_numpy(dtype=float)
            if t in norm_calibrators:
                # Apply external pre-fitted calibrator
                calibrated_vals = apply_single_calibrator(norm_calibrators[t], raw_vals)
            else:
                # Keep raw if no specific calibrator for this threshold
                calibrated_vals = np.clip(raw_vals, 0.0, 1.0)
            calibrated_cols[t] = calibrated_vals
        else:
            # Unavailable threshold remains strictly NaN
            calibrated_cols[t] = np.full(n_rows, np.nan, dtype=float)

    # Re-enforce monotonicity after calibration: P(>= 115.6) <= P(>= 64.5) <= P(>= 15.6)
    p15 = calibrated_cols[15.6]
    p64 = calibrated_cols[64.5] if not np.all(np.isnan(calibrated_cols[64.5])) else None
    p115 = calibrated_cols[115.6] if not np.all(np.isnan(calibrated_cols[115.6])) else None

    p15_corr, p64_corr, p115_corr = enforce_probability_monotonicity(p15, p64, p115)

    return pd.DataFrame(
        {
            "p_ge_15_6": p15_corr,
            "p_ge_64_5": p64_corr if p64 is not None else np.nan,
            "p_ge_115_6": p115_corr if p115 is not None else np.nan,
        },
        index=raw_prob_df.index,
    )
