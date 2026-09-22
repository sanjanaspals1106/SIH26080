"""Probability ordering constraints per PRD §13.3.

Rule:
After prediction and calibration, enforce for every cell:
P(>=115.6) <= P(>=64.5) <= P(>=15.6)
Probabilities are kept clipped to [0.0, 1.0].
Null / unavailable thresholds are strictly preserved without inventing probabilities.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


def enforce_probability_ordering(
    p_15_6: Optional[Union[float, np.ndarray]],
    p_64_5: Optional[Union[float, np.ndarray]] = None,
    p_115_6: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[
    Optional[Union[float, np.ndarray]],
    Optional[Union[float, np.ndarray]],
    Optional[Union[float, np.ndarray]],
]:
    """Enforces monotonicity: P(>=115.6) <= P(>=64.5) <= P(>=15.6).

    Args:
        p_15_6: Moderate rainfall probability (scalar or array).
        p_64_5: Heavy rainfall probability (scalar or array, or None if unavailable).
        p_115_6: Very-heavy rainfall probability (scalar or array, or None if unavailable).

    Returns:
        Tuple of (p_15_6, p_64_5, p_115_6) with ordering enforced and values in [0, 1].
        Unavailable (None/NaN) thresholds are preserved as None/NaN.
    """
    is_array = (
        isinstance(p_15_6, np.ndarray)
        or isinstance(p_64_5, np.ndarray)
        or isinstance(p_115_6, np.ndarray)
    )

    if is_array:
        p15 = None if p_15_6 is None else np.asarray(p_15_6, dtype=float).copy()
        p64 = None if p_64_5 is None else np.asarray(p_64_5, dtype=float).copy()
        p115 = None if p_115_6 is None else np.asarray(p_115_6, dtype=float).copy()

        # Clip valid non-NaN values to [0.0, 1.0]
        if p15 is not None:
            mask15 = ~np.isnan(p15)
            p15[mask15] = np.clip(p15[mask15], 0.0, 1.0)
        if p64 is not None:
            mask64 = ~np.isnan(p64)
            p64[mask64] = np.clip(p64[mask64], 0.0, 1.0)
        if p115 is not None:
            mask115 = ~np.isnan(p115)
            p115[mask115] = np.clip(p115[mask115], 0.0, 1.0)

        # Enforce p_64_5 <= p_15_6
        if p15 is not None and p64 is not None:
            valid_pair = (~np.isnan(p15)) & (~np.isnan(p64))
            p64[valid_pair] = np.minimum(p64[valid_pair], p15[valid_pair])

        # Enforce p_115_6 <= p_64_5
        if p64 is not None and p115 is not None:
            valid_pair = (~np.isnan(p64)) & (~np.isnan(p115))
            p115[valid_pair] = np.minimum(p115[valid_pair], p64[valid_pair])
        elif p15 is not None and p115 is not None:
            valid_pair = (~np.isnan(p15)) & (~np.isnan(p115))
            p115[valid_pair] = np.minimum(p115[valid_pair], p15[valid_pair])

        return p15, p64, p115

    # Scalar handling
    p15 = None if p_15_6 is None else float(np.clip(p_15_6, 0.0, 1.0))
    p64 = None if p_64_5 is None else float(np.clip(p_64_5, 0.0, 1.0))
    p115 = None if p_115_6 is None else float(np.clip(p_115_6, 0.0, 1.0))

    if p15 is not None and p64 is not None:
        p64 = min(p64, p15)

    if p64 is not None and p115 is not None:
        p115 = min(p115, p64)
    elif p15 is not None and p115 is not None:
        p115 = min(p115, p15)

    return p15, p64, p115


def enforce_probability_ordering_dict(
    probabilities: Dict[str, Optional[Union[float, np.ndarray]]],
) -> Dict[str, Optional[Union[float, np.ndarray]]]:
    """Helper to enforce ordering on dictionary representations.

    Keys supported:
    - 'p_ge_15_6' / 'p_15_6'
    - 'p_ge_64_5' / 'p_64_5'
    - 'p_ge_115_6' / 'p_115_6'
    """
    key_15 = "p_ge_15_6" if "p_ge_15_6" in probabilities else "p_15_6"
    key_64 = "p_ge_64_5" if "p_ge_64_5" in probabilities else "p_64_5"
    key_115 = "p_ge_115_6" if "p_ge_115_6" in probabilities else "p_115_6"

    val_15 = probabilities.get(key_15)
    val_64 = probabilities.get(key_64)
    val_115 = probabilities.get(key_115)

    out_15, out_64, out_115 = enforce_probability_ordering(val_15, val_64, val_115)

    res = dict(probabilities)
    if key_15 in res:
        res[key_15] = out_15
    if key_64 in res:
        res[key_64] = out_64
    if key_115 in res:
        res[key_115] = out_115

    return res
