"""B0 Raw NWP Baseline (PRD §12.1).

Definition:
    B0: Raw NWP - The aligned window rain (§8.3). No change.

Rules:
    - Input: rain_mm
    - Output: prediction equal to rain_mm exactly
    - Preserves row/index alignment
    - Validates required column exists
    - No training/model fitting required
"""

from typing import Union
import numpy as np
import pandas as pd


def predict_b0(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    rain_col: str = "rain_mm",
) -> Union[pd.Series, np.ndarray]:
    """Pass-through prediction for raw NWP forecast without modification.

    Args:
        data: DataFrame containing 'rain_mm', or Series / ndarray of forecast rainfall.
        rain_col: Name of raw forecast column if data is a DataFrame (default: 'rain_mm').

    Returns:
        Exact copy of raw rainfall values, preserving row and index alignment.

    Raises:
        ValueError: If data is a DataFrame and required rain_col is missing.
        TypeError: If data is not a DataFrame, Series, or ndarray.
    """
    if isinstance(data, pd.DataFrame):
        if rain_col not in data.columns:
            raise ValueError(
                f"Missing required column '{rain_col}' for B0 raw baseline. "
                f"Available columns: {list(data.columns)}"
            )
        return data[rain_col].copy()
    elif isinstance(data, pd.Series):
        return data.copy()
    elif isinstance(data, np.ndarray):
        return np.copy(data)
    else:
        raise TypeError(
            f"Unsupported input type for B0 prediction: {type(data)}. "
            "Expected pandas DataFrame, Series, or numpy ndarray."
        )


class B0RawBaseline:
    """Class wrapper for B0 Raw NWP Baseline."""

    def __init__(self, rain_col: str = "rain_mm"):
        self.rain_col = rain_col

    def fit(self, *args, **kwargs) -> "B0RawBaseline":
        """No-op fit method (B0 requires no fitting)."""
        return self

    def predict(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Generate B0 raw predictions."""
        return predict_b0(data, rain_col=self.rain_col)
