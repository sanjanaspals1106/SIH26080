"""B1 Quantile Mapping Baseline (PRD §12.1).

Specification (PRD §12.1):
    For each lead and each region_code (§14.3), use all training cell-days to make
    the forecast-rain distribution and the observed-rain distribution.
    For a raw value x:
        p = share of training forecast values <= x (at ties, use the middle of the tie);
        corrected = the observed value at probability p.
    Use 100 probability steps with straight-line interpolation.
    Above the largest step, multiply by the ratio of the 99.5th percentiles (observed / forecast).
    Corrected values must be non-negative (>= 0).
    Models are fitted on training seasons only (never holdout or validation seasons).
"""

from typing import Dict, Tuple, List, Optional, Union, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd

REQUIRED_FIT_COLUMNS = ["lead_day", "region_code", "rain_mm", "obs_mm"]
REQUIRED_PREDICT_COLUMNS = ["lead_day", "region_code", "rain_mm"]


def compute_mid_tie_probabilities(f_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute empirical cumulative probabilities with mid-tie adjustment.

    For a raw value x:
        p = share of training forecast values <= x (at ties, use the middle of the tie).

    Formula:
        p = (count(F < x) + count(F <= x)) / (2 * N)
    """
    n = len(f_sorted)
    if n == 0:
        return np.zeros_like(x, dtype=float)

    idx_left = np.searchsorted(f_sorted, x, side="left")
    idx_right = np.searchsorted(f_sorted, x, side="right")
    return (idx_left + idx_right) / (2.0 * float(n))


class QuantileMappingModel:
    """Empirical Quantile Mapping model per (lead_day, region_code).

    Stores 100 probability steps, empirical transfer tables, and 99.5th percentile tail ratios.
    """

    def __init__(self, n_steps: int = 100):
        self.n_steps = n_steps
        # 100 probability steps spanning [0.0, 0.99] with step 99 as largest mapping step
        self.prob_steps = np.linspace(0.0, 0.99, n_steps)
        self.largest_step_p = float(self.prob_steps[-1])
        # Mapping table key: f"{lead_day}_{region_code}"
        self.mappings: Dict[str, Dict[str, Any]] = {}
        self.training_seasons: List[int] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        training_seasons: Optional[List[int]] = None,
    ) -> "QuantileMappingModel":
        """Fit empirical distributions on training cell-days per (lead_day, region_code).

        Args:
            train_df: DataFrame containing required columns.
            training_seasons: Optional list of seasons to filter by.
                              Ensures no held-out seasons are accessed.

        Returns:
            self
        """
        # Validate columns
        missing = [c for c in REQUIRED_FIT_COLUMNS if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for B1 fit: {missing}")

        data = train_df
        if training_seasons is not None:
            if "season" not in train_df.columns:
                raise ValueError("Cannot filter by training_seasons: 'season' column missing in DataFrame.")
            data = train_df[train_df["season"].isin(training_seasons)].copy()
            self.training_seasons = sorted(list(set(training_seasons)))
        elif "season" in train_df.columns:
            self.training_seasons = sorted(list(train_df["season"].unique()))

        self.mappings = {}

        # Group by (lead_day, region_code)
        grouped = data.groupby(["lead_day", "region_code"])

        for (lead, region), group in grouped:
            f_vals = group["rain_mm"].dropna().to_numpy(dtype=float)
            o_vals = group["obs_mm"].dropna().to_numpy(dtype=float)

            if len(f_vals) == 0 or len(o_vals) == 0:
                continue

            f_sorted = np.sort(f_vals)
            o_sorted = np.sort(o_vals)

            # 100 probability steps for observed rain
            obs_quantiles = np.percentile(o_sorted, 100.0 * self.prob_steps)

            # Value at the largest mapping step (p = 0.99)
            f_max_step = float(np.percentile(f_sorted, 100.0 * self.largest_step_p))

            # 99.5th percentiles for tail extrapolation
            f_p99_5 = float(np.percentile(f_sorted, 99.5))
            o_p99_5 = float(np.percentile(o_sorted, 99.5))

            if f_p99_5 > 0:
                tail_ratio = float(o_p99_5 / f_p99_5)
            else:
                tail_ratio = 1.0

            key = f"{lead}_{region}"
            self.mappings[key] = {
                "lead_day": int(lead),
                "region_code": str(region),
                "f_sorted": f_sorted.tolist(),
                "obs_quantiles": obs_quantiles.tolist(),
                "f_max_step": f_max_step,
                "tail_ratio": tail_ratio,
                "f_p99_5": f_p99_5,
                "o_p99_5": o_p99_5,
            }

        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """Apply quantile mapping to raw forecast values.

        Preserves input index and guarantees non-negative predictions.
        """
        missing = [c for c in REQUIRED_PREDICT_COLUMNS if c not in test_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for B1 predict: {missing}")

        predictions = pd.Series(index=test_df.index, dtype=float, name="corrected_mean_mm")

        # Group by (lead_day, region_code) for fast vectorized transformation
        for (lead, region), group_idx in test_df.groupby(["lead_day", "region_code"]).groups.items():
            key = f"{lead}_{region}"
            sub_rain = test_df.loc[group_idx, "rain_mm"].to_numpy(dtype=float)

            if key not in self.mappings:
                # Unseen lead/region fallback to raw rain
                predictions.loc[group_idx] = np.maximum(sub_rain, 0.0)
                continue

            mapping = self.mappings[key]
            f_sorted = np.array(mapping["f_sorted"], dtype=float)
            obs_quantiles = np.array(mapping["obs_quantiles"], dtype=float)
            f_max_step = float(mapping["f_max_step"])
            tail_ratio = float(mapping["tail_ratio"])

            # 1. Compute empirical probability with middle-of-tie
            p_vals = compute_mid_tie_probabilities(f_sorted, sub_rain)

            # 2. Check for values above the largest mapping step (p > 0.99 or raw > f_max_step)
            is_above_largest = (p_vals > self.largest_step_p) | (sub_rain > f_max_step)

            # 3. Linear interpolation along 100 probability steps
            corrected = np.interp(p_vals, self.prob_steps, obs_quantiles)

            # 4. Tail extrapolation above largest step
            corrected[is_above_largest] = sub_rain[is_above_largest] * tail_ratio

            # 5. Non-negativity constraint
            corrected = np.maximum(corrected, 0.0)

            predictions.loc[group_idx] = corrected

        return predictions

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted quantile mapping model to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        def _to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        payload = {
            "n_steps": int(self.n_steps),
            "prob_steps": [float(p) for p in self.prob_steps],
            "largest_step_p": float(self.largest_step_p),
            "training_seasons": [int(s) for s in self.training_seasons],
            "mappings": self.mappings,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_to_serializable)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "QuantileMappingModel":
        """Deserialize fitted quantile mapping model from JSON."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        instance = cls(n_steps=payload.get("n_steps", 100))
        instance.prob_steps = np.array(payload["prob_steps"], dtype=float)
        instance.largest_step_p = float(payload["largest_step_p"])
        instance.training_seasons = payload.get("training_seasons", [])
        instance.mappings = payload.get("mappings", {})
        return instance
