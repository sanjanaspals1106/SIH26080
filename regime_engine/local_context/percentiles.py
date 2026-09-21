"""Percentile ranking and influence scores for Layer C (coast and mountains).

PRD Section 11.5 & Appendix B:
- orographic_influence = 0 if upslope_flux == 0, else percentile rank among positive training values in (0, 1].
- coastal_influence = 0 if onshore_flux == 0, else percentile rank among positive training values in (0, 1].
- Display flags: orographic_favorable = (orographic_influence >= 0.90),
                 coastal_favorable = (coastal_influence >= 0.90).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class InfluencePercentileTable:
    """Percentile calibration table fitted strictly on positive training values."""

    def __init__(
        self,
        favorable_threshold: float = 0.90,
    ):
        self.favorable_threshold = favorable_threshold
        self.positive_upslope_sorted: np.ndarray = np.array([])
        self.positive_onshore_sorted: np.ndarray = np.array([])
        self.is_fitted = False

    def fit(
        self,
        training_upslope_flux: np.ndarray,
        training_onshore_flux: np.ndarray,
    ) -> "InfluencePercentileTable":
        """Fit empirical percentile distributions on positive training flux values."""
        pos_up = training_upslope_flux[training_upslope_flux > 1e-9]
        pos_on = training_onshore_flux[training_onshore_flux > 1e-9]

        if len(pos_up) > 0:
            self.positive_upslope_sorted = np.sort(pos_up.astype(float))
        else:
            self.positive_upslope_sorted = np.array([1.0])

        if len(pos_on) > 0:
            self.positive_onshore_sorted = np.sort(pos_on.astype(float))
        else:
            self.positive_onshore_sorted = np.array([1.0])

        self.is_fitted = True
        return self

    def compute_influence(
        self,
        upslope_flux: Union[float, np.ndarray],
        onshore_flux: Union[float, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute influence scores in [0, 1] and boolean favorable flags.

        Returns:
            (orographic_influence, coastal_influence, orographic_favorable, coastal_favorable)
        """
        if not self.is_fitted:
            # If not fitted yet, use simple max-based normalization
            up_arr = np.asarray(upslope_flux, dtype=float)
            on_arr = np.asarray(onshore_flux, dtype=float)
            orog_inf = np.where(up_arr > 0, np.clip(up_arr / 10.0, 0.01, 1.0), 0.0)
            coast_inf = np.where(on_arr > 0, np.clip(on_arr / 10.0, 0.01, 1.0), 0.0)
            return (
                orog_inf,
                coast_inf,
                orog_inf >= self.favorable_threshold,
                coast_inf >= self.favorable_threshold,
            )

        up_arr = np.asarray(upslope_flux, dtype=float)
        on_arr = np.asarray(onshore_flux, dtype=float)

        # 1. Orographic influence
        orog_inf = np.zeros_like(up_arr, dtype=float)
        pos_up_mask = up_arr > 1e-9
        if np.any(pos_up_mask):
            ranks = np.searchsorted(self.positive_upslope_sorted, up_arr[pos_up_mask], side="right")
            orog_inf[pos_up_mask] = np.clip(ranks / len(self.positive_upslope_sorted), 0.001, 1.0)

        # 2. Coastal influence
        coast_inf = np.zeros_like(on_arr, dtype=float)
        pos_on_mask = on_arr > 1e-9
        if np.any(pos_on_mask):
            ranks = np.searchsorted(self.positive_onshore_sorted, on_arr[pos_on_mask], side="right")
            coast_inf[pos_on_mask] = np.clip(ranks / len(self.positive_onshore_sorted), 0.001, 1.0)

        orog_fav = orog_inf >= self.favorable_threshold
        coast_fav = coast_inf >= self.favorable_threshold

        return orog_inf, coast_inf, orog_fav, coast_fav

    def to_dict(self) -> Dict[str, Any]:
        """Serialize percentile distributions to dict for storage."""
        # Store representative quantiles (0 to 100 in steps of 1)
        quantiles = np.linspace(0, 100, 101)
        return {
            "favorable_threshold": self.favorable_threshold,
            "upslope_quantiles": np.percentile(self.positive_upslope_sorted, quantiles).tolist(),
            "onshore_quantiles": np.percentile(self.positive_onshore_sorted, quantiles).tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfluencePercentileTable":
        """Deserialize from dict."""
        table = cls(favorable_threshold=data.get("favorable_threshold", 0.90))
        table.positive_upslope_sorted = np.array(data["upslope_quantiles"], dtype=float)
        table.positive_onshore_sorted = np.array(data["onshore_quantiles"], dtype=float)
        table.is_fitted = True
        return table
