"""Confidence and out-of-distribution (OOD) indicators for Layer A.

PRD Section 11.3 & Appendix B:
- regime_confidence = 1 - H / ln(3), where H is entropy of the 3 probabilities (active, normal, break).
- confidence_band: High if >= 0.50, Medium if 0.20 to < 0.50, Low if < 0.20.
- ood_flag = True if >= 2 of features A1-A6 lie outside the 1st-99th percentile range of training seasons.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_regime_confidence(
    probabilities: np.ndarray,
    high_threshold: float = 0.50,
    medium_threshold: float = 0.20,
) -> Tuple[float, str]:
    """Compute normalized Shannon entropy confidence and classification band.

    Args:
        probabilities: 1D array of 3 probabilities [p_active, p_normal, p_break].
        high_threshold: Minimum confidence for 'high' (default 0.50).
        medium_threshold: Minimum confidence for 'medium' (default 0.20).

    Returns:
        (confidence_score, confidence_band) where confidence_score in [0.0, 1.0].
    """
    probs = np.clip(probabilities, 1e-12, 1.0)
    # Normalize to ensure sum is exactly 1
    probs = probs / np.sum(probs)

    # Shannon entropy H = - sum(p * ln(p))
    H = -float(np.sum(probs * np.log(probs)))
    ln3 = np.log(3.0)

    confidence = 1.0 - (H / ln3)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence >= high_threshold:
        band = "high"
    elif confidence >= medium_threshold:
        band = "medium"
    else:
        band = "low"

    return confidence, band


class PhaseOODDetector:
    """Out-of-distribution detector for features A1-A6 based on training percentiles."""

    def __init__(
        self,
        min_features_outside: int = 2,
        percentile_range: Tuple[float, float] = (1.0, 99.0),
    ):
        self.min_features_outside = min_features_outside
        self.percentile_range = percentile_range
        self.thresholds: Dict[str, Tuple[float, float]] = {}

    def fit(self, training_a_features: Dict[str, np.ndarray]) -> "PhaseOODDetector":
        """Fit 1st and 99th percentiles on training seasons for each feature A1-A6.

        Args:
            training_a_features: Mapping 'A1'..'A6' to 1D numpy array of training values.
        """
        low_p, high_p = self.percentile_range
        for key in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            if key in training_a_features and len(training_a_features[key]) > 0:
                vals = training_a_features[key]
                p_low = float(np.percentile(vals, low_p))
                p_high = float(np.percentile(vals, high_p))
                self.thresholds[key] = (p_low, p_high)
            else:
                self.thresholds[key] = (-np.inf, np.inf)
        return self

    def predict(self, a_features: Dict[str, float]) -> bool:
        """Check if >= min_features_outside of A1-A6 fall outside training percentiles.

        Returns:
            True if OOD, False otherwise.
        """
        outside_count = 0
        for key, (p_low, p_high) in self.thresholds.items():
            val = a_features.get(key, 0.0)
            if val < p_low or val > p_high:
                outside_count += 1

        return outside_count >= self.min_features_outside
