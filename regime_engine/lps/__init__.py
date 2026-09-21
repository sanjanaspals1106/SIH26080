"""Layer B Low-Pressure System (LPS) package.

PRD Section 11.4 & Appendix B.
Rule-based detector finding low-pressure systems and computing cell-level influence scores.
"""

from regime_engine.lps.geo_utils import (
    haversine_distance_km,
    compute_bearing_and_components,
)
from regime_engine.lps.detector import LPSDetector
from regime_engine.lps.cell_features import compute_lps_cell_features
from regime_engine.lps.tuning import tune_lps_detector

__all__ = [
    "haversine_distance_km",
    "compute_bearing_and_components",
    "LPSDetector",
    "compute_lps_cell_features",
    "tune_lps_detector",
]
