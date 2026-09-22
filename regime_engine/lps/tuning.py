"""Tuning module for Low-Pressure System detector.

PRD Section 11.4 & Appendix B:
Grid search over:
- zeta_min in {1.0e-5, 1.5e-5, 2.0e-5}
- dp_min_hpa in {1, 2, 3}
- sigma_deg in {1.0, 1.5}
Match criterion: distance <= 300 km + 100 km * (lead - 1).
Metric: Critical Success Index (CSI = hits / (hits + false_alarms + misses)).
Tie-break: larger zeta_min.
"""

from typing import Any, Dict, List, Optional, Tuple
import itertools
import numpy as np

from regime_engine.lps.detector import LPSDetector
from regime_engine.lps.geo_utils import haversine_distance_km


def evaluate_lps_matches(
    detections: List[Dict[str, Any]],
    catalog_systems: List[Dict[str, Any]],
    match_threshold_km: float = 300.0,
) -> Tuple[int, int, int]:
    """Evaluate hits, false alarms, and misses between detections and catalogue.

    Returns:
        (hits, false_alarms, misses)
    """
    if not catalog_systems:
        return 0, len(detections), 0
    if not detections:
        return 0, 0, len(catalog_systems)

    matched_cat = set()
    hits = 0
    false_alarms = 0

    for det in detections:
        # Find closest catalogue system
        best_dist = float("inf")
        best_cat_idx = -1
        for c_idx, cat in enumerate(catalog_systems):
            d = haversine_distance_km(det["lat"], det["lon"], cat["lat"], cat["lon"])
            if d < best_dist:
                best_dist = d
                best_cat_idx = c_idx

        if best_dist <= match_threshold_km and best_cat_idx not in matched_cat:
            hits += 1
            matched_cat.add(best_cat_idx)
        else:
            false_alarms += 1

    misses = len(catalog_systems) - len(matched_cat)
    return hits, false_alarms, misses


def tune_lps_detector(
    sample_cases: List[Dict[str, Any]],
    zeta_grid: List[float] = [1.0e-5, 1.5e-5, 2.0e-5],
    dp_grid: List[float] = [1.0, 2.0, 3.0],
    sigma_grid: List[float] = [1.0, 1.5],
) -> Dict[str, Any]:
    """Run grid search to find optimal LPS detector parameters maximizing CSI.

    Each case in sample_cases should contain:
        'vort850': 2D array
        'msl': 2D array
        'lats': 1D array
        'lons': 1D array
        'lead_day': int (1, 2, or 3)
        'catalog_systems': [{'lat': float, 'lon': float}]

    Returns:
        Dict with 'best_params', 'best_csi', 'pod', 'far', 'settings_label'.
    """
    best_csi = -1.0
    best_params = {"zeta_min": 1.5e-5, "dp_min_hpa": 2.0, "sigma_deg": 1.5}
    best_pod = 0.0
    best_far = 0.0

    combos = list(itertools.product(zeta_grid, dp_grid, sigma_grid))

    for zeta_val, dp_val, sigma_val in combos:
        detector = LPSDetector(zeta_min=zeta_val, dp_min_hpa=dp_val, sigma_deg=sigma_val)
        total_hits = 0
        total_fa = 0
        total_miss = 0

        for case in sample_cases:
            lead = case.get("lead_day", 1)
            match_dist = 300.0 + 100.0 * (lead - 1)
            dets = detector.detect(case["vort850"], case["msl"], case["lats"], case["lons"])
            hits, fa, miss = evaluate_lps_matches(dets, case["catalog_systems"], match_threshold_km=match_dist)
            total_hits += hits
            total_fa += fa
            total_miss += miss

        denom = total_hits + total_fa + total_miss
        csi = total_hits / denom if denom > 0 else 0.0

        # Tie breaking: larger zeta_min preferred
        is_better = (csi > best_csi) or (abs(csi - best_csi) < 1e-6 and zeta_val > best_params["zeta_min"])
        if is_better:
            best_csi = csi
            best_params = {"zeta_min": zeta_val, "dp_min_hpa": dp_val, "sigma_deg": sigma_val}
            best_pod = total_hits / (total_hits + total_miss) if (total_hits + total_miss) > 0 else 0.0
            best_far = total_fa / (total_hits + total_fa) if (total_hits + total_fa) > 0 else 0.0

    return {
        "best_params": best_params,
        "best_csi": best_csi,
        "pod": best_pod,
        "far": best_far,
        "settings_label": "tuned" if best_csi > 0 else "untuned",
    }
