"""Spatial package for geographical processing, clustering, and feature extraction (PRD §15 F6)."""

from spatial.hotspots import (
    DEFAULT_CELL_SIZE,
    DEFAULT_CORRECTED_MEAN_MIN_MM,
    DEFAULT_FALLBACK_THRESHOLD_MM,
    DEFAULT_PRIMARY_THRESHOLD_MM,
    DEFAULT_PROBABILITY_MIN,
    Hotspot,
    build_hotspot_outline,
    compute_hotspots,
    group_hotspot_cells_8conn,
    is_hotspot_cell,
    load_hotspot_thresholds,
)

__all__ = [
    "Hotspot",
    "compute_hotspots",
    "is_hotspot_cell",
    "group_hotspot_cells_8conn",
    "build_hotspot_outline",
    "load_hotspot_thresholds",
    "DEFAULT_PRIMARY_THRESHOLD_MM",
    "DEFAULT_FALLBACK_THRESHOLD_MM",
    "DEFAULT_PROBABILITY_MIN",
    "DEFAULT_CORRECTED_MEAN_MIN_MM",
    "DEFAULT_CELL_SIZE",
]
