"""Verification statistical evaluation package (PRD §16.6)."""

from verification.stats.bootstrap import (
    BrierComponents,
    BootstrapResult,
    RMSEComponents,
    paired_block_bootstrap,
)
from verification.stats.comparison_tests import (
    DieboldMarianoResult,
    SeasonConsistencyResult,
    WilcoxonResult,
    compute_diebold_mariano,
    compute_season_consistency,
    compute_season_wilcoxon,
    get_dm_lag,
)

__all__ = [
    "BrierComponents",
    "BootstrapResult",
    "RMSEComponents",
    "paired_block_bootstrap",
    "SeasonConsistencyResult",
    "WilcoxonResult",
    "DieboldMarianoResult",
    "compute_season_consistency",
    "compute_season_wilcoxon",
    "compute_diebold_mariano",
    "get_dm_lag",
]
