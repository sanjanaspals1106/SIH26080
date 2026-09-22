"""Layer C local context (coast and mountains) package.

PRD Section 11.5 & Appendix B.
Computes raw upslope and onshore fluxes, and calibrated influence scores.
"""

from regime_engine.local_context.indices import (
    compute_terrain_gradients,
    compute_raw_fluxes,
)
from regime_engine.local_context.percentiles import (
    InfluencePercentileTable,
)

__all__ = [
    "compute_terrain_gradients",
    "compute_raw_fluxes",
    "InfluencePercentileTable",
]
