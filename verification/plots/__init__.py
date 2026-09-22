"""PRD §16.9 standard verification plot and data preparation.

Provides:
- Pure data preparation functions (independent of matplotlib).
- Optional matplotlib figure rendering wrappers.
"""

from verification.plots.data import (
    B0ToB3Row,
    CellImprovementPoint,
    FSSNeighbourhoodPoint,
    PerformanceDiagramPoint,
    RawRainBiasPoint,
    ReliabilityBinPoint,
    SeasonImprovementDot,
    prepare_average_improvement_map_data,
    prepare_b0_to_b3_table_data,
    prepare_fss_neighbourhood_data,
    prepare_per_season_improvement_data,
    prepare_performance_diagram_data,
    prepare_raw_rain_bias_data,
    prepare_reliability_diagram_data,
)
from verification.plots.matplotlib import (
    plot_average_improvement_map,
    plot_b0_to_b3_comparison,
    plot_fss_vs_neighbourhood,
    plot_per_season_improvement_dots,
    plot_performance_diagram,
    plot_raw_rain_bias,
    plot_reliability_diagram,
)

__all__ = [
    "PerformanceDiagramPoint",
    "FSSNeighbourhoodPoint",
    "ReliabilityBinPoint",
    "RawRainBiasPoint",
    "CellImprovementPoint",
    "SeasonImprovementDot",
    "B0ToB3Row",
    "prepare_performance_diagram_data",
    "prepare_fss_neighbourhood_data",
    "prepare_reliability_diagram_data",
    "prepare_raw_rain_bias_data",
    "prepare_average_improvement_map_data",
    "prepare_per_season_improvement_data",
    "prepare_b0_to_b3_table_data",
    "plot_performance_diagram",
    "plot_fss_vs_neighbourhood",
    "plot_reliability_diagram",
    "plot_raw_rain_bias",
    "plot_average_improvement_map",
    "plot_per_season_improvement_dots",
    "plot_b0_to_b3_comparison",
]
