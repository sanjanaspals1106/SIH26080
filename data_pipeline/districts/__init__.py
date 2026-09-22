"""District boundaries, area weights, district products, priority logic, and bias tables."""

from data_pipeline.districts.bias_table import (
    MIN_DATES_THRESHOLD,
    BiasRecord,
    BiasTable,
    compute_bias_statistics,
    compute_bias_table,
    compute_bias_table_entry,
    compute_district_bias,
    get_bias_table_entry,
)
from data_pipeline.districts.loader import load_districts
from data_pipeline.districts.priority import (
    DistrictForecast,
    assign_district_priorities,
    determine_attention_level,
    load_priority_thresholds,
)
from data_pipeline.districts.products import district_forecasts, district_history
from data_pipeline.districts.weights import (
    compute_district_weights,
    get_district_weights,
)

__all__ = [
    # M1 district loading / weights / products
    "compute_district_weights",
    "district_forecasts",
    "district_history",
    "get_district_weights",
    "load_districts",

    # M4 priority logic
    "DistrictForecast",
    "assign_district_priorities",
    "determine_attention_level",
    "load_priority_thresholds",

    # M4 bias-table logic
    "MIN_DATES_THRESHOLD",
    "BiasRecord",
    "BiasTable",
    "compute_bias_statistics",
    "compute_bias_table",
    "compute_bias_table_entry",
    "compute_district_bias",
    "get_bias_table_entry",
]