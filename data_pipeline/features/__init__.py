"""Stage 3: feature library, static geography and climatology (PRD 8.6, 9.4 to 9.6). Owner: M1.

`build_features` (library.py) is the one feature function; training and serving both call it.
"""

from data_pipeline.features.climatology import (
    assert_no_holdout,
    compute_climatology,
    get_climatology,
)
from data_pipeline.features.library import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    FEATURE_VERSION,
    TABLE_COLUMNS,
    TARGET_COLUMNS,
    build_features,
    feature_matrix,
)
from data_pipeline.features.pipeline import (
    build_stage3_season,
    read_district_forecasts,
    read_district_history,
    read_features,
)
from data_pipeline.features.static import compute_static_geography, get_static_geography

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_SCHEMA",
    "FEATURE_VERSION",
    "TABLE_COLUMNS",
    "TARGET_COLUMNS",
    "assert_no_holdout",
    "build_features",
    "build_stage3_season",
    "compute_climatology",
    "compute_static_geography",
    "feature_matrix",
    "get_climatology",
    "get_static_geography",
    "read_district_forecasts",
    "read_district_history",
    "read_features",
]
