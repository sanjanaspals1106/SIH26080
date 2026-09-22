"""Inference and serving modules for rainfall models (PRD §12, §17, §20.1)."""

from ml.inference.predict_correction import (
    compute_extrapolation_threshold,
    check_extrapolation,
    select_fallback_product,
    predict_b2,
    predict_b3,
    apply_rainfall_correction,
)
from ml.inference.serving_grid_writer import (
    SERVING_GRID_COLUMNS,
    VALID_PRODUCT_TYPES,
    VALID_FALLBACK_REASONS,
    validate_serving_grid_schema,
    validate_serving_grid_contracts,
    assemble_serving_grid,
    write_serving_grid_file,
    read_serving_grid_file,
)

__all__ = [
    "compute_extrapolation_threshold",
    "check_extrapolation",
    "select_fallback_product",
    "predict_b2",
    "predict_b3",
    "apply_rainfall_correction",
    "SERVING_GRID_COLUMNS",
    "VALID_PRODUCT_TYPES",
    "VALID_FALLBACK_REASONS",
    "validate_serving_grid_schema",
    "validate_serving_grid_contracts",
    "assemble_serving_grid",
    "write_serving_grid_file",
    "read_serving_grid_file",
]
