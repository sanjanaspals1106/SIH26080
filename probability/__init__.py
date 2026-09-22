"""Probability and range models, calibration, constraints, and coverage (PRD §13)."""

from probability.calibration import (
    ProbabilityCalibrator,
    apply_probability_calibrators,
    apply_single_calibrator,
    get_isotonic_min_events,
)
from probability.classifiers import (
    THRESHOLDS,
    ProbabilityModelAvailability,
    RainfallProbabilityModels,
    determine_model_availability,
    enforce_probability_monotonicity,
    evaluate_probability_settings_search,
    predict_probabilities,
    train_probability_classifier,
)
from probability.constraints import (
    enforce_probability_ordering,
    enforce_probability_ordering_dict,
)
from probability.coverage import (
    LeadCoverageResult,
    check_range_coverage,
    load_coverage_parameters,
)
from probability.range_models import (
    QUANTILES,
    QuantileRangeModels,
    calculate_range_coverage,
    predict_range,
    sort_and_clip_quantiles,
    train_quantile_model,
)

__all__ = [
    # M3 probability models
    "THRESHOLDS",
    "ProbabilityModelAvailability",
    "RainfallProbabilityModels",
    "determine_model_availability",
    "enforce_probability_monotonicity",
    "evaluate_probability_settings_search",
    "predict_probabilities",
    "train_probability_classifier",

    # M4 calibration / post-processing
    "ProbabilityCalibrator",
    "apply_probability_calibrators",
    "apply_single_calibrator",
    "get_isotonic_min_events",
    "enforce_probability_ordering",
    "enforce_probability_ordering_dict",

    # M3 quantile/range models
    "QUANTILES",
    "QuantileRangeModels",
    "calculate_range_coverage",
    "predict_range",
    "sort_and_clip_quantiles",
    "train_quantile_model",

    # M4 range coverage checks
    "LeadCoverageResult",
    "check_range_coverage",
    "load_coverage_parameters",
]