"""SIH26080 Regime Engine package (Owner: M2).

PRD Section 11, Section 15 (F3, F4) & Appendix B.
Implements:
- Layer A: Monsoon phase labels (Rajeevan et al. 2010), A1-A6 features, and logistic regression model.
- Layer B: Rule-based Low-Pressure System detector and cell-level influence.
- Layer C: Local context (coast and mountain) orographic & coastal moisture flux influence scores.
- F3: Historical Analog Finder.
- F4: Regime Transition Detection (phase shifts and LPS lifecycle events).
- Regime Contracts (§11.6) and the 14 regime features for ML model B3 (§11.7).
"""

from regime_engine.engine import RegimeEngine
from regime_engine.contract import (
    REGIME_14_FEATURES,
    build_domain_regime_contract,
    build_cell_regime_contract,
    assemble_14_regime_features,
    validate_training_regime_source,
)
from regime_engine.labels import (
    compute_core_zone_daily_mean,
    compute_doy_climatology,
    compute_standardized_anomalies,
    detect_spells,
    check_july_august_statistics,
    compare_with_published_list,
    generate_phase_labels,
)
from regime_engine.phase import (
    extract_single_lead_a_features,
    build_phase_feature_vector,
    compute_regime_confidence,
    PhaseOODDetector,
    PhaseModel,
    select_best_c_loso,
    generate_oof_phase_predictions,
    fit_final_phase_model,
    CLASSES,
)
from regime_engine.lps import (
    haversine_distance_km,
    compute_bearing_and_components,
    LPSDetector,
    compute_lps_cell_features,
    tune_lps_detector,
)
from regime_engine.local_context import (
    compute_terrain_gradients,
    compute_raw_fluxes,
    InfluencePercentileTable,
)
from regime_engine.analogs import (
    AnalogFinder,
    ANALOG_VECTOR_KEYS,
)
from regime_engine.transitions import (
    TransitionDetector,
    PHASES,
)

__all__ = [
    "RegimeEngine",
    "REGIME_14_FEATURES",
    "build_domain_regime_contract",
    "build_cell_regime_contract",
    "assemble_14_regime_features",
    "validate_training_regime_source",
    "compute_core_zone_daily_mean",
    "compute_doy_climatology",
    "compute_standardized_anomalies",
    "detect_spells",
    "check_july_august_statistics",
    "compare_with_published_list",
    "generate_phase_labels",
    "extract_single_lead_a_features",
    "build_phase_feature_vector",
    "compute_regime_confidence",
    "PhaseOODDetector",
    "PhaseModel",
    "select_best_c_loso",
    "generate_oof_phase_predictions",
    "fit_final_phase_model",
    "CLASSES",
    "haversine_distance_km",
    "compute_bearing_and_components",
    "LPSDetector",
    "compute_lps_cell_features",
    "tune_lps_detector",
    "compute_terrain_gradients",
    "compute_raw_fluxes",
    "InfluencePercentileTable",
    "AnalogFinder",
    "ANALOG_VECTOR_KEYS",
    "TransitionDetector",
    "PHASES",
]
