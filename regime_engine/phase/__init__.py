"""Layer A monsoon phase classification package.

PRD Section 11.3 & Appendix B.
Features A1-A6, multinomial logistic regression, confidence and out-of-fold prediction.
"""

from regime_engine.phase.features import (
    extract_single_lead_a_features,
    build_phase_feature_vector,
    compute_season_position,
    compute_box_mean,
    compute_trough_position,
)
from regime_engine.phase.confidence import (
    compute_regime_confidence,
    PhaseOODDetector,
)
from regime_engine.phase.model import (
    PhaseModel,
    select_best_c_loso,
    CLASSES,
)
from regime_engine.phase.oof import (
    generate_oof_phase_predictions,
    fit_final_phase_model,
)

__all__ = [
    "extract_single_lead_a_features",
    "build_phase_feature_vector",
    "compute_season_position",
    "compute_box_mean",
    "compute_trough_position",
    "compute_regime_confidence",
    "PhaseOODDetector",
    "PhaseModel",
    "select_best_c_loso",
    "generate_oof_phase_predictions",
    "fit_final_phase_model",
    "CLASSES",
]
