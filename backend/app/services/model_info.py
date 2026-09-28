"""Read-only content for `/model-info` (PRD 17-20). Model facts come from the small `metadata.json` files saved
next to the trained models (never the model files themselves) and from the evaluation protocol in `m3_dev` /
`m3_holdout` metrics; the rest is fixed project scope. Nothing is trained, loaded or written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.services import verification as verification_service

# What the system does not do or claim. Written once, stated plainly; shown on the Model Info page.
LIMITATIONS = [
    "Replay mode only: forecasts are past ECMWF TIGGE control runs (June to September, 2021 to 2025) replayed against IMD "
    "observations. This is not a live forecasting service.",
    "Spatial resolution is the 0.25 degree IMD grid (about 28 km). Each value is a cell or district average, not a point "
    "forecast, and heavy-rain location is only verified to a neighbourhood of about 140 km (FSS, 5x5 cells).",
    "District values use 2011 census boundaries (641 districts). Districts created after 2011 are not included.",
    "Heavy-rain caveat: on the 2025 holdout the mean-corrected forecast issues far fewer 64.5 mm exceedances than were "
    "observed, and its threat score at 64.5 mm is below raw NWP. Use the calibrated heavy-rain probabilities and the "
    "q10-q90 range for heavy-rain guidance.",
    "Model settings are XGBoost defaults; the hyperparameter search specified in the plan has not been run.",
    "Verification values are point estimates pooled over lead days 1 to 3 and all valid cells; confidence intervals and "
    "significance tests are not part of the current report.",
    "Attention levels and hotspots are model-based guidance derived from calibrated probabilities. They are not official "
    "warnings.",
    "Only one NWP source (ECMWF control member) and one truth dataset (IMD gridded rainfall) are used; there is no "
    "ensemble spread.",
]


def _meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _model(role: str, algorithm: str, feature_set: str, n_features: int, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role, "version": "v1", "algorithm": algorithm,
        "feature_set_version": feature_set, "n_features": n_features,
        "training_seasons": meta.get("train_seasons"),
    }  # fmt: skip


def get_model_info(
    data_dir: Path, *, alignment_method: str, lead_days: list[int], district_source: str, census_basis: int
) -> dict[str, Any]:  # fmt: skip
    final = data_dir / "models" / "m3" / "final"
    b2, b3 = _meta(final / "b2" / "metadata.json"), _meta(final / "b3" / "metadata.json")
    prob, rng = _meta(final / "probability" / "metadata.json"), _meta(final / "range" / "metadata.json")
    n = {k: len(v.get("feature_names", [])) for k, v in (("b2", b2), ("b3", b3), ("prob", prob), ("rng", rng))}
    models = [
        _model("B1: Quantile Mapping", "Empirical quantile calibration with tail ratios", "raw rainfall only", 1, {}),
        _model("B2: Global ML", "XGBoost (Tweedie objective)", "static and dynamic features", n["b2"], b2),
        _model("B3: Regime-Aware ML", "XGBoost (Tweedie objective)", "static, dynamic and regime features", n["b3"], b3),
        _model("Heavy-Rain Probability", "XGBoost classifiers at 15.6, 64.5, 115.6 mm, calibrated", "regime-aware set", n["prob"], prob),
        _model("Rainfall Range (q10 / q50 / q90)", "XGBoost quantile regressors", "regime-aware set", n["rng"], rng),
    ]  # fmt: skip

    try:
        dev, holdout = verification_service._files(data_dir / "verification")
        dev_seasons = [int(s) for s in dev["development_seasons"]]
        holdout_seasons = [int(s) for s in dev["holdout_seasons"]]
        locked = holdout is not None
    except Exception:  # metrics not generated: still describe the protocol from the model metadata
        dev_seasons = [int(s) for s in b3.get("train_seasons", [])]
        holdout_seasons, locked = [], False

    return {
        "models": models,
        "data": {
            "nwp": "ECMWF TIGGE control forecast", "truth": "IMD 0.25 degree gridded rainfall",
            "alignment_method": alignment_method, "lead_days": lead_days,
            "season_scope": "Monsoon (June to September), 2021 to 2025, 122 forecast runs per season",
            "district_file": f"{district_source} ({census_basis} census basis)",
        },
        "protocol": {
            "n_seasons": len(dev_seasons) + len(holdout_seasons), "development_seasons": dev_seasons,
            "holdout_seasons": holdout_seasons, "holdout_locked": locked,
            "status": "LOCKED EVALUATION" if locked else "DEVELOPMENT ONLY",
        },
        "limitations": LIMITATIONS,
        "fallback_share_recent": None,
    }  # fmt: skip
