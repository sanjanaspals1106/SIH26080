"""M3 Orchestration and Integration Pipeline (PRD §10, §12, §13, §17, §20.1, §21.1).

Responsibilities:
1. Validate upstream inputs from M1 (features, cells, leads) and M2 (regimes, sources).
2. Development out-of-fold (OOF) orchestration across development seasons (LOSO).
3. Production of OOF evaluation table for M4 (uncalibrated, prediction_source='oof').
4. Final M3 model training on all development seasons and persistence (model_store).
5. Inference runner on new/holdout runs producing the 18-column serving grid Parquet.
"""

from typing import Dict, Any, List, Optional, Union, Sequence, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import xgboost as xgb

from ml.feature_contracts import (
    BASE_FEATURES,
    REGIME_FEATURES,
    B2_FEATURES,
    B3_FEATURES,
    validate_feature_columns,
)
from ml.baselines.b0_raw import predict_b0
from ml.baselines.b1_quantile_mapping import QuantileMappingModel
from ml.training.train_correction import (
    train_b2_model,
    train_b3_model,
    fit_final_b2_model,
    fit_final_b3_model,
    validate_b3_training_data,
)
from ml.inference.predict_correction import (
    predict_b2,
    predict_b3,
    compute_extrapolation_threshold,
    check_extrapolation,
    select_fallback_product,
)
from ml.inference.serving_grid_writer import (
    SERVING_GRID_COLUMNS,
    assemble_serving_grid,
    write_serving_grid_file,
)
from ml.models.model_store import (
    ModelMetadata,
    save_model,
    load_model,
    save_range_models,
    load_range_models,
    save_probability_models,
    load_probability_models,
)
from probability.classifiers import (
    RainfallProbabilityModels,
    train_probability_classifier,
)
from probability.range_models import (
    QuantileRangeModels,
    train_quantile_model,
)
from probability.calibration import apply_probability_calibrators

REQUIRED_M1_COLUMNS = [
    "run_id",
    "season",
    "lead_day",
    "cell_id",
    "region_code",
    "rain_mm",
]


@dataclass
class FinalM3Models:
    """Container for all fitted final M3 models and parameters."""
    b1_model: QuantileMappingModel
    b2_model: xgb.XGBRegressor
    b3_model: xgb.XGBRegressor
    prob_models: RainfallProbabilityModels
    range_models: QuantileRangeModels
    p99_9_threshold: float
    training_seasons: List[int]
    feature_set_version: str = "v1"
    git_commit: str = "test"
    model_version_id: Optional[int] = None
    dev_scores: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save all models and metadata to directory."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save B1
        self.b1_model.save(out_dir / "b1" / "b1_model.json")

        # 2. Save B2
        meta_b2 = ModelMetadata(
            model_type="b2",
            feature_names=B2_FEATURES,
            train_seasons=self.training_seasons,
            git_commit=self.git_commit,
            p99_9_threshold=self.p99_9_threshold,
            model_version_id=self.model_version_id,
            metrics=self.dev_scores.get("b2", {}),
        )
        save_model(self.b2_model, meta_b2, out_dir / "b2")

        # 3. Save B3
        meta_b3 = ModelMetadata(
            model_type="b3",
            feature_names=B3_FEATURES,
            train_seasons=self.training_seasons,
            git_commit=self.git_commit,
            p99_9_threshold=self.p99_9_threshold,
            model_version_id=self.model_version_id,
            metrics=self.dev_scores.get("b3", {}),
        )
        save_model(self.b3_model, meta_b3, out_dir / "b3")

        # 4. Save Probability Models
        save_probability_models(
            self.prob_models,
            out_dir / "probability",
            metadata={
                "train_seasons": self.training_seasons,
                "git_commit": self.git_commit,
                "metrics": self.dev_scores.get("probability", {}),
            },
        )

        # 5. Save Range Models
        save_range_models(
            self.range_models,
            out_dir / "range",
            metadata={
                "train_seasons": self.training_seasons,
                "git_commit": self.git_commit,
                "metrics": self.dev_scores.get("range", {}),
            },
        )

        return out_dir

    @classmethod
    def load(cls, input_dir: Union[str, Path]) -> "FinalM3Models":
        """Load all models and metadata from directory."""
        in_dir = Path(input_dir)

        b1_file = in_dir / "b1" / "b1_model.json"
        b1_model = QuantileMappingModel.load(b1_file)

        b2_model, meta_b2 = load_model(in_dir / "b2")
        b3_model, meta_b3 = load_model(in_dir / "b3")

        prob_models = load_probability_models(in_dir / "probability")
        range_models = load_range_models(in_dir / "range")

        p99_9 = float(meta_b3.get("p99_9_threshold") or meta_b2.get("p99_9_threshold") or 150.0)
        train_seasons = meta_b3.get("train_seasons", [])
        m_v_id = meta_b3.get("model_version_id")

        return cls(
            b1_model=b1_model,
            b2_model=b2_model,
            b3_model=b3_model,
            prob_models=prob_models,
            range_models=range_models,
            p99_9_threshold=p99_9,
            training_seasons=train_seasons,
            model_version_id=m_v_id,
        )


def validate_m1_m2_inputs(
    df: pd.DataFrame,
    is_training: bool = False,
    require_regime: bool = True,
) -> Dict[str, Any]:
    """Validate upstream input contract from M1 and M2.

    Returns contract status dict:
    - regime_available: bool
    - ood_flag: bool
    - feature_set_version: str
    """
    # 1. Validate M1 required columns
    missing_m1 = [col for col in REQUIRED_M1_COLUMNS if col not in df.columns]
    if missing_m1:
        raise ValueError(f"Missing required M1 columns in input: {missing_m1}")

    # Validate 27 base features
    validate_feature_columns(df, model_type="B2")

    if is_training and "obs_mm" not in df.columns:
        raise ValueError("Missing required target column 'obs_mm' for training.")

    # 2. Validate M2 inputs
    regime_available = True
    missing_regime = [col for col in REGIME_FEATURES if col not in df.columns]
    if missing_regime:
        regime_available = False

    if "regime_available" in df.columns and not df["regime_available"].all():
        regime_available = False

    if "regime_source" not in df.columns:
        regime_available = False

    ood_flag = False
    if "ood_flag" in df.columns:
        ood_flag = bool(df["ood_flag"].any())

    # If training B3 and regime is required, enforce strict regime_source='oof'
    if is_training and require_regime:
        if not regime_available:
            raise ValueError("Training B3 models strictly requires valid regime features and regime_source.")
        validate_b3_training_data(df)

    feature_set_version = str(df["feature_set_version"].iloc[0]) if "feature_set_version" in df.columns else "v1"

    return {
        "regime_available": regime_available,
        "ood_flag": ood_flag,
        "feature_set_version": feature_set_version,
    }


def run_development_oof_pipeline(
    dev_df: pd.DataFrame,
    development_seasons: Optional[Sequence[int]] = None,
    b2_params: Optional[Dict[str, Any]] = None,
    b3_params: Optional[Dict[str, Any]] = None,
    prob_params: Optional[Dict[str, Any]] = None,
    range_params: Optional[Dict[str, Any]] = None,
    event_counts: Optional[Dict[float, int]] = None,
    train_cell_stride: int = 1,
) -> pd.DataFrame:
    """Run Leave-One-Season-Out (LOSO) cross-validation on development seasons (PRD §10.3).

    For each held-out development season h:
    - Train on all development seasons except h.
    - Evaluate on h.
    - Strictly no holdout leakage.
    - Uses B0, B1, B2, B3, probability, and range models.
    - Does NOT calibrate probabilities here (calibration is owned by M4).

    Returns:
        OOF prediction DataFrame formatted for M4 handoff.
    """
    # 1. Validate inputs
    validate_m1_m2_inputs(dev_df, is_training=True, require_regime=True)

    if development_seasons is None:
        if "season" not in dev_df.columns:
            raise ValueError("Missing 'season' column in development dataframe.")
        development_seasons = sorted(dev_df["season"].unique())

    if len(development_seasons) < 2:
        raise ValueError(f"Need at least 2 development seasons for OOF evaluation, got {len(development_seasons)}.")

    # Default fast parameters if not provided
    cfg_b2 = b2_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    cfg_b3 = b3_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    cfg_prob = prob_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    cfg_range = range_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    counts = event_counts or {15.6: 100, 64.5: 50, 115.6: 35}

    oof_records = []

    for h in development_seasons:
        train_fold = dev_df[dev_df["season"] != h].copy()
        eval_fold = dev_df[dev_df["season"] == h].copy()

        if len(eval_fold) == 0 or len(train_fold) == 0:
            continue

        # Verify OOF isolation: h never in train_fold
        assert h not in train_fold["season"].values, f"Season {h} leaked into training fold!"

        # 1. B0 Raw
        b0_raw = predict_b0(eval_fold)

        # 2. B1 Quantile Mapping (fitted strictly on training seasons)
        b1_model = QuantileMappingModel().fit(train_fold)
        b1_preds = b1_model.predict(eval_fold)

        # 3. B2 Tweedie (27 base features)
        b2_model = train_b2_model(train_fold, params=cfg_b2, train_cell_stride=train_cell_stride)
        b2_preds = predict_b2(b2_model, eval_fold)

        # 4. B3 Tweedie (41 regime-aware features, requires regime_source='oof')
        validate_b3_training_data(train_fold)
        b3_model = train_b3_model(train_fold, params=cfg_b3, train_cell_stride=train_cell_stride)
        b3_preds = predict_b3(b3_model, eval_fold)

        # 5. Heavy-rain probabilities (uncalibrated)
        prob_models = RainfallProbabilityModels().fit(
            train_fold,
            event_counts=counts,
            params_map=cfg_prob,
            train_cell_stride=train_cell_stride,
        )
        prob_preds_df = prob_models.predict_probabilities(eval_fold)

        # 6. Quantile range models (q10, q50, q90)
        range_models = QuantileRangeModels().fit(
            train_fold,
            params_map=cfg_range,
            train_cell_stride=train_cell_stride,
        )
        range_preds_df = range_models.predict_range(eval_fold)

        # Assemble fold records for M4
        fold_out = pd.DataFrame({
            "run_id": eval_fold["run_id"].values,
            "season": eval_fold["season"].values,
            "lead_day": eval_fold["lead_day"].values,
            "cell_id": eval_fold["cell_id"].values,
            "obs_mm": eval_fold["obs_mm"].values,
            "raw_mm": b0_raw.values,
            "b1_corrected_mm": b1_preds.values,
            "b2_corrected_mean_mm": b2_preds.values,
            "b3_corrected_mean_mm": b3_preds.values,
            "uncalibrated_p_ge_15_6": prob_preds_df["p_ge_15_6"].values,
            "uncalibrated_p_ge_64_5": prob_preds_df["p_ge_64_5"].values,
            "uncalibrated_p_ge_115_6": prob_preds_df["p_ge_115_6"].values,
            "q10_mm": range_preds_df["q10_mm"].values,
            "q50_mm": range_preds_df["q50_mm"].values,
            "q90_mm": range_preds_df["q90_mm"].values,
            "prediction_source": "oof",
        }, index=eval_fold.index)

        oof_records.append(fold_out)

    if not oof_records:
        raise RuntimeError("No OOF predictions could be generated across development seasons.")

    oof_df = pd.concat(oof_records, axis=0).sort_values(by=["season", "lead_day", "cell_id"])
    return oof_df


def fit_final_m3_models(
    dev_df: pd.DataFrame,
    development_seasons: Sequence[int],
    b2_params: Optional[Dict[str, Any]] = None,
    b3_params: Optional[Dict[str, Any]] = None,
    prob_params: Optional[Dict[str, Any]] = None,
    range_params: Optional[Dict[str, Any]] = None,
    event_counts: Optional[Dict[float, int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    feature_set_version: str = "v1",
    git_commit: str = "test",
    model_version_id: Optional[int] = None,
    dev_scores: Optional[Dict[str, Any]] = None,
    train_cell_stride: int = 1,
) -> FinalM3Models:
    """Fit all final M3 models on ALL development seasons and optionally persist artifacts (PRD §12.4).

    Requirements:
    - Fits on ALL development seasons.
    - Strictly excludes any holdout season.
    - Computes training-season 99.9th percentile extrapolation threshold.
    - Persists models and metadata if output_dir provided.
    """
    train_df = dev_df[dev_df["season"].isin(development_seasons)].copy()
    if len(train_df) == 0:
        raise ValueError(f"No training data matching development seasons: {list(development_seasons)}")

    # 1. Validate training inputs
    validate_m1_m2_inputs(train_df, is_training=True, require_regime=True)

    cfg_b2 = b2_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    cfg_b3 = b3_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    cfg_prob = prob_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    cfg_range = range_params or {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05}
    counts = event_counts or {15.6: 100, 64.5: 50, 115.6: 35}

    # 2. Fit B1
    b1_model = QuantileMappingModel().fit(train_df)

    # 3. Fit B2
    b2_model = fit_final_b2_model(
        train_df,
        development_seasons=development_seasons,
        selected_config=cfg_b2,
        train_cell_stride=train_cell_stride,
    )

    # 4. Fit B3 (strictly requires regime_source='oof')
    b3_model = fit_final_b3_model(
        train_df,
        development_seasons=development_seasons,
        selected_config=cfg_b3,
        train_cell_stride=train_cell_stride,
    )

    # 5. Fit Probability Models
    prob_models = RainfallProbabilityModels().fit(
        train_df,
        event_counts=counts,
        params_map=cfg_prob,
        train_cell_stride=train_cell_stride,
    )

    # 6. Fit Range Models
    range_models = QuantileRangeModels().fit(
        train_df,
        params_map=cfg_range,
        train_cell_stride=train_cell_stride,
    )

    # 7. Compute Extrapolation Threshold from training seasons only (PRD §12.3)
    p99_9 = compute_extrapolation_threshold(train_df, rain_col="rain_mm", percentile=99.9)
    b2_model.p99_9_threshold = p99_9
    b3_model.p99_9_threshold = p99_9

    final_models = FinalM3Models(
        b1_model=b1_model,
        b2_model=b2_model,
        b3_model=b3_model,
        prob_models=prob_models,
        range_models=range_models,
        p99_9_threshold=p99_9,
        training_seasons=list(development_seasons),
        feature_set_version=feature_set_version,
        git_commit=git_commit,
        model_version_id=model_version_id,
        dev_scores=dev_scores or {},
    )

    if output_dir is not None:
        final_models.save(output_dir)

    return final_models


def predict_final_m3(
    models: FinalM3Models,
    features_df: pd.DataFrame,
    regime_available: Optional[bool] = None,
    ood_flag: bool = False,
    ml_available: bool = True,
    validation_failed: bool = False,
    no_correction: bool = False,
    model_version_id: Optional[int] = None,
    output_serving_dir: Optional[Union[str, Path]] = None,
    calibrators: Optional[Dict[Any, Any]] = None,
    allow_uncalibrated: bool = False,
) -> pd.DataFrame:
    """Execute final M3 inference pipeline producing the canonical 18-column serving grid (PRD §20.1).

    Accepts regime_source='final' for inference.
    Does NOT train models or fit calibrators.
    Applies external pre-fitted M4 calibrators if supplied.
    If calibrators are omitted and allow_uncalibrated is False, raises ValueError.
    Applies extrapolation detection and PRD §17.1 fallback selection.
    Optionally writes data/serving/grid/run_id=<run_id>/part.parquet.
    """
    # 1. Validate M1 features
    validate_feature_columns(features_df, model_type="B2")

    # 2. Check regime availability
    if regime_available is None:
        has_regime_cols = all(col in features_df.columns for col in REGIME_FEATURES)
        flag_val = features_df["regime_available"].all() if "regime_available" in features_df.columns else True
        regime_available = bool(has_regime_cols and flag_val)

    if "ood_flag" in features_df.columns:
        ood_flag = bool(features_df["ood_flag"].any() or ood_flag)

    # 3. Generate predictions
    b0_raw = predict_b0(features_df)
    b1_preds = models.b1_model.predict(features_df) if ml_available or not no_correction else None
    b2_preds = predict_b2(models.b2_model, features_df) if ml_available and not no_correction else None

    # B3 requires regime features
    b3_preds = None
    if regime_available and ml_available and not no_correction:
        validate_feature_columns(features_df, model_type="B3")
        b3_preds = predict_b3(models.b3_model, features_df)

    prob_df = None
    range_df = None
    if regime_available and ml_available and not no_correction:
        raw_prob_df = models.prob_models.predict_probabilities(features_df)
        prob_df = apply_probability_calibrators(
            raw_prob_df,
            calibrators=calibrators,
            allow_uncalibrated=allow_uncalibrated,
        )
        range_df = models.range_models.predict_range(features_df)

    # 4. Check extrapolation flags
    extrap_flags = check_extrapolation(b0_raw.values, models.p99_9_threshold)

    # 5. Assemble serving grid
    m_id = model_version_id if model_version_id is not None else models.model_version_id
    serving_df = assemble_serving_grid(
        base_df=features_df,
        b0_raw_mm=b0_raw.values,
        b1_mm=b1_preds.values if b1_preds is not None else None,
        b2_mm=b2_preds.values if b2_preds is not None else None,
        b3_mm=b3_preds.values if b3_preds is not None else None,
        range_df=range_df,
        prob_df=prob_df,
        extrapolation_flags=extrap_flags,
        regime_available=regime_available,
        ood_flag=ood_flag,
        ml_available=ml_available,
        validation_failed=validation_failed,
        no_correction=no_correction,
        model_version_id=m_id,
    )

    # 6. Optionally write parquet partition per run_id
    if output_serving_dir is not None:
        if "run_id" in features_df.columns:
            unique_runs = features_df["run_id"].unique()
            for r_id in unique_runs:
                run_mask = features_df["run_id"] == r_id
                write_serving_grid_file(
                    serving_df[run_mask].copy(),
                    run_id=str(r_id),
                    output_base_dir=output_serving_dir,
                )
        else:
            write_serving_grid_file(
                serving_df,
                run_id="default_run",
                output_base_dir=output_serving_dir,
            )

    return serving_df
