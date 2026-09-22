"""Training Pipeline for B2 and B3 XGBoost Rainfall Correction Models (PRD §10, §12).

Specifications:
- Objective: reg:tweedie
- Target: observed rain in mm (obs_mm)
- Monotone constraint: +1 on rain_mm
- Tree method: hist
- Never remove dry rows from training (§10.11)
- Stop with error if any training row has regime_source = 'final' (§11.6, §21.1)
"""

from typing import List, Dict, Any, Optional, Sequence
import pandas as pd
import numpy as np
import xgboost as xgb

from ml.feature_contracts import (
    BASE_FEATURES,
    REGIME_FEATURES,
    B2_FEATURES,
    B3_FEATURES,
    validate_feature_columns,
)

# Aliases for backward-compatibility
BASE_FEATURES_27 = BASE_FEATURES
REGIME_FEATURES_14 = REGIME_FEATURES
REGIME_AWARE_FEATURES_41 = B3_FEATURES


def validate_b3_training_data(df: pd.DataFrame) -> None:
    """Validate data integrity and protocol restrictions for B3 training (PRD §11.6, §21.1).

    Rules:
    - Training data MUST contain 'regime_source'.
    - If 'regime_source' is missing, raise a clear error.
    - For EVERY row used to train B3:
        regime_source must equal 'oof'.
        If ANY training row has regime_source == 'final' (or anything other than 'oof'),
        raise a clear hard error.
    """
    if "regime_source" not in df.columns:
        raise ValueError(
            "Protocol violation (PRD §11.6): Missing required 'regime_source' column in B3 training data."
        )

    final_count = (df["regime_source"] == "final").sum()
    if final_count > 0:
        raise ValueError(
            f"Protocol violation (PRD §11.6): Found {final_count} rows with regime_source='final' in training data. "
            "All B3 training rows must strictly use out-of-fold regime predictions ('oof')."
        )

    non_oof = df["regime_source"] != "oof"
    if non_oof.any():
        invalid_sources = df.loc[non_oof, "regime_source"].unique().tolist()
        raise ValueError(
            f"Protocol violation (PRD §11.6): All B3 training rows must have regime_source='oof'. "
            f"Found invalid regime_source values: {invalid_sources}"
        )


def validate_training_dataframe(df: pd.DataFrame, is_b3: bool = False) -> None:
    """Validate data integrity and protocol restrictions before training."""
    if is_b3:
        validate_b3_training_data(df)
    elif "regime_source" in df.columns:
        # For non-B3, still check that final regime features are not inappropriately mixed
        final_rows = (df["regime_source"] == "final").sum()
        if final_rows > 0:
            raise ValueError(
                f"Protocol violation: Found {final_rows} rows with regime_source='final' in training data."
            )


def build_b2_monotone_constraints() -> tuple:
    """Build monotone constraints tuple for B2 features: +1 on rain_mm, 0 on all others."""
    return tuple(1 if f == "rain_mm" else 0 for f in B2_FEATURES)


def build_b3_monotone_constraints() -> tuple:
    """Build monotone constraints tuple for B3 features: +1 on rain_mm, 0 on all 40 others."""
    return tuple(1 if f == "rain_mm" else 0 for f in B3_FEATURES)


def _build_tweedie_regressor(params: Dict[str, Any], monotone_constraints: tuple) -> xgb.XGBRegressor:
    """Construct configured XGBRegressor for Tweedie rainfall correction."""
    tweedie_p = float(params.get("tweedie_variance_power", 1.5))
    max_d = int(params.get("max_depth", 6))
    min_child_w = float(params.get("min_child_weight", 100))
    lr = float(params.get("learning_rate", 0.05))
    n_est = int(params.get("n_estimators", 400))
    subsample = float(params.get("subsample", 0.8))
    colsample = float(params.get("colsample_bytree", 0.8))
    reg_lambda = float(params.get("reg_lambda", 10.0))
    seed = int(params.get("random_state", 42))
    n_jobs = int(params.get("n_jobs", -1))

    return xgb.XGBRegressor(
        objective="reg:tweedie",
        tree_method="hist",
        monotone_constraints=monotone_constraints,
        tweedie_variance_power=tweedie_p,
        max_depth=max_d,
        min_child_weight=min_child_w,
        learning_rate=lr,
        n_estimators=n_est,
        subsample=subsample,
        colsample_bytree=colsample,
        reg_lambda=reg_lambda,
        random_state=seed,
        n_jobs=n_jobs,
    )


def train_b2_model(
    train_df: pd.DataFrame,
    params: Dict[str, Any],
    train_cell_stride: int = 1,
) -> xgb.XGBRegressor:
    """Train B2 global machine learning model with 27 features without regime features.

    Specifications (PRD §10.11, §12.2):
    - Features: B2_FEATURES exactly (27 features in PRD order)
    - Target: obs_mm (observed rain in mm)
    - Objective: reg:tweedie
    - Tree method: hist
    - Monotone constraint: +1 on rain_mm, 0 on all others
    - Preserves dry/zero-rain rows (no base-rate distortion)
    - NaNs handled directly by XGBoost without imputation
    - No early stopping
    """
    # 1. Validate features (ensures 27 features, no forbidden columns)
    validate_feature_columns(train_df, model_type="B2")

    # 2. Validate target column
    if "obs_mm" not in train_df.columns:
        raise ValueError("Missing required target column 'obs_mm' in training dataframe.")

    # 3. Optional memory stride for training cells (§10.11)
    df = train_df
    if train_cell_stride > 1:
        df = df.iloc[::train_cell_stride]

    X = df[B2_FEATURES]
    y = df["obs_mm"].to_numpy(dtype=float)

    monotone_constraints = build_b2_monotone_constraints()
    model = _build_tweedie_regressor(params, monotone_constraints)
    model.fit(X, y)
    return model


def train_b3_model(
    train_df: pd.DataFrame,
    params: Dict[str, Any],
    train_cell_stride: int = 1,
) -> xgb.XGBRegressor:
    """Train B3 regime-aware model with 41 features (PRD §10.3, §11.6, §11.7, §12).

    Specifications:
    - Features: B3_FEATURES exactly (41 features: 27 base + 14 regime in PRD order)
    - Target: obs_mm
    - Objective: reg:tweedie
    - Tree method: hist
    - Monotone constraint: +1 on rain_mm, 0 on all other 40 features
    - Preserves dry rows (no filtering)
    - NaNs handled directly by XGBoost without imputation
    - Protocol guard: hard error if regime_source is missing or not 'oof' (§11.6)
    - No early stopping
    """
    # 1. Validate features (ensures 41 features in PRD order)
    validate_feature_columns(train_df, model_type="B3")

    # 2. Validate target
    if "obs_mm" not in train_df.columns:
        raise ValueError("Missing required target column 'obs_mm' in training dataframe.")

    # 3. Protocol guard on regime_source (PRD §11.6, §21.1)
    validate_b3_training_data(train_df)

    # 4. Optional memory stride for training cells (§10.11)
    df = train_df
    if train_cell_stride > 1:
        df = df.iloc[::train_cell_stride]

    X = df[B3_FEATURES]
    y = df["obs_mm"].to_numpy(dtype=float)

    monotone_constraints = build_b3_monotone_constraints()
    model = _build_tweedie_regressor(params, monotone_constraints)
    model.fit(X, y)
    return model


def fit_final_b2_model(
    df: pd.DataFrame,
    development_seasons: Sequence[int],
    selected_config: Dict[str, Any],
    train_cell_stride: int = 1,
) -> xgb.XGBRegressor:
    """Fit final B2 model on ALL development seasons using the selected settings.

    Inputs must explicitly receive development seasons (does not infer splits).
    """
    if not development_seasons:
        raise ValueError("development_seasons list cannot be empty.")

    if "season" not in df.columns:
        raise ValueError("Missing 'season' column required to filter development seasons.")

    dev_df = df[df["season"].isin(development_seasons)].copy()
    if len(dev_df) == 0:
        raise ValueError(
            f"No data found matching development seasons: {list(development_seasons)}"
        )

    return train_b2_model(dev_df, params=selected_config, train_cell_stride=train_cell_stride)


def fit_final_b3_model(
    df: pd.DataFrame,
    development_seasons: Sequence[int],
    selected_config: Dict[str, Any],
    train_cell_stride: int = 1,
) -> xgb.XGBRegressor:
    """Fit final B3 model on ALL development seasons using the selected settings.

    Inputs must explicitly receive development seasons.
    All selected training rows must strictly have regime_source == 'oof'.
    """
    if not development_seasons:
        raise ValueError("development_seasons list cannot be empty.")

    if "season" not in df.columns:
        raise ValueError("Missing 'season' column required to filter development seasons.")

    dev_df = df[df["season"].isin(development_seasons)].copy()
    if len(dev_df) == 0:
        raise ValueError(
            f"No data found matching development seasons: {list(development_seasons)}"
        )

    # Validate regime source on the filtered development data
    validate_b3_training_data(dev_df)

    return train_b3_model(dev_df, params=selected_config, train_cell_stride=train_cell_stride)
