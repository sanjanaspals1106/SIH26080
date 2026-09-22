"""Model-Estimated Range: q10, q50, q90 Quantile Models (PRD §13.7).

Specifications:
- Three XGBoost models with reg:quantileerror.
- Quantile alphas: 0.1, 0.5, 0.9.
- Features: exactly B3_FEATURES (41 features in canonical PRD order).
- Target: log1p(obs_mm) (natural log of 1 + observed rain).
- Inverted via expm1(pred) to return to mm scale.
- Quantile crossing resolution: sort values per cell so q10 <= q50 <= q90.
- Clip values at 0 (non-negative rainfall).
- Tree method: hist (strictly do not use exact).
- Natural base rate preserved: keep dry/zero-rain rows (obs_mm == 0 -> log1p(0) == 0).
- NaNs passed natively to XGBoost without imputation.
- Training safety: regime_source == 'oof' strictly required for training rows.
- Inference: accepts regime_source == 'final' or 'oof'.
- Important PRD rule (§7.4, §13.7):
    q50 is the median of the forecast distribution.
    It is NOT the same as corrected_mean_mm (Tweedie mean).
    Never replace corrected_mean_mm with q50.
"""

from typing import Dict, Any, Tuple, Optional, Union, List
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from ml.feature_contracts import B3_FEATURES, validate_feature_columns
from ml.training.train_correction import validate_b3_training_data

QUANTILES = [0.1, 0.5, 0.9]


def sort_and_clip_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort quantiles per cell to guarantee q10 <= q50 <= q90 and clip at 0.

    Resolves quantile crossing by sorting the three values for each cell,
    then clipping any small numerical negative predictions at 0.0 mm.
    """
    stacked = np.stack([q10, q50, q90], axis=-1)
    sorted_q = np.sort(stacked, axis=-1)
    clipped = np.clip(sorted_q, 0.0, None)
    return clipped[..., 0], clipped[..., 1], clipped[..., 2]


def _build_quantile_regressor(
    alpha: float,
    params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBRegressor:
    """Build single XGBRegressor with reg:quantileerror and hist tree method.

    Strictly forbids tree_method='exact' (PRD §13.7).
    """
    if params is None:
        params = {}

    tree_method = str(params.get("tree_method", "hist"))
    if tree_method == "exact":
        raise ValueError("tree_method='exact' is forbidden for range models; use 'hist' (PRD §13.7).")

    max_d = int(params.get("max_depth", 4))
    min_child_w = float(params.get("min_child_weight", 20.0))
    lr = float(params.get("learning_rate", 0.05))
    n_est = int(params.get("n_estimators", 100))
    subsample = float(params.get("subsample", 0.8))
    colsample = float(params.get("colsample_bytree", 0.8))
    reg_lambda = float(params.get("reg_lambda", 10.0))
    seed = int(params.get("random_state", 42))
    n_jobs = int(params.get("n_jobs", -1))

    return xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=float(alpha),
        tree_method="hist",
        eval_metric="quantile",
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


def train_quantile_model(
    train_df: pd.DataFrame,
    alpha: float,
    params: Optional[Dict[str, Any]] = None,
    train_cell_stride: int = 1,
) -> xgb.XGBRegressor:
    """Train a single quantile regression model for a specified quantile_alpha.

    Specifications (PRD §13.7):
    - Features: B3_FEATURES exactly (41 features in PRD order)
    - Target: log1p(obs_mm)
    - Objective: reg:quantileerror with quantile_alpha = alpha
    - Tree method: hist
    - Dry rows preserved (obs_mm == 0 -> log1p(0) == 0)
    - NaNs passed directly to XGBoost
    - Training safety: regime_source == 'oof' strictly required
    """
    # 1. Feature contract validation (41 features, no forbidden columns)
    validate_feature_columns(train_df, model_type="B3")

    if "obs_mm" not in train_df.columns:
        raise ValueError("Missing required target column 'obs_mm' in training dataframe.")

    # 2. Protocol guard on regime_source (PRD §11.6, §21.1)
    validate_b3_training_data(train_df)

    # 3. Optional memory stride for training
    df = train_df
    if train_cell_stride > 1:
        df = df.iloc[::train_cell_stride]

    X = df[B3_FEATURES]
    # log1p transformation keeps 0 -> 0 and squashes high rainfall tails
    y = np.log1p(df["obs_mm"].to_numpy(dtype=float))

    model = _build_quantile_regressor(alpha, params)
    model.fit(X, y)
    return model


class QuantileRangeModels:
    """Manages training and inference of q10, q50, q90 range estimation models (PRD §13.7)."""

    def __init__(self, models: Optional[Dict[float, xgb.XGBRegressor]] = None):
        self.models: Dict[float, xgb.XGBRegressor] = models or {}

    def fit(
        self,
        train_df: pd.DataFrame,
        params_map: Optional[Union[Dict[str, Any], Dict[float, Dict[str, Any]]]] = None,
        train_cell_stride: int = 1,
    ) -> "QuantileRangeModels":
        """Train 3 quantile regression models (q10, q50, q90) on log1p(obs_mm).

        Args:
            train_df: DataFrame with 41 B3 features, obs_mm, and regime_source='oof'.
            params_map: Optional single parameter dict or dict mapping alpha -> param dict.
            train_cell_stride: Optional spatial stride for memory-efficient training.
        """
        for alpha in QUANTILES:
            cfg: Dict[str, Any] = {}
            if isinstance(params_map, dict):
                if alpha in params_map and isinstance(params_map[alpha], dict):
                    cfg = params_map[alpha]
                elif all(isinstance(k, str) for k in params_map.keys()):
                    cfg = params_map

            self.models[alpha] = train_quantile_model(
                train_df=train_df,
                alpha=alpha,
                params=cfg,
                train_cell_stride=train_cell_stride,
            )

        return self

    def predict_range(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict expm1 transformed quantiles with crossing resolution and clipping.

        Guarantees:
        - Output pd.DataFrame with columns: ['q10_mm', 'q50_mm', 'q90_mm'].
        - Row and index alignment preserved with features_df.
        - Monotonic ordering: 0 <= q10_mm <= q50_mm <= q90_mm.
        - Values in mm scale (via expm1 inverse transformation).
        - Note: q50_mm is the median, distinct from corrected_mean_mm.
        """
        validate_feature_columns(features_df, model_type="B3")
        X = features_df[B3_FEATURES]

        for alpha in QUANTILES:
            if alpha not in self.models:
                raise RuntimeError(f"Quantile model for alpha={alpha} has not been fitted.")

        # Raw predictions are in log1p space
        q10_log = self.models[0.1].predict(X)
        q50_log = self.models[0.5].predict(X)
        q90_log = self.models[0.9].predict(X)

        # Invert to mm scale: expm1(y) = exp(y) - 1
        q10_raw = np.expm1(q10_log)
        q50_raw = np.expm1(q50_log)
        q90_raw = np.expm1(q90_log)

        # Resolve quantile crossing and clip at 0
        q10_out, q50_out, q90_out = sort_and_clip_quantiles(q10_raw, q50_raw, q90_raw)

        return pd.DataFrame(
            {
                "q10_mm": q10_out,
                "q50_mm": q50_out,
                "q90_mm": q90_out,
            },
            index=features_df.index,
        )

    def save(self, dir_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save all 3 quantile models and metadata to a directory."""
        from ml.models.model_store import save_range_models
        return save_range_models(self, dir_path, metadata)

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "QuantileRangeModels":
        """Load 3 quantile models from a directory."""
        from ml.models.model_store import load_range_models
        return load_range_models(dir_path)


def predict_range(
    models: Union[QuantileRangeModels, Dict[float, xgb.XGBRegressor]],
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict model-estimated quantiles q10, q50, q90 (PRD §13.7).

    Accepts QuantileRangeModels instance or dict of models {0.1: m1, 0.5: m2, 0.9: m3}.
    """
    if isinstance(models, QuantileRangeModels):
        return models.predict_range(features_df)
    elif isinstance(models, dict):
        wrapper = QuantileRangeModels(models=models)
        return wrapper.predict_range(features_df)
    else:
        raise TypeError(f"Expected QuantileRangeModels or dict of models, got {type(models)}")


def calculate_range_coverage(
    df: Optional[pd.DataFrame] = None,
    q10_col: str = "q10_mm",
    q90_col: str = "q90_mm",
    obs_col: str = "obs_mm",
    lead_col: str = "lead_day",
    group_by_lead: bool = True,
    q10: Optional[Union[pd.Series, np.ndarray]] = None,
    q90: Optional[Union[pd.Series, np.ndarray]] = None,
    obs: Optional[Union[pd.Series, np.ndarray]] = None,
    lead: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Calculate empirical coverage of model-estimated range [q10, q90] (PRD §13.7).

    Coverage is defined as the fraction of observations where:
        q10 <= obs_mm <= q90
    The nominal / target coverage is 80% (0.80).

    Computes the true empirical coverage without faking or forcing 80%.
    Per PRD §13.7: if any lead differs from 80% by more than 10 percentage points
    (i.e. coverage < 70% or coverage > 90%), deviation_exceeded is True.

    Args:
        df: Optional DataFrame containing quantiles, obs_mm, and lead_day.
        q10_col: Name of q10 column in df.
        q90_col: Name of q90 column in df.
        obs_col: Name of observed rainfall column in df.
        lead_col: Name of lead_day column in df.
        group_by_lead: Whether to compute coverage broken down by lead_day.
        q10, q90, obs, lead: Optional direct arrays/Series if df is not provided.

    Returns:
        Dict with keys:
        - overall_coverage: float
        - lead_coverage: Dict[int, float]
        - nominal_coverage: 0.80
        - total_samples: int
        - within_range_count: int
        - deviation_exceeded: bool
        - lead_samples: Dict[int, int]
    """
    if df is not None:
        q10_vals = df[q10_col].to_numpy(dtype=float)
        q90_vals = df[q90_col].to_numpy(dtype=float)
        obs_vals = df[obs_col].to_numpy(dtype=float)
        lead_vals = df[lead_col].to_numpy(dtype=int) if lead_col in df.columns else None
    else:
        if q10 is None or q90 is None or obs is None:
            raise ValueError("Must provide either 'df' or all of ('q10', 'q90', 'obs').")
        q10_vals = np.asarray(q10, dtype=float)
        q90_vals = np.asarray(q90, dtype=float)
        obs_vals = np.asarray(obs, dtype=float)
        lead_vals = np.asarray(lead, dtype=int) if lead is not None else None

    # Filter out NaNs (e.g. unknown observations)
    valid_mask = (~np.isnan(obs_vals)) & (~np.isnan(q10_vals)) & (~np.isnan(q90_vals))
    if not np.any(valid_mask):
        raise ValueError("No valid rows found with non-NaN observations and quantiles.")

    q10_valid = q10_vals[valid_mask]
    q90_valid = q90_vals[valid_mask]
    obs_valid = obs_vals[valid_mask]

    inside = (q10_valid <= obs_valid) & (obs_valid <= q90_valid)
    overall_cov = float(np.mean(inside))

    lead_coverage: Dict[int, float] = {}
    lead_samples: Dict[int, int] = {}

    if group_by_lead and lead_vals is not None:
        lead_valid = lead_vals[valid_mask]
        unique_leads = np.unique(lead_valid)
        for ld in sorted(unique_leads):
            ld_mask = lead_valid == ld
            lead_coverage[int(ld)] = float(np.mean(inside[ld_mask]))
            lead_samples[int(ld)] = int(np.sum(ld_mask))

    # Check PRD §13.7 rule: differs from 80% by more than 10 percentage points
    if lead_coverage:
        deviation_exceeded = any(abs(cov - 0.80) > 0.10 for cov in lead_coverage.values())
    else:
        deviation_exceeded = abs(overall_cov - 0.80) > 0.10

    return {
        "overall_coverage": overall_cov,
        "lead_coverage": lead_coverage,
        "nominal_coverage": 0.80,
        "total_samples": int(len(inside)),
        "within_range_count": int(np.sum(inside)),
        "deviation_exceeded": bool(deviation_exceeded),
        "lead_samples": lead_samples,
    }
