"""Heavy-Rainfall Probability Classifiers (PRD §13.1, §13.2, §13.3).

Specifications:
- Exactly B3_FEATURES (41 features in canonical PRD order).
- XGBoost objective: binary:logistic
- Tree method: hist
- Natural base rate: strictly no oversampling, no dry-row removal.
- Target derived from obs_mm threshold:
    15.6 mm (moderate or more)
    64.5 mm (heavy)
    115.6 mm (very heavy)
- Event-count branching (inputs provided by M4):
    - threshold >= 30 events -> standalone classifier
    - 115.6 < 30 events and 64.5 >= 30 -> chained form:
        P(>=115.6) = P(>=64.5) * P(>=115.6 | >=64.5)
    - 64.5 < 30 events -> heavy and very heavy models disabled, only 15.6 available
- Training safety:
    - regime_source == 'oof' strictly required for training rows.
    - regime_source == 'final' or missing raises hard error.
- Post-prediction probability ordering:
    P(>=115.6) <= P(>=64.5) <= P(>=15.6) for every cell, values in [0, 1].
- Hyperparameter search: scored by lowest mean Brier score across newest 3 validation seasons.
"""

from typing import Dict, Any, Optional, Tuple, Sequence, Union, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb

from ml.feature_contracts import B3_FEATURES, validate_feature_columns
from ml.training.train_correction import validate_b3_training_data
from ml.training.hyperparameter_search import (
    generate_shared_hyperparameter_configs,
    get_validation_seasons,
)

THRESHOLDS = [15.6, 64.5, 115.6]


@dataclass
class ProbabilityModelAvailability:
    """Status metadata for probability models based on M4 event counts (PRD §13.2)."""
    has_15_6: bool
    has_64_5: bool
    has_115_6: bool
    chained_115_6: bool
    message: Optional[str] = None


def determine_model_availability(event_counts: Dict[float, int]) -> ProbabilityModelAvailability:
    """Determine heavy-rain model branching from M4 event counts (PRD §13.2).

    Rules:
    - Threshold has >= 30 events -> build own classifier.
    - 115.6 has < 30 events AND 64.5 has >= 30 -> chained form:
        P(>=115.6) = P(>=64.5) * P(>=115.6 | >=64.5).
    - 64.5 has < 30 events -> no heavy-rain model, message 'Not enough events'.
    """
    n_15 = int(event_counts.get(15.6, 0))
    n_64 = int(event_counts.get(64.5, 0))
    n_115 = int(event_counts.get(115.6, 0))

    has_15 = n_15 >= 30

    if n_64 < 30:
        return ProbabilityModelAvailability(
            has_15_6=has_15,
            has_64_5=False,
            has_115_6=False,
            chained_115_6=False,
            message="Not enough events",
        )

    # 64.5 has >= 30 events
    if n_115 >= 30:
        return ProbabilityModelAvailability(
            has_15_6=has_15,
            has_64_5=True,
            has_115_6=True,
            chained_115_6=False,
            message=None,
        )
    else:
        # Chained form
        return ProbabilityModelAvailability(
            has_15_6=has_15,
            has_64_5=True,
            has_115_6=True,
            chained_115_6=True,
            message="Chained form: P(>=115.6) = P(>=64.5) * P(>=115.6 | >=64.5)",
        )


def enforce_probability_monotonicity(
    p_15_6: np.ndarray,
    p_64_5: Optional[np.ndarray],
    p_115_6: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Enforce P(>=115.6) <= P(>=64.5) <= P(>=15.6) across all cells and clip in [0, 1]."""
    p_15 = np.clip(p_15_6, 0.0, 1.0)
    p_64 = None
    p_115 = None

    if p_64_5 is not None:
        p_64 = np.minimum(p_15, np.clip(p_64_5, 0.0, 1.0))
        if p_115_6 is not None:
            p_115 = np.minimum(p_64, np.clip(p_115_6, 0.0, 1.0))

    return p_15, p_64, p_115


def _build_binary_classifier(params: Dict[str, Any]) -> xgb.XGBClassifier:
    """Instantiate XGBClassifier configured with binary:logistic and hist tree method."""
    max_d = int(params.get("max_depth", 6))
    min_child_w = float(params.get("min_child_weight", 100))
    lr = float(params.get("learning_rate", 0.05))
    n_est = int(params.get("n_estimators", 400))
    subsample = float(params.get("subsample", 0.8))
    colsample = float(params.get("colsample_bytree", 0.8))
    reg_lambda = float(params.get("reg_lambda", 10.0))
    seed = int(params.get("random_state", 42))
    n_jobs = int(params.get("n_jobs", -1))

    return xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
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


def train_probability_classifier(
    train_df: pd.DataFrame,
    threshold: float,
    params: Dict[str, Any],
    is_conditional: bool = False,
    train_cell_stride: int = 1,
) -> xgb.XGBClassifier:
    """Train single binary probability classifier using 41 B3 features.

    Specifications (PRD §13.3):
    - Features: B3_FEATURES exactly (41 features)
    - Objective: binary:logistic
    - Tree method: hist
    - Natural base rate preserved (no oversampling, no dry-row removal)
    - If is_conditional=True: trained strictly on rows with obs_mm >= 64.5
    - Training safety: regime_source == 'oof' strictly enforced.
    """
    # 1. Feature contract validation (41 features, no forbidden columns)
    validate_feature_columns(train_df, model_type="B3")

    if "obs_mm" not in train_df.columns:
        raise ValueError("Missing required target column 'obs_mm' in training dataframe.")

    # 2. Protocol guard on regime_source (PRD §11.6, §21.1)
    validate_b3_training_data(train_df)

    df = train_df
    # For conditional model: P(>= 115.6 | >= 64.5)
    if is_conditional:
        df = df[df["obs_mm"] >= 64.5].copy()
        if len(df) == 0:
            raise ValueError("No rows found matching conditional criteria obs_mm >= 64.5")

    # 3. Optional memory stride
    if train_cell_stride > 1:
        df = df.iloc[::train_cell_stride]

    X = df[B3_FEATURES]
    y = (df["obs_mm"] >= threshold).astype(int).to_numpy()

    clf = _build_binary_classifier(params)
    clf.fit(X, y)
    return clf


def evaluate_probability_settings_search(
    df: pd.DataFrame,
    development_seasons: Sequence[int],
    threshold: float,
    combinations: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    n_validation_seasons: int = 3,
    is_conditional: bool = False,
    train_cell_stride: int = 1,
) -> Dict[str, Any]:
    """Evaluate candidate parameter combinations for probability models using Brier score (PRD §13.3).

    Brier score = mean((p - y)^2).
    Averages Brier scores across the newest 3 development validation seasons.
    Selects combination with lowest mean Brier score.
    """
    if "season" not in df.columns:
        raise ValueError("Missing 'season' column required for season-based CV.")

    dev_df = df[df["season"].isin(development_seasons)].copy()
    if len(dev_df) == 0:
        raise ValueError("No data found matching provided development_seasons.")

    val_seasons = get_validation_seasons(development_seasons, n_validation=n_validation_seasons)

    if combinations is None:
        combinations = generate_shared_hyperparameter_configs(seed=seed, n_configs=20)

    all_results: List[Dict[str, Any]] = []

    for idx, config in enumerate(combinations):
        bs_by_season: Dict[int, float] = {}

        for val_s in val_seasons:
            train_fold = dev_df[dev_df["season"] != val_s]
            val_fold = dev_df[dev_df["season"] == val_s]

            # Fit classifier on other development seasons
            clf = train_probability_classifier(
                train_fold,
                threshold=threshold,
                params=config,
                is_conditional=is_conditional,
                train_cell_stride=train_cell_stride,
            )

            # Predict probabilities on validation season
            X_val = val_fold[B3_FEATURES]
            p_val = clf.predict_proba(X_val)[:, 1]
            y_val = (val_fold["obs_mm"] >= threshold).astype(float).to_numpy()

            # Compute Brier score: mean((p - y)^2)
            bs = float(np.mean((p_val - y_val) ** 2))
            bs_by_season[int(val_s)] = bs

        mean_bs = float(np.mean(list(bs_by_season.values())))
        all_results.append({
            "config_index": idx,
            "config": config,
            "brier_score_by_season": bs_by_season,
            "mean_brier_score": mean_bs,
        })

    # Pick lowest mean Brier score (tie -> earliest generated order)
    best_entry = min(all_results, key=lambda r: r["mean_brier_score"])

    return {
        "best_config": best_entry["config"],
        "best_config_index": best_entry["config_index"],
        "best_mean_brier_score": best_entry["mean_brier_score"],
        "validation_seasons": val_seasons,
        "all_results": all_results,
    }


class RainfallProbabilityModels:
    """Manages heavy-rainfall probability classifiers at 15.6, 64.5, and 115.6 mm."""

    def __init__(
        self,
        availability: Optional[ProbabilityModelAvailability] = None,
        models: Optional[Dict[str, xgb.XGBClassifier]] = None,
    ):
        self.availability = availability
        self.models: Dict[str, xgb.XGBClassifier] = models or {}
        self.training_seasons: List[int] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        event_counts: Dict[float, int],
        params_map: Union[Dict[str, Any], Dict[float, Dict[str, Any]]],
        training_seasons: Optional[Sequence[int]] = None,
        train_cell_stride: int = 1,
    ) -> "RainfallProbabilityModels":
        """Fit probability models according to event-count branching (PRD §13.2, §13.3).

        Args:
            train_df: DataFrame containing 41 B3 features and obs_mm.
            event_counts: Official M4 event counts for thresholds 15.6, 64.5, 115.6.
            params_map: Either single config dict or dict mapping threshold -> config dict.
            training_seasons: Optional list of seasons to filter training data.
            train_cell_stride: Memory stride (default 1).
        """
        data = train_df
        if training_seasons is not None:
            if "season" not in train_df.columns:
                raise ValueError("Missing 'season' column in DataFrame to filter training_seasons.")
            data = train_df[train_df["season"].isin(training_seasons)].copy()
            self.training_seasons = sorted(list(set(training_seasons)))
        elif "season" in train_df.columns:
            self.training_seasons = sorted(list(train_df["season"].unique()))

        # Determine branching based on M4 event counts
        self.availability = determine_model_availability(event_counts)
        self.models = {}

        def get_params(t: float) -> Dict[str, Any]:
            if isinstance(params_map, dict) and t in params_map:
                return params_map[t]
            return params_map  # Common parameters dictionary

        # 1. Moderate threshold: 15.6 mm
        if self.availability.has_15_6:
            self.models["15.6"] = train_probability_classifier(
                data,
                threshold=15.6,
                params=get_params(15.6),
                is_conditional=False,
                train_cell_stride=train_cell_stride,
            )

        # 2. Heavy threshold: 64.5 mm
        if self.availability.has_64_5:
            self.models["64.5"] = train_probability_classifier(
                data,
                threshold=64.5,
                params=get_params(64.5),
                is_conditional=False,
                train_cell_stride=train_cell_stride,
            )

        # 3. Very heavy threshold: 115.6 mm (standalone or chained)
        if self.availability.has_115_6:
            if self.availability.chained_115_6:
                # Conditional model: P(>= 115.6 | >= 64.5)
                self.models["115.6_cond"] = train_probability_classifier(
                    data,
                    threshold=115.6,
                    params=get_params(115.6),
                    is_conditional=True,
                    train_cell_stride=train_cell_stride,
                )
            else:
                # Standalone model
                self.models["115.6"] = train_probability_classifier(
                    data,
                    threshold=115.6,
                    params=get_params(115.6),
                    is_conditional=False,
                    train_cell_stride=train_cell_stride,
                )

        return self

    def predict_probabilities(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict calibrated probabilities and enforce monotonic ordering (PRD §13.3).

        Guarantees:
        - Output pd.DataFrame with columns: ['p_ge_15_6', 'p_ge_64_5', 'p_ge_115_6'].
        - Preserves input row and index order.
        - Monotonicity: P(>= 115.6) <= P(>= 64.5) <= P(>= 15.6).
        - All values bounded in [0.0, 1.0].
        - Unavailable thresholds return NaN.
        """
        validate_feature_columns(features_df, model_type="B3")
        X = features_df[B3_FEATURES]

        n_rows = len(features_df)
        p_15_raw = np.full(n_rows, np.nan, dtype=float)
        p_64_raw = np.full(n_rows, np.nan, dtype=float)
        p_115_raw = np.full(n_rows, np.nan, dtype=float)

        if self.availability is None:
            raise RuntimeError("Probability models have not been fitted.")

        if self.availability.has_15_6 and "15.6" in self.models:
            p_15_raw = self.models["15.6"].predict_proba(X)[:, 1]

        if self.availability.has_64_5 and "64.5" in self.models:
            p_64_raw = self.models["64.5"].predict_proba(X)[:, 1]

        if self.availability.has_115_6:
            if self.availability.chained_115_6 and "115.6_cond" in self.models:
                p_cond = self.models["115.6_cond"].predict_proba(X)[:, 1]
                p_115_raw = p_64_raw * p_cond
            elif "115.6" in self.models:
                p_115_raw = self.models["115.6"].predict_proba(X)[:, 1]

        # Enforce probability monotonicity
        p15_arr = p_15_raw if self.availability.has_15_6 else None
        p64_arr = p_64_raw if self.availability.has_64_5 else None
        p115_arr = p_115_raw if self.availability.has_115_6 else None

        if p15_arr is not None:
            p15_corr, p64_corr, p115_corr = enforce_probability_monotonicity(
                p15_arr, p64_arr, p115_arr
            )
        else:
            p15_corr, p64_corr, p115_corr = p15_raw, p64_raw, p115_raw

        return pd.DataFrame(
            {
                "p_ge_15_6": p15_corr,
                "p_ge_64_5": p64_corr if self.availability.has_64_5 else np.nan,
                "p_ge_115_6": p115_corr if self.availability.has_115_6 else np.nan,
            },
            index=features_df.index,
        )


def predict_probabilities(
    models: RainfallProbabilityModels,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Helper to predict probabilities using fitted probability models."""
    return models.predict_probabilities(features_df)
