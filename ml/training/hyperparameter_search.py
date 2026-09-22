"""Deterministic Shared Hyperparameter Generator for B2 and B3 (PRD §12.2).

Rules (PRD §12.2):
- Exactly 20 unique random parameter combinations from the specified grid.
- Deterministic for the same seed.
- Seed is read from config/protocol.yaml if available, or supplied explicitly.
  Does NOT silently invent a production seed.
- B2 and B3 consume the exact SAME generated list for a fair comparison.
- Evaluated on 3 validation seasons (newest 3 development seasons).
- Fully serializable to JSON.
"""

from typing import List, Dict, Any, Optional, Sequence, Union, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Exact search space from PRD §12.2
PARAM_GRID: Dict[str, List[Any]] = {
    "tweedie_variance_power": [1.2, 1.5, 1.8],
    "max_depth": [4, 5, 6, 7, 8],
    "min_child_weight": [50, 100, 200, 400],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [200, 400, 800],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_lambda": [5, 10, 20],
}


def resolve_protocol_seed(
    seed: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> int:
    """Resolve random seed from argument or config/protocol.yaml.

    Raises:
        ValueError: If seed is not provided and config file cannot supply one.
    """
    if seed is not None:
        return int(seed)

    target_config = Path(config_path) if config_path is not None else Path("config/protocol.yaml")
    if target_config.is_file():
        try:
            import yaml
            with open(target_config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                for key in ["seed", "random_seed", "protocol_seed"]:
                    if key in cfg and cfg[key] is not None:
                        return int(cfg[key])
        except Exception as e:
            raise ValueError(f"Failed to read seed from config file '{target_config}': {e}")

    raise ValueError(
        "No seed provided and config/protocol.yaml is not present. "
        "A seed must be explicitly provided (do not invent a silent fallback seed)."
    )


def generate_shared_hyperparameter_configs(
    seed: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
    n_configs: int = 20,
) -> List[Dict[str, Any]]:
    """Generate exactly n_configs UNIQUE parameter dictionaries deterministically.

    Both B2 and B3 must evaluate the same generated list (PRD §12.2).
    """
    resolved_seed = resolve_protocol_seed(seed=seed, config_path=config_path)
    rng = np.random.RandomState(resolved_seed)
    keys = sorted(PARAM_GRID.keys())

    unique_combinations: List[Dict[str, Any]] = []
    seen_signatures = set()

    # Total combinations in grid is 4,320; 20 unique items are found quickly
    max_draws = 10000
    draws = 0

    while len(unique_combinations) < n_configs and draws < max_draws:
        draws += 1
        combo: Dict[str, Any] = {}
        for k in keys:
            options = PARAM_GRID[k]
            choice_idx = int(rng.choice(len(options)))
            val = options[choice_idx]
            combo[k] = float(val) if isinstance(val, float) else int(val)

        # Signature to guarantee uniqueness
        sig: Tuple[Tuple[str, Any], ...] = tuple((k, combo[k]) for k in keys)
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_combinations.append(combo)

    if len(unique_combinations) != n_configs:
        raise RuntimeError(
            f"Failed to sample {n_configs} unique configurations after {draws} draws."
        )

    # Ensure JSON serializability
    json.dumps(unique_combinations)
    return unique_combinations


# Backward-compatible alias
sample_hyperparameter_combinations = generate_shared_hyperparameter_configs


def get_validation_seasons(
    development_seasons: Sequence[int],
    n_validation: int = 3,
) -> List[int]:
    """Select the newest 3 development seasons for hyperparameter validation.

    PRD §12.2:
        'Each combination is scored on 3 validation seasons (the newest 3 development
        seasons, one at a time, trained on the other development seasons).'

    Args:
        development_seasons: Sequence of development seasons.
        n_validation: Number of validation seasons to select (default: 3).

    Returns:
        List of selected validation seasons in ascending order.

    Raises:
        ValueError: If fewer than n_validation development seasons are provided.
    """
    if len(development_seasons) < n_validation:
        raise ValueError(
            f"At least {n_validation} development seasons are required for validation, "
            f"but only {len(development_seasons)} provided: {list(development_seasons)}"
        )

    sorted_seasons = sorted(list(development_seasons))
    return sorted_seasons[-n_validation:]


def evaluate_settings_search(
    df: pd.DataFrame,
    development_seasons: Sequence[int],
    combinations: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    model_type: str = "B2",
    n_validation_seasons: int = 3,
    train_cell_stride: int = 1,
) -> Dict[str, Any]:
    """Evaluate candidate parameter combinations across 3 validation seasons (PRD §12.2).

    Rules:
    - Validation seasons = newest 3 development seasons.
    - Each combination is evaluated on each validation season separately.
    - For validation season s, trained on development seasons except s.
    - Never trains on the validation season.
    - Metric: RMSE = sqrt(mean((y_pred - y_true)^2)).
    - Mean RMSE = average across the validation seasons.
    - Chooses config with lowest mean RMSE (first in original order on tie).
    - Uses development seasons only (never holdout seasons).

    Returns:
        Dict containing:
        - 'best_config': Dict[str, Any]
        - 'best_config_index': int
        - 'best_mean_rmse': float
        - 'validation_seasons': List[int]
        - 'all_results': List[Dict[str, Any]]
    """
    m_type = model_type.upper()
    if m_type == "B2":
        from ml.training.train_correction import train_b2_model as train_fn
        from ml.inference.predict_correction import predict_b2 as predict_fn
    elif m_type == "B3":
        from ml.training.train_correction import train_b3_model as train_fn
        from ml.inference.predict_correction import predict_b3 as predict_fn
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'B2' or 'B3'.")

    if "season" not in df.columns:
        raise ValueError("Missing 'season' column in DataFrame for season-based splitting.")

    # Strictly isolate development seasons (PRD §10.1, §10.4)
    dev_df = df[df["season"].isin(development_seasons)].copy()
    if len(dev_df) == 0:
        raise ValueError("No data rows found matching provided development_seasons.")

    # Select newest 3 development seasons
    val_seasons = get_validation_seasons(development_seasons, n_validation=n_validation_seasons)

    # Use or generate the 20 shared configurations
    if combinations is None:
        combinations = generate_shared_hyperparameter_configs(seed=seed, n_configs=20)

    all_results: List[Dict[str, Any]] = []

    for idx, config in enumerate(combinations):
        rmse_by_season: Dict[int, float] = {}

        for val_season in val_seasons:
            # Leave-one-season-out within development seasons
            train_fold = dev_df[dev_df["season"] != val_season]
            val_fold = dev_df[dev_df["season"] == val_season]

            # Fit model on the other development seasons
            model = train_fn(
                train_fold,
                params=config,
                train_cell_stride=train_cell_stride,
            )

            # Predict on validation season
            preds = predict_fn(model, val_fold)
            y_true = val_fold["obs_mm"].to_numpy(dtype=float)

            # Calculate seasonal RMSE
            diff = preds.to_numpy() - y_true
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            rmse_by_season[int(val_season)] = rmse

        mean_rmse = float(np.mean(list(rmse_by_season.values())))
        all_results.append({
            "config_index": idx,
            "config": config,
            "rmse_per_season": rmse_by_season,
            "mean_rmse": mean_rmse,
        })

    # Pick lowest mean RMSE (min preserves first occurrence on ties)
    best_entry = min(all_results, key=lambda r: r["mean_rmse"])

    return {
        "best_config": best_entry["config"],
        "best_config_index": best_entry["config_index"],
        "best_mean_rmse": best_entry["mean_rmse"],
        "validation_seasons": val_seasons,
        "all_results": all_results,
    }
