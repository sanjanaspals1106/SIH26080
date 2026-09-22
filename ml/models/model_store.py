"""Model Save and Load Utilities with Metadata Tracking (PRD §12.4, §13.7, §20.2).

Saves models alongside complete model metadata:
- model_type (b2, b3, prob_15_6, prob_64_5, prob_115_6, q10, q50, q90)
- feature_names
- train_seasons
- created_at
- git_commit or 'test'
- best_params
- metrics

Uses standard XGBoost JSON format (save_model / load_model).
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
import numpy as np
import xgboost as xgb

VALID_MODEL_TYPES = {
    "b2",
    "b3",
    "prob_15_6",
    "prob_64_5",
    "prob_115_6",
    "prob_115_6_cond",
    "probability",
    "q10",
    "q50",
    "q90",
    "range",  # Composite container for q10, q50, q90
}


@dataclass
class ModelVersionRecord:
    """Record schema matching model_versions table (PRD §20.2)."""
    name: str
    model_role: str
    algorithm: str
    feature_set_version: str
    training_seasons: List[int]
    git_commit: str
    settings: Dict[str, Any]
    dev_scores: Dict[str, Any]
    model_version_id: Optional[int] = None
    is_active: bool = False


@dataclass
class ModelMetadata:
    """Standard metadata contract for all M3 models (PRD §12.4, §13, §20.2)."""
    model_type: str
    feature_names: List[str]
    train_seasons: List[int]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str = "test"
    best_params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_version_id: Optional[int] = None
    p99_9_threshold: Optional[float] = None

    def validate(self) -> None:
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. Expected one of: {sorted(VALID_MODEL_TYPES)}"
            )
        if not isinstance(self.feature_names, list):
            raise TypeError(f"feature_names must be a list of str, got {type(self.feature_names)}")
        if not isinstance(self.train_seasons, list):
            raise TypeError(f"train_seasons must be a list of int, got {type(self.train_seasons)}")


class LoadedModel(tuple):
    """Container for loaded model and metadata.

    Supports both unpacking as (model, metadata) and direct attribute / method forwarding.
    """

    def __new__(cls, model: Any, metadata: Dict[str, Any]):
        return super().__new__(cls, (model, metadata))

    @property
    def model(self) -> Any:
        return self[0]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self[1]

    def __getattr__(self, name: str):
        # Forward attribute/method access to the underlying model (e.g. predict, predict_proba)
        return getattr(self[0], name)

    def __repr__(self) -> str:
        return f"LoadedModel(model={type(self[0]).__name__}, model_type='{self[1].get('model_type', '')}')"


def _normalize_metadata(metadata: Union[ModelMetadata, ModelVersionRecord, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert metadata dataclass or dict to standard JSON-serializable dictionary."""
    if isinstance(metadata, ModelMetadata):
        metadata.validate()
        return asdict(metadata)
    elif isinstance(metadata, ModelVersionRecord):
        return {
            "model_type": metadata.model_role or metadata.name,
            "feature_names": [],
            "train_seasons": list(metadata.training_seasons),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": metadata.git_commit or "test",
            "best_params": metadata.settings,
            "metrics": metadata.dev_scores,
            "model_version_id": metadata.model_version_id,
            "is_active": metadata.is_active,
        }
    elif isinstance(metadata, dict):
        mtype = metadata.get("model_type")
        if mtype is None:
            # Fallback to model_role or name if provided
            mtype = metadata.get("model_role") or metadata.get("name")
        if mtype not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{mtype}'. Expected one of: {sorted(VALID_MODEL_TYPES)}"
            )

        created_at = metadata.get("created_at")
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()

        meta_dict = {
            "model_type": str(mtype),
            "feature_names": list(metadata.get("feature_names", [])),
            "train_seasons": [int(s) for s in metadata.get("train_seasons", [])],
            "created_at": str(created_at),
            "git_commit": str(metadata.get("git_commit", "test")),
            "best_params": dict(metadata.get("best_params", {})),
            "metrics": dict(metadata.get("metrics", {})),
        }
        if "model_version_id" in metadata:
            meta_dict["model_version_id"] = metadata["model_version_id"]
        if "p99_9_threshold" in metadata and metadata["p99_9_threshold"] is not None:
            meta_dict["p99_9_threshold"] = float(metadata["p99_9_threshold"])
        return meta_dict
    else:
        raise TypeError(f"Unsupported metadata type: {type(metadata)}")


def save_model(
    model: Any,
    metadata: Union[ModelMetadata, ModelVersionRecord, Dict[str, Any]],
    path: Union[str, Path],
) -> Path:
    """Save model and metadata using standard XGBoost JSON format.

    Args:
        model: Trained XGBoost model or composite model with save_model method.
        metadata: Model metadata record or dictionary.
        path: Target file path (ending in .json) or directory path.

    Returns:
        Path to saved model file.
    """
    path = Path(path)
    if path.suffix == ".json":
        model_file = path
        target_dir = path.parent
        meta_file = target_dir / "metadata.json"
    else:
        target_dir = path
        model_file = target_dir / "model.json"
        meta_file = target_dir / "metadata.json"

    target_dir.mkdir(parents=True, exist_ok=True)
    meta_dict = _normalize_metadata(metadata)
    if meta_dict.get("p99_9_threshold") is None and hasattr(model, "p99_9_threshold") and model.p99_9_threshold is not None:
        meta_dict["p99_9_threshold"] = float(model.p99_9_threshold)

    # Save XGBoost model to JSON
    if hasattr(model, "save_model"):
        model.save_model(str(model_file))
    else:
        raise TypeError(
            f"Object of type {type(model)} does not implement 'save_model'. Standard XGBoost models required."
        )

    # Save metadata JSON
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return model_file


def load_model(path: Union[str, Path]) -> LoadedModel:
    """Load model and metadata from standard XGBoost JSON format.

    Returns:
        LoadedModel: Can be unpacked as (model, metadata) or used directly.
    """
    path = Path(path)
    if path.is_dir():
        model_file = path / "model.json"
        meta_file = path / "metadata.json"
    else:
        model_file = path
        meta_file = path.parent / "metadata.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found at: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    model_type = meta_dict.get("model_type", "")
    if model_type in ("prob_15_6", "prob_64_5", "prob_115_6", "prob_115_6_cond"):
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()

    model.load_model(str(model_file))
    model.metadata = meta_dict
    if "p99_9_threshold" in meta_dict:
        model.p99_9_threshold = meta_dict["p99_9_threshold"]

    return LoadedModel(model, meta_dict)


def save_range_models(
    range_models: Any,
    output_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save composite QuantileRangeModels (q10, q50, q90) to directory."""
    from ml.feature_contracts import B3_FEATURES

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_meta = metadata or {}
    train_seasons = base_meta.get("train_seasons", [])
    git_commit = base_meta.get("git_commit", "test")

    for alpha, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        if alpha in range_models.models:
            sub_dir = out_path / name
            sub_meta = {
                "model_type": name,
                "feature_names": B3_FEATURES,
                "train_seasons": train_seasons,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit,
                "best_params": base_meta.get("best_params", {}).get(name, {}),
                "metrics": base_meta.get("metrics", {}).get(name, {}),
            }
            save_model(range_models.models[alpha], sub_meta, sub_dir)

    top_meta = {
        "model_type": "range",
        "feature_names": B3_FEATURES,
        "train_seasons": train_seasons,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "best_params": base_meta.get("best_params", {}),
        "metrics": base_meta.get("metrics", {}),
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(top_meta, f, indent=2)

    return out_path


def load_range_models(input_dir: Union[str, Path]) -> Any:
    """Load composite QuantileRangeModels from directory."""
    from probability.range_models import QuantileRangeModels

    in_path = Path(input_dir)
    models: Dict[float, Any] = {}

    for alpha, name in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        sub_dir = in_path / name
        if sub_dir.exists():
            loaded_m, _ = load_model(sub_dir)
            models[alpha] = loaded_m
        elif (in_path / f"{name}.json").exists():
            loaded_m, _ = load_model(in_path / f"{name}.json")
            models[alpha] = loaded_m

    return QuantileRangeModels(models=models)


def save_probability_models(
    prob_models: Any,
    output_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save composite RainfallProbabilityModels to directory."""
    from ml.feature_contracts import B3_FEATURES

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_meta = metadata or {}
    train_seasons = base_meta.get("train_seasons", [])
    git_commit = base_meta.get("git_commit", "test")

    # Save individual fitted models
    for key, model in prob_models.models.items():
        sub_name = f"prob_{str(key).replace('.', '_')}"
        sub_dir = out_path / sub_name
        sub_meta = {
            "model_type": sub_name,
            "feature_names": B3_FEATURES,
            "train_seasons": train_seasons,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "best_params": base_meta.get("best_params", {}).get(key, {}),
            "metrics": base_meta.get("metrics", {}).get(key, {}),
        }
        save_model(model, sub_meta, sub_dir)

    # Save top-level metadata with availability
    top_meta = {
        "model_type": "probability",
        "feature_names": B3_FEATURES,
        "train_seasons": train_seasons,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "availability": {
            "has_15_6": prob_models.availability.has_15_6 if prob_models.availability else True,
            "has_64_5": prob_models.availability.has_64_5 if prob_models.availability else False,
            "has_115_6": prob_models.availability.has_115_6 if prob_models.availability else False,
            "chained_115_6": prob_models.availability.chained_115_6 if prob_models.availability else False,
            "message": prob_models.availability.message if prob_models.availability else None,
        } if hasattr(prob_models, "availability") and prob_models.availability else None,
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(top_meta, f, indent=2)

    return out_path


def load_probability_models(input_dir: Union[str, Path]) -> Any:
    """Load composite RainfallProbabilityModels from directory."""
    from probability.classifiers import RainfallProbabilityModels, ProbabilityModelAvailability

    in_path = Path(input_dir)
    models: Dict[str, Any] = {}

    for key in ["15.6", "64.5", "115.6", "115.6_cond"]:
        sub_name = f"prob_{key.replace('.', '_')}"
        sub_dir = in_path / sub_name
        if sub_dir.exists():
            loaded_m, _ = load_model(sub_dir)
            models[key] = loaded_m

    # Load availability if exists
    availability = None
    meta_path = in_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            top_meta = json.load(f)
        avail_dict = top_meta.get("availability")
        if avail_dict:
            availability = ProbabilityModelAvailability(
                has_15_6=avail_dict.get("has_15_6", True),
                has_64_5=avail_dict.get("has_64_5", False),
                has_115_6=avail_dict.get("has_115_6", False),
                chained_115_6=avail_dict.get("chained_115_6", False),
                message=avail_dict.get("message"),
            )

    return RainfallProbabilityModels(models=models, availability=availability)


# Backward-compatible aliases matching existing signatures
def save_model_artifact(
    model: Any,
    metadata: Union[ModelMetadata, ModelVersionRecord, Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Save model artifact and corresponding metadata JSON (PRD §12.4, §20.2)."""
    return save_model(model, metadata, output_dir)


def load_model_artifact(model_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load model artifact and its metadata record (PRD §12.4, §20.2)."""
    loaded = load_model(model_dir)
    return loaded.model, loaded.metadata
