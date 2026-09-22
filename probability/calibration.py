"""Probability calibration module (PRD §13.4, §10.4, §10.6).

Rules:
- Calibrators are fitted strictly on pooled out-of-fold (OOF) predictions of development seasons.
- No holdout data may influence calibrator fitting (PRD §10.6 L4/L6).
- Rejects development rows unless prediction_source/regime_source is 'oof'.
- If pooled positive events >= 200: Isotonic Regression.
- Otherwise: Platt scaling (Logistic Regression on raw probabilities).
- Outputs strictly clipped to [0.0, 1.0].
"""

from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import yaml


def get_isotonic_min_events(config_path: Optional[Union[str, Path]] = None) -> int:
    """Reads the minimum positive events threshold for isotonic regression from config/thresholds.yaml."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        min_ev = data.get("calibration", {}).get("isotonic_min_events")
        if min_ev is not None:
            return int(min_ev)
    return 200


def get_protocol_seed_from_config(config_path: Optional[Union[str, Path]] = None) -> int:
    """Reads project random seed from config/protocol.yaml."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "protocol.yaml"

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        seed = data.get("random_seed")
        if seed is not None:
            return int(seed)
    return 42



class ProbabilityCalibrator:
    """Calibrator for raw rainfall probability forecasts (PRD §13.4)."""

    def __init__(
        self,
        threshold_mm: Optional[float] = None,
        method: Optional[str] = None,
        min_events_for_isotonic: Optional[int] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initializes calibrator.

        Args:
            threshold_mm: Rainfall threshold in mm (e.g. 15.6, 64.5, 115.6).
            method: Optional override: 'isotonic' or 'platt'. If None, chosen automatically.
            min_events_for_isotonic: Minimum positive events needed to use Isotonic Regression.
            config_path: Path to thresholds.yaml.
        """
        self.threshold_mm = threshold_mm
        self._config_path = config_path
        self.method_override = method.lower() if method else None
        if min_events_for_isotonic is not None:
            self.min_events_for_isotonic = int(min_events_for_isotonic)
        else:
            self.min_events_for_isotonic = get_isotonic_min_events(config_path)

        self.method_used: Optional[str] = None
        self._model: Optional[Union[IsotonicRegression, LogisticRegression]] = None
        self.n_samples_: Optional[int] = None
        self.n_positives_: Optional[int] = None
        self.is_fitted: bool = False

    def fit(
        self,
        probabilities: Sequence[float] | np.ndarray,
        observed_binary: Sequence[Union[int, float, bool]] | np.ndarray,
        evaluation_set: Union[str, Sequence[str]] = "development",
        prediction_source: Union[str, Sequence[str]] = "oof",
        regime_source: Optional[Union[str, Sequence[str]]] = "oof",
    ) -> "ProbabilityCalibrator":
        """Fits calibrator on pooled OOF predictions of development seasons.

        Rejects holdout data and non-OOF predictions per PRD §10.4, §10.6, §13.4.

        Args:
            probabilities: Raw forecast probabilities in [0, 1].
            observed_binary: Binary observed event targets (1 = event, 0 = no event).
            evaluation_set: Must be strictly 'development'.
            prediction_source: Must be strictly 'oof'.
            regime_source: Must be 'oof' if provided.

        Returns:
            Fitted ProbabilityCalibrator instance.
        """
        # Rule check 1: Reject holdout data (PRD §10.4, §10.6 L4/L6)
        if isinstance(evaluation_set, str):
            if evaluation_set.strip().lower() == "holdout":
                raise ValueError(
                    "Holdout data cannot be used for calibrator fitting per PRD §10.4, §10.6 L4."
                )
            if evaluation_set.strip().lower() != "development":
                raise ValueError(
                    f"Calibrator must be fitted only on 'development' set, got: {evaluation_set!r}."
                )
        else:
            arr_eval = np.char.lower(np.asarray(evaluation_set, dtype=str))
            if np.any(arr_eval == "holdout"):
                raise ValueError(
                    "Holdout data detected in evaluation_set; forbidden per PRD §10.4, §10.6 L4."
                )
            if np.any(arr_eval != "development"):
                raise ValueError(
                    "All rows for calibrator fitting must belong to the 'development' evaluation set."
                )

        # Rule check 2: Reject non-OOF predictions (PRD §10.4, §13.4)
        if isinstance(prediction_source, str):
            if prediction_source.strip().lower() != "oof":
                raise ValueError(
                    f"prediction_source must be 'oof', got {prediction_source!r}. "
                    "PRD §13.4 requires calibrators to be fitted strictly on out-of-fold predictions."
                )
        else:
            arr_pred = np.char.lower(np.asarray(prediction_source, dtype=str))
            if np.any(arr_pred != "oof"):
                raise ValueError("All rows for calibration must have prediction_source='oof'.")

        # Rule check 3: Reject non-OOF regime source
        if regime_source is not None:
            if isinstance(regime_source, str):
                if regime_source.strip().lower() != "oof":
                    raise ValueError(
                        f"regime_source must be 'oof', got {regime_source!r}. "
                        "PRD §10.3 / §10.6 requires out-of-fold regime probabilities."
                    )
            else:
                arr_reg = np.char.lower(np.asarray(regime_source, dtype=str))
                if np.any(arr_reg != "oof"):
                    raise ValueError("All rows for calibration must have regime_source='oof'.")

        p = np.asarray(probabilities, dtype=float).ravel()
        y = np.asarray(observed_binary, dtype=float).ravel()

        if p.shape != y.shape:
            raise ValueError(f"Shape mismatch: probabilities {p.shape}, observed {y.shape}.")

        # Exclude NaNs
        valid_mask = ~(np.isnan(p) | np.isnan(y))
        p_clean = p[valid_mask]
        y_clean = y[valid_mask]

        n = len(p_clean)
        if n == 0:
            raise ValueError("Cannot fit calibrator: no valid non-NaN samples provided.")

        if not np.all(np.isin(y_clean, [0.0, 1.0])):
            raise ValueError("Observed binary targets must contain only 0 and 1.")

        n_pos = int(np.count_nonzero(y_clean == 1.0))

        # Select method per PRD §13.4
        if self.method_override:
            method = self.method_override
        else:
            if n_pos >= self.min_events_for_isotonic:
                method = "isotonic"
            else:
                method = "platt"

        if method == "isotonic":
            # Isotonic Regression bounded in [0.0, 1.0]
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(p_clean, y_clean)
        elif method == "platt":
            # Platt scaling: Logistic regression on raw scores
            seed = get_protocol_seed_from_config(self._config_path)
            model = LogisticRegression(solver="lbfgs", random_state=seed)
            model.fit(p_clean.reshape(-1, 1), y_clean)
        else:
            raise ValueError(f"Unsupported calibration method: {method}")

        self._model = model
        self.method_used = method
        self.n_samples_ = n
        self.n_positives_ = n_pos
        self.is_fitted = True

        return self

    def predict(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        """Applies fitted calibrator to raw probabilities, returning calibrated probabilities in [0, 1]."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("ProbabilityCalibrator is not fitted. Call fit() before predict().")

        p = np.asarray(probabilities, dtype=float)
        orig_shape = p.shape
        p_flat = p.ravel()

        valid_mask = ~np.isnan(p_flat)
        out = np.full_like(p_flat, np.nan, dtype=float)

        if np.any(valid_mask):
            p_valid = p_flat[valid_mask]
            if self.method_used == "isotonic":
                preds = self._model.predict(p_valid)
            elif self.method_used == "platt":
                preds = self._model.predict_proba(p_valid.reshape(-1, 1))[:, 1]
            else:
                raise RuntimeError(f"Invalid fitted method: {self.method_used}")

            out[valid_mask] = np.clip(preds, 0.0, 1.0)

        return out.reshape(orig_shape)

    def save(self, path: Union[str, Path]) -> Path:
        """Saves fitted calibrator and metadata to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold_mm": self.threshold_mm,
            "method_used": self.method_used,
            "min_events_for_isotonic": self.min_events_for_isotonic,
            "n_samples": self.n_samples_,
            "n_positives": self.n_positives_,
            "is_fitted": self.is_fitted,
            "model": self._model,
        }
        with open(target, "wb") as f:
            pickle.dump(payload, f)
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProbabilityCalibrator":
        """Loads a fitted calibrator from disk."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Calibrator file not found: {target}")
        with open(target, "rb") as f:
            payload = pickle.load(f)

        calibrator = cls(
            threshold_mm=payload.get("threshold_mm"),
            min_events_for_isotonic=payload.get("min_events_for_isotonic"),
        )
        calibrator.method_used = payload.get("method_used")
        calibrator.n_samples_ = payload.get("n_samples")
        calibrator.n_positives_ = payload.get("n_positives")
        calibrator.is_fitted = payload.get("is_fitted", False)
        calibrator._model = payload.get("model")
        return calibrator
