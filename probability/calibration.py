"""Probability calibration and serving integration (PRD §10.4, §13.3, §13.4).

Responsibilities:
1. M4 fits calibrators strictly on pooled out-of-fold development predictions.
2. Holdout data must never be used to fit calibrators.
3. Use isotonic regression when enough positive events exist, otherwise Platt scaling.
4. M3 may apply already-fitted calibrators during serving, but must never fit them.
5. Preserve unavailable thresholds as NaN.
6. Clip calibrated probabilities to [0, 1].
7. Re-enforce:
   P(>=115.6) <= P(>=64.5) <= P(>=15.6)
   after calibration.
"""

from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import yaml

from probability.classifiers import enforce_probability_monotonicity


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_isotonic_min_events(
    config_path: Optional[Union[str, Path]] = None,
) -> int:
    """Read the isotonic-regression event threshold from thresholds config."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        min_events = data.get("calibration", {}).get("isotonic_min_events")
        if min_events is not None:
            return int(min_events)

    return 200


def get_protocol_seed_from_config(
    config_path: Optional[Union[str, Path]] = None,
) -> int:
    """Read the project random seed from protocol config."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "protocol.yaml"

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        seed = data.get("random_seed")
        if seed is not None:
            return int(seed)

    return 42


# ---------------------------------------------------------------------------
# M4 calibrator fitting
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """Calibrator for raw rainfall probability forecasts (PRD §13.4)."""

    def __init__(
        self,
        threshold_mm: Optional[float] = None,
        method: Optional[str] = None,
        min_events_for_isotonic: Optional[int] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        self.threshold_mm = threshold_mm
        self._config_path = config_path
        self.method_override = method.lower() if method else None

        if min_events_for_isotonic is not None:
            self.min_events_for_isotonic = int(min_events_for_isotonic)
        else:
            self.min_events_for_isotonic = get_isotonic_min_events(config_path)

        self.method_used: Optional[str] = None
        self._model: Optional[
            Union[IsotonicRegression, LogisticRegression]
        ] = None
        self.n_samples_: Optional[int] = None
        self.n_positives_: Optional[int] = None
        self.is_fitted = False

    def fit(
        self,
        probabilities: Sequence[float] | np.ndarray,
        observed_binary: Sequence[Union[int, float, bool]] | np.ndarray,
        evaluation_set: Union[str, Sequence[str]] = "development",
        prediction_source: Union[str, Sequence[str]] = "oof",
        regime_source: Optional[Union[str, Sequence[str]]] = "oof",
    ) -> "ProbabilityCalibrator":
        """Fit strictly on pooled OOF development predictions."""

        # Evaluation-set protection.
        if isinstance(evaluation_set, str):
            eval_value = evaluation_set.strip().lower()

            if eval_value == "holdout":
                raise ValueError(
                    "Holdout data cannot be used for calibrator fitting."
                )

            if eval_value != "development":
                raise ValueError(
                    "Calibrator fitting requires evaluation_set='development', "
                    f"got {evaluation_set!r}."
                )
        else:
            arr_eval = np.char.lower(np.asarray(evaluation_set, dtype=str))

            if np.any(arr_eval == "holdout"):
                raise ValueError(
                    "Holdout data detected in calibrator fitting input."
                )

            if np.any(arr_eval != "development"):
                raise ValueError(
                    "All calibrator-fitting rows must be development rows."
                )

        # Prediction must be out-of-fold.
        if isinstance(prediction_source, str):
            if prediction_source.strip().lower() != "oof":
                raise ValueError(
                    "prediction_source must be 'oof' for calibrator fitting."
                )
        else:
            arr_pred = np.char.lower(np.asarray(prediction_source, dtype=str))

            if np.any(arr_pred != "oof"):
                raise ValueError(
                    "All calibrator-fitting rows must have "
                    "prediction_source='oof'."
                )

        # Regime features used by B3 must also be OOF.
        if regime_source is not None:
            if isinstance(regime_source, str):
                if regime_source.strip().lower() != "oof":
                    raise ValueError(
                        "regime_source must be 'oof' for calibrator fitting."
                    )
            else:
                arr_reg = np.char.lower(np.asarray(regime_source, dtype=str))

                if np.any(arr_reg != "oof"):
                    raise ValueError(
                        "All calibrator-fitting rows must have regime_source='oof'."
                    )

        p = np.asarray(probabilities, dtype=float).ravel()
        y = np.asarray(observed_binary, dtype=float).ravel()

        if p.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: probabilities {p.shape}, observed {y.shape}."
            )

        valid_mask = np.isfinite(p) & np.isfinite(y)
        p_clean = p[valid_mask]
        y_clean = y[valid_mask]

        if len(p_clean) == 0:
            raise ValueError(
                "Cannot fit calibrator: no valid samples were provided."
            )

        if not np.all(np.isin(y_clean, [0.0, 1.0])):
            raise ValueError(
                "Observed binary targets must contain only 0 and 1."
            )

        n_positive = int(np.count_nonzero(y_clean == 1.0))

        if self.method_override:
            method = self.method_override
        elif n_positive >= self.min_events_for_isotonic:
            method = "isotonic"
        else:
            method = "platt"

        if method == "isotonic":
            model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            model.fit(p_clean, y_clean)

        elif method == "platt":
            seed = get_protocol_seed_from_config(self._config_path)

            model = LogisticRegression(
                solver="lbfgs",
                random_state=seed,
            )
            model.fit(p_clean.reshape(-1, 1), y_clean)

        else:
            raise ValueError(
                f"Unsupported calibration method: {method!r}"
            )

        self._model = model
        self.method_used = method
        self.n_samples_ = len(p_clean)
        self.n_positives_ = n_positive
        self.is_fitted = True

        return self

    def predict(
        self,
        probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Apply the fitted calibrator while preserving NaNs."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError(
                "ProbabilityCalibrator is not fitted. Call fit() first."
            )

        p = np.asarray(probabilities, dtype=float)
        original_shape = p.shape
        flat = p.ravel()

        valid_mask = np.isfinite(flat)
        output = np.full_like(flat, np.nan, dtype=float)

        if np.any(valid_mask):
            values = flat[valid_mask]

            if self.method_used == "isotonic":
                preds = self._model.predict(values)

            elif self.method_used == "platt":
                preds = self._model.predict_proba(
                    values.reshape(-1, 1)
                )[:, 1]

            else:
                raise RuntimeError(
                    f"Invalid fitted calibration method: {self.method_used!r}"
                )

            output[valid_mask] = np.clip(
                np.asarray(preds, dtype=float),
                0.0,
                1.0,
            )

        return output.reshape(original_shape)

    def save(self, path: Union[str, Path]) -> Path:
        """Persist fitted calibrator and metadata."""
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
    def load(
        cls,
        path: Union[str, Path],
    ) -> "ProbabilityCalibrator":
        """Load a previously saved calibrator."""
        target = Path(path)

        if not target.is_file():
            raise FileNotFoundError(
                f"Calibrator file not found: {target}"
            )

        with open(target, "rb") as f:
            payload = pickle.load(f)

        calibrator = cls(
            threshold_mm=payload.get("threshold_mm"),
            min_events_for_isotonic=payload.get(
                "min_events_for_isotonic"
            ),
        )

        calibrator.method_used = payload.get("method_used")
        calibrator.n_samples_ = payload.get("n_samples")
        calibrator.n_positives_ = payload.get("n_positives")
        calibrator.is_fitted = payload.get("is_fitted", False)
        calibrator._model = payload.get("model")

        return calibrator


# ---------------------------------------------------------------------------
# M3/M4 integration seam: applying already-fitted calibrators
# ---------------------------------------------------------------------------

def apply_single_calibrator(
    calibrator: Any,
    raw_p: np.ndarray,
) -> np.ndarray:
    """Apply an already-fitted calibrator without fitting or modifying it."""
    raw_arr = np.asarray(raw_p, dtype=float)

    if raw_arr.size == 0:
        return raw_arr.copy()

    flat = raw_arr.ravel()
    valid_mask = np.isfinite(flat)

    if not np.any(valid_mask):
        return raw_arr.copy()

    valid_inputs = flat[valid_mask]

    if hasattr(calibrator, "predict_proba"):
        try:
            result = calibrator.predict_proba(
                valid_inputs.reshape(-1, 1)
            )
        except Exception:
            result = calibrator.predict_proba(valid_inputs)

        result = np.asarray(result)

        if result.ndim == 2 and result.shape[1] >= 2:
            output_values = result[:, 1]
        else:
            output_values = result.ravel()

    elif hasattr(calibrator, "predict"):
        try:
            output_values = calibrator.predict(valid_inputs)
        except Exception:
            output_values = calibrator.predict(
                valid_inputs.reshape(-1, 1)
            )

    elif hasattr(calibrator, "transform"):
        try:
            output_values = calibrator.transform(valid_inputs)
        except Exception:
            output_values = calibrator.transform(
                valid_inputs.reshape(-1, 1)
            )

    elif callable(calibrator):
        output_values = calibrator(valid_inputs)

    else:
        raise TypeError(
            f"Unsupported calibrator object: {type(calibrator)!r}"
        )

    output_values = np.clip(
        np.asarray(output_values, dtype=float).ravel(),
        0.0,
        1.0,
    )

    if len(output_values) != len(valid_inputs):
        raise ValueError(
            "Calibrator returned a different number of predictions "
            "than inputs."
        )

    calibrated = np.array(flat, dtype=float, copy=True)
    calibrated[valid_mask] = output_values

    return calibrated.reshape(raw_arr.shape)


def normalize_calibrator_key(key: Any) -> Optional[float]:
    """Normalize supported calibrator dictionary keys to a rain threshold."""
    if isinstance(key, (int, float)):
        value = float(key)

        for threshold in (15.6, 64.5, 115.6):
            if np.isclose(value, threshold):
                return threshold

    key_str = str(key).strip().lower()

    if "115" in key_str:
        return 115.6
    if "64" in key_str:
        return 64.5
    if "15" in key_str:
        return 15.6

    return None


def apply_probability_calibrators(
    raw_prob_df: pd.DataFrame,
    calibrators: Optional[Dict[Any, Any]] = None,
    allow_uncalibrated: bool = False,
) -> pd.DataFrame:
    """Apply externally fitted M4 calibrators to probability predictions.

    M3 may call this function during serving/inference, but must never fit the
    calibrators here.
    """
    has_calibrators = bool(calibrators)

    if not has_calibrators:
        if not allow_uncalibrated:
            raise ValueError(
                "Final/holdout serving requires pre-fitted M4 probability calibrators. "
                "Set allow_uncalibrated=True only for explicit development use."
            )

        output = pd.DataFrame(index=raw_prob_df.index)

        for threshold in (15.6, 64.5, 115.6):
            col = f"p_ge_{str(threshold).replace('.', '_')}"
            raw_col = f"uncalibrated_{col}"

            if col in raw_prob_df.columns:
                output[col] = np.clip(
                    raw_prob_df[col].to_numpy(dtype=float),
                    0.0,
                    1.0,
                )
            elif raw_col in raw_prob_df.columns:
                output[col] = np.clip(
                    raw_prob_df[raw_col].to_numpy(dtype=float),
                    0.0,
                    1.0,
                )
            else:
                output[col] = np.nan

        # Even explicit uncalibrated development output should respect the
        # probability nesting constraint.
        p15 = output["p_ge_15_6"].to_numpy(dtype=float)
        p64 = output["p_ge_64_5"].to_numpy(dtype=float)
        p115 = output["p_ge_115_6"].to_numpy(dtype=float)

        p64_arg = None if np.all(np.isnan(p64)) else p64
        p115_arg = None if np.all(np.isnan(p115)) else p115

        p15, p64_corrected, p115_corrected = (
            enforce_probability_monotonicity(
                p15,
                p64_arg,
                p115_arg,
            )
        )

        output["p_ge_15_6"] = p15
        output["p_ge_64_5"] = (
            p64_corrected if p64_arg is not None else np.nan
        )
        output["p_ge_115_6"] = (
            p115_corrected if p115_arg is not None else np.nan
        )

        return output

    normalized_calibrators: Dict[float, Any] = {}

    for key, calibrator in calibrators.items():
        threshold = normalize_calibrator_key(key)

        if threshold is not None and calibrator is not None:
            normalized_calibrators[threshold] = calibrator

    n_rows = len(raw_prob_df)
    calibrated_columns: Dict[float, np.ndarray] = {}

    for threshold in (15.6, 64.5, 115.6):
        col = f"p_ge_{str(threshold).replace('.', '_')}"
        raw_col = f"uncalibrated_{col}"

        if col in raw_prob_df.columns:
            source_col = col
        elif raw_col in raw_prob_df.columns:
            source_col = raw_col
        else:
            source_col = None

        if (
            source_col is not None
            and not raw_prob_df[source_col].isna().all()
        ):
            raw_values = raw_prob_df[source_col].to_numpy(dtype=float)

            if threshold in normalized_calibrators:
                calibrated_values = apply_single_calibrator(
                    normalized_calibrators[threshold],
                    raw_values,
                )
            else:
                calibrated_values = np.clip(
                    raw_values,
                    0.0,
                    1.0,
                )

            calibrated_columns[threshold] = calibrated_values

        else:
            calibrated_columns[threshold] = np.full(
                n_rows,
                np.nan,
                dtype=float,
            )

    # Re-enforce nested exceedance probabilities after calibration.
    p15 = calibrated_columns[15.6]

    p64 = calibrated_columns[64.5]
    p64_arg = None if np.all(np.isnan(p64)) else p64

    p115 = calibrated_columns[115.6]
    p115_arg = None if np.all(np.isnan(p115)) else p115

    p15_corrected, p64_corrected, p115_corrected = (
        enforce_probability_monotonicity(
            p15,
            p64_arg,
            p115_arg,
        )
    )

    return pd.DataFrame(
        {
            "p_ge_15_6": p15_corrected,
            "p_ge_64_5": (
                p64_corrected if p64_arg is not None else np.nan
            ),
            "p_ge_115_6": (
                p115_corrected if p115_arg is not None else np.nan
            ),
        },
        index=raw_prob_df.index,
    )