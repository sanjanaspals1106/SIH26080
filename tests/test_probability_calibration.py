"""Unit tests for probability calibration and probability ordering constraints (PRD §13.3, §13.4)."""

import math
from pathlib import Path
import numpy as np
import pytest

from probability.calibration import ProbabilityCalibrator, get_isotonic_min_events
from probability.constraints import (
    enforce_probability_ordering,
    enforce_probability_ordering_dict,
)


def test_min_events_from_config():
    """Verify event threshold 200 is read from config/thresholds.yaml."""
    min_ev = get_isotonic_min_events()
    assert min_ev == 200


def test_ge_200_positives_selects_isotonic():
    """Test 1: >= 200 positive observed events selects Isotonic Regression."""
    rng = np.random.default_rng(42)
    n_pos = 200
    n_neg = 200

    y = np.array([1] * n_pos + [0] * n_neg)
    p = np.concatenate([rng.uniform(0.4, 0.9, n_pos), rng.uniform(0.1, 0.5, n_neg)])

    calibrator = ProbabilityCalibrator()
    calibrator.fit(p, y, evaluation_set="development", prediction_source="oof")

    assert calibrator.method_used == "isotonic"
    assert calibrator.is_fitted is True
    assert calibrator.n_positives_ == 200


def test_lt_200_positives_selects_platt():
    """Test 2: < 200 positive observed events selects Platt scaling."""
    rng = np.random.default_rng(42)
    n_pos = 199
    n_neg = 200

    y = np.array([1] * n_pos + [0] * n_neg)
    p = np.concatenate([rng.uniform(0.4, 0.9, n_pos), rng.uniform(0.1, 0.5, n_neg)])

    calibrator = ProbabilityCalibrator()
    calibrator.fit(p, y, evaluation_set="development", prediction_source="oof")

    assert calibrator.method_used == "platt"
    assert calibrator.is_fitted is True
    assert calibrator.n_positives_ == 199


def test_holdout_input_rejected_during_fit():
    """Test 3: Holdout data is strictly rejected during fit per PRD §10.4, §10.6 L4."""
    p = np.array([0.2, 0.8])
    y = np.array([0, 1])

    calibrator = ProbabilityCalibrator()

    # String input
    with pytest.raises(ValueError, match="Holdout data cannot be used"):
        calibrator.fit(p, y, evaluation_set="holdout", prediction_source="oof")

    # Array input with holdout
    with pytest.raises(ValueError, match="Holdout data detected"):
        calibrator.fit(
            p, y, evaluation_set=["development", "holdout"], prediction_source=["oof", "oof"]
        )


def test_non_oof_development_input_rejected():
    """Test 4: Non-OOF development data is strictly rejected per PRD §10.4, §13.4."""
    p = np.array([0.2, 0.8])
    y = np.array([0, 1])

    calibrator = ProbabilityCalibrator()

    # prediction_source = 'train'
    with pytest.raises(ValueError, match="prediction_source must be 'oof'"):
        calibrator.fit(p, y, evaluation_set="development", prediction_source="train")

    # regime_source = 'final'
    with pytest.raises(ValueError, match="regime_source must be 'oof'"):
        calibrator.fit(
            p,
            y,
            evaluation_set="development",
            prediction_source="oof",
            regime_source="final",
        )


def test_calibrated_outputs_stay_in_unit_interval():
    """Test 5: Calibrated probabilities stay strictly within [0.0, 1.0]."""
    rng = np.random.default_rng(42)
    y = np.array([1] * 50 + [0] * 50)
    p = np.concatenate([rng.uniform(0.5, 0.9, 50), rng.uniform(0.1, 0.4, 50)])

    calibrator = ProbabilityCalibrator()
    calibrator.fit(p, y, evaluation_set="development", prediction_source="oof")

    # Extreme test inputs
    test_p = np.array([-10.0, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 100.0])
    preds = calibrator.predict(test_p)

    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_ordering_constraint_fixes_crossing_probabilities():
    """Test 6: Monotonicity constraint fixes crossing probabilities P(115.6) <= P(64.5) <= P(15.6)."""
    # Scalar crossing: 15.6 is 0.4, 64.5 is 0.6, 115.6 is 0.8
    p15, p64, p115 = enforce_probability_ordering(0.4, 0.6, 0.8)
    assert math.isclose(p15, 0.4)
    assert math.isclose(p64, 0.4)
    assert math.isclose(p115, 0.4)
    assert p115 <= p64 <= p15

    # Array crossing
    arr15 = np.array([0.5, 0.3])
    arr64 = np.array([0.7, 0.4])
    arr115 = np.array([0.9, 0.5])

    o15, o64, o115 = enforce_probability_ordering(arr15, arr64, arr115)
    assert np.all(o115 <= o64)
    assert np.all(o64 <= o15)
    assert np.allclose(o15, [0.5, 0.3])
    assert np.allclose(o64, [0.5, 0.3])
    assert np.allclose(o115, [0.5, 0.3])


def test_already_valid_probabilities_remain_unchanged():
    """Test 7: Already valid probabilities remain completely unchanged."""
    p15, p64, p115 = enforce_probability_ordering(0.8, 0.4, 0.1)
    assert math.isclose(p15, 0.8)
    assert math.isclose(p64, 0.4)
    assert math.isclose(p115, 0.1)


def test_null_unavailable_probability_remains_null():
    """Test 8: Preserves null/unavailable thresholds without inventing probabilities."""
    # Only 15.6 available
    s15, s64, s115 = enforce_probability_ordering(0.7, None, None)
    assert math.isclose(s15, 0.7)
    assert s64 is None
    assert s115 is None

    # 15.6 and 64.5 available, 115.6 unavailable
    s15, s64, s115 = enforce_probability_ordering(0.5, 0.8, None)
    assert math.isclose(s15, 0.5)
    assert math.isclose(s64, 0.5)  # fixed crossing
    assert s115 is None

    # All None
    n15, n64, n115 = enforce_probability_ordering(None, None, None)
    assert n15 is None
    assert n64 is None
    assert n115 is None

    # Arrays with NaNs
    arr_a = np.array([0.5, np.nan])
    arr_b = np.array([0.7, 0.3])
    arr_c = np.array([0.9, np.nan])
    out_a, out_b, out_c = enforce_probability_ordering(arr_a, arr_b, arr_c)
    assert math.isclose(out_a[0], 0.5)
    assert math.isclose(out_b[0], 0.5)
    assert math.isclose(out_c[0], 0.5)
    assert np.isnan(out_a[1])
    assert math.isclose(out_b[1], 0.3)
    assert np.isnan(out_c[1])

    # Dict helper test
    d = {"p_ge_15_6": 0.5, "p_ge_64_5": 0.8, "p_ge_115_6": None}
    d_out = enforce_probability_ordering_dict(d)
    assert math.isclose(d_out["p_ge_15_6"], 0.5)
    assert math.isclose(d_out["p_ge_64_5"], 0.5)
    assert d_out["p_ge_115_6"] is None


def test_save_load_gives_same_predictions(tmp_path):
    """Test 9: Saving and loading calibrator gives identical predictions."""
    rng = np.random.default_rng(42)
    y = np.array([1] * 250 + [0] * 250)
    p = np.concatenate([rng.uniform(0.5, 0.9, 250), rng.uniform(0.1, 0.5, 250)])

    calibrator = ProbabilityCalibrator(threshold_mm=64.5)
    calibrator.fit(p, y, evaluation_set="development", prediction_source="oof")

    model_file = tmp_path / "calibrator_64_5.pkl"
    calibrator.save(model_file)
    assert model_file.is_file()

    loaded = ProbabilityCalibrator.load(model_file)
    assert loaded.is_fitted is True
    assert loaded.threshold_mm == 64.5
    assert loaded.method_used == "isotonic"

    test_p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    orig_preds = calibrator.predict(test_p)
    loaded_preds = loaded.predict(test_p)

    assert np.array_equal(orig_preds, loaded_preds)
