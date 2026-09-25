"""Regression: `select_best_c_loso` must score probabilities against the right classes.

`PhaseModel.predict_proba` returns columns in the order ['active', 'normal', 'break']. sklearn's `log_loss` sorts
`labels` alphabetically (['active', 'break', 'normal']), so passing the columns unsorted swapped 'normal' and
'break': a near-perfect forecast scored about 3.0 instead of about 0.1, and the choice of C was made on that.
No model is trained here: `PhaseModel` is replaced by a stub whose probabilities are known.
"""

import numpy as np
import pytest

import regime_engine.phase.model as phase_model
from regime_engine.phase.model import CLASSES, select_best_c_loso


class StubPhaseModel:
    """C == 1: near-perfect probabilities (0.9 on the true class); any other C: uniform 1/3 probabilities.
    The true class of a row is encoded in X[:, 0] as its index in CLASSES."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        idx = X[:, 0].astype(int)
        if self.C != 1.0:
            return np.full((len(X), 3), 1.0 / 3.0)
        probs = np.full((len(X), 3), 0.05)
        probs[np.arange(len(X)), idx] = 0.90  # columns are in CLASSES order, like the real PhaseModel
        return probs


@pytest.fixture()
def data():
    """3 seasons x 30 days, every class in every season."""
    idx = np.tile(np.arange(3), 30)  # 0 active, 1 normal, 2 break
    X = np.column_stack([idx, np.arange(len(idx))]).astype(float)
    y = np.array([CLASSES[i] for i in idx])
    seasons = np.repeat([2021, 2022, 2023], 30)
    return X, y, seasons


def test_near_perfect_forecast_gets_a_low_loss(data, monkeypatch):
    losses = []
    real = phase_model.log_loss
    monkeypatch.setattr(phase_model, "log_loss", lambda *a, **k: losses.append(real(*a, **k)) or losses[-1])
    monkeypatch.setattr(phase_model, "PhaseModel", StubPhaseModel)
    X, y, seasons = data
    select_best_c_loso(X, y, seasons, c_grid=[1.0])
    assert len(losses) == 3  # one per held-out season
    assert np.allclose(losses, -np.log(0.90), atol=1e-6)  # 0.105, not ~3.0 (the swapped-column value)


def test_the_better_model_wins_the_c_selection(data, monkeypatch):
    monkeypatch.setattr(phase_model, "PhaseModel", StubPhaseModel)
    X, y, seasons = data
    # uniform probabilities cost ln 3 = 1.10; near-perfect ones cost 0.105. With the old column order the
    # "perfect" model looked worse (about 3.0) and the uniform one was chosen.
    assert select_best_c_loso(X, y, seasons, c_grid=[0.1, 1.0, 10.0]) == 1.0
    assert select_best_c_loso(X, y, seasons, c_grid=[10.0, 1.0, 0.1]) == 1.0  # not an order effect


def test_real_phase_model_on_separable_data_has_a_low_loss(monkeypatch):
    rng = np.random.default_rng(0)
    centres = {"active": 3.0, "normal": 0.0, "break": -3.0}
    y = np.array(list(centres) * 60)
    X = np.array([[centres[c]] for c in y]) + rng.normal(0, 0.3, size=(len(y), 1))
    seasons = np.repeat([2021, 2022, 2023], 60)
    losses = []
    real = phase_model.log_loss
    monkeypatch.setattr(phase_model, "log_loss", lambda *a, **k: losses.append(real(*a, **k)) or losses[-1])
    select_best_c_loso(X, y, seasons, c_grid=[1.0])
    assert max(losses) < 0.2
