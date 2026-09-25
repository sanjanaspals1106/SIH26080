"""Fold-safe climatology: `clim_provider` recomputes clim_mean / clim_p95 from each fold's training seasons only."""

import numpy as np
import pandas as pd
import pytest
from tests.synthetic_fixtures import generate_synthetic_m3_data

from ml.fold_climatology import memoize_provider, with_fold_climatology
from ml.orchestration import run_development_oof_pipeline
from ml.training.hyperparameter_search import evaluate_settings_search

FAST = {"n_estimators": 2, "max_depth": 2, "learning_rate": 0.05}


class Provider:
    """A climatology whose value encodes WHICH seasons it was made from, and that records every request."""

    def __init__(self, cells):
        self.cells, self.requests = np.asarray(cells), []

    def __call__(self, seasons):
        self.requests.append(list(seasons))
        code = float(sum(seasons))  # e.g. seasons [2019] -> 2019.0, [2018, 2019] -> 4037.0
        return pd.DataFrame({"cell_id": self.cells, "clim_mean": code, "clim_p95": code + 0.5})


def test_none_changes_nothing():
    df = generate_synthetic_m3_data(seasons=[2018, 2019], dates_per_season=1)
    a, b = df[df["season"] == 2018], df[df["season"] == 2019]
    out_a, out_b = with_fold_climatology(a, b, None)
    assert out_a is a and out_b is b


def test_train_and_heldout_rows_get_the_climatology_of_the_training_seasons_only():
    df = generate_synthetic_m3_data(seasons=[2018, 2019, 2020], dates_per_season=1)
    train, held = df[df["season"] != 2020], df[df["season"] == 2020]
    p = Provider(df["cell_id"].unique())
    new_train, new_held = with_fold_climatology(train, held, p)
    assert p.requests == [[2018, 2019]]  # the held-out season was never offered to the provider
    assert (new_train["clim_mean"] == 2018 + 2019).all() and (new_held["clim_mean"] == 2018 + 2019).all()
    assert (new_held["clim_p95"] == 2018 + 2019 + 0.5).all()
    assert (train["clim_mean"] != 2018 + 2019).any()  # the inputs were copied, not edited
    assert list(new_held.index) == list(held.index) and new_held["obs_mm"].equals(held["obs_mm"])


def test_a_leaky_fold_is_refused():
    df = generate_synthetic_m3_data(seasons=[2018, 2019], dates_per_season=1)
    with pytest.raises(ValueError, match="held-out season is in the training fold"):
        with_fold_climatology(df, df[df["season"] == 2019], Provider(df["cell_id"].unique()))


def test_memoize_asks_once_per_season_set():
    p = Provider([1, 2])
    m = memoize_provider(p)
    m([2019, 2018]), m([2018, 2019]), m([2018])
    assert p.requests == [[2018, 2019], [2018]] and memoize_provider(None) is None


def test_oof_pipeline_uses_a_different_training_climatology_per_fold():
    seasons = [2018, 2019, 2020]
    df = generate_synthetic_m3_data(seasons=seasons, dates_per_season=2)
    p = Provider(df["cell_id"].unique())
    oof = run_development_oof_pipeline(
        dev_df=df, development_seasons=seasons, b2_params=FAST, b3_params=FAST, prob_params=FAST, range_params=FAST,
        clim_provider=p,
    )
    assert sorted(p.requests) == [[2018, 2019], [2018, 2020], [2019, 2020]]  # one per fold, never all three seasons
    assert set(oof["season"]) == set(seasons) and len(oof) == len(df)
    # without a provider the behaviour is exactly what it was
    plain = run_development_oof_pipeline(
        dev_df=df, development_seasons=seasons, b2_params=FAST, b3_params=FAST, prob_params=FAST, range_params=FAST,
    )
    assert list(plain.columns) == list(oof.columns) and len(plain) == len(oof)


def test_settings_search_uses_fold_climatology_too():
    seasons = [2018, 2019, 2020]
    df = generate_synthetic_m3_data(seasons=seasons, dates_per_season=2)
    p = Provider(df["cell_id"].unique())
    evaluate_settings_search(df, seasons, combinations=[FAST, {**FAST, "max_depth": 3}], model_type="B2",
                             n_validation_seasons=3, clim_provider=p)
    assert sorted(p.requests) == [[2018, 2019], [2018, 2020], [2019, 2020]]  # cached across the two settings
