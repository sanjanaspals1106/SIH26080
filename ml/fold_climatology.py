"""Fold-safe cell climatology (PRD 10.4: `clim_mean` / `clim_p95` are fitted on the training seasons of each fold).

The feature table holds one climatology per row, made from all development seasons. Inside a leave-one-season-out
fold that means the held-out season's own rain sits in its `clim_*` features. `clim_provider` fixes it without
touching the feature table: for every fold, both the training and the held-out rows get `clim_mean` / `clim_p95`
recomputed from the training seasons of that fold only.

`clim_provider(seasons) -> DataFrame[cell_id, clim_mean, clim_p95]`. In practice (needs the IMD years on disk; the
results are cached per season set by `get_climatology`):

    clim_provider = lambda seasons: get_climatology(seasons, config)

`None` (the default everywhere) leaves the columns as they are, so nothing changes unless it is passed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

ClimProvider = Callable[[Sequence[int]], pd.DataFrame]
CLIM_COLUMNS = ["clim_mean", "clim_p95"]


def memoize_provider(provider: ClimProvider | None) -> ClimProvider | None:
    """Ask the provider once per set of seasons, however many folds and settings ask for it."""
    if provider is None:
        return None
    memo: dict[tuple[int, ...], pd.DataFrame] = {}

    def cached(seasons: Sequence[int]) -> pd.DataFrame:
        key = tuple(sorted(int(s) for s in seasons))
        if key not in memo:
            memo[key] = provider(list(key))
        return memo[key]

    return cached


def with_fold_climatology(
    train_fold: pd.DataFrame, eval_fold: pd.DataFrame, provider: ClimProvider | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train_fold, eval_fold)` with `clim_mean` / `clim_p95` taken from the training seasons of the fold.

    Both frames use the same table (made only from the seasons present in `train_fold`); the held-out season is
    never part of it. Returns the inputs untouched if `provider` is None. Cells the provider does not know stay NaN.
    """
    if provider is None:
        return train_fold, eval_fold
    seasons = sorted(int(s) for s in train_fold["season"].unique())
    if set(seasons) & set(int(s) for s in eval_fold["season"].unique()):
        raise ValueError("The held-out season is in the training fold; refusing to compute its climatology.")
    clim = provider(seasons).set_index("cell_id")

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in CLIM_COLUMNS:
            out[c] = clim[c].reindex(df["cell_id"]).to_numpy().astype(np.float32)
        return out

    return apply(train_fold), apply(eval_fold)
