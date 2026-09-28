"""Historical analogs (PRD 15 F3), reduced vector. Reads `data/regime/` and `data/features/district_history/`;
never recomputes, retrains or writes anything back.

The PRD's 10-number vector needs the raw A1-A6 atmospheric indicators, which were never persisted (see
`services/regime.py`). This module is therefore honest about what it does: similarity on the four numbers that
ARE saved for every run and lead -- `p_active`, `p_break`, `lps_present` and `lps_strength` (`p_normal` is
1 - p_active - p_break, so it adds nothing). It never claims A1-A6.

This module does NOT import `regime_engine` (its package `__init__` pulls in scikit-learn; the API process must
not). `find_analogs` below is a local port of the selection logic in `regime_engine/analogs/finder.py`:
same-lead library, the query's own season excluded, standardized Euclidean distance, distance percentile among
all candidates, greedy pick of the nearest case then the next nearest at least `MIN_SEPARATION_DAYS` from every
case already picked, K = 5.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from backend.app.errors import ApiError

VECTOR_KEYS = ["p_active", "p_break", "lps_present", "lps_strength"]
K = 5
MIN_SEPARATION_DAYS = 5

HISTORY_COLUMNS = [
    "run_id", "lead_day", "imd_date", "season", "observed_mean_mm", "observed_max_cell_mm", "raw_mean_mm",
]  # fmt: skip


@lru_cache(maxsize=1)
def _library(regime_dir: Path) -> pd.DataFrame:
    """One row per (run_id, lead_day): the phase probabilities and the run's low-pressure-system numbers."""
    domain_path = regime_dir / "regime_domain.parquet"
    if not domain_path.is_file():
        raise ApiError(503, "REGIME_DATA_UNAVAILABLE", "Regime domain data has not been generated.")
    domain = pd.read_parquet(domain_path)
    domain["imd_date"] = pd.to_datetime(domain["imd_date"])
    domain = domain.drop(columns=["lps_strength"], errors="ignore")  # always taken from regime_features below

    strengths = []
    for path in sorted((regime_dir / "regime_features").glob("season_*.parquet")):
        table = pq.read_table(path, columns=["run_id", "lead_day", "lps_strength"]).to_pandas()
        # The number is identical for every cell of one run and lead; `max` just collapses them.
        strengths.append(table.groupby(["run_id", "lead_day"], as_index=False)["lps_strength"].max())
    if strengths:
        domain = domain.merge(pd.concat(strengths), on=["run_id", "lead_day"], how="left")
    else:
        domain["lps_strength"] = np.nan

    domain["lps_present"] = domain["lps_detected"].astype(float)
    domain["lps_strength"] = domain["lps_strength"].fillna(0.0).where(domain["lps_present"] > 0, 0.0)
    keep = ["run_id", "lead_day", "season", "imd_date", "regime_available", *VECTOR_KEYS]
    return domain[keep].reset_index(drop=True)


@lru_cache(maxsize=64)
def _district_history(features_dir: Path, district_id: str) -> pd.DataFrame:
    """Every (run, lead) outcome of one district, all seasons."""
    frames = []
    for path in sorted((features_dir / "district_history").glob("season_*/*.parquet")):
        table = pq.read_table(path, columns=HISTORY_COLUMNS + ["district_id"], filters=[("district_id", "=", district_id)])
        if table.num_rows:
            frames.append(table.drop_columns(["district_id"]).to_pandas())
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["observed_mean_mm", "raw_mean_mm"])


def _round(value: float | None, digits: int = 1) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def find_analogs(
    library: pd.DataFrame, query: dict[str, float], lead_day: int, query_season: int, k: int = K
) -> pd.DataFrame:
    """The (up to) `k` closest library cases, nearest first, with `distance` and `distance_percentile` columns."""
    cand = library[(library["lead_day"] == lead_day) & (library["season"] != query_season)].copy()
    cand = cand.dropna(subset=VECTOR_KEYS)
    if cand.empty:
        return cand.assign(distance=[], distance_percentile=[])

    x = cand[VECTOR_KEYS].to_numpy(dtype=float)
    q = np.array([query[key] for key in VECTOR_KEYS], dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    dist = np.sqrt((((x - mean) / std - (q - mean) / std) ** 2).sum(axis=1))

    cand["distance"] = dist
    cand["distance_percentile"] = (np.argsort(np.argsort(dist)) + 1) / len(cand) * 100.0
    cand = cand.sort_values("distance", kind="stable")

    picked: list[int] = []
    picked_dates: list[dt.date] = []
    for idx, row in cand.iterrows():
        day = row["imd_date"].date()
        if all(abs((day - other).days) >= MIN_SEPARATION_DAYS for other in picked_dates):
            picked.append(idx)
            picked_dates.append(day)
            if len(picked) == k:
                break
    return cand.loc[picked]


def get_analogs(
    regime_dir: Path,
    features_dir: Path,
    run_id: str,
    lead_day: int,
    district_id: str,
    query_season: int,
    holdout_seasons: set[int],
) -> dict:
    lib = _library(regime_dir)
    hist = _district_history(features_dir, district_id)
    if hist.empty:
        raise ApiError(404, "DISTRICT_NOT_FOUND", "No historical outcomes exist for this district.")

    row = lib[(lib["run_id"] == run_id) & (lib["lead_day"] == lead_day)]
    if row.empty or not bool(row.iloc[0]["regime_available"]):
        raise ApiError(404, "REGIME_NOT_FOUND", "No regime state exists for this run and lead day.")
    row = row.iloc[0]
    query = {key: float(row[key]) for key in VECTOR_KEYS}

    # Library: development seasons only (PRD F3), cases whose district outcome is on file.
    have_outcome = hist[["run_id", "lead_day"]].drop_duplicates()
    pool = lib[lib["regime_available"] & ~lib["season"].isin(holdout_seasons)]
    pool = pool.merge(have_outcome, on=["run_id", "lead_day"], how="inner")
    chosen = find_analogs(pool, query, lead_day, query_season)

    outcomes = hist.set_index(["run_id", "lead_day"])
    analogs: list[dict] = []
    errors: list[float] = []
    for rank, (_, case) in enumerate(chosen.iterrows(), start=1):
        out = outcomes.loc[(case["run_id"], lead_day)]
        if isinstance(out, pd.DataFrame):
            out = out.iloc[0]
        error = float(out["observed_mean_mm"] - out["raw_mean_mm"])
        errors.append(error)
        analogs.append({
            "rank": rank,
            "analog_run_id": case["run_id"],
            "imd_date": case["imd_date"].strftime("%Y-%m-%d"),
            "season": int(case["season"]),
            "distance": round(float(case["distance"]), 3),
            "distance_percentile": round(float(case["distance_percentile"]), 1),
            "p_active": _round(case["p_active"], 3),
            "p_break": _round(case["p_break"], 3),
            "lps_present": bool(case["lps_present"] > 0),
            "lps_strength": _round(case["lps_strength"], 3),
            "observed_mean_mm": _round(out["observed_mean_mm"]),
            "observed_wettest_cell_mm": _round(out["observed_max_cell_mm"]),
            "raw_mean_mm": _round(out["raw_mean_mm"]),
            "error_observed_minus_raw_mm": _round(error),
        })  # fmt: skip

    return {
        "lead_day": lead_day,
        "district_id": district_id,
        "query": {
            "imd_date": row["imd_date"].strftime("%Y-%m-%d"),
            "p_active": _round(query["p_active"], 3),
            "p_break": _round(query["p_break"], 3),
            "lps_present": bool(query["lps_present"] > 0),
            "lps_strength": _round(query["lps_strength"], 3),
        },
        "analogs": analogs,
        "median_error_observed_minus_raw_mm": _round(float(np.median(errors))) if errors else None,
        "n_analogs": len(analogs),
        "library_size": int(((pool["lead_day"] == lead_day) & (pool["season"] != query_season)).sum()),
    }
