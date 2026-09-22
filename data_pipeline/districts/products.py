"""District products: `district_forecasts` and `district_history` (PRD 14.4, 20.2). Owner: M1.

Built from the Golden Dataset rows and the cached district weights, with the PRD 14.4 definitions:

* `raw_mean_mm = sum_i w_i * raw_i` (all valid cells of the district; the weights add up to 1)
* `observed_mean_mm = sum_i w_i * obs_i`, `observed_max_cell_mm = max over S of obs_i` and
  `observed_heavy_area_fraction = sum_i w_i * 1[obs_i >= 64.5]`, with the same weights and the same set S of main
  cells (`is_main`). If IMD is missing in any cell that enters a value, that value is NULL (no guessing).
* `n_effective_cells = 1 / sum_i w_i^2`, `is_small = n_effective_cells < 4`.
* `raw_wettest_cell_id`, `raw_wettest_cell_mm`: the main cell with the most **raw** rain (ties: lowest cell_id).

**What does not exist yet is NULL, on purpose.** There is no bias-corrected model at this stage, so
`corrected_mean_mm`, the PRD `wettest_cell_*` fields (they are defined on the corrected mean, PRD 14.4) and the
probability fields (`heavy_prob_max_cell`, `very_heavy_prob_max_cell`, `*_area_fraction_expected`) are present in
the schema and NULL. Stage 4 fills them; nothing is estimated here. A district with no main cell (all weights
below `w_min`) has NULL for every max-over-S field.

Files: `<DATA_DIR>/features/{district_forecasts,district_history}/season_<YYYY>/<table>_<YYYYMM>.parquet`.
`district_history` (past outcomes for the analogs) keeps the rows where `observed_mean_mm` is known;
`prediction_source`, `phase`, `lps_near` and `corrected_mean_mm` are NULL until the later stages own them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import ValidationError

KEY = ["run_id", "lead_day", "district_id"]
NULLABLE_INTS = ["raw_wettest_cell_id", "wettest_cell_id"]  # pandas "Int32": NULL is pd.NA
_NULL_FLOATS = [
    "corrected_mean_mm", "wettest_cell_mean_mm", "wettest_cell_q10_mm", "wettest_cell_q50_mm", "wettest_cell_q90_mm",
    "heavy_prob_max_cell", "very_heavy_prob_max_cell", "heavy_area_fraction_expected",
    "very_heavy_area_fraction_expected",
]  # fmt: skip

FORECAST_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("lead_day", pa.int8()),
        ("district_id", pa.string()),
        ("imd_date", pa.date32()),
        ("season", pa.int16()),
        ("raw_mean_mm", pa.float32()),
        ("corrected_mean_mm", pa.float32()),
        ("observed_mean_mm", pa.float32()),
        ("observed_max_cell_mm", pa.float32()),
        ("observed_heavy_area_fraction", pa.float32()),
        ("raw_wettest_cell_id", pa.int32()),
        ("raw_wettest_cell_mm", pa.float32()),
        ("wettest_cell_id", pa.int32()),
        ("wettest_cell_mean_mm", pa.float32()),
        ("wettest_cell_q10_mm", pa.float32()),
        ("wettest_cell_q50_mm", pa.float32()),
        ("wettest_cell_q90_mm", pa.float32()),
        ("heavy_prob_max_cell", pa.float32()),
        ("very_heavy_prob_max_cell", pa.float32()),
        ("heavy_area_fraction_expected", pa.float32()),
        ("very_heavy_area_fraction_expected", pa.float32()),
        ("n_effective_cells", pa.float32()),
        ("n_main_cells", pa.int32()),
        ("is_small", pa.bool_()),
        ("alignment_offset_hours", pa.int8()),
        ("imd_stamp", pa.string()),
    ]
)
HISTORY_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("lead_day", pa.int8()),
        ("district_id", pa.string()),
        ("imd_date", pa.date32()),
        ("season", pa.int16()),
        ("observed_mean_mm", pa.float32()),
        ("observed_max_cell_mm", pa.float32()),
        ("raw_mean_mm", pa.float32()),
        ("corrected_mean_mm", pa.float32()),
        ("prediction_source", pa.string()),
        ("phase", pa.string()),
        ("lps_near", pa.bool_()),
    ]
)
FORECAST_COLUMNS, HISTORY_COLUMNS = FORECAST_SCHEMA.names, HISTORY_SCHEMA.names
_GOLDEN_NEEDED = [
    "run_id", "lead_day", "cell_id", "imd_date", "season", "rain_mm", "obs_mm", "alignment_offset_hours", "imd_stamp",
]  # fmt: skip


def district_forecasts(
    golden: pd.DataFrame, weights: pd.DataFrame, summary: pd.DataFrame, config: IngestionConfig | None = None
) -> pd.DataFrame:
    """District products for a batch of Golden Dataset rows (whole runs). See the module docstring."""
    config = config or load_config()
    heavy_mm = config.features.thresholds_mm[1]  # 64.5 mm (D8)
    missing = [c for c in _GOLDEN_NEEDED if c not in golden.columns]
    if missing:
        raise ValidationError(f"Golden rows are missing columns {missing}.")

    g = golden[_GOLDEN_NEEDED].merge(
        weights[["district_id", "cell_id", "area_weight", "is_main"]], on="cell_id", how="inner"
    )
    if g.empty:
        raise ValidationError(
            "No golden cell belongs to any district (do the weights use the same grid and valid cells?)."
        )
    w, rain, obs = g["area_weight"].to_numpy(), g["rain_mm"].to_numpy(), g["obs_mm"].to_numpy()
    main, obs_nan = g["is_main"].to_numpy(), np.isnan(obs)
    g["w"] = w
    g["w_rain"] = w * rain
    g["w_obs"] = w * np.where(obs_nan, 0.0, obs)
    g["w_obs_heavy"] = w * np.where(obs_nan, 0.0, obs >= heavy_mm)
    g["obs_nan"] = obs_nan
    g["main_obs_nan"] = obs_nan & main
    g["main_obs"] = np.where(main, obs, np.nan)

    by = g.groupby(KEY, sort=True)
    agg = by.agg(
        imd_date=("imd_date", "first"),
        season=("season", "first"),
        alignment_offset_hours=("alignment_offset_hours", "first"),
        imd_stamp=("imd_stamp", "first"),
        w_sum=("w", "sum"),
        raw_mean_mm=("w_rain", "sum"),
        obs_sum=("w_obs", "sum"),
        heavy_sum=("w_obs_heavy", "sum"),
        any_obs_nan=("obs_nan", "any"),
        any_main_obs_nan=("main_obs_nan", "any"),
        n_main=("is_main", "sum"),
        observed_max_cell_mm=("main_obs", "max"),
    )
    if not np.allclose(agg["w_sum"], 1.0, atol=1e-6):
        raise ValidationError(
            "Golden rows do not cover every valid cell of some district (weights add up to "
            f"{float(agg['w_sum'].min()):.6f}..{float(agg['w_sum'].max()):.6f}). The weights and the Golden Dataset "
            "must use the same valid-cell mask."
        )
    out = agg.reset_index()
    out["observed_mean_mm"] = np.where(out["any_obs_nan"], np.nan, out["obs_sum"])
    out["observed_heavy_area_fraction"] = np.where(out["any_obs_nan"], np.nan, out["heavy_sum"])
    out["observed_max_cell_mm"] = np.where(out["any_main_obs_nan"], np.nan, out["observed_max_cell_mm"])

    # wettest RAW cell among the main cells: most rain, ties -> lowest cell_id
    m = g[g["is_main"]].sort_values(
        KEY[:2] + ["district_id", "rain_mm", "cell_id"], ascending=[True, True, True, False, True]
    )
    top = m.drop_duplicates(KEY)[KEY + ["cell_id", "rain_mm"]].rename(
        columns={"cell_id": "raw_wettest_cell_id", "rain_mm": "raw_wettest_cell_mm"}
    )
    out = out.merge(top, on=KEY, how="left")

    out = out.merge(summary[["district_id", "n_effective_cells", "is_small"]], on="district_id", how="left")
    out["n_main_cells"] = out["n_main"]
    for c in _NULL_FLOATS:
        out[c] = np.nan
    out["wettest_cell_id"] = np.nan  # PRD wettest cell is defined on the corrected mean: not available yet
    out["imd_date"] = pd.to_datetime(out["imd_date"])
    out = out[FORECAST_COLUMNS].astype(_dtypes(FORECAST_SCHEMA)).astype(dict.fromkeys(NULLABLE_INTS, "Int32"))
    out = out.sort_values(KEY, kind="stable").reset_index(drop=True)
    check_forecasts(out, config)
    return out


def district_history(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Past outcomes per (run, lead, district) for the analogs (PRD 20.2): rows with a known observed mean."""
    h = forecasts.loc[forecasts["observed_mean_mm"].notna()].copy()
    h["corrected_mean_mm"] = np.nan  # later stages
    h["prediction_source"] = None  # 'oof' or 'final': set when a corrected prediction exists
    h["phase"] = None  # regime engine (M2)
    h["lps_near"] = pd.array([pd.NA] * len(h), dtype="boolean")  # regime engine (M2)
    return h[HISTORY_COLUMNS].reset_index(drop=True)


def _dtypes(schema: pa.Schema) -> dict[str, str]:
    return {
        f.name: f.type.to_pandas_dtype()
        for f in schema
        if (pa.types.is_floating(f.type) or pa.types.is_integer(f.type)) and f.name not in NULLABLE_INTS
    }


def check_forecasts(df: pd.DataFrame, config: IngestionConfig | None = None) -> None:
    """Schema and key checks."""
    if list(df.columns) != FORECAST_COLUMNS:
        raise ValidationError("district_forecasts columns differ from the schema.")
    if df.duplicated(KEY).any():
        raise ValidationError(
            f"{int(df.duplicated(KEY).sum())} duplicate (run_id, lead_day, district_id) rows."
        )
    for c in ("raw_mean_mm", "n_effective_cells"):
        if df[c].isna().any():
            raise ValidationError(f"{c} must always be defined.")
    if (df["raw_mean_mm"] < 0).any():
        raise ValidationError("raw_mean_mm is negative.")
