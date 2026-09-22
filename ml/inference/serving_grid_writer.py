"""Serving Grid Writer and Assembly Helper (PRD §17.1-§17.2, §20.1).

Storage layout:
    data/serving/grid/run_id=<run_id>/part.parquet
    (one file per run: all leads, all valid cells)

Required columns (EXACTLY 18 columns, in exact PRD §20.1 order):
    1. cell_id (int)
    2. lead_day (int)
    3. raw_mm (float)
    4. corrected_mean_mm (float)
    5. q10_mm (float, nullable)
    6. q50_mm (float, nullable)
    7. q90_mm (float, nullable)
    8. p_ge_15_6 (float, nullable)
    9. p_ge_64_5 (float, nullable)
    10. p_ge_115_6 (float, nullable)
    11. obs_mm (float, nullable - empty if unknown)
    12. lps_influence (float)
    13. orographic_influence (float)
    14. coastal_influence (float)
    15. product_type (string: 'raw_nwp' | 'quantile_mapping' | 'global_ml' | 'regime_aware_ml')
    16. fallback_reason (string, nullable)
    17. extrapolation_flag (bool)
    18. model_version_id (int, nullable)

Guarantees:
- Strict 18-column contract (no extra columns).
- Unique (cell_id, lead_day) keys.
- lead_day in {1, 2, 3}.
- Non-negative rainfall values.
- Valid probability bounds in [0, 1].
- Quantile ordering q10 <= q50 <= q90.
- Fallback rows have null range and probabilities (PRD §17.2).
- Cell extrapolation overrides and serves raw NWP forecast.
- PyArrow / Pandas parquet format.
- Deterministic output.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from ml.inference.predict_correction import select_fallback_product, check_extrapolation

SERVING_GRID_COLUMNS: List[str] = [
    "cell_id",
    "lead_day",
    "raw_mm",
    "corrected_mean_mm",
    "q10_mm",
    "q50_mm",
    "q90_mm",
    "p_ge_15_6",
    "p_ge_64_5",
    "p_ge_115_6",
    "obs_mm",
    "lps_influence",
    "orographic_influence",
    "coastal_influence",
    "product_type",
    "fallback_reason",
    "extrapolation_flag",
    "model_version_id",
]

VALID_PRODUCT_TYPES = {
    "raw_nwp",
    "quantile_mapping",
    "global_ml",
    "regime_aware_ml",
}

VALID_FALLBACK_REASONS = {
    "REGIME_UNAVAILABLE",
    "ML_UNAVAILABLE",
    "OOD_INPUT",
    "EXTRAPOLATION",
    "VALIDATION_FAILED",
    "NO_CORRECTION",
    None,
}


def validate_serving_grid_schema(df: pd.DataFrame) -> None:
    """Verify that DataFrame matches the exact 18-column PRD §20.1 contract."""
    missing = [c for c in SERVING_GRID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Serving grid DataFrame missing required columns: {missing}")

    extra = [c for c in df.columns if c not in SERVING_GRID_COLUMNS]
    if extra:
        raise ValueError(f"Serving grid contains unexpected extra columns: {extra}")

    if list(df.columns) != SERVING_GRID_COLUMNS:
        raise ValueError(
            f"Columns must match exact PRD §20.1 order.\nExpected: {SERVING_GRID_COLUMNS}\nGot: {list(df.columns)}"
        )


def validate_serving_grid_contracts(df: pd.DataFrame) -> None:
    """Verify data-level validation contracts (PRD §17.1-§17.2, §20.1)."""
    validate_serving_grid_schema(df)

    # 1. Uniqueness of (cell_id, lead_day)
    if df.duplicated(subset=["cell_id", "lead_day"]).any():
        dups = df[df.duplicated(subset=["cell_id", "lead_day"], keep=False)]
        raise ValueError(f"Duplicate (cell_id, lead_day) found in serving grid ({len(dups)} rows).")

    # 2. lead_day in {1, 2, 3}
    invalid_leads = set(df["lead_day"].dropna().unique()) - {1, 2, 3}
    if invalid_leads:
        raise ValueError(f"Invalid lead_day values: {invalid_leads}. Expected subset of {{1, 2, 3}}.")

    # 3. Rain values non-negative when non-null
    for r_col in ["raw_mm", "corrected_mean_mm"]:
        vals = df[r_col].dropna()
        if (vals < -1e-6).any():
            raise ValueError(f"Negative values detected in '{r_col}'.")

    if "obs_mm" in df.columns:
        obs_vals = df["obs_mm"].dropna()
        if (obs_vals < -1e-6).any():
            raise ValueError("Negative values detected in 'obs_mm'.")

    # 4. Probabilities bounded in [0, 1] when non-null
    for p_col in ["p_ge_15_6", "p_ge_64_5", "p_ge_115_6"]:
        p_vals = df[p_col].dropna()
        if ((p_vals < -1e-6) | (p_vals > 1.0 + 1e-6)).any():
            raise ValueError(f"Probabilities in '{p_col}' must be bounded in [0.0, 1.0].")

    # 5. Quantile ordering: q10 <= q50 <= q90 for non-null ranges
    has_range = df["q10_mm"].notna() & df["q50_mm"].notna() & df["q90_mm"].notna()
    if has_range.any():
        sub = df[has_range]
        if (sub["q10_mm"] > sub["q50_mm"] + 1e-5).any():
            raise ValueError("Quantile crossing detected: q10_mm > q50_mm.")
        if (sub["q50_mm"] > sub["q90_mm"] + 1e-5).any():
            raise ValueError("Quantile crossing detected: q50_mm > q90_mm.")

    # 6. Extrapolation rows must serve raw NWP for that cell
    extrap_mask = df["extrapolation_flag"] == True
    if extrap_mask.any():
        extrap_rows = df[extrap_mask]
        diff = np.abs(extrap_rows["corrected_mean_mm"].to_numpy() - extrap_rows["raw_mm"].to_numpy())
        if np.any(diff > 1e-4):
            raise ValueError("Extrapolation rows must serve raw_mm exactly.")
        if not (extrap_rows["product_type"] == "raw_nwp").all():
            raise ValueError("Extrapolation rows must have product_type='raw_nwp'.")
        if not (extrap_rows["fallback_reason"] == "EXTRAPOLATION").all():
            raise ValueError("Extrapolation rows must have fallback_reason='EXTRAPOLATION'.")

    # 7. Fallback rows have null range and probabilities (PRD §17.2)
    fallback_mask = df["fallback_reason"].notna() | (df["product_type"] != "regime_aware_ml")
    if fallback_mask.any():
        fb_rows = df[fallback_mask]
        for col in ["q10_mm", "q50_mm", "q90_mm", "p_ge_15_6", "p_ge_64_5", "p_ge_115_6"]:
            if fb_rows[col].notna().any():
                raise ValueError(
                    f"Fallback rows must have null {col} (PRD §17.2). Found non-null values."
                )


def assemble_serving_grid(
    base_df: pd.DataFrame,
    b0_raw_mm: Optional[Union[pd.Series, np.ndarray]] = None,
    b1_mm: Optional[Union[pd.Series, np.ndarray]] = None,
    b2_mm: Optional[Union[pd.Series, np.ndarray]] = None,
    b3_mm: Optional[Union[pd.Series, np.ndarray]] = None,
    range_df: Optional[pd.DataFrame] = None,
    prob_df: Optional[pd.DataFrame] = None,
    extrapolation_flags: Optional[Union[pd.Series, np.ndarray]] = None,
    p99_9_threshold: Optional[float] = None,
    regime_available: bool = True,
    ood_flag: bool = False,
    ml_available: bool = True,
    validation_failed: bool = False,
    no_correction: bool = False,
    model_version_id: Optional[int] = None,
    obs_mm: Optional[Union[pd.Series, np.ndarray]] = None,
) -> pd.DataFrame:
    """Assemble the canonical 18-column serving grid DataFrame from existing predictions.

    Does NOT train or run models; only combines already-produced predictions.

    Precedence ladder (PRD §17.1):
    1. no_correction -> ("raw_nwp", "NO_CORRECTION", raw_mm)
    2. validation_failed -> ("raw_nwp", "VALIDATION_FAILED", raw_mm)
    3. cell extrapolation_flag -> ("raw_nwp", "EXTRAPOLATION", raw_mm)
    4. not ml_available -> ("quantile_mapping", "ML_UNAVAILABLE", b1_mm)
    5. ood_flag -> ("quantile_mapping", "OOD_INPUT", b1_mm)
    6. not regime_available -> ("global_ml", "REGIME_UNAVAILABLE", b2_mm)
    7. normal -> ("regime_aware_ml", None, b3_mm)
    """
    n_rows = len(base_df)
    if n_rows == 0:
        raise ValueError("base_df is empty; cannot assemble serving grid.")

    # 1. Resolve raw_mm
    if b0_raw_mm is not None:
        raw_mm = np.asarray(b0_raw_mm, dtype=float)
    elif "raw_mm" in base_df.columns:
        raw_mm = base_df["raw_mm"].to_numpy(dtype=float)
    elif "rain_mm" in base_df.columns:
        raw_mm = base_df["rain_mm"].to_numpy(dtype=float)
    else:
        raise ValueError("Cannot resolve raw forecast; provide b0_raw_mm or rain_mm in base_df.")

    # 2. Resolve cell extrapolation flags
    if extrapolation_flags is not None:
        extrap_arr = np.asarray(extrapolation_flags, dtype=bool)
    elif p99_9_threshold is not None:
        extrap_arr = check_extrapolation(raw_mm, p99_9_threshold)
    elif "extrapolation_flag" in base_df.columns:
        extrap_arr = base_df["extrapolation_flag"].to_numpy(dtype=bool)
    else:
        extrap_arr = np.zeros(n_rows, dtype=bool)

    # 3. Determine run-level fallback baseline
    run_product, run_reason = select_fallback_product(
        has_b3=(b3_mm is not None),
        has_b2=(b2_mm is not None),
        has_b1=(b1_mm is not None),
        regime_available=regime_available,
        ml_available=ml_available,
        ood_flag=ood_flag,
        validation_failed=validation_failed,
        no_correction=no_correction,
        extrapolation_flag=False,  # Cell-level handled next
    )

    # Select default baseline corrected rain
    if run_product == "regime_aware_ml":
        baseline_corrected = np.asarray(b3_mm, dtype=float)
    elif run_product == "global_ml":
        baseline_corrected = np.asarray(b2_mm, dtype=float)
    elif run_product == "quantile_mapping":
        baseline_corrected = np.asarray(b1_mm, dtype=float) if b1_mm is not None else raw_mm
    else:
        baseline_corrected = raw_mm

    # 4. Construct cell-level products with extrapolation override
    final_corrected = np.array(baseline_corrected, dtype=float, copy=True)
    product_types = np.array([run_product] * n_rows, dtype=object)
    fallback_reasons = np.array([run_reason] * n_rows, dtype=object)

    for i in range(n_rows):
        if extrap_arr[i]:
            final_corrected[i] = raw_mm[i]
            product_types[i] = "raw_nwp"
            fallback_reasons[i] = "EXTRAPOLATION"

    # Ensure non-negativity
    final_corrected = np.maximum(final_corrected, 0.0)

    # 5. Populate range and probability outputs (null for ANY fallback row, PRD §17.2)
    q10 = np.full(n_rows, np.nan, dtype=float)
    q50 = np.full(n_rows, np.nan, dtype=float)
    q90 = np.full(n_rows, np.nan, dtype=float)
    p15 = np.full(n_rows, np.nan, dtype=float)
    p64 = np.full(n_rows, np.nan, dtype=float)
    p115 = np.full(n_rows, np.nan, dtype=float)

    # Eligible rows: normal regime_aware_ml with no fallback reason
    normal_indices = [
        i for i in range(n_rows)
        if product_types[i] == "regime_aware_ml" and fallback_reasons[i] is None
    ]

    if normal_indices:
        norm_idx = np.array(normal_indices)
        if range_df is not None:
            if "q10_mm" in range_df.columns:
                q10[norm_idx] = range_df["q10_mm"].iloc[norm_idx].to_numpy(dtype=float)
            if "q50_mm" in range_df.columns:
                q50[norm_idx] = range_df["q50_mm"].iloc[norm_idx].to_numpy(dtype=float)
            if "q90_mm" in range_df.columns:
                q90[norm_idx] = range_df["q90_mm"].iloc[norm_idx].to_numpy(dtype=float)

        if prob_df is not None:
            if "p_ge_15_6" in prob_df.columns:
                p15[norm_idx] = prob_df["p_ge_15_6"].iloc[norm_idx].to_numpy(dtype=float)
            if "p_ge_64_5" in prob_df.columns:
                p64[norm_idx] = prob_df["p_ge_64_5"].iloc[norm_idx].to_numpy(dtype=float)
            if "p_ge_115_6" in prob_df.columns:
                p115[norm_idx] = prob_df["p_ge_115_6"].iloc[norm_idx].to_numpy(dtype=float)

    # 6. Resolve obs_mm
    if obs_mm is not None:
        obs_vals = np.asarray(obs_mm, dtype=float)
    elif "obs_mm" in base_df.columns:
        obs_vals = base_df["obs_mm"].to_numpy(dtype=float)
    else:
        obs_vals = np.full(n_rows, np.nan, dtype=float)

    # 7. Static / regime influences
    lps_inf = (
        base_df["lps_influence"].to_numpy(dtype=float)
        if "lps_influence" in base_df.columns
        else np.zeros(n_rows, dtype=float)
    )
    orog_inf = (
        base_df["orographic_influence"].to_numpy(dtype=float)
        if "orographic_influence" in base_df.columns
        else np.zeros(n_rows, dtype=float)
    )
    coast_inf = (
        base_df["coastal_influence"].to_numpy(dtype=float)
        if "coastal_influence" in base_df.columns
        else np.zeros(n_rows, dtype=float)
    )

    # 8. Cell ID and Lead Day
    if "cell_id" not in base_df.columns or "lead_day" not in base_df.columns:
        raise ValueError("base_df must contain 'cell_id' and 'lead_day'.")

    cell_ids = base_df["cell_id"].to_numpy(dtype=int)
    lead_days = base_df["lead_day"].to_numpy(dtype=int)

    # 9. Model Version ID
    m_v_id = model_version_id
    if m_v_id is None and "model_version_id" in base_df.columns:
        m_v_ids = base_df["model_version_id"].values
    else:
        m_v_ids = [m_v_id] * n_rows

    serving_df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "lead_day": lead_days,
            "raw_mm": raw_mm,
            "corrected_mean_mm": final_corrected,
            "q10_mm": q10,
            "q50_mm": q50,
            "q90_mm": q90,
            "p_ge_15_6": p15,
            "p_ge_64_5": p64,
            "p_ge_115_6": p115,
            "obs_mm": obs_vals,
            "lps_influence": lps_inf,
            "orographic_influence": orog_inf,
            "coastal_influence": coast_inf,
            "product_type": product_types,
            "fallback_reason": fallback_reasons,
            "extrapolation_flag": extrap_arr,
            "model_version_id": m_v_ids,
        }
    )

    validate_serving_grid_contracts(serving_df)
    return serving_df


def write_serving_grid_file(
    df: pd.DataFrame,
    run_id: str,
    output_base_dir: Union[str, Path] = "data/serving",
    validate_contracts: bool = True,
) -> Path:
    """Write serving grid partition for run_id as Parquet.

    Target path:
        {output_base_dir}/grid/run_id={run_id}/part.parquet
    """
    if validate_contracts:
        validate_serving_grid_contracts(df)
    else:
        validate_serving_grid_schema(df)

    base = Path(output_base_dir)
    if base.name == "grid":
        partition_dir = base / f"run_id={run_id}"
    else:
        partition_dir = base / "grid" / f"run_id={run_id}"

    partition_dir.mkdir(parents=True, exist_ok=True)
    target_file = partition_dir / "part.parquet"

    # Ensure deterministic row sorting by lead_day and cell_id
    out_df = (
        df[SERVING_GRID_COLUMNS]
        .sort_values(by=["lead_day", "cell_id"])
        .reset_index(drop=True)
    )
    out_df.to_parquet(target_file, engine="pyarrow", index=False)

    return target_file


def read_serving_grid_file(
    run_id: str,
    output_base_dir: Union[str, Path] = "data/serving",
) -> pd.DataFrame:
    """Read serving grid partition for a specific run_id."""
    base = Path(output_base_dir)
    if base.name == "grid":
        target_file = base / f"run_id={run_id}" / "part.parquet"
    else:
        target_file = base / "grid" / f"run_id={run_id}" / "part.parquet"

    if not target_file.exists():
        raise FileNotFoundError(f"Serving grid file not found at: {target_file}")

    return pd.read_parquet(target_file, engine="pyarrow")
