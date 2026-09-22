"""F1 Raw vs AI-Corrected Improvement backend logic (PRD §15 F1).

Cell improvement definition:
    improvement = |raw_mm - observed_mm| - |corrected_mean_mm - observed_mm|
Interpretation:
    > 0 => corrected closer
    < 0 => raw closer
    == 0 => tie

District improvement comparison:
    |raw_mean_mm - observed_mean_mm| vs |corrected_mean_mm - observed_mean_mm|

Rules:
1. Only compare rows with real observations.
2. Missing observation, missing raw, or missing corrected must be excluded.
3. Fallback / raw-only products without corrected must be excluded.
4. Ties must NOT be counted as corrected wins.
5. Mixed run IDs or mixed lead days must fail fast (raise ValueError).
6. Pure functions only; no API, database, or frontend code yet.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

CLOSER_CORRECTED = "corrected"
CLOSER_RAW = "raw"
CLOSER_TIE = "tie"

RAW_ONLY_PRODUCT_TYPES = {
    "raw_nwp",
    "raw_only",
    "fallback_raw",
    "ecmwf_raw",
}


@dataclass
class CellImprovement:
    """Per-cell improvement record suitable for the map improvement layer."""
    cell_id: Union[int, str]
    improvement_mm: float
    closer_system: str  # "corrected" | "raw" | "tie"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DistrictImprovement:
    """Per-district improvement record."""
    district_id: Union[int, str]
    improvement_mm: float
    closer_system: str  # "corrected" | "raw" | "tie"
    raw_mean_mm: float
    corrected_mean_mm: float
    observed_mean_mm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class CellImprovementSummary:
    """Cell improvement count summary."""
    corrected_closer_cells: int = 0
    compared_cells: int = 0
    raw_closer_cells: int = 0
    tied_cells: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DistrictImprovementSummary:
    """District improvement count summary."""
    corrected_closer_districts: int = 0
    compared_districts: int = 0
    raw_closer_districts: int = 0
    tied_districts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ImprovementSummary:
    """Improvement summary for one run and lead, combining cell and district counts."""
    run_id: Optional[str] = None
    lead_day: Optional[int] = None
    corrected_closer_cells: int = 0
    compared_cells: int = 0
    raw_closer_cells: int = 0
    tied_cells: int = 0
    corrected_closer_districts: int = 0
    compared_districts: int = 0
    raw_closer_districts: int = 0
    tied_districts: int = 0
    cell_improvements: List[CellImprovement] = field(default_factory=list)
    district_improvements: List[DistrictImprovement] = field(default_factory=list)

    @property
    def summary_line(self) -> str:
        return format_f1_summary_line(
            corrected_closer_cells=self.corrected_closer_cells,
            compared_cells=self.compared_cells,
            corrected_closer_districts=self.corrected_closer_districts,
            compared_districts=self.compared_districts,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "lead_day": self.lead_day,
            "corrected_closer_cells": self.corrected_closer_cells,
            "compared_cells": self.compared_cells,
            "raw_closer_cells": self.raw_closer_cells,
            "tied_cells": self.tied_cells,
            "corrected_closer_districts": self.corrected_closer_districts,
            "compared_districts": self.compared_districts,
            "raw_closer_districts": self.raw_closer_districts,
            "tied_districts": self.tied_districts,
            "summary_line": self.summary_line,
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def format_f1_summary_line(
    corrected_closer_cells: int,
    compared_cells: int,
    corrected_closer_districts: int,
    compared_districts: int,
) -> str:
    """Formats the F1 single-day summary line per PRD §15 F1 item 4.
    
    Example:
        'Corrected is closer to the observation in 42 of 60 cells and 12 of 15 districts. One day only, not evidence.'
    """
    return (
        f"Corrected is closer to the observation in {corrected_closer_cells} of {compared_cells} cells "
        f"and {corrected_closer_districts} of {compared_districts} districts. One day only, not evidence."
    )


def _is_missing(val: Any) -> bool:
    """Check if value is None, NaN, inf, or missing string."""
    if val is None:
        return True
    if isinstance(val, (float, int)):
        return math.isnan(val) or math.isinf(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "none", "nan", "null"):
            return True
    return False


def _to_float(val: Any) -> Optional[float]:
    """Converts value to float or returns None if missing/invalid."""
    if _is_missing(val):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _extract_field(record: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Extracts field from dict or object using multiple potential keys."""
    if isinstance(record, dict):
        for k in keys:
            if k in record and record[k] is not None:
                return record[k]
        return default
    for k in keys:
        if hasattr(record, k):
            val = getattr(record, k)
            if val is not None:
                return val
    return default


def _is_fallback_without_corrected(record: Any, corrected_val: Optional[float]) -> bool:
    """Checks whether the record represents a fallback or raw-only product without corrected forecast."""
    product_type = _extract_field(record, ["product_type", "type"])
    if product_type is not None and str(product_type).lower() in RAW_ONLY_PRODUCT_TYPES:
        return True

    has_corrected = _extract_field(record, ["has_corrected"])
    if has_corrected is False:
        return True

    fallback_used = _extract_field(record, ["fallback_used", "is_fallback"], default=False)
    if fallback_used and corrected_val is None:
        return True

    return False


def _validate_run_and_lead(
    records: Sequence[Any],
    expected_run_id: Optional[str] = None,
    expected_lead_day: Optional[int] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Validates that all records belong to the same run_id and lead_day.
    
    Fails fast (raises ValueError) if mixed run IDs or mixed lead days are encountered.
    """
    seen_run_ids = set()
    seen_lead_days = set()

    if expected_run_id is not None:
        seen_run_ids.add(str(expected_run_id))
    if expected_lead_day is not None:
        seen_lead_days.add(int(expected_lead_day))

    for rec in records:
        r_id = _extract_field(rec, ["run_id", "run"])
        if r_id is not None:
            seen_run_ids.add(str(r_id))

        l_day = _extract_field(rec, ["lead_day", "lead"])
        if l_day is not None:
            try:
                seen_lead_days.add(int(l_day))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid lead_day value: {l_day}")

        if len(seen_run_ids) > 1:
            raise ValueError(f"Mixed run IDs rejected: found multiple run IDs {sorted(list(seen_run_ids))}")

        if len(seen_lead_days) > 1:
            raise ValueError(f"Mixed lead days rejected: found multiple lead days {sorted(list(seen_lead_days))}")

    final_run_id = next(iter(seen_run_ids)) if seen_run_ids else expected_run_id
    final_lead_day = next(iter(seen_lead_days)) if seen_lead_days else expected_lead_day
    return final_run_id, final_lead_day


def evaluate_cell_improvement(
    raw_mm: Optional[float],
    corrected_mean_mm: Optional[float],
    observed_mm: Optional[float],
    cell_id: Union[int, str] = 0,
    tol: float = 1e-9,
) -> Optional[CellImprovement]:
    """Calculates improvement for a single cell.
    
    improvement = |raw_mm - observed_mm| - |corrected_mean_mm - observed_mm|
    
    Interpretation:
        > 0 => corrected closer
        < 0 => raw closer
        == 0 => tie
        
    Returns:
        CellImprovement instance, or None if any required value is missing/NaN.
    """
    raw = _to_float(raw_mm)
    corr = _to_float(corrected_mean_mm)
    obs = _to_float(observed_mm)

    if raw is None or corr is None or obs is None:
        return None

    raw_err = abs(raw - obs)
    corr_err = abs(corr - obs)

    if math.isclose(raw_err, corr_err, abs_tol=tol, rel_tol=tol):
        closer_system = CLOSER_TIE
        improvement_mm = 0.0
    elif raw_err > corr_err:
        closer_system = CLOSER_CORRECTED
        improvement_mm = raw_err - corr_err
    else:
        closer_system = CLOSER_RAW
        improvement_mm = raw_err - corr_err

    return CellImprovement(
        cell_id=cell_id,
        improvement_mm=improvement_mm,
        closer_system=closer_system,
    )


def evaluate_district_improvement(
    raw_mean_mm: Optional[float],
    corrected_mean_mm: Optional[float],
    observed_mean_mm: Optional[float],
    district_id: Union[int, str] = "",
    tol: float = 1e-9,
) -> Optional[DistrictImprovement]:
    """Calculates improvement for a single district.
    
    Compares:
        |raw_mean_mm - observed_mean_mm| vs |corrected_mean_mm - observed_mean_mm|
        
    Returns:
        DistrictImprovement instance, or None if any required value is missing/NaN.
    """
    raw = _to_float(raw_mean_mm)
    corr = _to_float(corrected_mean_mm)
    obs = _to_float(observed_mean_mm)

    if raw is None or corr is None or obs is None:
        return None

    raw_err = abs(raw - obs)
    corr_err = abs(corr - obs)

    if math.isclose(raw_err, corr_err, abs_tol=tol, rel_tol=tol):
        closer_system = CLOSER_TIE
        improvement_mm = 0.0
    elif raw_err > corr_err:
        closer_system = CLOSER_CORRECTED
        improvement_mm = raw_err - corr_err
    else:
        closer_system = CLOSER_RAW
        improvement_mm = raw_err - corr_err

    return DistrictImprovement(
        district_id=district_id,
        improvement_mm=improvement_mm,
        closer_system=closer_system,
        raw_mean_mm=raw,
        corrected_mean_mm=corr,
        observed_mean_mm=obs,
    )


def compute_cell_improvements(
    cells: Sequence[Any],
    run_id: Optional[str] = None,
    lead_day: Optional[int] = None,
) -> List[CellImprovement]:
    """Computes improvement for each valid cell, excluding missing and fallback-without-corrected cells.
    
    Fails fast if mixed run IDs or mixed lead days are detected.
    """
    _validate_run_and_lead(cells, expected_run_id=run_id, expected_lead_day=lead_day)

    results: List[CellImprovement] = []
    for rec in cells:
        raw_val = _extract_field(rec, ["raw_mm", "raw", "raw_rain_mm"])
        corr_val = _extract_field(rec, ["corrected_mean_mm", "corrected_mm", "corrected", "corrected_rain_mm"])
        obs_val = _extract_field(rec, ["observed_mm", "obs_mm", "observed", "obs", "truth_mm"])
        cell_id = _extract_field(rec, ["cell_id", "id", "cell"], default=len(results))

        corr_f = _to_float(corr_val)
        if _is_fallback_without_corrected(rec, corr_f):
            continue

        item = evaluate_cell_improvement(
            raw_mm=raw_val,
            corrected_mean_mm=corr_val,
            observed_mm=obs_val,
            cell_id=cell_id,
        )
        if item is not None:
            results.append(item)

    return results


def compute_district_improvements(
    districts: Sequence[Any],
    run_id: Optional[str] = None,
    lead_day: Optional[int] = None,
) -> List[DistrictImprovement]:
    """Computes improvement for each valid district, excluding missing and fallback-without-corrected.
    
    Fails fast if mixed run IDs or mixed lead days are detected.
    """
    _validate_run_and_lead(districts, expected_run_id=run_id, expected_lead_day=lead_day)

    results: List[DistrictImprovement] = []
    for rec in districts:
        raw_val = _extract_field(rec, ["raw_mean_mm", "raw_mean", "raw_mm", "raw"])
        corr_val = _extract_field(rec, ["corrected_mean_mm", "corrected_mean", "corrected_mm", "corrected"])
        obs_val = _extract_field(rec, ["observed_mean_mm", "obs_mean_mm", "observed_mean", "observed_mm", "observed", "obs"])
        district_id = _extract_field(rec, ["district_id", "id", "district"], default=str(len(results)))

        corr_f = _to_float(corr_val)
        if _is_fallback_without_corrected(rec, corr_f):
            continue

        item = evaluate_district_improvement(
            raw_mean_mm=raw_val,
            corrected_mean_mm=corr_val,
            observed_mean_mm=obs_val,
            district_id=district_id,
        )
        if item is not None:
            results.append(item)

    return results


def compute_cell_summary(
    cells: Union[Sequence[CellImprovement], Sequence[Any]],
    run_id: Optional[str] = None,
    lead_day: Optional[int] = None,
) -> CellImprovementSummary:
    """Computes summary counts for cells: corrected_closer, raw_closer, tied, compared."""
    if not cells:
        return CellImprovementSummary()

    if isinstance(cells[0], CellImprovement):
        items = list(cells)
    else:
        items = compute_cell_improvements(cells, run_id=run_id, lead_day=lead_day)

    corrected_closer = sum(1 for c in items if c.closer_system == CLOSER_CORRECTED)
    raw_closer = sum(1 for c in items if c.closer_system == CLOSER_RAW)
    tied = sum(1 for c in items if c.closer_system == CLOSER_TIE)
    compared = len(items)

    return CellImprovementSummary(
        corrected_closer_cells=corrected_closer,
        compared_cells=compared,
        raw_closer_cells=raw_closer,
        tied_cells=tied,
    )


def compute_district_summary(
    districts: Union[Sequence[DistrictImprovement], Sequence[Any]],
    run_id: Optional[str] = None,
    lead_day: Optional[int] = None,
) -> DistrictImprovementSummary:
    """Computes summary counts for districts: corrected_closer, raw_closer, tied, compared."""
    if not districts:
        return DistrictImprovementSummary()

    if isinstance(districts[0], DistrictImprovement):
        items = list(districts)
    else:
        items = compute_district_improvements(districts, run_id=run_id, lead_day=lead_day)

    corrected_closer = sum(1 for d in items if d.closer_system == CLOSER_CORRECTED)
    raw_closer = sum(1 for d in items if d.closer_system == CLOSER_RAW)
    tied = sum(1 for d in items if d.closer_system == CLOSER_TIE)
    compared = len(items)

    return DistrictImprovementSummary(
        corrected_closer_districts=corrected_closer,
        compared_districts=compared,
        raw_closer_districts=raw_closer,
        tied_districts=tied,
    )


def compute_improvement_summary(
    run_id: Any = None,
    lead_day: Any = None,
    cells: Optional[Sequence[Any]] = None,
    districts: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> ImprovementSummary:
    """Computes full improvement summary for one run and lead day across cells and districts.
    
    Supports flexible arguments:
        compute_improvement_summary(run_id="run_1", lead_day=1, cells=[...], districts=[...])
        compute_improvement_summary(cells=[...], districts=[...])
        compute_improvement_summary(cell_data=[...], district_data=[...])
        
    Fails fast if mixed run IDs or mixed lead days are detected across cells or districts.
    """
    # Handle positional invocation where cells or districts might be passed first
    if isinstance(run_id, (list, tuple)) and cells is None:
        cells = run_id
        run_id = None
    if isinstance(lead_day, (list, tuple)) and districts is None:
        districts = lead_day
        lead_day = None

    if cells is None and "cell_data" in kwargs:
        cells = kwargs["cell_data"]
    if districts is None and "district_data" in kwargs:
        districts = kwargs["district_data"]

    cell_list = list(cells) if cells is not None else []
    dist_list = list(districts) if districts is not None else []

    # Validate combined records for consistent run_id and lead_day
    combined = cell_list + dist_list
    final_run_id, final_lead_day = _validate_run_and_lead(
        combined,
        expected_run_id=run_id,
        expected_lead_day=lead_day,
    )

    cell_items = compute_cell_improvements(cell_list, run_id=final_run_id, lead_day=final_lead_day) if cell_list else []
    dist_items = compute_district_improvements(dist_list, run_id=final_run_id, lead_day=final_lead_day) if dist_list else []

    cell_sum = compute_cell_summary(cell_items)
    dist_sum = compute_district_summary(dist_items)

    return ImprovementSummary(
        run_id=final_run_id,
        lead_day=final_lead_day,
        corrected_closer_cells=cell_sum.corrected_closer_cells,
        compared_cells=cell_sum.compared_cells,
        raw_closer_cells=cell_sum.raw_closer_cells,
        tied_cells=cell_sum.tied_cells,
        corrected_closer_districts=dist_sum.corrected_closer_districts,
        compared_districts=dist_sum.compared_districts,
        raw_closer_districts=dist_sum.raw_closer_districts,
        tied_districts=dist_sum.tied_districts,
        cell_improvements=cell_items,
        district_improvements=dist_items,
    )


def compute_grid_improvements(
    raw_grid: np.ndarray,
    corrected_grid: np.ndarray,
    observed_grid: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, CellImprovementSummary]:
    """Computes improvement arrays and summary from 2D spatial rain fields.
    
    Returns:
        improvement_grid: 2D float array (NaN for invalid/missing cells)
        closer_system_grid: 2D object array with 'corrected', 'raw', 'tie', or None
        summary: CellImprovementSummary
    """
    raw_arr = np.asarray(raw_grid, dtype=float)
    corr_arr = np.asarray(corrected_grid, dtype=float)
    obs_arr = np.asarray(observed_grid, dtype=float)

    if not (raw_arr.shape == corr_arr.shape == obs_arr.shape):
        raise ValueError(
            f"Shape mismatch: raw {raw_arr.shape}, corrected {corr_arr.shape}, observed {obs_arr.shape}"
        )

    valid = (
        ~np.isnan(raw_arr)
        & ~np.isnan(corr_arr)
        & ~np.isnan(obs_arr)
        & ~np.isinf(raw_arr)
        & ~np.isinf(corr_arr)
        & ~np.isinf(obs_arr)
    )
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)

    raw_err = np.abs(raw_arr - obs_arr)
    corr_err = np.abs(corr_arr - obs_arr)
    diff = raw_err - corr_err

    improvement_grid = np.full(raw_arr.shape, np.nan, dtype=float)
    closer_grid = np.full(raw_arr.shape, None, dtype=object)

    improvement_grid[valid] = diff[valid]

    ties = valid & (np.abs(diff) <= tol)
    corr_wins = valid & (diff > tol)
    raw_wins = valid & (diff < -tol)

    improvement_grid[ties] = 0.0
    closer_grid[ties] = CLOSER_TIE
    closer_grid[corr_wins] = CLOSER_CORRECTED
    closer_grid[raw_wins] = CLOSER_RAW

    summary = CellImprovementSummary(
        corrected_closer_cells=int(np.sum(corr_wins)),
        compared_cells=int(np.sum(valid)),
        raw_closer_cells=int(np.sum(raw_wins)),
        tied_cells=int(np.sum(ties)),
    )

    return improvement_grid, closer_grid, summary
