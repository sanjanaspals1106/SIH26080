"""Verification component persistence engine (PRD §16.4 and §20.1).

Saves and loads VerificationComponent records as Parquet under:
data/verification/components/

Partitioning:
Partitions by evaluation_set and forecast_type:
<base_dir>/evaluation_set=<eval_set>/forecast_type=<forecast_type>.parquet

Rules:
- Preserves all VerificationComponent metadata and additive fields.
- Round-trips None values faithfully without converting to 0 or NaN.
- Deterministic column schema and order.
- Rejects duplicate identities unless explicit replace=True is specified.
- Replaces only matching identities on replace=True.
- Append mode preserves existing records.
- Incompatible or corrupt Parquet schemas fail clearly.
- Supports filtering reads by forecast_type, evaluation_set, lead_day,
  threshold_mm, neighbourhood_cells, group_type, group_value.
- No metric recomputation or bootstrap during IO.
"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from verification.components import VerificationComponent

# Sentinel for distinguishing omitted filter argument from explicit None filter
_UNSET = object()

DEFAULT_DATA_DIR: Path = Path("data/verification/components")

# Identity metadata dimensions per PRD §16.4 & user specification
IDENTITY_DIMENSIONS: Tuple[str, ...] = (
    "imd_date",
    "lead_day",
    "forecast_type",
    "evaluation_set",
    "threshold_mm",
    "neighbourhood_cells",
    "group_type",
    "group_value",
)

# PyArrow Schema defining deterministic columns and nullability
COMPONENT_PYARROW_SCHEMA: pa.Schema = pa.schema([
    pa.field("imd_date", pa.string(), nullable=True),
    pa.field("lead_day", pa.int32(), nullable=False),
    pa.field("forecast_type", pa.string(), nullable=False),
    pa.field("evaluation_set", pa.string(), nullable=False),
    pa.field("threshold_mm", pa.float64(), nullable=True),
    pa.field("neighbourhood_cells", pa.int32(), nullable=True),
    pa.field("group_type", pa.string(), nullable=False),
    pa.field("group_value", pa.string(), nullable=False),
    pa.field("bin_lower", pa.float64(), nullable=True),
    pa.field("bin_upper", pa.float64(), nullable=True),
    pa.field("n_samples", pa.int64(), nullable=False),
    pa.field("n_events", pa.int64(), nullable=True),
    pa.field("n", pa.int64(), nullable=False),
    pa.field("sum_error", pa.float64(), nullable=False),
    pa.field("sum_abs_error", pa.float64(), nullable=False),
    pa.field("sum_squared_error", pa.float64(), nullable=False),
    pa.field("a", pa.int64(), nullable=False),
    pa.field("b", pa.int64(), nullable=False),
    pa.field("c", pa.int64(), nullable=False),
    pa.field("d", pa.int64(), nullable=False),
    pa.field("fss_numerator_sum", pa.float64(), nullable=False),
    pa.field("fss_forecast_fraction_sq_sum", pa.float64(), nullable=False),
    pa.field("fss_observed_fraction_sq_sum", pa.float64(), nullable=False),
    pa.field("brier_n", pa.int64(), nullable=False),
    pa.field("sum_brier_terms", pa.float64(), nullable=False),
    pa.field("sum_reference_brier_terms", pa.float64(), nullable=False),
    pa.field("sum_predicted_probability", pa.float64(), nullable=False),
    pa.field("n_observed_events", pa.int64(), nullable=False),
    pa.field("pinball_q10_sum", pa.float64(), nullable=False),
    pa.field("pinball_q50_sum", pa.float64(), nullable=False),
    pa.field("pinball_q90_sum", pa.float64(), nullable=False),
    pa.field("range_n", pa.int64(), nullable=False),
    pa.field("coverage_count", pa.int64(), nullable=False),
])

SCHEMA_COLUMNS: Tuple[str, ...] = tuple(COMPONENT_PYARROW_SCHEMA.names)


class ComponentIOError(Exception):
    """Base exception for verification component IO errors."""
    pass


class DuplicateComponentError(ComponentIOError, ValueError):
    """Raised when an existing or duplicate component identity is encountered without replace=True."""
    pass


class IncompatibleSchemaError(ComponentIOError, ValueError):
    """Raised when a Parquet file has missing columns, type mismatches, or corrupt format."""
    pass


def get_default_components_dir() -> Path:
    """Returns the default directory for verification component Parquet storage."""
    return DEFAULT_DATA_DIR


def get_component_identity(
    item: Union[VerificationComponent, Dict[str, Any]],
) -> Tuple[Optional[str], int, str, str, Optional[float], Optional[int], str, str, Optional[float], Optional[float]]:
    """Extracts the canonical 10-dimensional metadata identity for a component record."""
    if isinstance(item, VerificationComponent):
        d_str = str(item.imd_date) if item.imd_date is not None else None
        t_val = round(float(item.threshold_mm), 4) if item.threshold_mm is not None else None
        n_val = int(item.neighbourhood_cells) if item.neighbourhood_cells is not None else None
        b_low = round(float(item.bin_lower), 4) if item.bin_lower is not None else None
        b_upp = round(float(item.bin_upper), 4) if item.bin_upper is not None else None
        return (
            d_str,
            int(item.lead_day),
            str(item.forecast_type),
            str(item.evaluation_set),
            t_val,
            n_val,
            str(item.group_type),
            str(item.group_value),
            b_low,
            b_upp,
        )
    else:
        d = item.get("imd_date")
        d_str = str(d) if d is not None and not (isinstance(d, float) and np.isnan(d)) else None
        t = item.get("threshold_mm")
        t_val = (
            round(float(t), 4)
            if t is not None and not (isinstance(t, float) and np.isnan(t))
            else None
        )
        n = item.get("neighbourhood_cells")
        n_val = (
            int(n)
            if n is not None and not (isinstance(n, float) and np.isnan(n))
            else None
        )
        bl = item.get("bin_lower")
        b_low = (
            round(float(bl), 4)
            if bl is not None and not (isinstance(bl, float) and np.isnan(bl))
            else None
        )
        bu = item.get("bin_upper")
        b_upp = (
            round(float(bu), 4)
            if bu is not None and not (isinstance(bu, float) and np.isnan(bu))
            else None
        )
        return (
            d_str,
            int(item["lead_day"]),
            str(item["forecast_type"]),
            str(item["evaluation_set"]),
            t_val,
            n_val,
            str(item["group_type"]),
            str(item["group_value"]),
            b_low,
            b_upp,
        )


def component_to_dict(c: VerificationComponent) -> Dict[str, Any]:
    """Converts a VerificationComponent to a dict matching COMPONENT_PYARROW_SCHEMA."""
    return {
        "imd_date": str(c.imd_date) if c.imd_date is not None else None,
        "lead_day": int(c.lead_day),
        "forecast_type": str(c.forecast_type),
        "evaluation_set": str(c.evaluation_set),
        "threshold_mm": float(c.threshold_mm) if c.threshold_mm is not None else None,
        "neighbourhood_cells": int(c.neighbourhood_cells) if c.neighbourhood_cells is not None else None,
        "group_type": str(c.group_type),
        "group_value": str(c.group_value),
        "bin_lower": float(c.bin_lower) if c.bin_lower is not None else None,
        "bin_upper": float(c.bin_upper) if c.bin_upper is not None else None,
        "n_samples": int(c.n_samples),
        "n_events": int(c.n_events) if c.n_events is not None else None,
        "n": int(c.n),
        "sum_error": float(c.sum_error),
        "sum_abs_error": float(c.sum_abs_error),
        "sum_squared_error": float(c.sum_squared_error),
        "a": int(c.a),
        "b": int(c.b),
        "c": int(c.c),
        "d": int(c.d),
        "fss_numerator_sum": float(c.fss_numerator_sum),
        "fss_forecast_fraction_sq_sum": float(c.fss_forecast_fraction_sq_sum),
        "fss_observed_fraction_sq_sum": float(c.fss_observed_fraction_sq_sum),
        "brier_n": int(c.brier_n),
        "sum_brier_terms": float(c.sum_brier_terms),
        "sum_reference_brier_terms": float(c.sum_reference_brier_terms),
        "sum_predicted_probability": float(c.sum_predicted_probability),
        "n_observed_events": int(c.n_observed_events),
        "pinball_q10_sum": float(c.pinball_q10_sum),
        "pinball_q50_sum": float(c.pinball_q50_sum),
        "pinball_q90_sum": float(c.pinball_q90_sum),
        "range_n": int(c.range_n),
        "coverage_count": int(c.coverage_count),
    }


def dict_to_component(d: Dict[str, Any]) -> VerificationComponent:
    """Converts a Parquet row dict back to a VerificationComponent instance."""
    def _opt_float(v: Any) -> Optional[float]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    def _opt_int(v: Any) -> Optional[int]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)

    def _opt_str(v: Any) -> Optional[str]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return str(v)

    return VerificationComponent(
        imd_date=_opt_str(d.get("imd_date")),
        lead_day=int(d.get("lead_day", 1)),
        forecast_type=str(d.get("forecast_type", "regime_aware_ml")),
        evaluation_set=str(d.get("evaluation_set", "development")),
        threshold_mm=_opt_float(d.get("threshold_mm")),
        neighbourhood_cells=_opt_int(d.get("neighbourhood_cells")),
        group_type=str(d.get("group_type", "all")),
        group_value=str(d.get("group_value", "all")),
        bin_lower=_opt_float(d.get("bin_lower")),
        bin_upper=_opt_float(d.get("bin_upper")),
        n_samples=int(d.get("n_samples", 0)),
        n_events=_opt_int(d.get("n_events")),
        n=int(d.get("n", 0)),
        sum_error=float(d.get("sum_error", 0.0)),
        sum_abs_error=float(d.get("sum_abs_error", 0.0)),
        sum_squared_error=float(d.get("sum_squared_error", 0.0)),
        a=int(d.get("a", 0)),
        b=int(d.get("b", 0)),
        c=int(d.get("c", 0)),
        d=int(d.get("d", 0)),
        fss_numerator_sum=float(d.get("fss_numerator_sum", 0.0)),
        fss_forecast_fraction_sq_sum=float(d.get("fss_forecast_fraction_sq_sum", 0.0)),
        fss_observed_fraction_sq_sum=float(d.get("fss_observed_fraction_sq_sum", 0.0)),
        brier_n=int(d.get("brier_n", 0)),
        sum_brier_terms=float(d.get("sum_brier_terms", 0.0)),
        sum_reference_brier_terms=float(d.get("sum_reference_brier_terms", 0.0)),
        sum_predicted_probability=float(d.get("sum_predicted_probability", 0.0)),
        n_observed_events=int(d.get("n_observed_events", 0)),
        pinball_q10_sum=float(d.get("pinball_q10_sum", 0.0)),
        pinball_q50_sum=float(d.get("pinball_q50_sum", 0.0)),
        pinball_q90_sum=float(d.get("pinball_q90_sum", 0.0)),
        range_n=int(d.get("range_n", 0)),
        coverage_count=int(d.get("coverage_count", 0)),
    )


def validate_parquet_schema(
    file_schema: pa.Schema,
    expected_schema: pa.Schema = COMPONENT_PYARROW_SCHEMA,
) -> None:
    """Validates that a file's schema matches the deterministic component schema."""
    expected_names = set(expected_schema.names)
    file_names = set(file_schema.names)

    missing = expected_names - file_names
    if missing:
        raise IncompatibleSchemaError(
            f"Parquet file schema is missing required columns: {sorted(list(missing))}"
        )

    for field in expected_schema:
        idx = file_schema.get_field_index(field.name)
        if idx == -1:
            raise IncompatibleSchemaError(f"Missing column '{field.name}' in Parquet schema.")
        file_field = file_schema.field(idx)
        if file_field.type != field.type:
            raise IncompatibleSchemaError(
                f"Column '{field.name}' type mismatch: expected {field.type}, got {file_field.type}"
            )


def save_components(
    components: Union[Sequence[VerificationComponent], Iterable[VerificationComponent]],
    target: Optional[Union[str, Path]] = None,
    replace: bool = False,
) -> int:
    """Saves a collection of VerificationComponent records to Parquet files.
    
    Args:
        components: List or iterable of VerificationComponent objects.
        target: Target directory or file path. If None, defaults to data/verification/components/.
        replace: If True, replaces records with matching identities. If False, raises DuplicateComponentError.
        
    Returns:
        The total count of components saved.
    """
    comp_list: List[VerificationComponent] = list(components)
    if not comp_list:
        return 0

    # 1. Check for duplicates within the input batch
    seen_in_batch: Dict[Tuple[Any, ...], VerificationComponent] = {}
    for i, c in enumerate(comp_list):
        ident = get_component_identity(c)
        if ident in seen_in_batch:
            if not replace:
                raise DuplicateComponentError(
                    f"Duplicate component identity within input batch at index {i}: {ident}"
                )
        seen_in_batch[ident] = c

    batch_to_write = list(seen_in_batch.values()) if replace else comp_list

    # 2. Determine file partition targets
    base_path = Path(target) if target is not None else get_default_components_dir()
    is_direct_file = str(base_path).endswith(".parquet") or (base_path.exists() and base_path.is_file())

    file_groups: Dict[Path, List[VerificationComponent]] = defaultdict(list)
    if is_direct_file:
        file_groups[base_path].extend(batch_to_write)
    else:
        for c in batch_to_write:
            f_path = (
                base_path
                / f"evaluation_set={c.evaluation_set}"
                / f"forecast_type={c.forecast_type}.parquet"
            )
            file_groups[f_path].append(c)

    total_saved = 0

    # 3. Write/append to each target partition file
    for file_path, group_comps in file_groups.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)

        existing_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        if file_path.exists():
            try:
                table = pq.read_table(file_path, partitioning=None)
            except Exception as e:
                raise IncompatibleSchemaError(
                    f"Failed to read existing Parquet file '{file_path}': {e}"
                ) from e

            validate_parquet_schema(table.schema)

            for row in table.to_pylist():
                ident = get_component_identity(row)
                existing_map[ident] = row

        # Apply append / replace logic
        for c in group_comps:
            ident = get_component_identity(c)
            if ident in existing_map and not replace:
                raise DuplicateComponentError(
                    f"Component identity already exists in '{file_path}': {ident}. Use replace=True to overwrite."
                )
            existing_map[ident] = component_to_dict(c)

        combined_rows = list(existing_map.values())

        # Build PyArrow Table with strictly deterministic schema
        out_table = pa.Table.from_pylist(combined_rows, schema=COMPONENT_PYARROW_SCHEMA)
        pq.write_table(out_table, file_path)
        total_saved += len(group_comps)

    return total_saved


def _matches_filter(val: Any, filter_val: Any, is_float: bool = False) -> bool:
    """Helper for evaluating read filters."""
    if filter_val is _UNSET:
        return True
    if filter_val is None:
        return val is None
    if isinstance(filter_val, (list, tuple, set)):
        if is_float:
            return any(
                val is not None and abs(float(val) - float(fv)) < 1e-4
                for fv in filter_val
            )
        else:
            return val in filter_val
    if is_float:
        if val is None:
            return False
        return abs(float(val) - float(filter_val)) < 1e-4
    return val == filter_val


def load_components(
    target: Optional[Union[str, Path]] = None,
    forecast_type: Any = _UNSET,
    evaluation_set: Any = _UNSET,
    lead_day: Any = _UNSET,
    threshold_mm: Any = _UNSET,
    neighbourhood_cells: Any = _UNSET,
    group_type: Any = _UNSET,
    group_value: Any = _UNSET,
) -> List[VerificationComponent]:
    """Reads VerificationComponent records from Parquet storage with optional filtering.
    
    Args:
        target: Parquet file or root directory to search. Defaults to data/verification/components/.
        forecast_type: Filter by forecast_type (str or collection of str).
        evaluation_set: Filter by evaluation_set (str or collection of str).
        lead_day: Filter by lead_day (int or collection of int).
        threshold_mm: Filter by threshold_mm (float, collection, or None for continuous).
        neighbourhood_cells: Filter by neighbourhood_cells (int, collection, or None).
        group_type: Filter by group_type (str or collection).
        group_value: Filter by group_value (str or collection).
        
    Returns:
        List of matching VerificationComponent instances.
    """
    base_path = Path(target) if target is not None else get_default_components_dir()
    if not base_path.exists():
        return []

    if base_path.is_file():
        files_to_read = [base_path]
    else:
        files_to_read = sorted(list(base_path.rglob("*.parquet")))

    if not files_to_read:
        return []

    # Optional partition path level pruning
    candidate_files = []
    for fp in files_to_read:
        p_str = str(fp)
        if evaluation_set is not _UNSET and evaluation_set is not None:
            eval_list = [evaluation_set] if isinstance(evaluation_set, str) else list(evaluation_set)
            if "evaluation_set=" in p_str and not any(f"evaluation_set={e}" in p_str for e in eval_list):
                continue
        if forecast_type is not _UNSET and forecast_type is not None:
            ft_list = [forecast_type] if isinstance(forecast_type, str) else list(forecast_type)
            if "forecast_type=" in p_str and not any(f"forecast_type={ft}" in p_str for ft in ft_list):
                continue
        candidate_files.append(fp)

    results: List[VerificationComponent] = []

    for file_path in candidate_files:
        try:
            table = pq.read_table(file_path, partitioning=None)
        except Exception as e:
            raise IncompatibleSchemaError(
                f"Failed to read Parquet file '{file_path}': {e}"
            ) from e

        validate_parquet_schema(table.schema)

        rows = table.to_pylist()
        for row in rows:
            # Apply row-level filters
            if not _matches_filter(row.get("forecast_type"), forecast_type):
                continue
            if not _matches_filter(row.get("evaluation_set"), evaluation_set):
                continue
            if not _matches_filter(row.get("lead_day"), lead_day):
                continue
            if not _matches_filter(row.get("threshold_mm"), threshold_mm, is_float=True):
                continue
            if not _matches_filter(row.get("neighbourhood_cells"), neighbourhood_cells):
                continue
            if not _matches_filter(row.get("group_type"), group_type):
                continue
            if not _matches_filter(row.get("group_value"), group_value):
                continue

            results.append(dict_to_component(row))

    return results
