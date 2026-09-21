"""Small Parquet helpers for the Stage 3 tables. Owner: M1.

Same layout rule as the Golden Dataset: one file per (season, start month) under
`<DATA_DIR>/features/<table>/season_<YYYY>/<table>_<YYYYMM>.parquet`, fixed schema, rows sorted by the key,
atomic replace. Writing a month again replaces its file, so re-running never creates duplicates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import MissingInputError, ValidationError

log = logging.getLogger(__name__)


def write_month_tables(
    df: pd.DataFrame,
    table: str,
    schema: pa.Schema,
    key: list[str],
    month_of: pd.Series,
    config: IngestionConfig | None = None,
) -> list[Path]:
    """Write `df` (already in `schema` column order) as one file per season and month. `month_of` gives 'YYYYMM'."""
    config = config or load_config()
    if list(df.columns) != schema.names:
        raise ValidationError(f"{table}: columns differ from the schema {schema.names}.")
    if df.duplicated(key).any():
        raise ValidationError(f"{table}: duplicate rows for key {key}.")
    df = df.sort_values(key, kind="stable")
    paths = []
    for yyyymm, part in df.groupby(month_of.loc[df.index], sort=True):
        path = config.features.dir / table / f"season_{yyyymm[:4]}" / f"{table}_{yyyymm}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        pq.write_table(
            pa.Table.from_pandas(part, schema=schema, preserve_index=False).replace_schema_metadata(None),
            tmp,
            compression="zstd",
        )
        tmp.replace(path)
        paths.append(path)
        log.info("wrote %s (%d rows)", path, len(part))
    return paths


def read_month_tables(
    table: str, schema: pa.Schema, config: IngestionConfig | None = None, seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    """Read a Stage 3 table (all seasons or the given ones) in its schema."""
    config = config or load_config()
    root = config.features.dir / table
    dirs = [root / f"season_{s}" for s in seasons] if seasons is not None else sorted(root.glob("season_*"))
    files = sorted(f for d in dirs for f in d.glob(f"{table}_*.parquet"))
    if not files:
        raise MissingInputError(
            f"No '{table}' files under {root}. Build them with scripts/build_features.py."
        )
    table = pads.dataset([str(f) for f in files], schema=schema, format="parquet").to_table()
    # nullable booleans (targets, flags) come back as pandas "boolean", so NULL is pd.NA, not None/NaN
    return table.to_pandas(date_as_object=False, types_mapper={pa.bool_(): pd.BooleanDtype()}.get)
