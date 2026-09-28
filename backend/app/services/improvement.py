"""F1 improvement summary (PRD 15 F1): on one run and lead, in how many cells and districts is the corrected
forecast closer to the IMD observation than the raw NWP. The comparison logic is `verification/improvement.py`
(pure numpy, tie-safe); this module only loads the three cell arrays (raw and observed from the Golden Dataset,
corrected from the served store) and the district means from the database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
from sqlalchemy import Connection, select

from backend.app.db.tables import district_forecasts as f
from backend.app.errors import ApiError
from backend.app.services.grid import _read, corrected_file, golden_file
from verification.improvement import (
    CLOSER_CORRECTED,
    CLOSER_RAW,
    compute_grid_improvements,
    evaluate_district_improvement,
)


def _by_cell(table: pa.Table, column: str) -> dict[int, float]:
    return dict(zip(table["cell_id"].to_pylist(), table[column].to_pylist(), strict=True))


def improvement_summary(
    conn: Connection, data_dir: Path, golden_dir: Path, run_id: str, lead_day: int
) -> dict:
    gold = _read(golden_file(golden_dir, run_id), ["cell_id", "rain_mm", "obs_mm", "imd_date"], run_id, lead_day)
    cor = _read(corrected_file(data_dir / "serving" / "corrected", run_id), ["cell_id", "corrected_mean_mm"], run_id, lead_day)
    if gold is None or cor is None:
        raise ApiError(503, "GRID_DATA_UNAVAILABLE", "Cell-level data for this run is not available.")

    ids = gold["cell_id"].to_pylist()
    corrected = _by_cell(cor, "corrected_mean_mm")
    raw = np.array(gold["rain_mm"].to_pylist(), dtype=float)
    obs = np.array(gold["obs_mm"].to_pylist(), dtype=float)  # nulls -> NaN: excluded from the comparison
    corr = np.array([corrected.get(c, np.nan) if corrected.get(c) is not None else np.nan for c in ids], dtype=float)
    _, _, cells = compute_grid_improvements(raw, corr, obs)

    rows = conn.execute(
        select(f.c.raw_mean_mm, f.c.corrected_mean_mm, f.c.observed_mean_mm).where(
            f.c.run_id == run_id, f.c.lead_day == lead_day
        )
    ).all()
    closer = [
        d.closer_system
        for d in (evaluate_district_improvement(r.raw_mean_mm, r.corrected_mean_mm, r.observed_mean_mm) for r in rows)
        if d is not None
    ]
    return {
        "imd_date": gold["imd_date"][0].as_py(),
        "total_valid_cells": cells.compared_cells,
        "cells_corrected_closer": cells.corrected_closer_cells,
        "cells_raw_closer": cells.raw_closer_cells,
        "cells_no_change": cells.tied_cells,
        "total_districts": len(closer),
        "districts_corrected_closer": closer.count(CLOSER_CORRECTED),
        "districts_raw_closer": closer.count(CLOSER_RAW),
    }
