"""Stage 2: temporal + spatial alignment, valid cells and the Golden Dataset (PRD 8, 9.3). Owner: M1.

Consumes the normalised objects of `data_pipeline.ingestion`; never talks to ECDS or IMD itself.
"""

from data_pipeline.alignment.cells import (
    compute_valid_cells,
    get_valid_cells,
    imd_grid,
    load_valid_cells,
    save_valid_cells,
)
from data_pipeline.alignment.golden import (
    GOLDEN_COLUMNS,
    GOLDEN_SCHEMA,
    align_static,
    atmosphere_fullgrid_windows,
    build_golden_season,
    check_mask_consistency,
    golden_rows,
    read_golden,
    validate_golden,
    write_golden,
)
from data_pipeline.alignment.lag import LagTestResult, decide_imd_stamp, lag_test
from data_pipeline.alignment.spatial import (
    GridMap,
    build_grid_map,
    regrid_bilinear,
    regrid_rain,
)
from data_pipeline.alignment.temporal import (
    c1_atmosphere_windows,
    c1_rain_windows,
    imd_date,
    window_hours,
)

__all__ = [
    "GOLDEN_COLUMNS",
    "GOLDEN_SCHEMA",
    "GridMap",
    "LagTestResult",
    "align_static",
    "atmosphere_fullgrid_windows",
    "build_golden_season",
    "build_grid_map",
    "c1_atmosphere_windows",
    "check_mask_consistency",
    "c1_rain_windows",
    "compute_valid_cells",
    "decide_imd_stamp",
    "get_valid_cells",
    "golden_rows",
    "imd_date",
    "imd_grid",
    "lag_test",
    "load_valid_cells",
    "read_golden",
    "regrid_bilinear",
    "regrid_rain",
    "save_valid_cells",
    "validate_golden",
    "window_hours",
    "write_golden",
]
