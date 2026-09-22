# Golden Dataset (Stage 2 output)

Owner: **M1**. PRD 9.3 and 9.7. Code: `data_pipeline/alignment/golden.py` (schema in `GOLDEN_SCHEMA`).
If this file and the PRD differ, the PRD wins.

**One row per `(run_id, lead_day, cell_id)`**, valid IMD cells only (PRD 8.5). No features, no regime
labels, no model output. Read it with `read_golden(config, seasons=None)`.

## Files

`<DATA_DIR>/golden/season_<YYYY>/golden_<YYYYMM>.parquet`: one file per season and start month, zstd,
fixed schema, rows sorted by `(run_id, lead_day, cell_id)`. Writing a month again replaces its file, so
re-running never creates duplicates. `<DATA_DIR>/golden/grid_cells.parquet` holds the grid and valid-cell
mask (`cell_id`, `latitude`, `longitude`, `is_valid`, `valid_fraction`; base period and day count are in the
file metadata). All of it is git-ignored (`/data/`).

## Grid and `cell_id`

IMD 0.25 degree grid, 129 x 135, lat 6.5 to 38.5, lon 66.5 to 100.0.
`cell_id = i_lat * 135 + i_lon` (0-based, counted from lat 6.5 / lon 66.5, both ascending). It is defined on
the full grid, so it does not change when the valid-cell mask changes. `latitude`/`longitude` are identical
for a given `cell_id` in every table (T7).

## Columns

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | `tigge_ecmwf_cf_YYYYMMDDHH` (PRD 9.1) |
| `source` | string | `ECMWF-TIGGE-control` |
| `initialization_time` | timestamp[us, UTC] | forecast start, always 00 UTC |
| `season` | int16 | year of the start date |
| `lead_day` | int8 | 1, 2 or 3 |
| `imd_date` | date32 | IMD date paired with this lead: I + k (`end_date`) or I + k - 1 (`start_date`) |
| `window_start_utc`, `window_end_utc` | timestamp[us, UTC] | C1 window, hours 24(k-1)+3 to 24k+3 after the start |
| `alignment_method` | string | `C1` |
| `alignment_offset_hours` | int8 | 0 for C1 (3 for C0, which is not implemented) |
| `imd_stamp` | string | `end_date` or `start_date` (decided by the T4 lag test) |
| `unit` | string | `mm` (unit of `rain_mm` and `obs_mm`) |
| `grid` | string | `IMD_0.25` |
| `cell_id` | int32 | see above |
| `latitude`, `longitude` | float32 | cell centre, degrees |
| `rain_mm` | float32 | forecast window rain on the IMD grid, mm (kg m-2). Cumulative `tp` differenced, never below 0 |
| `obs_mm` | float32 | IMD observed rain for `imd_date`, mm. NaN where IMD is missing (-999) or the date is not in the IMD data |
| `msl` | float32 | mean sea-level pressure, window mean, Pa (as stored in TIGGE) |
| `u850`, `v850`, `u200`, `v200` | float32 | wind components, window mean, m s-1 |
| `q850` | float32 | specific humidity, window mean, kg kg-1 |
| `orog` | float32 | TIGGE orography on the IMD grid (bilinear). This is `elevation_m` of PRD 8.6 (gpm/m as stored) |
| `lsm` | float32 | TIGGE land-sea mask on the IMD grid (bilinear), 0 to 1 |
| `rain_clipped` | bool | a source window value was below 0 and was set to 0 (T2) |
| `obs_missing` | bool | `obs_mm` is NaN. Rows are kept; training code decides what to drop |
| `rain_regrid` | string | `area_mean` (source finer or equal to 0.25) or `containing_cell` (coarser) |

Units are as stored in TIGGE and IMD; nothing is converted. The window atmosphere is the mean of the
6-hourly steps inside the window: 6..24, 30..48, 54..72 h for leads 1, 2, 3.

## What is not here

`slope`, `aspect_*`, `dist_coast_km`, `coast_normal`, `region_code`, neighbourhood and gradient rain features,
vorticity, wind speed, shear, climatology, time features, regime features, districts (Stage 3 and later).
