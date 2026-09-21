# Features, static geography and district products (Stage 3)

Owner: **M1**. PRD 8.6, 9.4 to 9.6, 14. Code: `data_pipeline/features/` and `data_pipeline/districts/`.
If this file and the PRD differ, the PRD wins. Where the PRD leaves a detail open, the choice made is stated here.

**Feature version: `FEATURE_VERSION = "v1"`** (`data_pipeline/features/library.py`). It is stored in the column
`feature_set_version` of every feature row. Change it whenever a definition below changes.

## Metric convention (all distances, gradients, vorticity)

One 0.25 degree IMD cell is **27.75 km** north-south and **27.75 x cos(latitude) km** east-west, using the cell's
own latitude (PRD 8.6). Nothing is computed in raw degrees. The implied earth radius (27.75 km per 0.25 degree)
is `R = 6.36e6 m`. The one exception is district **areas**, which use the equal-area **EPSG:6933** projection.
Finite differences use the central difference; at the edge of the grid or next to a non-valid cell they use a
one-sided difference; if no neighbour exists on that axis the result is NaN.

## The 27 features (`FEATURE_COLUMNS`, PRD 9.4 order)

The Golden Dataset holds **valid cells only**, so a "missing" neighbour is a non-valid cell (sea, outside India).

| Feature | Definition | Unit | Null when |
|---|---|---|---|
| `rain_mm` | window rain of the run/lead on the IMD grid (C1) | mm | never |
| `nbr_mean_3`, `nbr_mean_5` | mean of `rain_mm` over the 3 x 3 / 5 x 5 **cells** around the cell (itself included), valid cells only | mm | never |
| `nbr_max_3`, `nbr_max_5` | maximum over the same windows | mm | never |
| `rain_grad` | size of the horizontal gradient of `rain_mm` (metric grid) | mm km-1 | no valid neighbour on one axis |
| `rain_prev_lead`, `rain_next_lead` | `rain_mm` of lead-1 / lead+1 of the **same run**, same cell | mm | lead 1 (prev), lead 3 (next), or that lead is absent |
| `u850`, `v850` | window-mean wind at 850 hPa | m s-1 | never |
| `wspd850` | `sqrt(u850^2 + v850^2)` | m s-1 | never |
| `vort850` | relative vorticity `dv/dx - du/dy + u tan(lat)/R` from `u850`, `v850` by finite differences | s-1 | no valid neighbour on one axis |
| `q850` | window-mean specific humidity | kg kg-1 | never |
| `msl` | window-mean sea-level pressure (as stored in TIGGE) | Pa | never |
| `shear_200_850` | `sqrt((u200-u850)^2 + (v200-v850)^2)` | m s-1 | never |
| `elevation_m` | TIGGE `orog` on the IMD grid | m | never |
| `slope` | size of the height gradient, central differences | m m-1 | never |
| `aspect_sin`, `aspect_cos` | direction of the **downhill** slope as a bearing clockwise from north: `sin = -dz/dx / slope`, `cos = -dz/dy / slope`; flat cell: 0 and 0 | 1 | never |
| `dist_coast_km` | distance from a land cell (`lsm >= 0.5`) to the nearest sea cell (`lsm < 0.5`); sea cells 0 | km | no sea cell on the grid |
| `clim_mean`, `clim_p95` | mean and 95th percentile of observed IMD rain on JJAS days of the **training seasons**, missing days ignored | mm | the cell has no observed day |
| `doy_sin`, `doy_cos` | `sin/cos(2 pi doy / 365.25)`, `doy` = day of year of the **forecast start date** (the same for all three leads of a run) | 1 | never |
| `lead_day` | 1, 2, 3 | day | never |
| `latitude`, `longitude` | cell centre | degrees | never |

Choices where the PRD is open: neighbourhood windows are counted in cells and only valid cells inside the window
are used (as in PRD 16); `rain_grad` is in mm per km; `doy` uses the start date, not the IMD date, so it does not
depend on the `imd_stamp` decision; distance to coast is an exact nearest-cell search under the PRD's km-per-cell
conversion (a distance transform cannot use an east-west cell size that changes with latitude).

## Feature table (PRD 9.6) and files

Parquet, one row per `(run_id, lead_day, cell_id)`, files
`<DATA_DIR>/features/cell_features/season_<YYYY>/cell_features_<YYYYMM>.parquet`. Columns (`TABLE_COLUMNS`):
`run_id`, `lead_day`, `cell_id`, `season`, `feature_set_version`, the other 26 features, then the **targets**
`obs_mm`, `obs_ge_15_6`, `obs_ge_64_5`, `obs_ge_115_6` (bool, NULL where `obs_mm` is missing). Targets are not
inputs: `feature_matrix(table)` returns exactly the 27 model inputs in order. Types: float32, `lead_day` int8,
`cell_id` int32, `season` int16. Regime features are not here (M2).

## Cached static products (`<DATA_DIR>/features/`)

| File | Content | Recomputed when |
|---|---|---|
| `static/static_geography.parquet` | `cell_id, elevation_m, slope, aspect_sin, aspect_cos, dist_coast_km` for all 17,415 cells | the TIGGE `orog`/`lsm` fields change (input hash) |
| `climatology/climatology_<hash>.parquet` | `cell_id, clim_mean, clim_p95`; the seasons used are in the file metadata | never for the same season set |
| `districts/cell_district_weights.parquet`, `districts/district_summary.parquet` | see below | districts, valid cells or settings change (input hash) |

**Climatology has no default seasons.** Pass the training seasons; `assert_no_holdout` (also run by
`build_stage3_season(holdout_seasons=...)`) refuses a climatology that used a holdout season (PRD L4). Real
climatology needs the IMD files of the training seasons on disk; nothing is invented when they are missing.

## District weights (PRD 14.2)

`w_ij = area(cell i ∩ district j) / area(district j)`, areas in **EPSG:6933**, cells are exact 0.25 degree squares.
Non-valid cells are dropped and the rest **renormalised to sum to 1** per district. `is_main = area_weight >= 0.05`.

`cell_district_weights`: `district_id, cell_id, overlap_km2, area_weight, is_main`.
`district_summary`: `district_id, name, state, district_area_km2, valid_area_fraction, n_cells, n_main_cells,
n_effective_cells (= 1 / sum w^2), is_small (= n_effective_cells < 4), centroid_lat, centroid_lon`.
Districts with no valid cell get no rows (listed in a warning).

**PRD observation:** with `w_min = 0.05`, a district of more than about 20 similar cells has **no main cell**
(every weight is below 0.05), so its max-over-S fields are NULL. See the report for the decision this needs.

## District products

`district_forecasts`, key `(run_id, lead_day, district_id)`, files
`<DATA_DIR>/features/district_forecasts/season_<YYYY>/district_forecasts_<YYYYMM>.parquet`.

| Column | Definition | Now |
|---|---|---|
| `imd_date`, `season` | from the Golden Dataset | filled |
| `raw_mean_mm` | `sum w_i * rain_i` | filled |
| `observed_mean_mm` | `sum w_i * obs_i`; NULL if IMD is missing in any cell of the district | filled |
| `observed_max_cell_mm` | max `obs_i` over the main cells; NULL if IMD is missing in a main cell or there is no main cell | filled |
| `observed_heavy_area_fraction` | `sum w_i * 1[obs_i >= 64.5]`; NULL if IMD is missing in any cell | filled |
| `raw_wettest_cell_id`, `raw_wettest_cell_mm` | main cell with the most **raw** rain (ties: lowest `cell_id`) | filled |
| `n_effective_cells`, `n_main_cells`, `is_small` | from the weights | filled |
| `alignment_offset_hours`, `imd_stamp` | provenance | filled |
| `corrected_mean_mm` | needs the bias-corrected model | **NULL** (Stage 4) |
| `wettest_cell_id`, `wettest_cell_mean_mm`, `wettest_cell_q10/q50/q90_mm` | PRD 14.4 defines them on the corrected mean | **NULL** (Stage 4) |
| `heavy_prob_max_cell`, `very_heavy_prob_max_cell`, `heavy_area_fraction_expected`, `very_heavy_area_fraction_expected` | need calibrated cell probabilities | **NULL** (Stage 4) |

`attention_level`, `priority_rank`, `product_type`, `fallback_*`, `model_version_id` (M4) are not in this table.

`district_history` (PRD 20.2), key `(run_id, lead_day, district_id)`, same layout: `imd_date, season,
observed_mean_mm, observed_max_cell_mm, raw_mean_mm`, plus `corrected_mean_mm`, `prediction_source`, `phase`,
`lps_near`, which are **NULL** until the later stages fill them. Only rows with a known `observed_mean_mm` are kept.
