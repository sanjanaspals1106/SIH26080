# backend/

FastAPI service (PRD section 18). Owner: **M1**. It **reads stored results**; it never trains a model, builds
features, computes weights or regrids anything, and has no endpoint that changes data.

```text
OFFLINE (batch scripts)                                  ONLINE (this service)
raw data -> Stage 1 -> Stage 2 Golden -> Stage 3   ->   scripts/load_m1.py -> PostgreSQL -> FastAPI -> JSON
                                          |                                                  ^
                                          +---- Golden Parquet (cell level, PRD D16) --------+
```

- **District level** (runs, district forecasts, districts, map outlines) comes from **PostgreSQL**.
- **Cell level** (`/forecasts/grid`) comes from the **Golden Dataset Parquet** files (`<DATA_DIR>/golden/`), read
  with `pyarrow`. Cell data is not copied into the database (PRD D16).
- Layout: `app/api/` routes, `app/services/` queries, `app/schemas/` response models, `app/db/` tables, session and
  the offline loader (`loader.py`; the API never imports it).

## Environment variables

| Variable | Used for | Example |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection (API and loader) | `postgresql+psycopg://sih:CHANGE_ME@localhost:5432/sih_rain` |
| `DATA_DIR` | where the Parquet files live (default `./data`) | `./data` |
| `TEST_DATABASE_URL` | optional, tests only: also run the database tests on a real PostgreSQL | `postgresql+psycopg://sih@localhost:5432/sih_rain_test` |

Set them in your shell or in `.env` (copy `.env.example`; never commit `.env`). Nothing is hard-coded.

## Set up the database (once)

Install PostgreSQL 14+ locally, then follow [`database/README.md`](../database/README.md):

```bash
psql -U sih -h localhost -d sih_rain -f database/schema.sql
```

## Load the M1 products (after the pipeline has run)

```bash
python scripts/load_m1.py                       # all seasons found on disk
python scripts/load_m1.py --seasons 2025        # only some seasons
python scripts/load_m1.py --districts <file>    # district file, if source.file is not yet set in config/districts.yaml
```

It reads the Golden Dataset and the Stage 3 tables/caches and upserts: **run it as often as you like**, nothing is
duplicated. Only M1's columns are written. Columns that later stages own (`corrected_mean_mm`, probabilities,
`attention_level`, `phase`, `lps_near`, ...) stay NULL and are never overwritten by a reload.

| Table | Content |
|---|---|
| `grid_cells` | all 17,415 IMD cells: `latitude`, `longitude`, `is_valid`, `elevation_m`, `slope`, `dist_coast_km` |
| `districts` | name, state, simplified GeoJSON outline (Feature; `properties.source_id` if the file has one), centroid, `n_effective_cells`, `is_small`, `source_year` (2011 census basis) |
| `cell_district_weights` | Stage 3 weights (`area_weight`, `is_main`); replaced as a whole on each load |
| `nwp_runs` | run id, source, initialization time, season, alignment method/offset, `imd_stamp`; `evaluation_set` is set later by the protocol |
| `district_forecasts` | `imd_date`, `raw_mean_mm`, `observed_mean_mm`, `is_small`; everything model-derived is NULL |
| `district_history` | past outcomes (`observed_mean_mm`, `observed_max_cell_mm`, `raw_mean_mm`) |

Keys: `district_forecasts` and `district_history` are unique per `(run_id, lead_day, district_id)` and point at
`nwp_runs` (`schema.sql`, "Stage 4 additions"). Stage 3 columns with no place in the PRD tables
(`observed_heavy_area_fraction`, `raw_wettest_cell_*`, `n_main_cells`) stay in Parquet.

## Start the API

```bash
uvicorn backend.app.main:app --port 8000        # docs: http://localhost:8000/docs
```

## Endpoints (base URL `/api/v1`, all `GET`)

| Endpoint | Returns |
|---|---|
| `/health` | status, version, `database`: `ok` / `unavailable` / `not_configured` |
| `/runs`, `/runs/{run_id}` | replay runs (filter `season`; paginated) and one run's details |
| `/forecasts/districts` | district table; filters `run_id`, `lead_day` (1-3), `district_id`, `state`, `imd_date`, `season`; `limit` (max 1000), `offset` |
| `/forecasts/districts/{district_id}` | one district and its forecasts (optionally for a run/lead/date) |
| `/forecasts/grid?run_id=&lead_day=&variable=` | one cell layer from Parquet. Only `raw` and `observed` exist at this stage; the other PRD layers answer `503 VARIABLE_NOT_AVAILABLE` |
| `/map/metadata` | grid bounds and size, layers and whether they exist, leads, thresholds, run/date ranges, district source (2011 census) |
| `/map/districts` | GeoJSON FeatureCollection of the stored, simplified outlines |

Errors follow PRD 18.3: `{"error": {"code": "...", "message": "..."}}` with 404 (`RUN_NOT_FOUND`,
`DISTRICT_NOT_FOUND`, `FORECAST_NOT_FOUND`), 422 (`INVALID_PARAMETER`), 503 (`DATABASE_NOT_CONFIGURED`,
`DATABASE_UNAVAILABLE`, `VARIABLE_NOT_AVAILABLE`, `GRID_DATA_UNAVAILABLE`). Database errors are logged, never returned.

```bash
curl localhost:8000/api/v1/health
curl "localhost:8000/api/v1/runs?season=2025&limit=10"
curl "localhost:8000/api/v1/forecasts/districts?run_id=tigge_ecmwf_cf_2025071500&lead_day=1&state=Kerala"
curl "localhost:8000/api/v1/forecasts/grid?run_id=tigge_ecmwf_cf_2025071500&lead_day=1&variable=observed"
curl localhost:8000/api/v1/map/metadata
```

Answers carry `"mode": "replay"`. Fields that later stages own are `null`, not invented; `evaluation_set` is `null`
until the protocol (M4) assigns it.

## Tests

`python -m pytest tests/test_backend_*.py` runs against SQLite with synthetic data (no server needed). With a local
PostgreSQL, add `TEST_DATABASE_URL=postgresql+psycopg://<user>@localhost:5432/<empty_db>`: the same tests then also run
against PostgreSQL, each test in its own schema built from the real `database/schema.sql`.
