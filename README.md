# SIH26080

Smart India Hackathon 2026, problem statement 26080: regime-aware AI post-processing of monsoon
rainfall forecasts. We take the ECMWF control forecast (TIGGE), correct its rainfall with
XGBoost models that also know the monsoon regime (active/break phase, low-pressure systems, coast
and mountain influence), and show district-level products with honest verification against IMD
rainfall. The system runs in **replay mode only**: it shows stored past forecasts.

Everything is specified in [`docs/PRD.md`](docs/PRD.md). If this README and the PRD disagree, the
PRD wins. The work plan and the gates (G0 to G7) are in PRD section 24.

## Setup (every builder)

From PRD section 22.2. macOS or Linux (Windows: WSL2 with Ubuntu). CPU only; 8 GB RAM is enough.

1. Install **Miniforge** (conda-forge). From the repository root create the environment
   (run it from the root, because `environment.yml` ends with `-e .`):
   `conda env create -f environment.yml`, then `conda activate sih-rain`.
2. Install **PostgreSQL 14 or newer** locally (macOS: Homebrew or Postgres.app; Linux: package
   manager). Create the database `sih_rain` and the user `sih`, and apply the schema. See
   [`database/README.md`](database/README.md). No Docker, no PostGIS.
3. Install **Node.js 20 LTS**.
4. Register on the **ECMWF Data Store (ECDS)**. From your profile page, copy the exact lines shown
   for the API into `~/.cdsapirc` (do not type them from memory; they change).
   **`~/.cdsapirc` holds your personal key. It must NEVER be committed** or pasted anywhere public.
5. Clone the repository. Copy `.env.example` to `.env` and edit it. Never commit `.env`.
6. Run the checks: `python scripts/check_setup.py` (imports, xgboost >= 2.0, PostgreSQL
   connection, Node >= 20), then `cd frontend && npm install && npm run dev`.

## Run

Backend (from the repository root, with the conda environment active):

```bash
uvicorn backend.app.main:app --port 8000
# check: curl http://localhost:8000/api/v1/health
```

Frontend:

```bash
cd frontend
cp .env.example .env        # VITE_API_BASE_URL=http://localhost:8000/api/v1
npm install
npm run dev
```

Tests: `python -m pytest`

## Who owns what

From PRD section 21.1. Folders the PRD does not assign to one person say so.

| Folder | Owner | Notes |
|---|---|---|
| `data_pipeline/` | M1 | Downloads, alignment, golden dataset, feature library, district weights and fields |
| `regime_engine/` | M2 | Labels, phase model, low-pressure detector, coast/mountain indices, analogs (F3), transition detection (F4) |
| `ml/` | M3 | B0 to B3, settings search, model records; `ml/models/` holds model files (git-ignored) |
| `probability/` | M3 and M4 | M3: probability and range models. M4: calibration and coverage checks (PRD 13) |
| `protocol/` | M4 | Split logic, cross-fit runner, `run_holdout.py` and the holdout lock (M3 helps with the models) |
| `verification/` | M4 | Metrics, statistics, reports, plots |
| `backend/` | M1 | From gate G2. Until then M5 uses the mock API |
| `database/` | M1 | `schema.sql` |
| `frontend/` | M5 | The whole frontend |
| `mock_api/` | M5 | Not in the PRD tree (see its README) |
| `config/` | shared | PRD gives no per-file owner. Each file belongs to the module it configures: `alignment`, `districts`, `regions` M1; `regime` M2; `protocol`, `thresholds`, `verification` M4 (this split is our reading of section 21.1) |
| `docs/` | shared | `data-status.md` and `data-sources.md` M1; `event-counts.md` M4; `data-contracts.md` everyone who uses a contract |
| `tests/` | shared | Each owner tests their own area (PRD section 23) |
| `scripts/` | shared | Not in the PRD tree (see its README) |

## Where to look next

- Full specification: [`docs/PRD.md`](docs/PRD.md). Start with section 2 (fixed decisions), 3
  (open checks) and 5 (wording and honesty rules).
- Gates and work plan: PRD section 24.
- How to contribute and who reviews what: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Contracts: [`docs/data-contracts.md`](docs/data-contracts.md).
