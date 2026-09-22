# scripts/

Helper scripts that are not part of any importable package.

**This folder is not in the PRD repository tree (section 22.5).** It was added at setup for
`check_setup.py`. It has no single owner in the PRD; treat it as shared.

| Script | What it does |
|---|---|
| `check_setup.py` | PRD 22.2 step 6: checks imports, `xgboost >= 2.0`, the PostgreSQL connection (`DATABASE_URL`) and `node --version >= 20`. Prints PASS or FAIL for each item. |
| `ingest_example.py` | Stage 1 (PRD 7, 9.1): runs the TIGGE, IMD and district ingestion on small synthetic files in a temporary folder. Needs no credentials and no network. |
| `golden_example.py` | Stage 2 (PRD 8, 9.3): synthetic Stage 1 files -> valid-cell mask -> C1 alignment -> IMD grid -> Golden Dataset, in a temporary folder. No credentials, no network. |
| `build_golden.py` | Stage 2 on real data: `mask` (valid cells from IMD 1981-2010), `season <year>` (Golden Dataset), `lag <year>` (T4 lag test). Reads the raw files Stage 1 downloaded; downloads nothing. |
| `features_example.py` | Stage 3 (PRD 9.4, 14): synthetic Stage 1 and 2 data -> static geography, climatology, district weights, the 27 features, district forecasts and history, in a temporary folder. No credentials, no network. |
| `build_features.py` | Stage 3 on real data: `static`, `climatology <training seasons>`, `weights`, `season <year> --train ...`. Reads the Golden Dataset; downloads nothing. |
| `load_m1.py` | Stage 4 (PRD 18.2): loads the Golden/Stage 3 products into the M1 database tables (`DATABASE_URL`). Offline, repeatable, upsert. See `backend/README.md`. |
