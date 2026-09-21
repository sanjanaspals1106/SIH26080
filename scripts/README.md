# scripts/

Helper scripts that are not part of any importable package.

**This folder is not in the PRD repository tree (section 22.5).** It was added at setup for
`check_setup.py`. It has no single owner in the PRD; treat it as shared.

| Script | What it does |
|---|---|
| `check_setup.py` | PRD 22.2 step 6: checks imports, `xgboost >= 2.0`, the PostgreSQL connection (`DATABASE_URL`) and `node --version >= 20`. Prints PASS or FAIL for each item. |
