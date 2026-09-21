## What and why

<!-- One or two sentences. Link the PRD section(s) this implements. -->

**Reviewer (PRD 21.2):** @

## Checklist

- [ ] Matches the contracts (PRD 9.7, 11.6, 16.11, 18). If a contract changed, everyone who uses it was told.
- [ ] No forbidden inputs (PRD 10.6):
  - [ ] L1 no observed rain from the forecast start or later as a feature
  - [ ] L2 no analysis or reanalysis fields as inputs (forecast fields only)
  - [ ] L3 no label-based or in-sample phase probabilities in training rows
  - [ ] L4 no statistic that uses holdout seasons (climatology, percentile tables, scalers, quantile-mapping tables)
  - [ ] L5 no random splits or shuffled cross-validation
  - [ ] L6 no settings, thresholds or calibration chosen on the holdout
  - [ ] L7 no `cell_id` as a feature
  - [ ] L8 no rain (forecast or observed) as an input of the phase model
  - [ ] L9 no columns that never change (for example start hour)
- [ ] No data or secrets committed (`data/`, GRIB/NetCDF files, model files, `.env`, `~/.cdsapirc`).
- [ ] Tests added or updated (PRD section 23).
