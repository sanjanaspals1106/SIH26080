# M3 Rainfall Models Handoff & Interface Contract

**Document Version:** 1.1  
**Status:** M3 Implementation Complete (Awaiting Real Upstream Data at Gate G1)  
**Owners:** M3 (Rainfall Models), interfacing with M1 (Pipeline & Serving), M2 (Regime Engine), M4 (Verification & Protocol)  
**Relevant PRD Sections:** §10, §12, §13, §17, §20.1, §21.1, §23  

---

## 1. M3 Inputs from M1 (Pipeline & Feature Engineering)

M1 supplies gridded forecast, static geography, and observation inputs.

### 1.1 Required Run and Grid Identifiers
- `run_id` (str): ECMWF control forecast initialization stamp (e.g. `tigge_ecmwf_cf_YYYYMMDDHH`).
- `season` (int): Monsoon year (e.g. `2016`). Total available seasons $N$ is fixed at Gate G1.
- `lead_day` (int): Forecast lead day, strictly in `{1, 2, 3}`.
- `cell_id` (int): 0.25° valid IMD grid point index.
- `region_code` (str): One of 6 PRD §14.3 regions:
  `HIMALAYA`, `NORTHEAST`, `WEST_COAST`, `NORTHWEST_WEST`, `SOUTH_EAST`, `CENTRAL_EAST`.
- `obs_mm` (float, nullable):
  - **Historical / evaluation rows:** `obs_mm` is required (target for model fitting and OOF validation).
  - **Replay / final inference rows:** `obs_mm` may be unavailable / null. M3 does NOT require observation truth to produce a forecast.
- `feature_set_version` (str): e.g. `"v1"`.

### 1.2 The 27 Base Features (Canonical PRD Order)
M1 must provide these exact 27 numerical features without renaming:
1. `rain_mm` (raw NWP control forecast)
2. `nbr_mean_3` (3×3 spatial neighborhood mean)
3. `nbr_max_3` (3×3 spatial neighborhood max)
4. `nbr_mean_5` (5×5 spatial neighborhood mean)
5. `nbr_max_5` (5×5 spatial neighborhood max)
6. `rain_grad` (spatial rainfall gradient magnitude)
7. `rain_prev_lead` (forecast rain at lead - 1, NaN for lead 1)
8. `rain_next_lead` (forecast rain at lead + 1, NaN for lead 3)
9. `u850` (850 hPa zonal wind)
10. `v850` (850 hPa meridional wind)
11. `wspd850` (850 hPa wind speed)
12. `vort850` (850 hPa relative vorticity)
13. `q850` (850 hPa specific humidity)
14. `msl` (mean sea level pressure)
15. `shear_200_850` (vertical wind shear magnitude between 200 and 850 hPa)
16. `elevation_m` (surface elevation)
17. `slope` (terrain slope)
18. `aspect_sin` (sine of terrain aspect)
19. `aspect_cos` (cosine of terrain aspect)
20. `dist_coast_km` (distance to coastline in km)
21. `clim_mean` (climatological mean rain)
22. `clim_p95` (climatological 95th percentile rain)
23. `doy_sin` (sine of day of year)
24. `doy_cos` (cosine of day of year)
25. `lead_day` (lead day integer: 1, 2, or 3)
26. `latitude` (cell latitude, EPSG:4326)
27. `longitude` (cell longitude, EPSG:4326)

*Note:* NaNs in temporal boundary features (`rain_prev_lead` for lead 1, `rain_next_lead` for lead 3) are handled natively by XGBoost `hist`. Do not impute with 0.

---

## 2. M3 Inputs from M2 (Regime Engine)

M2 provides domain-level monsoon phase classifications, low-pressure system (LPS) detections, and flux indices.

### 2.1 The 14 Regime Features (Canonical PRD Order)
1. `p_active` (probability of active monsoon phase)
2. `p_normal` (probability of normal monsoon phase)
3. `p_break` (probability of break monsoon phase)
4. `regime_confidence` (monsoon phase classification confidence)
5. `lps_present` (binary flag: LPS detected in domain)
6. `distance_to_lps_km` (distance to nearest LPS center, large sentinel if absent)
7. `bearing_sin` (sine of bearing angle to LPS)
8. `bearing_cos` (cosine of bearing angle to LPS)
9. `lps_strength` (vorticity/mslp intensity score of LPS)
10. `lps_influence` (exponential spatial influence kernel of LPS)
11. `upslope_flux` (orographic upslope moisture flux component)
12. `onshore_flux` (coastal onshore moisture flux component)
13. `orographic_influence` (combined orographic index)
14. `coastal_influence` (combined coastal index)

### 2.2 Regime Metadata & Protocol Guards
- `regime_source` (str):
  - **Training:** Must strictly equal `"oof"`. Any row with `"final"` or missing triggers a hard protocol error (`validate_b3_training_data`).
  - **Inference:** Accepts `"final"` or `"oof"`.
- `regime_available` (bool): `True` if M2 pipeline ran successfully. If `False`, M3 automatically falls back to B2 (`global_ml`).
- `ood_flag` (bool): Run-level out-of-distribution quality flag. If `True`, M3 automatically falls back to B1 (`quantile_mapping`).

---

## 3. Inputs from M4 (Protocol, Calibration, Verification)

M4 controls experiment boundaries, protocol logic, event counts, and post-processing calibration:

### 3.1 Season Partition Logic (PRD §10.2)
Total available seasons $N$ is determined by M1/M4 at Gate G1. M3 strictly adopts PRD §10.2 season partition rules:
- **$N \ge 8$:** Oldest $N - 2$ seasons = development, newest 2 = holdout (Full protocol).
- **$5 \le N \le 7$:** Oldest $N - 1$ seasons = development, newest 1 = holdout (One holdout season; intervals will be wide).
- **$3 \le N \le 4$:** All $N$ seasons for Leave-One-Season-Out (LOSO) development only; no holdout. Every result is labelled `DEVELOPMENT ONLY`.
- **$N \le 2$:** **Stop.** Project cannot be evaluated; team must acquire more seasons before continuing.

> [!NOTE]
> PRD §10.2 example: For the target of 10 seasons (2016–2025), development is 2016–2023 and holdout is 2024–2025. These are illustrative examples and must NOT be assumed as real until M1/M4 formally fix $N$ at Gate G1.

### 3.2 Primary Verification Checks (PRD §10.8)
Primary metrics fixed in `config/protocol.yaml`:
1. **P1 (RMSE):** Root mean squared error of corrected mean against observed rain (all leads combined).
2. **P2 (ETS):** Equitable Threat Score at 64.5 mm (or 15.6 mm if < 30 events).
3. **P3 (FSS):** Fractional Skill Score at 64.5 mm with a 5×5-cell neighborhood (~140 km).
4. **P4 (Brier Skill Score):** BSS for $P(\text{rain} \ge 64.5\text{ mm})$, compared against training-season climatology.
5. **P5 (Range Coverage):** Share of observations falling inside $q_{10}$–$q_{90}$ range (diagnostic check; target 80%).

*(Note: CRPS is not listed as a required primary PRD verification metric).*

### 3.3 Event Counts & Probability Branching (PRD §13.2)
- `event_counts` (Dict[float, int]): Official counts of threshold exceedances across development seasons: `{15.6: n15, 64.5: n64, 115.6: n115}`.
  - Controls branching: independent models ($\ge 30$), chained formulation for 115.6 mm, or disabling heavy models ($< 30$ events at 64.5 mm).

### 3.4 Calibration Integration Interface (PRD §10.4, §13.3)
- **M4 Ownership:** M4 owns fitting probability calibrators (Platt scaling / Isotonic regression) on pooled development OOF predictions.
- **M3 Never Fits:** M3 NEVER fits calibrators.
- **Serving Seam:** For final/holdout serving, M3 accepts external pre-fitted calibrators via `predict_final_m3(..., calibrators=calibrators_dict)`.
- **Enforcement:**
  - Calibrators transform uncalibrated probability predictions only.
  - Calibrated outputs clipped to $[0.0, 1.0]$.
  - Monotonicity $P(\ge 115.6) \le P(\ge 64.5) \le P(\ge 15.6)$ re-enforced after calibration.
  - Unavailable thresholds remain null.
  - Fallback rows have all probability outputs set to null (`NaN`).
  - If final serving is requested without required calibrators, M3 raises a clear error unless explicit development-only mode (`allow_uncalibrated=True`) is selected.

---

## 4. Outputs to M4 (OOF Prediction Table)

Generated by `run_development_oof_pipeline(dev_df, development_seasons)`.

### 4.1 Schema of `oof_predictions` Table
| Column Name | Type | Description |
|---|---|---|
| `run_id` | str | Forecast run identifier |
| `season` | int | Monsoon year |
| `lead_day` | int | Forecast lead day (1, 2, 3) |
| `cell_id` | int | Grid cell index |
| `obs_mm` | float | IMD ground truth rain |
| `raw_mm` | float | Raw NWP forecast (B0) |
| `b1_corrected_mm` | float | Quantile mapped prediction (B1) |
| `b2_corrected_mean_mm` | float | Global Tweedie prediction (B2) |
| `b3_corrected_mean_mm` | float | Regime-aware Tweedie prediction (B3) |
| `uncalibrated_p_ge_15_6` | float | Raw tree probability for $\ge 15.6\text{ mm}$ |
| `uncalibrated_p_ge_64_5` | float | Raw tree probability for $\ge 64.5\text{ mm}$ |
| `uncalibrated_p_ge_115_6` | float | Raw tree probability for $\ge 115.6\text{ mm}$ |
| `q10_mm` | float | Model-estimated 10th percentile |
| `q50_mm` | float | Model-estimated 50th percentile (median) |
| `q90_mm` | float | Model-estimated 90th percentile |
| `prediction_source` | str | Always `'oof'` |

> [!IMPORTANT]
> All probabilities emitted in development OOF handoff to M4 are explicitly named `uncalibrated_p_*`. M4 consumes these uncalibrated probabilities to fit calibrators.

---

## 5. Outputs to M1 (Serving Grid Parquet)

Written per run by `write_serving_grid_file(df, run_id, output_base_dir)` to:
`data/serving/grid/run_id=<run_id>/part.parquet`

### 5.1 Exact 18-Column Parquet Schema (PRD §20.1)
1. `cell_id` (int): Grid cell identifier.
2. `lead_day` (int): Forecast lead day (1, 2, or 3).
3. `raw_mm` (float): Raw NWP precipitation.
4. `corrected_mean_mm` (float): Headline served prediction ($\ge 0$).
5. `q10_mm` (float, nullable): 10th percentile range estimate (null in fallback).
6. `q50_mm` (float, nullable): 50th percentile median estimate (null in fallback).
7. `q90_mm` (float, nullable): 90th percentile range estimate (null in fallback).
8. `p_ge_15_6` (float, nullable): Calibrated probability of rain $\ge 15.6\text{ mm}$ (null in fallback).
9. `p_ge_64_5` (float, nullable): Calibrated probability of rain $\ge 64.5\text{ mm}$ (null in fallback).
10. `p_ge_115_6` (float, nullable): Calibrated probability of rain $\ge 115.6\text{ mm}$ (null in fallback).
11. `obs_mm` (float, nullable): Observed rain (null when unknown).
12. `lps_influence` (float): LPS spatial kernel weight.
13. `orographic_influence` (float): Mountain wind-flux index.
14. `coastal_influence` (float): Coastal wind-flux index.
15. `product_type` (str): `'regime_aware_ml'` | `'global_ml'` | `'quantile_mapping'` | `'raw_nwp'`.
16. `fallback_reason` (str, nullable): Reason code or `None` if normal.
17. `extrapolation_flag` (bool): `True` if `raw_mm > p99_9_threshold`.
18. `model_version_id` (int, nullable): PostgreSQL foreign key reference.

---

## 6. Fallback Ladder Precedence (PRD §17.1)

If anomalies occur during inference, M3 applies the fallback hierarchy:

| Priority | Situation | Served Product | `product_type` | `fallback_reason` | Range / Probabilities |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | Normal operation | B3 | `regime_aware_ml` | `None` | Populated (Calibrated) |
| 2 | Regime features missing/failed | B2 | `global_ml` | `REGIME_UNAVAILABLE` | **Null (NaN)** |
| 3 | ML model files missing | B1 | `quantile_mapping` | `ML_UNAVAILABLE` | **Null (NaN)** |
| 4 | Run-level OOD detected (`ood_flag = true`) | B1 | `quantile_mapping` | `OOD_INPUT` | **Null (NaN)** |
| 5 | **Cell `rain_mm > p99.9` threshold** | **Raw NWP for cell** | `raw_nwp` | `EXTRAPOLATION` | **Null (NaN)** |
| 6 | Validation / check failure | Raw NWP | `raw_nwp` | `VALIDATION_FAILED` | **Null (NaN)** |
| 7 | No correction available | Raw NWP | `raw_nwp` | `NO_CORRECTION` | **Null (NaN)** |

> [!NOTE]
> Cell-level `EXTRAPOLATION` strictly overrides the run-level product for that specific cell. In any fallback row, range and probability outputs are null (`NaN`) per PRD §17.2.

---

## 7. Model Artifacts & Persistence

Final trained models are saved under `models/m3/` via `FinalM3Models.save(output_dir)`:
- `b1/b1_model.json`: Quantile mapping lookup tables and tail ratios.
- `b2/model.json` & `b2/metadata.json`: XGBoost Tweedie regressor with 27 base features.
- `b3/model.json` & `b3/metadata.json`: XGBoost Tweedie regressor with 41 regime-aware features.
- `probability/metadata.json` & subfolders: Heavy-rain binary classifiers ($15.6, 64.5, 115.6\text{ mm}$).
- `range/metadata.json` & subfolders: Quantile regressors ($q_{10}, q_{50}, q_{90}$).
- **Extrapolation Threshold:** Stored in `metadata.json` under `p99_9_threshold`. Fitted strictly on development seasons.

---

## 8. Main Callable Functions in Codebase

All functions are importable from `ml` and `ml.orchestration`:

1. **OOF Development Pipeline (Uncalibrated for M4):**
   ```python
   from ml.orchestration import run_development_oof_pipeline
   oof_df = run_development_oof_pipeline(dev_df, development_seasons)
   ```
2. **Final Model Fit & Persistence:**
   ```python
   from ml.orchestration import fit_final_m3_models
   final_models = fit_final_m3_models(dev_df, development_seasons, output_dir="models/m3")
   ```
3. **Inference & Serving Grid Assembly (with M4 Calibrators):**
   ```python
   from ml.orchestration import predict_final_m3
   # Calibrated inference
   serving_df = predict_final_m3(
       final_models,
       features_df,
       calibrators=m4_calibrators_dict,
       output_serving_dir="data/serving",
   )
   # Explicit development uncalibrated mode
   dev_serving_df = predict_final_m3(
       final_models,
       features_df,
       allow_uncalibrated=True,
   )
   ```
4. **Calibration Application Helper:**
   ```python
   from probability.calibration import apply_probability_calibrators
   calibrated_probs_df = apply_probability_calibrators(raw_prob_df, calibrators=m4_calibrators)
   ```
5. **Parquet Read & Write Utilities:**
   ```python
   from ml.inference.serving_grid_writer import write_serving_grid_file, read_serving_grid_file
   target_path = write_serving_grid_file(serving_df, run_id="run_001", output_base_dir="data/serving")
   loaded_df = read_serving_grid_file(run_id="run_001", output_base_dir="data/serving")
   ```
