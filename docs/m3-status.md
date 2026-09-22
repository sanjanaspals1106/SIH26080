# M3 Rainfall Models Readiness & Status

**Document Version:** 1.1  
**Status:** Code Implementation Complete / Awaiting Real Upstream Data at Gate G1  
**Module:** M3 (Rainfall Models)  
**Relevant PRD Sections:** §10, §12, §13, §17, §20.1, §21.1, §23  

---

## M3 READINESS CHECKLIST

[x] B0 raw baseline implemented  
[x] B1 quantile mapping implemented  
[x] B2 Tweedie implemented  
[x] B3 Tweedie implemented  
[x] heavy-rain probability models implemented  
[x] q10/q50/q90 range models implemented  
[x] extrapolation threshold fit logic implemented  
[x] serving-grid parquet writer implemented  
[x] development OOF pipeline implemented  
[x] final model fit/predict implemented  
[ ] real TIGGE/IMD development data received  
[ ] official frozen development run completed  

---

## What is 100% Complete and Tested in Code

All M3 core algorithmic components, validation contracts, model serialization, and integration runners are fully implemented, verified, and backed by a comprehensive automated test suite:

1. **B0 Raw NWP Baseline (`ml/baselines/b0_raw.py`)**:
   - Zero-overhead identity pass-through of `rain_mm`.
   - Strict row/index preservation and missing column validation.

2. **B1 Quantile Mapping Baseline (`ml/baselines/b1_quantile_mapping.py`)**:
   - Empirical quantile mapping binned strictly by `(lead_day, region_code)` pairs across all 6 IMD regions.
   - 100-step linear interpolation with mid-tie handling.
   - 99.5th percentile tail ratio extrapolation for extreme values beyond observed empirical quantiles.
   - Strict non-negativity constraint and JSON serialization/deserialization.

3. **Feature Contracts & Schemas (`ml/feature_contracts.py`)**:
   - Canonical 27 `BASE_FEATURES` and 14 `REGIME_FEATURES` in strict PRD order.
   - Exact contract validation enforcing non-inclusion of `cell_id` and `obs_mm`.
   - Native preservation of missing values (NaNs) without artificial zero-imputation.

4. **Shared Hyperparameter Search (`ml/training/hyperparameter_search.py`)**:
   - Deterministic 20-configuration grid generator shared identically between B2 and B3.
   - Season-based temporal cross-validation evaluating the newest 3 development seasons without data leakage.

5. **B2 Global Rainfall Correction (`ml/training/train_correction.py`)**:
   - XGBoost `reg:tweedie` with `tree_method='hist'`.
   - Monotone constraint: `+1` on `rain_mm`, `0` on all remaining 26 features.
   - Non-negative corrected output enforcement ($\ge 0$).

6. **B3 Regime-Aware Rainfall Correction (`ml/training/train_correction.py`)**:
   - XGBoost `reg:tweedie` with 41 features (27 base + 14 regime).
   - Strict protocol safety guard (`validate_b3_training_data`): hard error if `regime_source != 'oof'` or if `'final'` leaks into training data.

7. **Heavy-Rain Probability Models (`probability/classifiers.py`)**:
   - Classifiers for 15.6 mm, 64.5 mm, and 115.6 mm using `binary:logistic` and natural class balance.
   - M4 event-count branching logic ($\ge 30$ events trigger independent classifiers; $< 30$ events trigger conditional chained formulation or safe deactivation).
   - Monotonicity preservation: $P(\ge 115.6) \le P(\ge 64.5) \le P(\ge 15.6)$ per grid cell.
   - Output emitted as uncalibrated raw tree probabilities bounded in $[0, 1]$ ready for M4 calibration.

8. **Calibration Integration Seam (`probability/calibration.py`)**:
   - M3 strictly NEVER fits calibrators (calibration is owned and fitted by M4 on OOF predictions).
   - Final inference receives external pre-fitted M4 calibrators, applies them, clips to $[0.0, 1.0]$, and re-enforces monotonicity.
   - Unavailable thresholds remain null. Fallback rows null all probabilities.
   - Refuses to silently expose uncalibrated probabilities as calibrated in final serving.

9. **Quantile Regression Range Models (`probability/range_models.py`)**:
   - Three independent models for $q_{10}, q_{50}, q_{90}$ with `reg:quantileerror`.
   - $\log(1 + \text{obs\_mm})$ target scaling and $\exp(y) - 1$ inverse transformation.
   - Per-cell crossing resolution enforcing $q_{10} \le q_{50} \le q_{90}$.
   - Range coverage metric utility (`calculate_range_coverage`).

10. **Extrapolation Threshold & Fallback Ladder (`ml/inference/predict_correction.py`)**:
    - Training-season $99.9\text{th}$ percentile threshold fitting and persistence.
    - Extrapolation detection via strict inequality (`raw_mm > p99_9_threshold`).
    - Priority fallback ladder adhering to PRD §17.1 (`regime_aware_ml` $\to$ `global_ml` $\to$ `quantile_mapping` $\to$ `raw_nwp`).
    - Nulling of range and probability values on any fallback occurrence (PRD §17.2).

11. **Serving Grid Assembly & Parquet Writer (`ml/inference/serving_grid_writer.py`)**:
    - Canonical 18-column Parquet schema assembly and validation (`validate_serving_grid_contracts`).
    - Partitioned Parquet read and write utilities for `data/serving/grid/run_id=<run_id>/part.parquet`.

12. **Model Persistence (`ml/models/model_store.py`)**:
    - Complete JSON-based model store for B1, B2, B3, probability classifiers, and range models with full hyperparameters and threshold metadata.

13. **End-to-End Orchestration Layer (`ml/orchestration.py`)**:
    - Input contract validation for M1/M2 payloads.
    - Leave-One-Season-Out (LOSO) development OOF pipeline emitting exact 16-column table for M4.
    - Production model fitting on development seasons and final inference pipeline.

---

## What CANNOT Be Run Yet Because Real Data is Pending

The following production and experimental steps are blocked pending ingestion of real historical TIGGE ECMWF NWP forecasts and IMD 0.25° gridded observational truth:

1. **Gate G1 Season Partition Fixing**:
   - Cannot fix the total number of available seasons $N$ and partition them according to PRD §10.2:
     - N >= 8: oldest N-2 = development, newest 2 = holdout.
     - 5 <= N <= 7: oldest N-1 = development, newest 1 = holdout.
     - 3 <= N <= 4: all N = development (LOSO only, no holdout).
     - N <= 2: stop.
2. **Official Development Cross-Validation**:
   - Cannot run full leave-one-season-out OOF cross-validation across genuine development monsoon seasons.
3. **Official Model Selection & Hyperparameter Freezing**:
   - Cannot select the optimal config from the 20-candidate grid based on real empirical validation RMSE and Brier scores.
4. **Official Production Model Fitting**:
   - Cannot train and save the frozen production model weights for B1, B2, B3, probability classifiers, and range models.
5. **Empirical Verification Claims**:
   - Cannot compute official primary verification checks (PRD §10.8):
     - P1: RMSE against observed rain
     - P2: ETS at 64.5 mm
     - P3: FSS at 64.5 mm (5×5 neighborhood)
     - P4: Brier skill score for $P(\ge 64.5\text{ mm})$ vs climatology
     - P5: $q_{10}$–$q_{90}$ coverage (target 80%)
6. **Final Holdout Gate Evaluation**:
   - Cannot evaluate the frozen model pipeline on the locked holdout seasons.

---

## Exact Blocking Prerequisites from M1/M2/M4

To unblock the official development run, M3 requires the following deliverables:

### From M1 (Data Pipeline & Feature Engineering)
- **Historical Development Feature Tables**:
  - Full historical seasons with total $N$ determined at Gate G1, containing all 27 canonical `BASE_FEATURES` on the 0.25° IMD grid.
  - Required identifiers: `run_id`, `season`, `lead_day` (1, 2, 3), `cell_id`, `region_code` (6 IMD regions).
  - Ground truth target column: `obs_mm` (required for historical/evaluation rows; nullable for live replay/inference).
  - Feature set version identifier (e.g. `feature_set_version="v1"`).

### From M2 (Regime Classification Engine)
- **Monsoon Phase & LPS Features**:
  - All 14 canonical `REGIME_FEATURES` aligned by `(run_id, lead_day, cell_id)`.
  - Development dataset must have `regime_source == 'oof'` strictly.
  - Runtime flags: `regime_available` (bool) and `ood_flag` (bool).

### From M4 (Protocol, Calibration & Verification)
- **Protocol Specifications**:
  - Formal designation and freezing of development seasons and locked holdout seasons strictly per PRD §10.2 rules once $N$ is fixed at Gate G1.
  - Official protocol random seed.
  - Official threshold exceedance event counts across development seasons for 15.6, 64.5, and 115.6 mm.
- **Probability Calibrators**:
  - Pre-fitting of probability calibrators (Platt scaling / Isotonic regression) on pooled development OOF predictions, to be supplied to M3 at final serving.
