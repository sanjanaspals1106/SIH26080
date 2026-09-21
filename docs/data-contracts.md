# Data contracts

Copied from `docs/PRD.md` so that everyone can see the contracts in one place:
section 9.7 (internal data contract), 11.6 (regime output), 16.11 (verification output) and
18.5 (API example answers). **If this file and the PRD ever differ, the PRD wins.**

Contracts are frozen at gate G0 (PRD 21.3). A change needs a message to everyone who uses the
contract; see `CONTRIBUTING.md`. All values in the examples are mock values.

## 9.7 Internal data contract

```json
{
  "run_id": "tigge_ecmwf_cf_2024071500",
  "source": "ECMWF-TIGGE-control",
  "initialization_time": "2024-07-15T00:00:00Z",
  "lead_day": 1,
  "imd_date": "2024-07-16",
  "window_start_utc": "2024-07-15T03:00:00Z",
  "window_end_utc": "2024-07-16T03:00:00Z",
  "alignment_method": "C1",
  "alignment_offset_hours": 0,
  "unit": "mm",
  "grid": "IMD_0.25",
  "season": 2024
}
```

## 11.6 Regime output contract

Domain level (one per run and lead):

```json
{
  "run_id": "tigge_ecmwf_cf_2024071500",
  "lead_day": 1,
  "phase": {
    "active_probability": 0.42,
    "normal_probability": 0.46,
    "break_probability": 0.12,
    "regime_confidence": 0.18,
    "confidence_band": "low",
    "regime_source": "final"
  },
  "lps": {
    "detected": true,
    "centres": [{ "lat": 20.4, "lon": 86.1, "zeta_max": 2.1e-5, "mslp_min_hpa": 996.0, "strength_percentile": 0.74 }],
    "settings": "tuned"
  },
  "quality": { "ood_flag": false, "regime_available": true }
}
```

Cell level (one per run, lead and cell):

```json
{
  "cell_id": 8123,
  "lps_present": true, "distance_to_lps_km": 260, "bearing_to_lps_deg": 235, "lps_influence": 0.52,
  "orographic_influence": 0.88, "coastal_influence": 0.35,
  "orographic_favorable": false, "coastal_favorable": false
}
```

The numbers are examples. `regime_source` is `oof` for training rows and `final` for everything else. **Training code must stop with an error if any training row has `regime_source = final`.**

## 16.11 Output contract

Values are `null` here **on purpose**. No result exists until the engine has run on real data.

```json
{
  "_example": true,
  "forecast_type": "regime_aware_ml",
  "comparison_to": "raw_nwp",
  "evaluation_set": "holdout",
  "lead_day": 1,
  "threshold_mm": 64.5,
  "neighbourhood_cells": 5,
  "group": { "type": "all", "value": "all" },
  "metrics": {
    "rmse":           { "value": null, "ci_low": null, "ci_high": null },
    "pod":            { "value": null, "ci_low": null, "ci_high": null },
    "far":            { "value": null, "ci_low": null, "ci_high": null },
    "csi":            { "value": null, "ci_low": null, "ci_high": null },
    "ets":            { "value": null, "ci_low": null, "ci_high": null },
    "fss":            { "value": null, "ci_low": null, "ci_high": null },
    "frequency_bias": { "value": null, "ci_low": null, "ci_high": null }
  },
  "paired_difference": { "ets": { "value": null, "ci_low": null, "ci_high": null } },
  "n_samples": null,
  "n_events": null,
  "bootstrap": { "block_days": 7, "resamples": 2000 },
  "model_version_id": null
}
```

## 18.5 Example answers (all values are mock values)

**District forecast** (`GET /forecasts/districts`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500", "mode": "replay", "evaluation_set": "holdout",
  "lead_day": 1, "imd_date": "2024-07-16",
  "model_version": { "correction": "regime_xgb_v1", "heavy_rain": "hr_xgb_v1", "phase": "phase_lr_v1" },
  "districts": [{
    "district_id": "D001", "district_name": "Example District", "state": "Example State",
    "is_small": false, "product_type": "regime_aware_ml",
    "raw_mean_mm": 18.0, "corrected_mean_mm": 24.5, "observed_mean_mm": 27.0,
    "wettest_cell_mean_mm": 41.0, "wettest_cell_q10_mm": 12.0, "wettest_cell_q50_mm": 33.0, "wettest_cell_q90_mm": 78.0,
    "heavy_prob_max_cell": 0.22, "very_heavy_prob_max_cell": 0.05,
    "heavy_area_fraction_expected": 0.06, "very_heavy_area_fraction_expected": 0.01,
    "attention_level": "WATCH", "priority_rank": 37,
    "flags": { "fallback_used": false, "fallback_reason": null, "ood_flag": false, "extrapolation_flag": false }
  }]
}
```

**Audit trail** (`GET /forecasts/{forecast_id}/audit`):

```json
{
  "_mock": true, "forecast_id": "tigge_ecmwf_cf_2024071500_L1_D001", "evaluation_set": "holdout",
  "steps": {
    "raw": { "district_mean_mm": 18.0, "wettest_cell_mm": 33.0 },
    "regime": {
      "phase": { "active": 0.42, "normal": 0.46, "break": 0.12, "confidence_band": "low" },
      "nearest_lps": { "distance_km": 260, "bearing_deg": 235, "influence": 0.52, "settings": "tuned" },
      "orographic_influence": 0.88, "coastal_influence": 0.35
    },
    "history": { "phase": "normal", "lps_near": true, "n_dates": 14, "median_diff_mm": 3.1, "q25_diff_mm": -2.0, "q75_diff_mm": 9.4, "note": "few past cases" },
    "correction": { "district_mean_mm": 6.5, "wettest_cell_mm": 8.0 },
    "corrected": { "district_mean_mm": 24.5, "wettest_cell_mm": 41.0 },
    "confidence": {
      "heavy_prob_max_cell": 0.22, "very_heavy_prob_max_cell": 0.05,
      "range_wettest_cell_mm": { "q10": 12.0, "q50": 33.0, "q90": 78.0 },
      "measured_coverage_q10_q90": 0.78
    },
    "record": { "model_version": "regime_xgb_v1", "feature_set_version": "fs_v1", "alignment_method": "C1", "fallback_used": false }
  },
  "summary": "The model raised the district mean from 18.0 to 24.5 mm (+6.5). Most likely phase: normal (confidence low)."
}
```

**Analogs** (`GET /analogs`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500", "lead_day": 1, "district_id": "D001",
  "analogs": [{
    "rank": 1, "imd_date": "2019-07-28", "season": 2019,
    "distance": 1.31, "distance_percentile": 0.6,
    "observed_mean_mm": 22.0, "observed_wettest_cell_mm": 47.0,
    "raw_mean_mm": 15.0, "corrected_mean_mm": 21.0, "error_observed_minus_raw_mm": 7.0
  }],
  "median_error_observed_minus_raw_mm": 5.5, "n_analogs": 5
}
```

**Transitions** (`GET /regime/transitions`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500",
  "series": [{ "imd_date": "2024-07-14", "p_active": 0.20, "p_normal": 0.60, "p_break": 0.20, "lps_present": false }],
  "events": [{
    "event_type": "PHASE:normal->active", "event_date": "2024-07-17", "confirmed": false,
    "confidence": 0.58, "district_id": null,
    "domain_mean_corrected_change_mm": 3.2, "n_dates_before": 3, "n_dates_after": 1
  }]
}
```

**Hotspots** (`GET /hotspots`):

```json
{
  "_mock": true, "run_id": "tigge_ecmwf_cf_2024071500", "lead_day": 1, "threshold_mm": 64.5,
  "hotspots": [{
    "hotspot_id": 1, "n_cells": 6, "max_probability": 0.63, "max_corrected_mean_mm": 58.0,
    "centroid": { "lat": 17.4, "lon": 73.6 }, "district_ids": ["D001", "D002"],
    "outline": { "type": "Polygon", "coordinates": [] }
  }]
}
```

**Priority table** (`GET /districts/priority`): the same fields as the district forecast example, sorted by the rule in F5, with `attention_level`, `priority_rank` and the note `"Model-based attention level. Not an official warning."`.

**Verification** (`GET /verification`): the structure in §16.11. Query parameters: `forecast_type`, `comparison_to`, `evaluation_set`, `lead_day`, `threshold`, `neighbourhood`, `group_type`, `group_value`.

**Model information** (`GET /model-info`):

```json
{
  "_mock": true,
  "models": [{ "role": "correction_mean", "version": "regime_xgb_v1", "algorithm": "XGBoost Tweedie", "feature_set_version": "fs_v1", "n_features": 41 }],
  "data": {
    "nwp": "ECMWF-TIGGE-control", "truth": "IMD 0.25 degree", "alignment_method": "C1",
    "lead_days": [1, 2, 3], "season_scope": "JJAS", "district_file": "DataMeet, 2011 districts"
  },
  "protocol": { "n_seasons": null, "development_seasons": [], "holdout_seasons": [], "holdout_locked": false, "status": "DEVELOPMENT ONLY" },
  "limitations": [
    "Districts created after 2011 are not included.",
    "Western disturbances are not modelled (June-September only).",
    "Influence values are percentile or simple indices, not probabilities.",
    "The range is model-estimated, not guaranteed."
  ],
  "fallback_share_recent": null
}
```
