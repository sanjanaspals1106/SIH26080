-- SIH26080 database schema. Copied verbatim from docs/PRD.md section 20.2.
-- No PostGIS. District outlines are stored as GeoJSON (JSONB).
-- Apply with: psql -U sih -d sih_rain -f database/schema.sql   (see database/README.md)

CREATE TABLE districts (
  district_id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
  geojson JSONB, centroid_lat REAL, centroid_lon REAL,
  n_effective_cells REAL, is_small BOOLEAN, source_year INT
);
CREATE TABLE grid_cells (
  cell_id INT PRIMARY KEY, latitude NUMERIC(6,3) NOT NULL, longitude NUMERIC(6,3) NOT NULL,
  is_valid BOOLEAN, elevation_m REAL, slope REAL, dist_coast_km REAL, region_code TEXT,
  UNIQUE (latitude, longitude)
);
CREATE TABLE cell_district_weights (
  cell_id INT REFERENCES grid_cells, district_id TEXT REFERENCES districts,
  area_weight REAL NOT NULL, is_main BOOLEAN,
  PRIMARY KEY (cell_id, district_id)
);
CREATE TABLE model_versions (
  model_version_id SERIAL PRIMARY KEY, name TEXT NOT NULL, model_role TEXT, algorithm TEXT,
  feature_set_version TEXT, training_seasons INT[], git_commit TEXT,
  settings JSONB, dev_scores JSONB, created_at TIMESTAMPTZ DEFAULT now(), is_active BOOLEAN DEFAULT FALSE
);
CREATE TABLE experiment_runs (
  experiment_id SERIAL PRIMARY KEY, evaluation_set TEXT NOT NULL,   -- development | holdout
  git_commit TEXT, config_hash TEXT, locked BOOLEAN DEFAULT FALSE, forced_rerun BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE nwp_runs (
  run_id TEXT PRIMARY KEY, source TEXT, initialization_time TIMESTAMPTZ NOT NULL, season INT,
  evaluation_set TEXT,                     -- development | holdout
  mode TEXT DEFAULT 'replay', alignment_method TEXT, alignment_offset_hours SMALLINT DEFAULT 0,
  imd_stamp TEXT, status TEXT, created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE regime_predictions (
  run_id TEXT REFERENCES nwp_runs, lead_day SMALLINT, imd_date DATE,
  active_probability REAL, normal_probability REAL, break_probability REAL,
  regime_confidence REAL, confidence_band TEXT, regime_source TEXT,   -- oof | final
  regime_available BOOLEAN, ood_flag BOOLEAN, indicators JSONB,
  domain_mean_raw_mm REAL, domain_mean_corrected_mm REAL, lps_settings TEXT,
  model_version_id INT REFERENCES model_versions,
  PRIMARY KEY (run_id, lead_day)
);
CREATE TABLE lps_detections (
  run_id TEXT, lead_day SMALLINT, lps_id SMALLINT, latitude REAL, longitude REAL,
  zeta_max REAL, mslp_min_hpa REAL, strength_percentile REAL,
  PRIMARY KEY (run_id, lead_day, lps_id)
);
CREATE TABLE district_forecasts (
  run_id TEXT, lead_day SMALLINT, district_id TEXT REFERENCES districts, imd_date DATE,
  raw_mean_mm REAL, corrected_mean_mm REAL, observed_mean_mm REAL,
  wettest_cell_id INT, wettest_cell_mean_mm REAL,
  wettest_cell_q10_mm REAL, wettest_cell_q50_mm REAL, wettest_cell_q90_mm REAL,
  heavy_prob_max_cell REAL, very_heavy_prob_max_cell REAL,
  heavy_area_fraction_expected REAL, very_heavy_area_fraction_expected REAL,
  attention_level TEXT, priority_rank INT, is_small BOOLEAN,
  product_type TEXT, fallback_used BOOLEAN, fallback_reason TEXT,
  model_version_id INT REFERENCES model_versions,
  PRIMARY KEY (run_id, lead_day, district_id)
);
CREATE TABLE hotspots (
  run_id TEXT, lead_day SMALLINT, hotspot_id INT, threshold_mm REAL,
  n_cells INT, max_probability REAL, max_corrected_mean_mm REAL,
  centroid_lat REAL, centroid_lon REAL, outline JSONB, district_ids TEXT[],
  PRIMARY KEY (run_id, lead_day, hotspot_id)
);
CREATE TABLE district_history (              -- past outcomes, used by analogs (F3)
  run_id TEXT, lead_day SMALLINT, district_id TEXT, imd_date DATE, season INT,
  observed_mean_mm REAL, observed_max_cell_mm REAL, raw_mean_mm REAL, corrected_mean_mm REAL,
  prediction_source TEXT,                    -- oof | final
  phase TEXT, lps_near BOOLEAN,
  PRIMARY KEY (run_id, lead_day, district_id)
);
CREATE TABLE bias_table (                    -- history shown in the audit trail (F2)
  district_id TEXT, lead_day SMALLINT, phase TEXT, lps_near BOOLEAN,
  season_excluded INT,                       -- NULL = all development seasons
  n_dates INT, median_diff_mm REAL, q25_diff_mm REAL, q75_diff_mm REAL
);
CREATE TABLE analog_results (
  run_id TEXT, lead_day SMALLINT, rank SMALLINT,
  analog_run_id TEXT, analog_imd_date DATE, analog_season INT,
  distance REAL, distance_percentile REAL,
  PRIMARY KEY (run_id, lead_day, rank)
);
CREATE TABLE transitions (
  run_id TEXT, event_id INT, event_type TEXT, from_state TEXT, to_state TEXT,
  event_date DATE, confirmed BOOLEAN, confidence REAL, district_id TEXT,
  domain_mean_corrected_change_mm REAL, n_dates_before INT, n_dates_after INT,
  PRIMARY KEY (run_id, event_id)
);
CREATE TABLE verification_results (
  result_id BIGSERIAL PRIMARY KEY, experiment_id INT REFERENCES experiment_runs,
  forecast_type TEXT, comparison_to TEXT, evaluation_set TEXT,
  lead_day SMALLINT, threshold_mm REAL, neighbourhood_cells SMALLINT,
  group_type TEXT, group_value TEXT, metric TEXT,
  value REAL, ci_low REAL, ci_high REAL, n_samples BIGINT, n_events BIGINT, undefined_reason TEXT,
  model_version_id INT REFERENCES model_versions
);
CREATE TABLE pipeline_events (
  event_id BIGSERIAL PRIMARY KEY, run_id TEXT, stage TEXT, level TEXT, message TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes listed after the DDL in PRD section 20.2 (index names are ours; the PRD gives columns only).
CREATE INDEX idx_district_forecasts_run_district ON district_forecasts (run_id, district_id);
CREATE INDEX idx_district_history_district_lead_season ON district_history (district_id, lead_day, season);
CREATE INDEX idx_verification_results_type_lead_metric ON verification_results (forecast_type, lead_day, metric);
CREATE INDEX idx_bias_table_district_lead_excluded ON bias_table (district_id, lead_day, season_excluded);
