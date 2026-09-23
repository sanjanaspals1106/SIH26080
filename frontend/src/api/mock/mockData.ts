import type {
  RunsResponse,
  DistrictForecastsResponse,
  PriorityTableResponse,
  GridResponse,
  ImprovementSummaryResponse,
  AuditTrailResponse,
  HotspotsResponse,
  RegimeResponse,
  RegimeTransitionsResponse,
  AnalogsResponse,
  VerificationResponse,
  ReliabilityResponse,
  ModelInfoResponse,
  MapMetadataResponse,
  HealthResponse,
  GeoJsonFeatureCollection,
  GridVariable,
  GridCellData,
} from '../../types'

export const MOCK_RUNS_RESPONSE: RunsResponse = {
  _mock: true,
  mode: 'replay',
  evaluation_set: 'holdout',
  runs: [
    {
      run_id: 'tigge_ecmwf_cf_2024071500',
      source: 'ECMWF-TIGGE-control',
      initialization_time: '2024-07-15T00:00:00Z',
      season: 2024,
      evaluation_set: 'holdout',
      mode: 'replay',
      alignment_method: 'C1',
      alignment_offset_hours: 0,
      imd_stamp: 'end_date',
      status: 'completed',
      created_at: '2026-09-21T00:00:00Z',
    },
    {
      run_id: 'tigge_ecmwf_cf_2024071600',
      source: 'ECMWF-TIGGE-control',
      initialization_time: '2024-07-16T00:00:00Z',
      season: 2024,
      evaluation_set: 'holdout',
      mode: 'replay',
      alignment_method: 'C1',
      alignment_offset_hours: 0,
      imd_stamp: 'end_date',
      status: 'completed',
      created_at: '2026-09-21T00:00:00Z',
    },
    {
      run_id: 'tigge_ecmwf_cf_2023081000',
      source: 'ECMWF-TIGGE-control',
      initialization_time: '2023-08-10T00:00:00Z',
      season: 2023,
      evaluation_set: 'development',
      mode: 'replay',
      alignment_method: 'C1',
      alignment_offset_hours: 0,
      imd_stamp: 'end_date',
      status: 'completed',
      created_at: '2026-09-21T00:00:00Z',
    },
  ],
}

export const MOCK_DISTRICTS_FORECAST: DistrictForecastsResponse = {
  _mock: true,
  run_id: 'tigge_ecmwf_cf_2024071500',
  mode: 'replay',
  evaluation_set: 'holdout',
  lead_day: 1,
  imd_date: '2024-07-16',
  model_version: {
    correction: 'regime_xgb_v1',
    heavy_rain: 'hr_xgb_v1',
    phase: 'phase_lr_v1',
  },
  districts: [
    {
      district_id: 'D001',
      district_name: 'Ratnagiri',
      state: 'Maharashtra',
      is_small: false,
      product_type: 'regime_aware_ml',
      raw_mean_mm: 18.0,
      corrected_mean_mm: 24.5,
      observed_mean_mm: 27.0,
      wettest_cell_mean_mm: 41.0,
      wettest_cell_q10_mm: 12.0,
      wettest_cell_q50_mm: 33.0,
      wettest_cell_q90_mm: 78.0,
      heavy_prob_max_cell: 0.58,
      very_heavy_prob_max_cell: 0.22,
      heavy_area_fraction_expected: 0.25,
      very_heavy_area_fraction_expected: 0.08,
      attention_level: 'HIGH',
      priority_rank: 1,
      flags: {
        fallback_used: false,
        fallback_reason: null,
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
    {
      district_id: 'D002',
      district_name: 'Sindhudurg',
      state: 'Maharashtra',
      is_small: false,
      product_type: 'regime_aware_ml',
      raw_mean_mm: 22.0,
      corrected_mean_mm: 31.0,
      observed_mean_mm: 34.0,
      wettest_cell_mean_mm: 52.0,
      wettest_cell_q10_mm: 18.0,
      wettest_cell_q50_mm: 45.0,
      wettest_cell_q90_mm: 92.0,
      heavy_prob_max_cell: 0.52,
      very_heavy_prob_max_cell: 0.18,
      heavy_area_fraction_expected: 0.20,
      very_heavy_area_fraction_expected: 0.05,
      attention_level: 'HIGH',
      priority_rank: 2,
      flags: {
        fallback_used: false,
        fallback_reason: null,
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
    {
      district_id: 'D003',
      district_name: 'Uttara Kannada',
      state: 'Karnataka',
      is_small: false,
      product_type: 'regime_aware_ml',
      raw_mean_mm: 14.5,
      corrected_mean_mm: 19.8,
      observed_mean_mm: 21.0,
      wettest_cell_mean_mm: 38.0,
      wettest_cell_q10_mm: 9.0,
      wettest_cell_q50_mm: 28.0,
      wettest_cell_q90_mm: 64.0,
      heavy_prob_max_cell: 0.35,
      very_heavy_prob_max_cell: 0.08,
      heavy_area_fraction_expected: 0.12,
      very_heavy_area_fraction_expected: 0.02,
      attention_level: 'WATCH',
      priority_rank: 3,
      flags: {
        fallback_used: false,
        fallback_reason: null,
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
    {
      district_id: 'D004',
      district_name: 'Pune',
      state: 'Maharashtra',
      is_small: false,
      product_type: 'regime_aware_ml',
      raw_mean_mm: 6.5,
      corrected_mean_mm: 7.8,
      observed_mean_mm: 8.0,
      wettest_cell_mean_mm: 14.2,
      wettest_cell_q10_mm: 3.0,
      wettest_cell_q50_mm: 11.0,
      wettest_cell_q90_mm: 22.0,
      heavy_prob_max_cell: 0.08,
      very_heavy_prob_max_cell: 0.01,
      heavy_area_fraction_expected: 0.01,
      very_heavy_area_fraction_expected: 0.0,
      attention_level: 'NORMAL',
      priority_rank: 4,
      flags: {
        fallback_used: false,
        fallback_reason: null,
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
    {
      district_id: 'D005',
      district_name: 'Wayanad',
      state: 'Kerala',
      is_small: true,
      product_type: 'regime_aware_ml',
      raw_mean_mm: 26.0,
      corrected_mean_mm: 38.5,
      observed_mean_mm: 42.0,
      wettest_cell_mean_mm: 68.0,
      wettest_cell_q10_mm: 24.0,
      wettest_cell_q50_mm: 55.0,
      wettest_cell_q90_mm: 112.0,
      heavy_prob_max_cell: 0.65,
      very_heavy_prob_max_cell: 0.28,
      heavy_area_fraction_expected: 0.35,
      very_heavy_area_fraction_expected: 0.12,
      attention_level: 'HIGH',
      priority_rank: 5,
      flags: {
        fallback_used: false,
        fallback_reason: null,
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
    {
      district_id: 'D006',
      district_name: 'Dakshina Kannada',
      state: 'Karnataka',
      is_small: false,
      product_type: 'regime_aware_ml',
      raw_mean_mm: 16.0,
      corrected_mean_mm: 22.4,
      observed_mean_mm: 25.0,
      wettest_cell_mean_mm: 44.0,
      wettest_cell_q10_mm: 12.0,
      wettest_cell_q50_mm: 36.0,
      wettest_cell_q90_mm: 72.0,
      heavy_prob_max_cell: 0.42,
      very_heavy_prob_max_cell: 0.11,
      heavy_area_fraction_expected: 0.16,
      very_heavy_area_fraction_expected: 0.03,
      attention_level: 'WATCH',
      priority_rank: 6,
      flags: {
        fallback_used: false,
        fallback_reason: null,
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
    {
      district_id: 'D007',
      district_name: 'Puri',
      state: 'Odisha',
      is_small: false,
      product_type: 'regime_aware_ml',
      raw_mean_mm: 19.5,
      corrected_mean_mm: 25.0,
      observed_mean_mm: 28.0,
      wettest_cell_mean_mm: 46.0,
      wettest_cell_q10_mm: 14.0,
      wettest_cell_q50_mm: 38.0,
      wettest_cell_q90_mm: 80.0,
      heavy_prob_max_cell: 0.45,
      very_heavy_prob_max_cell: 0.14,
      heavy_area_fraction_expected: 0.18,
      very_heavy_area_fraction_expected: 0.04,
      attention_level: 'WATCH',
      priority_rank: 7,
      flags: {
        fallback_used: false,
        fallback_reason: null,
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
    {
      district_id: 'D008',
      district_name: 'Kangra',
      state: 'Himachal Pradesh',
      is_small: false,
      product_type: 'raw_nwp',
      raw_mean_mm: 12.0,
      corrected_mean_mm: 12.0,
      observed_mean_mm: null,
      wettest_cell_mean_mm: 21.0,
      wettest_cell_q10_mm: null,
      wettest_cell_q50_mm: null,
      wettest_cell_q90_mm: null,
      heavy_prob_max_cell: null,
      very_heavy_prob_max_cell: null,
      heavy_area_fraction_expected: null,
      very_heavy_area_fraction_expected: null,
      attention_level: 'UNAVAILABLE',
      priority_rank: 8,
      flags: {
        fallback_used: true,
        fallback_reason: 'REGIME_UNAVAILABLE',
        ood_flag: false,
        extrapolation_flag: false,
      },
    },
  ],
}

export const MOCK_PRIORITY_TABLE: PriorityTableResponse = {
  ...MOCK_DISTRICTS_FORECAST,
  note: 'Model-based attention level. Not an official warning.',
}

export const MOCK_IMPROVEMENT_SUMMARY: ImprovementSummaryResponse = {
  _mock: true,
  run_id: 'tigge_ecmwf_cf_2024071500',
  lead_day: 1,
  imd_date: '2024-07-16',
  total_valid_cells: 4620,
  cells_corrected_closer: 3140,
  cells_raw_closer: 1220,
  cells_no_change: 260,
  total_districts: 640,
  districts_corrected_closer: 452,
  districts_raw_closer: 188,
  note: 'One day only, not evidence.',
}

export const MOCK_AUDIT_TRAIL: AuditTrailResponse = {
  _mock: true,
  forecast_id: 'tigge_ecmwf_cf_2024071500_L1_D001',
  evaluation_set: 'holdout',
  steps: {
    raw: { district_mean_mm: 18.0, wettest_cell_mm: 33.0 },
    regime: {
      phase: { active: 0.42, normal: 0.46, break: 0.12, confidence_band: 'low' },
      nearest_lps: {
        distance_km: 260,
        bearing_deg: 235,
        influence: 0.52,
        settings: 'tuned',
      },
      orographic_influence: 0.88,
      coastal_influence: 0.35,
    },
    history: {
      phase: 'normal',
      lps_near: true,
      n_dates: 14,
      median_diff_mm: 3.1,
      q25_diff_mm: -2.0,
      q75_diff_mm: 9.4,
      note: 'few past cases',
    },
    correction: { district_mean_mm: 6.5, wettest_cell_mm: 8.0 },
    corrected: { district_mean_mm: 24.5, wettest_cell_mm: 41.0 },
    confidence: {
      heavy_prob_max_cell: 0.58,
      very_heavy_prob_max_cell: 0.22,
      range_wettest_cell_mm: { q10: 12.0, q50: 33.0, q90: 78.0 },
      measured_coverage_q10_q90: 0.78,
    },
    record: {
      model_version: 'regime_xgb_v1',
      feature_set_version: 'fs_v1',
      alignment_method: 'C1',
      fallback_used: false,
    },
  },
  summary:
    'The model raised the district mean from 18.0 to 24.5 mm (+6.5). Most likely phase: normal (confidence low).',
}

export const MOCK_REGIME: RegimeResponse = {
  _mock: true,
  run_id: 'tigge_ecmwf_cf_2024071500',
  mode: 'replay',
  evaluation_set: 'holdout',
  lead_day: 1,
  imd_date: '2024-07-16',
  phase: {
    active: 0.42,
    normal: 0.46,
    break: 0.12,
    confidence: 0.46,
    confidence_band: 'low',
    dominant_phase: 'normal',
  },
  nearest_lps: {
    present: true,
    latitude: 19.5,
    longitude: 86.2,
    distance_km: 260,
    bearing_deg: 235,
    influence: 0.52,
    settings: 'tuned',
  },
  indicators: [
    {
      name: 'Monsoon Trough Latitude',
      code: 'A1',
      value: 23.2,
      unit: '°N',
      training_percentile: 65,
      description: 'Position of minimum sea level pressure across central India',
    },
    {
      name: '850 hPa Zonal Wind',
      code: 'A2',
      value: 12.4,
      unit: 'm/s',
      training_percentile: 78,
      description: 'Core westerly low-level jet strength across peninsular India',
    },
    {
      name: 'Tibetan High Geopotential',
      code: 'A3',
      value: 14320,
      unit: 'gpm',
      training_percentile: 55,
      description: 'Upper tropospheric anticyclone strength over Tibetan plateau',
    },
    {
      name: 'Offshore Trough Index',
      code: 'A4',
      value: 0.74,
      unit: 'idx',
      training_percentile: 82,
      description: 'Pressure gradient along the Konkan-Goa-Karnataka coastline',
    },
    {
      name: 'Cross-Equatorial Moisture Flux',
      code: 'A5',
      value: 320,
      unit: 'kg/(m·s)',
      training_percentile: 71,
      description: 'Somali jet moisture transport into Arabian Sea',
    },
    {
      name: 'Monsoon Zone Relative Vorticity',
      code: 'A6',
      value: 2.8e-5,
      unit: 's⁻¹',
      training_percentile: 84,
      description: 'Cyclonic shear along monsoon trough line',
    },
  ],
  domain_mean_raw_mm: 8.2,
  domain_mean_corrected_mm: 9.6,
  regime_available: true,
  ood_flag: false,
}

export const MOCK_TRANSITIONS: RegimeTransitionsResponse = {
  _mock: true,
  run_id: 'tigge_ecmwf_cf_2024071500',
  mode: 'replay',
  evaluation_set: 'holdout',
  series: [
    { imd_date: '2024-07-10', p_active: 0.15, p_normal: 0.75, p_break: 0.1, lps_present: false },
    { imd_date: '2024-07-11', p_active: 0.2, p_normal: 0.7, p_break: 0.1, lps_present: false },
    { imd_date: '2024-07-12', p_active: 0.25, p_normal: 0.65, p_break: 0.1, lps_present: false },
    { imd_date: '2024-07-13', p_active: 0.35, p_normal: 0.55, p_break: 0.1, lps_present: true },
    { imd_date: '2024-07-14', p_active: 0.42, p_normal: 0.48, p_break: 0.1, lps_present: true },
    { imd_date: '2024-07-15', p_active: 0.55, p_normal: 0.38, p_break: 0.07, lps_present: true },
    { imd_date: '2024-07-16', p_active: 0.62, p_normal: 0.32, p_break: 0.06, lps_present: true },
    { imd_date: '2024-07-17', p_active: 0.68, p_normal: 0.28, p_break: 0.04, lps_present: true },
  ],
  events: [
    {
      event_id: 1,
      event_type: 'PHASE:normal->active',
      from_state: 'normal',
      to_state: 'active',
      event_date: '2024-07-15',
      confirmed: true,
      confidence: 0.585,
      district_id: null,
      domain_mean_corrected_change_mm: 3.4,
      n_dates_before: 3,
      n_dates_after: 3,
    },
    {
      event_id: 2,
      event_type: 'LPS_FORMS',
      event_date: '2024-07-13',
      confirmed: true,
      confidence: null,
      district_id: null,
      domain_mean_corrected_change_mm: 2.8,
      n_dates_before: 3,
      n_dates_after: 3,
    },
  ],
}

export const MOCK_ANALOGS: AnalogsResponse = {
  _mock: true,
  run_id: 'tigge_ecmwf_cf_2024071500',
  mode: 'replay',
  evaluation_set: 'holdout',
  lead_day: 1,
  district_id: 'D001',
  analogs: [
    {
      rank: 1,
      imd_date: '2019-07-28',
      season: 2019,
      distance: 1.31,
      distance_percentile: 0.6,
      observed_mean_mm: 22.0,
      observed_wettest_cell_mm: 47.0,
      raw_mean_mm: 15.0,
      corrected_mean_mm: 21.0,
      error_observed_minus_raw_mm: 7.0,
    },
    {
      rank: 2,
      imd_date: '2021-08-04',
      season: 2021,
      distance: 1.48,
      distance_percentile: 1.2,
      observed_mean_mm: 28.5,
      observed_wettest_cell_mm: 56.0,
      raw_mean_mm: 20.0,
      corrected_mean_mm: 26.5,
      error_observed_minus_raw_mm: 8.5,
    },
    {
      rank: 3,
      imd_date: '2018-07-19',
      season: 2018,
      distance: 1.62,
      distance_percentile: 1.9,
      observed_mean_mm: 19.0,
      observed_wettest_cell_mm: 42.0,
      raw_mean_mm: 16.5,
      corrected_mean_mm: 18.2,
      error_observed_minus_raw_mm: 2.5,
    },
    {
      rank: 4,
      imd_date: '2020-08-14',
      season: 2020,
      distance: 1.75,
      distance_percentile: 2.4,
      observed_mean_mm: 31.0,
      observed_wettest_cell_mm: 62.0,
      raw_mean_mm: 24.0,
      corrected_mean_mm: 29.0,
      error_observed_minus_raw_mm: 7.0,
    },
    {
      rank: 5,
      imd_date: '2022-07-09',
      season: 2022,
      distance: 1.88,
      distance_percentile: 3.1,
      observed_mean_mm: 17.5,
      observed_wettest_cell_mm: 39.0,
      raw_mean_mm: 14.0,
      corrected_mean_mm: 16.8,
      error_observed_minus_raw_mm: 3.5,
    },
  ],
  median_error_observed_minus_raw_mm: 7.0,
  n_analogs: 5,
  note: '5 cases only',
}

export const MOCK_HOTSPOTS: HotspotsResponse = {
  _mock: true,
  run_id: 'tigge_ecmwf_cf_2024071500',
  mode: 'replay',
  evaluation_set: 'holdout',
  lead_day: 1,
  threshold_mm: 64.5,
  hotspots: [
    {
      hotspot_id: 1,
      n_cells: 6,
      max_probability: 0.63,
      max_corrected_mean_mm: 58.0,
      centroid: { lat: 17.4, lon: 73.6 },
      district_ids: ['D001', 'D002'],
      outline: {
        type: 'Polygon',
        coordinates: [
          [
            [73.3, 17.1],
            [73.8, 17.1],
            [73.8, 17.7],
            [73.3, 17.7],
            [73.3, 17.1],
          ],
        ],
      },
    },
  ],
}

export function getMockGridCells(
  variable: GridVariable = 'corrected',
  leadDay: number = 1
): GridCellData[] {
  const cells: GridCellData[] = []
  let id = 1000

  // 0.25° grid across West Coast demo region (Ratnagiri / Konkan / Goa / Coastal Karnataka)
  for (let lat = 14.5; lat <= 20.0; lat += 0.25) {
    for (let lon = 72.75; lon <= 76.0; lon += 0.25) {
      id++
      // Orographic Ghats peak simulation near lon 73.75
      const ghatsFactor = Math.max(0, 1 - Math.abs(lon - 73.75) * 1.5)
      const latFactor = Math.sin((lat - 14.0) / 2.0)
      const baseRain = Math.max(0, 15 + 45 * ghatsFactor * latFactor - (leadDay - 1) * 3)

      let value: number = baseRain

      if (variable === 'raw') {
        // Raw NWP often underforecasts heavy orographic rain on Western Ghats
        value = Math.max(0, baseRain * 0.72 - 4.0)
      } else if (variable === 'corrected') {
        // AI post-processed captures coastal/orographic enhancement
        value = Number(baseRain.toFixed(1))
      } else if (variable === 'observed') {
        // Truth observation with realistic variance
        value = Number((baseRain * 1.05 + ((id % 7) - 3)).toFixed(1))
        if (value < 0) value = 0
      } else if (variable === 'difference') {
        // corrected - raw
        const rawVal = Math.max(0, baseRain * 0.72 - 4.0)
        value = Number((baseRain - rawVal).toFixed(1))
      } else if (variable === 'improvement') {
        // |raw - observed| - |corrected - observed|
        const rawVal = Math.max(0, baseRain * 0.72 - 4.0)
        const obsVal = Math.max(0, baseRain * 1.05 + ((id % 7) - 3))
        const rawErr = Math.abs(rawVal - obsVal)
        const corrErr = Math.abs(baseRain - obsVal)
        value = Number((rawErr - corrErr).toFixed(1))
      } else if (variable === 'p_ge_15_6') {
        value = baseRain >= 15.6 ? 0.85 : 0.35
      } else if (variable === 'p_ge_64_5') {
        value = baseRain >= 45 ? 0.62 : 0.12
      } else if (variable === 'p_ge_115_6') {
        value = baseRain >= 65 ? 0.28 : 0.02
      }

      cells.push({
        cell_id: id,
        lat: Number(lat.toFixed(2)),
        lon: Number(lon.toFixed(2)),
        value: Number(value.toFixed(1)),
      })
    }
  }

  return cells
}

export const MOCK_GRID_CELLS: GridResponse = {
  _mock: true,
  run_id: 'tigge_ecmwf_cf_2024071500',
  mode: 'replay',
  evaluation_set: 'holdout',
  lead_day: 1,
  variable: 'corrected',
  cells: getMockGridCells('corrected', 1),
}

export const MOCK_VERIFICATION: VerificationResponse = {
  _mock: true,
  forecast_type: 'regime_aware_ml',
  comparison_to: 'raw_nwp',
  evaluation_set: 'holdout',
  lead_day: 1,
  threshold_mm: 64.5,
  neighbourhood_cells: 5,
  group: { type: 'all', value: 'all' },
  metrics: {
    rmse: { value: null, ci_low: null, ci_high: null },
    pod: { value: null, ci_low: null, ci_high: null },
    far: { value: null, ci_low: null, ci_high: null },
    csi: { value: null, ci_low: null, ci_high: null },
    ets: { value: null, ci_low: null, ci_high: null },
    fss: { value: null, ci_low: null, ci_high: null },
    frequency_bias: { value: null, ci_low: null, ci_high: null },
  },
  paired_difference: {
    ets: { value: null, ci_low: null, ci_high: null },
  },
  n_samples: null,
  n_events: null,
  bootstrap: { block_days: 7, resamples: 2000 },
  model_version_id: null,
}

export const MOCK_RELIABILITY: ReliabilityResponse = {
  _mock: true,
  forecast_type: 'regime_aware_ml',
  lead_day: 1,
  threshold_mm: 64.5,
  bins: [
    { bin_center: 0.1, observed_frequency: null, sample_count: 0 },
    { bin_center: 0.3, observed_frequency: null, sample_count: 0 },
    { bin_center: 0.5, observed_frequency: null, sample_count: 0 },
    { bin_center: 0.7, observed_frequency: null, sample_count: 0 },
    { bin_center: 0.9, observed_frequency: null, sample_count: 0 },
  ],
  brier_score: null,
  brier_skill_score: null,
}

export const MOCK_MODEL_INFO: ModelInfoResponse = {
  _mock: true,
  mode: 'replay',
  evaluation_set: 'holdout',
  models: [
    {
      role: 'correction_mean (B3)',
      version: 'regime_xgb_v1',
      algorithm: 'XGBoost Tweedie',
      feature_set_version: 'fs_v1',
      n_features: 41,
    },
    {
      role: 'heavy_rain_probability',
      version: 'hr_xgb_v1',
      algorithm: 'XGBoost Logistic / Monotone',
      feature_set_version: 'fs_v1',
      n_features: 41,
    },
    {
      role: 'monsoon_phase_classifier',
      version: 'phase_lr_v1',
      algorithm: 'Multinomial Logistic Regression',
      feature_set_version: 'fs_v1',
      n_features: 6,
    },
  ],
  data: {
    nwp: 'ECMWF-TIGGE-control (00 UTC)',
    truth: 'IMD 0.25° Gridded Rainfall',
    alignment_method: 'C1',
    lead_days: [1, 2, 3],
    season_scope: 'JJAS (June 1 – September 30)',
    district_file: 'DataMeet, 2011 Census Districts (~640 districts)',
  },
  protocol: {
    n_seasons: 10,
    development_seasons: [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    holdout_seasons: [2024, 2025],
    holdout_locked: true,
    status: 'LOCKED EVALUATION',
  },
  limitations: [
    'Districts created after 2011 are not included in boundary files (Census 2011 reference).',
    'Western disturbances are not modelled (system is scoped specifically for JJAS monsoon).',
    'Influence values are percentile or terrain-slope indices, not probabilities.',
    'q10–q90 rainfall range is model-estimated, not a guaranteed physical interval.',
    'System runs strictly in Replay mode on stored historical forecast archives.',
  ],
  fallback_share_recent: 0.0,
}

export const MOCK_MAP_METADATA: MapMetadataResponse = {
  grid_bounds: {
    north: 38.5,
    south: 6.5,
    west: 66.5,
    east: 100.0,
    step_deg: 0.25,
    n_lat: 129,
    n_lon: 135,
    total_valid_cells: 4620,
  },
  available_leads: [1, 2, 3],
  thresholds_mm: [15.6, 64.5, 115.6],
  district_source: 'DataMeet 2011 Census',
  district_count_2011: 640,
  layers: ['raw', 'corrected', 'difference', 'observed', 'improvement'],
}

export const MOCK_HEALTH: HealthResponse = {
  status: 'ok',
  version: '0.1.0',
}

export const MOCK_DISTRICTS_GEOJSON: GeoJsonFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      id: 'D001',
      properties: {
        district_id: 'D001',
        name: 'Ratnagiri',
        state: 'Maharashtra',
        source_year: 2011,
      },
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [73.1, 16.8],
            [73.6, 16.8],
            [73.5, 17.6],
            [73.1, 17.6],
            [73.1, 16.8],
          ],
        ],
      },
    },
    {
      type: 'Feature',
      id: 'D002',
      properties: {
        district_id: 'D002',
        name: 'Sindhudurg',
        state: 'Maharashtra',
        source_year: 2011,
      },
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [73.4, 15.6],
            [74.0, 15.6],
            [73.9, 16.4],
            [73.4, 16.4],
            [73.4, 15.6],
          ],
        ],
      },
    },
  ],
}
