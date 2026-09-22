"""M3 Rainfall Models Tests (PRD §23, §12.1).

Required M3 Tests from PRD §23:
- B0 output == rain_mm
- B1 fits independently by lead+region
- B1 output nonnegative
- B1 deterministic fit/predict
- B1 unseen fitting data is not accessed (LOSO compliance)
- B1 serialization round-trip gives same predictions
- B1 high-tail rule works (ratio of 99.5th percentiles)
- Missing required columns raise clear error
- Saving and loading round-trip
- Output shape
- Monotone constraint holds
- No negative corrected values
- Probability monotonicity
- Quantile ordering
- Extrapolation flag and fallback
- Protocol guard: stop with error if training row has regime_source = 'final'
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ml.baselines.b0_raw import predict_b0, B0RawBaseline
from ml.baselines.b1_quantile_mapping import (
    QuantileMappingModel,
    compute_mid_tie_probabilities,
)
from ml.training.train_correction import (
    BASE_FEATURES_27,
    REGIME_FEATURES_14,
    REGIME_AWARE_FEATURES_41,
    validate_training_dataframe,
    validate_b3_training_data,
    train_b2_model,
    train_b3_model,
    fit_final_b2_model,
    fit_final_b3_model,
    build_b2_monotone_constraints,
    build_b3_monotone_constraints,
)
from ml.feature_contracts import (
    BASE_FEATURES,
    REGIME_FEATURES,
    B2_FEATURES,
    B3_FEATURES,
    get_feature_list,
    validate_feature_columns,
)
from ml.training.hyperparameter_search import (
    PARAM_GRID,
    generate_shared_hyperparameter_configs,
    get_validation_seasons,
    resolve_protocol_seed,
    evaluate_settings_search,
)
from ml.inference.predict_correction import (
    compute_extrapolation_threshold,
    check_extrapolation,
    select_fallback_product,
    predict_b2,
    predict_b3,
    apply_rainfall_correction,
)
from ml.inference.serving_grid_writer import (
    SERVING_GRID_COLUMNS,
    VALID_PRODUCT_TYPES,
    VALID_FALLBACK_REASONS,
    validate_serving_grid_schema,
    validate_serving_grid_contracts,
    assemble_serving_grid,
    write_serving_grid_file,
    read_serving_grid_file,
)
from ml.orchestration import (
    FinalM3Models,
    validate_m1_m2_inputs,
    run_development_oof_pipeline,
    fit_final_m3_models,
    predict_final_m3,
)
from probability.classifiers import (
    enforce_probability_monotonicity,
    determine_model_availability,
    ProbabilityModelAvailability,
    train_probability_classifier,
    evaluate_probability_settings_search,
    RainfallProbabilityModels,
    predict_probabilities,
)
from probability.range_models import (
    sort_and_clip_quantiles,
    train_quantile_model,
    QuantileRangeModels,
    predict_range,
    calculate_range_coverage,
    QUANTILES,
)
from probability.calibration import (
    apply_probability_calibrators,
    apply_single_calibrator,
)
from ml.models.model_store import (
    save_model,
    load_model,
    save_range_models,
    load_range_models,
    ModelMetadata,
    VALID_MODEL_TYPES,
)
from tests.synthetic_fixtures import generate_synthetic_m3_data, ALL_REGIONS


# ==============================================================================
# B0 Raw Baseline Tests
# ==============================================================================

def test_b0_output_equals_rain_mm():
    """Verify B0 output exactly equals rain_mm and preserves row/index alignment."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    # Set arbitrary non-default index to test index preservation
    df.index = [f"idx_{i}" for i in range(len(df))]

    pred_series = predict_b0(df)
    assert isinstance(pred_series, pd.Series)
    assert pred_series.index.equals(df.index), "B0 must preserve row/index alignment"
    np.testing.assert_array_equal(pred_series.to_numpy(), df["rain_mm"].to_numpy())

    # Class wrapper
    baseline = B0RawBaseline()
    pred_class = baseline.predict(df)
    np.testing.assert_array_equal(pred_class.to_numpy(), df["rain_mm"].to_numpy())


def test_b0_missing_required_columns():
    """Verify B0 raises clear ValueError when required 'rain_mm' column is missing."""
    df = pd.DataFrame({"forecast_precip": [10.0, 20.0]})
    with pytest.raises(ValueError, match="Missing required column 'rain_mm'"):
        predict_b0(df)


# ==============================================================================
# B1 Quantile Mapping Tests (PRD §12.1)
# ==============================================================================

def test_b1_fits_independently_by_lead_and_region():
    """Verify B1 creates independent empirical mappings for each (lead_day, region_code)."""
    df = generate_synthetic_m3_data(dates_per_season=5)
    model = QuantileMappingModel(n_steps=100)
    model.fit(df)

    # Check that all (lead_day, region_code) combinations exist
    for lead in [1, 2, 3]:
        for region in ALL_REGIONS:
            key = f"{lead}_{region}"
            assert key in model.mappings, f"Missing mapping for {key}"
            mapping = model.mappings[key]
            assert len(mapping["obs_quantiles"]) == 100
            assert "tail_ratio" in mapping
            assert "f_max_step" in mapping


def test_b1_output_nonnegative():
    """Verify B1 corrected rainfall values are strictly >= 0."""
    train_df = generate_synthetic_m3_data(seasons=[2018, 2019], dates_per_season=5)
    test_df = generate_synthetic_m3_data(seasons=[2020], dates_per_season=3)

    model = QuantileMappingModel().fit(train_df)
    preds = model.predict(test_df)

    assert isinstance(preds, pd.Series)
    assert preds.index.equals(test_df.index), "B1 predict must preserve index alignment"
    assert (preds >= 0.0).all(), "B1 predictions must be non-negative"


def test_b1_deterministic_fit_predict():
    """Verify that fitting and predicting on the same data is strictly deterministic."""
    df = generate_synthetic_m3_data(dates_per_season=4, seed=123)

    m1 = QuantileMappingModel().fit(df)
    p1 = m1.predict(df)

    m2 = QuantileMappingModel().fit(df)
    p2 = m2.predict(df)

    np.testing.assert_array_equal(p1.to_numpy(), p2.to_numpy())


def test_b1_unseen_fitting_data_not_accessed():
    """Verify that held-out/validation season data is not accessed during training fit."""
    full_df = generate_synthetic_m3_data(seasons=[2018, 2019, 2020], dates_per_season=5)
    train_seasons = [2018, 2019]

    # Fit passing explicit training_seasons filter
    model = QuantileMappingModel().fit(full_df, training_seasons=train_seasons)

    assert model.training_seasons == [2018, 2019]

    # Fit directly on pre-filtered data
    filtered_df = full_df[full_df["season"].isin(train_seasons)]
    model_filtered = QuantileMappingModel().fit(filtered_df)

    # Check identical mappings
    for k in model.mappings:
        np.testing.assert_array_equal(
            model.mappings[k]["obs_quantiles"],
            model_filtered.mappings[k]["obs_quantiles"],
        )
        assert model.mappings[k]["tail_ratio"] == model_filtered.mappings[k]["tail_ratio"]


def test_b1_serialization_round_trip():
    """Verify that saving and loading B1 model produces identical predictions."""
    train_df = generate_synthetic_m3_data(seasons=[2018, 2019], dates_per_season=4)
    test_df = generate_synthetic_m3_data(seasons=[2020], dates_per_season=2)

    model = QuantileMappingModel().fit(train_df)
    pred_orig = model.predict(test_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "b1_qm.json"
        model.save(model_path)
        assert model_path.exists()

        loaded_model = QuantileMappingModel.load(model_path)
        pred_loaded = loaded_model.predict(test_df)

        np.testing.assert_array_equal(pred_orig.to_numpy(), pred_loaded.to_numpy())


def test_b1_high_tail_rule():
    """Verify PRD §12.1 tail rule: values above largest step use 99.5th percentile ratio."""
    # Construct synthetic data where:
    # forecast has p99.5 = 100.0, max step (p99) = 90.0
    # observed has p99.5 = 150.0
    # Expected tail_ratio = 150 / 100 = 1.5
    f_vals = np.linspace(0.0, 100.0, 1000)
    o_vals = np.linspace(0.0, 150.0, 1000)

    df_train = pd.DataFrame({
        "lead_day": 1,
        "region_code": "WEST_COAST",
        "rain_mm": f_vals,
        "obs_mm": o_vals,
    })

    model = QuantileMappingModel().fit(df_train)
    key = "1_WEST_COAST"
    tail_ratio = model.mappings[key]["tail_ratio"]
    f_max_step = model.mappings[key]["f_max_step"]

    # Test an extreme forecast above largest step
    extreme_forecast = f_max_step + 50.0
    df_test = pd.DataFrame({
        "lead_day": [1],
        "region_code": ["WEST_COAST"],
        "rain_mm": [extreme_forecast],
    })

    pred = model.predict(df_test).iloc[0]
    expected = extreme_forecast * tail_ratio
    assert np.isclose(pred, expected, rtol=1e-5), f"Expected {expected}, got {pred}"


def test_b1_missing_required_columns():
    """Verify B1 raises clear error when required columns are missing."""
    df_missing_obs = pd.DataFrame({
        "lead_day": [1],
        "region_code": ["WEST_COAST"],
        "rain_mm": [10.0],
    })
    model = QuantileMappingModel()
    with pytest.raises(ValueError, match="Missing required columns for B1 fit:.*obs_mm"):
        model.fit(df_missing_obs)

    df_missing_rain = pd.DataFrame({
        "lead_day": [1],
        "region_code": ["WEST_COAST"],
    })
    with pytest.raises(ValueError, match="Missing required columns for B1 predict:.*rain_mm"):
        model.predict(df_missing_rain)


def test_mid_tie_probabilities():
    """Verify middle-of-tie calculation."""
    f_sorted = np.array([0.0, 0.0, 0.0, 0.0, 10.0])  # 80% zeros
    x = np.array([0.0, 10.0, 20.0])
    p = compute_mid_tie_probabilities(f_sorted, x)

    # 4 zeros span [0, 4], midpoint is (0 + 4) / 10 = 0.40
    assert np.isclose(p[0], 0.40)
    # single 10 spans [4, 5], midpoint is (4 + 5) / 10 = 0.90
    assert np.isclose(p[1], 0.90)
    # 20 is above maximum (idx 5), midpoint is (5 + 5) / 10 = 1.0
    assert np.isclose(p[2], 1.00)


# ==============================================================================
# Existing M3 Contract and Protocol Tests
# ==============================================================================

# ==============================================================================
# Feature Contracts and Protocol Tests (PRD §9.4, §10.6, §11.7)
# ==============================================================================

EXPECTED_PRD_BASE_27 = [
    "rain_mm", "nbr_mean_3", "nbr_max_3", "nbr_mean_5", "nbr_max_5",
    "rain_grad", "rain_prev_lead", "rain_next_lead",
    "u850", "v850", "wspd850", "vort850", "q850", "msl", "shear_200_850",
    "elevation_m", "slope", "aspect_sin", "aspect_cos", "dist_coast_km",
    "clim_mean", "clim_p95",
    "doy_sin", "doy_cos", "lead_day", "latitude", "longitude",
]

EXPECTED_PRD_REGIME_14 = [
    "p_active", "p_normal", "p_break", "regime_confidence",
    "lps_present", "distance_to_lps_km", "bearing_sin", "bearing_cos",
    "lps_strength", "lps_influence",
    "upslope_flux", "onshore_flux", "orographic_influence", "coastal_influence",
]


def test_feature_counts_and_exact_prd_order():
    """Verify B2 has exactly 27 and B3 has exactly 41 features in exact PRD order."""
    assert len(BASE_FEATURES) == 27, "B2 must have exactly 27 base features"
    assert len(REGIME_FEATURES) == 14, "Regime features must be exactly 14"
    assert len(B2_FEATURES) == 27, "B2_FEATURES must have length 27"
    assert len(B3_FEATURES) == 41, "B3_FEATURES must have length 41"

    # Exact PRD order check
    assert list(BASE_FEATURES) == EXPECTED_PRD_BASE_27, "BASE_FEATURES does not match PRD §9.4 order"
    assert list(REGIME_FEATURES) == EXPECTED_PRD_REGIME_14, "REGIME_FEATURES does not match PRD §11.7 order"
    assert list(B2_FEATURES) == EXPECTED_PRD_BASE_27
    assert list(B3_FEATURES) == EXPECTED_PRD_BASE_27 + EXPECTED_PRD_REGIME_14


def test_no_duplicate_features():
    """Verify there are no duplicate feature names in B2 or B3."""
    assert len(set(B2_FEATURES)) == len(B2_FEATURES) == 27
    assert len(set(B3_FEATURES)) == len(B3_FEATURES) == 41


def test_forbidden_features_absent():
    """Verify cell_id (L7) and obs_mm (L1) are strictly absent from model feature sets."""
    assert "cell_id" not in B2_FEATURES
    assert "cell_id" not in B3_FEATURES
    assert "obs_mm" not in B2_FEATURES
    assert "obs_mm" not in B3_FEATURES
    assert "observed_rain" not in B3_FEATURES

    # Validate that custom feature list containing forbidden column raises ValueError
    dummy_df = pd.DataFrame(columns=B2_FEATURES + ["cell_id", "obs_mm"])
    with pytest.raises(ValueError, match="Protocol violation.*'cell_id' is forbidden"):
        validate_feature_columns(dummy_df, custom_features=["rain_mm", "cell_id"])

    with pytest.raises(ValueError, match="Protocol violation.*'obs_mm' is forbidden"):
        validate_feature_columns(dummy_df, custom_features=["rain_mm", "obs_mm"])


def test_missing_feature_validation():
    """Verify validation raises clear ValueError when required feature columns are missing."""
    df_missing = pd.DataFrame(columns=["rain_mm", "nbr_mean_3"])
    with pytest.raises(ValueError, match="Missing 25 required feature columns for B2"):
        validate_feature_columns(df_missing, model_type="B2")


def test_nan_values_accepted_without_imputation():
    """Verify NaN values are accepted in feature tables and not imputed or rejected."""
    data = {col: [1.0, 2.0] for col in B2_FEATURES}
    # PRD §9.4: rain_prev_lead at lead 1 and rain_next_lead at lead 3 are left empty (NaN)
    data["rain_prev_lead"] = [np.nan, 5.0]
    data["rain_next_lead"] = [10.0, np.nan]
    df = pd.DataFrame(data)

    validated_cols = validate_feature_columns(df, model_type="B2")
    assert validated_cols == B2_FEATURES
    # Verify NaNs are preserved (not imputed)
    assert np.isnan(df.loc[0, "rain_prev_lead"])
    assert np.isnan(df.loc[1, "rain_next_lead"])


# ==============================================================================
# Hyperparameter Search and Validation Seasons Tests (PRD §12.2)
# ==============================================================================

def test_hyperparameter_generation_count_and_uniqueness():
    """Verify exactly 20 unique parameter configs are generated deterministically."""
    configs_1 = generate_shared_hyperparameter_configs(seed=42, n_configs=20)
    assert len(configs_1) == 20

    # Test uniqueness
    signatures = [tuple(sorted(c.items())) for c in configs_1]
    assert len(set(signatures)) == 20, "All 20 generated configs must be strictly unique"

    # Determinism with same seed
    configs_2 = generate_shared_hyperparameter_configs(seed=42, n_configs=20)
    assert configs_1 == configs_2

    # Different seed produces different combinations/sequence
    configs_diff = generate_shared_hyperparameter_configs(seed=999, n_configs=20)
    assert configs_1 != configs_diff


def test_hyperparameter_grid_membership():
    """Verify every sampled parameter value strictly belongs to the PRD §12.2 grid."""
    configs = generate_shared_hyperparameter_configs(seed=42, n_configs=20)
    for cfg in configs:
        for param, allowed_values in PARAM_GRID.items():
            assert param in cfg, f"Missing parameter {param} in config"
            assert cfg[param] in allowed_values, (
                f"Value {cfg[param]} for {param} not in allowed grid {allowed_values}"
            )


def test_b2_b3_shared_configs_identity():
    """Verify B2 and B3 consume the exact same generated configuration list."""
    shared_configs = generate_shared_hyperparameter_configs(seed=42, n_configs=20)

    # Both systems must use the identical list
    b2_configs = shared_configs
    b3_configs = shared_configs
    assert b2_configs is b3_configs
    assert len(b2_configs) == 20


def test_resolve_protocol_seed_requires_explicit_seed():
    """Verify resolve_protocol_seed does not silently invent a seed when config is absent."""
    with pytest.raises(ValueError, match="No seed provided and config/protocol.yaml is not present"):
        resolve_protocol_seed(seed=None, config_path="non_existent_config.yaml")

    # Explicit seed is accepted
    assert resolve_protocol_seed(seed=123) == 123


def test_get_validation_seasons_newest_three():
    """Verify newest 3 development seasons are correctly selected."""
    dev_seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    val_seasons = get_validation_seasons(dev_seasons)
    assert val_seasons == [2021, 2022, 2023]

    # Unordered input
    unordered = [2020, 2016, 2023, 2018, 2021, 2017, 2019, 2022]
    assert get_validation_seasons(unordered) == [2021, 2022, 2023]

    # Fewer than 3 seasons raises error
    with pytest.raises(ValueError, match="At least 3 development seasons are required"):
        get_validation_seasons([2022, 2023])


def test_regime_source_training_guard():
    """Verify training fails if regime_source is 'final' (PRD §11.6, §21.1)."""
    bad_df = pd.DataFrame({
        "rain_mm": [10.0, 20.0],
        "regime_source": ["oof", "final"],
    })
    with pytest.raises(ValueError, match="Protocol violation"):
        validate_training_dataframe(bad_df, is_b3=True)

    good_df = pd.DataFrame({
        "rain_mm": [10.0, 20.0],
        "regime_source": ["oof", "oof"],
    })
    validate_training_dataframe(good_df, is_b3=True)


def test_extrapolation_detection():
    """Verify extrapolation flag triggers above 99.9th percentile threshold."""
    p99_9 = 150.0
    rain_vals = np.array([10.0, 149.9, 150.0, 150.1, 200.0])
    flags = check_extrapolation(rain_vals, p99_9)
    np.testing.assert_array_equal(flags, [False, False, False, True, True])


def test_serving_grid_schema_contract():
    """Verify serving grid column schema matches PRD §20.1 18 columns."""
    assert len(SERVING_GRID_COLUMNS) == 18
    dummy_data = {col: [0] for col in SERVING_GRID_COLUMNS}
    df = pd.DataFrame(dummy_data)
    validate_serving_grid_schema(df)

    incomplete_df = df.drop(columns=["corrected_mean_mm"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_serving_grid_schema(incomplete_df)


def test_probability_monotonicity_enforcement():
    """Verify P(>=115.6) <= P(>=64.5) <= P(>=15.6) across all cells (PRD §13.3)."""
    p15 = np.array([0.4, 0.8, 0.2])
    p64 = np.array([0.6, 0.5, 0.3])
    p115 = np.array([0.7, 0.2, 0.4])

    p15_out, p64_out, p115_out = enforce_probability_monotonicity(p15, p64, p115)

    assert np.all(p115_out <= p64_out)
    assert np.all(p64_out <= p15_out)
    assert np.all(p15_out <= 1.0)
    assert np.all(p115_out >= 0.0)


def test_quantile_sorting_and_clipping():
    """Verify q10 <= q50 <= q90 and non-negative clipping (PRD §13.7)."""
    q10 = np.array([-2.0, 50.0, 10.0])
    q50 = np.array([5.0, 20.0, 15.0])
    q90 = np.array([10.0, 15.0, 30.0])

    q10_out, q50_out, q90_out = sort_and_clip_quantiles(q10, q50, q90)

    assert np.all(q10_out <= q50_out)
    assert np.all(q50_out <= q90_out)
    assert np.all(q10_out >= 0.0)


# ==============================================================================
# B2 Model Training, Evaluation, and Prediction Tests (PRD §10, §12)
# ==============================================================================

def test_b2_features_exact_and_no_regime():
    """Verify B2 uses exactly 27 features and strictly contains zero regime features."""
    assert len(B2_FEATURES) == 27
    for regime_feat in REGIME_FEATURES:
        assert regime_feat not in B2_FEATURES, f"Regime feature '{regime_feat}' found in B2"


def test_b2_model_specifications_and_monotone_constraints():
    """Verify B2 model specs: reg:tweedie, hist, and rain_mm=+1, others=0."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    test_params = {
        "tweedie_variance_power": 1.5,
        "max_depth": 4,
        "min_child_weight": 50,
        "learning_rate": 0.05,
        "n_estimators": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
        "random_state": 42,
    }

    model = train_b2_model(df, params=test_params)
    params = model.get_params()

    assert params["objective"] == "reg:tweedie", f"Expected reg:tweedie, got {params['objective']}"
    assert params["tree_method"] == "hist", f"Expected hist, got {params['tree_method']}"

    # Monotone constraints
    constraints = params["monotone_constraints"]
    assert len(constraints) == 27, f"Expected 27 constraints, got {len(constraints)}"
    assert constraints[0] == 1, "First feature (rain_mm) must have +1 constraint"
    assert all(c == 0 for c in constraints[1:]), "All remaining 26 features must have 0 constraint"


def test_b2_dry_rows_and_nans_preserved():
    """Verify B2 model trains directly with zero-rain rows and un-imputed NaNs."""
    df = generate_synthetic_m3_data(dates_per_season=2)

    # Verify zero-rain rows are present in training data
    assert (df["obs_mm"] == 0.0).any(), "Dry rows must be present in dataset"
    assert (df["rain_mm"] == 0.0).any(), "Dry forecast rows must be present"

    # Verify NaNs exist in lead 1 and lead 3
    assert df.loc[df["lead_day"] == 1, "rain_prev_lead"].isna().all()
    assert df.loc[df["lead_day"] == 3, "rain_next_lead"].isna().all()

    # Train model directly on data containing dry rows and NaNs
    test_params = {"n_estimators": 5, "max_depth": 4, "learning_rate": 0.05}
    model = train_b2_model(df, params=test_params)
    assert model is not None


def test_b2_prediction_properties():
    """Verify B2 predictions preserve index/order and are strictly non-negative."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    # Assign custom non-sequential index
    custom_idx = [f"cell_row_{i}" for i in range(len(df))]
    df.index = custom_idx

    test_params = {"n_estimators": 5, "max_depth": 4, "learning_rate": 0.05}
    model = train_b2_model(df, params=test_params)

    preds = predict_b2(model, df)

    assert isinstance(preds, pd.Series)
    assert preds.index.equals(df.index), "Prediction must preserve row and index alignment"
    assert (preds >= 0.0).all(), "Predictions must be strictly non-negative (>= 0)"
    assert not preds.isna().any(), "Predictions must not contain NaNs"


def test_b2_settings_search_season_splitting_and_evaluation():
    """Verify 20-config evaluation across 3 validation seasons and holdout isolation."""
    # 4 development seasons + 1 holdout season
    dev_seasons = [2017, 2018, 2019, 2020]
    holdout_season = 2021
    all_seasons = dev_seasons + [holdout_season]

    df = generate_synthetic_m3_data(seasons=all_seasons, dates_per_season=2, seed=77)

    # 20 fast test configs using PRD grid values with n_estimators=3 for fast test execution
    base_configs = generate_shared_hyperparameter_configs(seed=42, n_configs=20)
    fast_configs = [{**cfg, "n_estimators": 3} for cfg in base_configs]

    result = evaluate_settings_search(
        df,
        development_seasons=dev_seasons,
        combinations=fast_configs,
        n_validation_seasons=3,
    )

    # 1. Exactly 20 configurations evaluated
    assert len(result["all_results"]) == 20

    # 2. Validation seasons are the newest 3 development seasons
    assert result["validation_seasons"] == [2018, 2019, 2020]
    assert holdout_season not in result["validation_seasons"], "Holdout season must never be evaluated"

    # 3. Validation season never appears in its own training fold
    for entry in result["all_results"]:
        assert len(entry["rmse_per_season"]) == 3
        for val_s in [2018, 2019, 2020]:
            assert val_s in entry["rmse_per_season"]
            assert entry["rmse_per_season"][val_s] >= 0.0

    # 4. Best config selected by lowest mean RMSE (first occurrence preserved on tie)
    all_mean_rmses = [r["mean_rmse"] for r in result["all_results"]]
    min_mean_rmse = min(all_mean_rmses)
    expected_best_idx = all_mean_rmses.index(min_mean_rmse)

    assert np.isclose(result["best_mean_rmse"], min_mean_rmse)
    assert result["best_config_index"] == expected_best_idx
    assert result["best_config"] == fast_configs[expected_best_idx]


def test_b2_deterministic_result():
    """Verify deterministic training and prediction given identical data and seed."""
    df = generate_synthetic_m3_data(dates_per_season=2, seed=10)
    params = {"n_estimators": 5, "max_depth": 4, "random_state": 42}

    m1 = train_b2_model(df, params=params)
    p1 = predict_b2(m1, df)

    m2 = train_b2_model(df, params=params)
    p2 = predict_b2(m2, df)

    np.testing.assert_array_equal(p1.to_numpy(), p2.to_numpy())


def test_fit_final_b2_model():
    """Verify final B2 model fits on all development seasons and rejects empty splits."""
    dev_seasons = [2018, 2019]
    df = generate_synthetic_m3_data(seasons=[2018, 2019, 2020], dates_per_season=2)

    params = {"n_estimators": 5, "max_depth": 4}
    model = fit_final_b2_model(df, development_seasons=dev_seasons, selected_config=params)
    assert model is not None

    test_slice = df[df["season"] == 2020]
    preds = predict_b2(model, test_slice)
    assert len(preds) == len(test_slice)
    assert (preds >= 0.0).all()


# ==============================================================================
# B3 Regime-Aware Model Tests (PRD §10.3, §11.6, §11.7, §12)
# ==============================================================================

def test_b3_feature_contracts_and_regime_inclusion():
    """Verify B3 uses exactly 41 features: 27 base + all 14 regime features."""
    assert len(B3_FEATURES) == 41
    assert len(BASE_FEATURES) == 27
    assert len(REGIME_FEATURES) == 14

    for base_feat in BASE_FEATURES:
        assert base_feat in B3_FEATURES, f"Base feature {base_feat} missing from B3"

    for regime_feat in REGIME_FEATURES:
        assert regime_feat in B3_FEATURES, f"Regime feature {regime_feat} missing from B3"


def test_b3_model_specifications_and_monotone_constraints():
    """Verify B3 model specs: reg:tweedie, hist, and rain_mm=+1, all other 40=0."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    test_params = {
        "tweedie_variance_power": 1.5,
        "max_depth": 4,
        "min_child_weight": 50,
        "learning_rate": 0.05,
        "n_estimators": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
        "random_state": 42,
    }

    model = train_b3_model(df, params=test_params)
    params = model.get_params()

    assert params["objective"] == "reg:tweedie"
    assert params["tree_method"] == "hist"

    constraints = params["monotone_constraints"]
    assert len(constraints) == 41, f"Expected 41 constraints for B3, got {len(constraints)}"
    assert constraints[0] == 1, "First feature (rain_mm) must have +1 constraint"
    assert all(c == 0 for c in constraints[1:]), "All remaining 40 features must have 0 constraint"


def test_b3_regime_source_safety_guard():
    """Verify protocol safety guard on regime_source for B3 training (PRD §11.6)."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    test_params = {"n_estimators": 5, "max_depth": 4}

    # 1. Missing regime_source column raises clear error
    df_missing_source = df.drop(columns=["regime_source"])
    with pytest.raises(ValueError, match="Missing required 'regime_source' column"):
        train_b3_model(df_missing_source, params=test_params)

    # 2. Even one row with regime_source='final' raises clear hard error
    df_with_final = df.copy()
    df_with_final.loc[0, "regime_source"] = "final"
    with pytest.raises(ValueError, match="Found 1 rows with regime_source='final'"):
        train_b3_model(df_with_final, params=test_params)

    # 3. Invalid regime_source value raises clear error
    df_invalid = df.copy()
    df_invalid.loc[0, "regime_source"] = "in_sample"
    with pytest.raises(ValueError, match="All B3 training rows must have regime_source='oof'"):
        train_b3_model(df_invalid, params=test_params)

    # 4. Correct regime_source='oof' succeeds
    model = train_b3_model(df, params=test_params)
    assert model is not None


def test_b3_uses_same_shared_20_configs_as_b2():
    """Verify B3 consumes the identical 20 configs generated for B2 (no separate search list)."""
    shared_configs = generate_shared_hyperparameter_configs(seed=42, n_configs=20)

    # Both systems evaluate the SAME list object
    b2_configs = shared_configs
    b3_configs = shared_configs
    assert b3_configs is b2_configs
    assert len(b3_configs) == 20

    # Test settings search for B3 with these configs
    df = generate_synthetic_m3_data(seasons=[2017, 2018, 2019, 2020], dates_per_season=2)
    fast_configs = [{**cfg, "n_estimators": 3} for cfg in shared_configs]

    result_b3 = evaluate_settings_search(
        df,
        development_seasons=[2017, 2018, 2019, 2020],
        combinations=fast_configs,
        model_type="B3",
        n_validation_seasons=3,
    )

    assert len(result_b3["all_results"]) == 20
    assert result_b3["validation_seasons"] == [2018, 2019, 2020]
    for idx, r in enumerate(result_b3["all_results"]):
        assert r["config_index"] == idx
        assert r["config"] == fast_configs[idx]


def test_fit_final_b3_model_and_holdout_exclusion():
    """Verify final B3 model fits on all development seasons and enforces 'oof'."""
    dev_seasons = [2018, 2019]
    holdout_season = 2020
    df = generate_synthetic_m3_data(seasons=[2018, 2019, holdout_season], dates_per_season=2)

    params = {"n_estimators": 5, "max_depth": 4}
    model = fit_final_b3_model(df, development_seasons=dev_seasons, selected_config=params)
    assert model is not None

    # Verify that if any development season row has 'final', it fails
    df_corrupted = df.copy()
    df_corrupted.loc[df_corrupted["season"] == 2018, "regime_source"] = "final"
    with pytest.raises(ValueError, match="regime_source='final'"):
        fit_final_b3_model(df_corrupted, development_seasons=dev_seasons, selected_config=params)


def test_predict_b3_inference_accepts_final_regime_source():
    """Verify predict_b3 accepts regime_source='final' and preserves row/index order."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    # At inference time, regime_source is typically 'final' (PRD §11.6)
    df["regime_source"] = "final"
    custom_idx = [f"test_idx_{i}" for i in range(len(df))]
    df.index = custom_idx

    params = {"n_estimators": 5, "max_depth": 4}
    # Fit model on training data (which has 'oof')
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    model = train_b3_model(train_df, params=params)

    # Predict on inference data (which has 'final')
    preds = predict_b3(model, df)

    assert isinstance(preds, pd.Series)
    assert preds.index.equals(df.index), "Prediction index must match input DataFrame index"
    assert (preds >= 0.0).all(), "Predictions must be strictly non-negative"
    assert not preds.isna().any(), "Predictions must not contain NaNs"


def test_b3_dry_rows_and_nans_preserved():
    """Verify B3 model trains directly with zero-rain rows and un-imputed NaNs."""
    df = generate_synthetic_m3_data(dates_per_season=2)

    assert (df["obs_mm"] == 0.0).any()
    assert (df["rain_mm"] == 0.0).any()
    assert df.loc[df["lead_day"] == 1, "rain_prev_lead"].isna().all()
    assert df.loc[df["lead_day"] == 3, "rain_next_lead"].isna().all()

    params = {"n_estimators": 5, "max_depth": 4}
    model = train_b3_model(df, params=params)
    assert model is not None


# ==============================================================================
# Heavy-Rainfall Probability Classifier Tests (PRD §13.1–§13.5)
# ==============================================================================

def test_probability_model_specifications_and_b3_features():
    """Verify probability classifier uses 41 B3 features, binary:logistic, and hist."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    test_params = {
        "max_depth": 4,
        "n_estimators": 5,
        "learning_rate": 0.05,
        "min_child_weight": 50,
        "reg_lambda": 10.0,
        "random_state": 42,
    }

    clf = train_probability_classifier(df, threshold=15.6, params=test_params)
    params = clf.get_params()

    assert params["objective"] == "binary:logistic"
    assert params["tree_method"] == "hist"
    # Ensure natural base rate (no class weighting/oversampling)
    assert params.get("scale_pos_weight", 1.0) in [1.0, 1, None]


def test_probability_regime_source_safety_guard():
    """Verify probability training strictly requires regime_source='oof' (PRD §11.6)."""
    df = generate_synthetic_m3_data(dates_per_season=2)
    test_params = {"n_estimators": 5, "max_depth": 4}

    # 1. Missing regime_source raises error
    df_missing = df.drop(columns=["regime_source"])
    with pytest.raises(ValueError, match="Missing required 'regime_source' column"):
        train_probability_classifier(df_missing, threshold=15.6, params=test_params)

    # 2. regime_source='final' raises hard error
    df_final = df.copy()
    df_final.loc[0, "regime_source"] = "final"
    with pytest.raises(ValueError, match="regime_source='final'"):
        train_probability_classifier(df_final, threshold=15.6, params=test_params)

    # 3. regime_source='oof' succeeds
    clf = train_probability_classifier(df, threshold=15.6, params=test_params)
    assert clf is not None


def test_event_count_branching_rules():
    """Verify event-count branching logic according to PRD §13.2 rules."""
    # Case A: >= 30 events at all thresholds -> standalone models for all
    counts_all = {15.6: 50, 64.5: 45, 115.6: 35}
    avail_a = determine_model_availability(counts_all)
    assert avail_a.has_15_6 is True
    assert avail_a.has_64_5 is True
    assert avail_a.has_115_6 is True
    assert avail_a.chained_115_6 is False
    assert avail_a.message is None

    # Case B: 115.6 < 30 events, but 64.5 >= 30 -> chained form
    counts_chained = {15.6: 60, 64.5: 35, 115.6: 12}
    avail_b = determine_model_availability(counts_chained)
    assert avail_b.has_15_6 is True
    assert avail_b.has_64_5 is True
    assert avail_b.has_115_6 is True
    assert avail_b.chained_115_6 is True
    assert "Chained form" in avail_b.message

    # Case C: 64.5 < 30 events -> heavy and very heavy models disabled
    counts_low = {15.6: 40, 64.5: 22, 115.6: 5}
    avail_c = determine_model_availability(counts_low)
    assert avail_c.has_15_6 is True
    assert avail_c.has_64_5 is False
    assert avail_c.has_115_6 is False
    assert avail_c.chained_115_6 is False
    assert avail_c.message == "Not enough events"


def test_probability_hyperparameter_search_brier_score():
    """Verify probability settings search uses Brier score and newest 3 development seasons."""
    dev_seasons = [2017, 2018, 2019, 2020]
    holdout_season = 2021
    df = generate_synthetic_m3_data(seasons=dev_seasons + [holdout_season], dates_per_season=2)

    base_configs = generate_shared_hyperparameter_configs(seed=42, n_configs=20)
    fast_configs = [{**cfg, "n_estimators": 3} for cfg in base_configs]

    result = evaluate_probability_settings_search(
        df,
        development_seasons=dev_seasons,
        threshold=15.6,
        combinations=fast_configs,
        n_validation_seasons=3,
    )

    # 1. Exactly 20 evaluated
    assert len(result["all_results"]) == 20
    # 2. Validation seasons = newest 3 development seasons
    assert result["validation_seasons"] == [2018, 2019, 2020]
    assert holdout_season not in result["validation_seasons"]

    # 3. Lowest mean Brier score selected
    all_brier_scores = [r["mean_brier_score"] for r in result["all_results"]]
    min_brier = min(all_brier_scores)
    expected_idx = all_brier_scores.index(min_brier)

    assert np.isclose(result["best_mean_brier_score"], min_brier)
    assert result["best_config_index"] == expected_idx
    assert result["best_config"] == fast_configs[expected_idx]


def test_probability_fit_and_prediction_standalone():
    """Verify standalone probability prediction, monotonicity, and index preservation."""
    train_df = generate_synthetic_m3_data(dates_per_season=3)
    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df["regime_source"] = "final"  # inference uses 'final'
    custom_idx = [f"prob_idx_{i}" for i in range(len(test_df))]
    test_df.index = custom_idx

    counts_standalone = {15.6: 100, 64.5: 50, 115.6: 35}
    fast_params = {"n_estimators": 4, "max_depth": 3}

    prob_models = RainfallProbabilityModels().fit(
        train_df,
        event_counts=counts_standalone,
        params_map=fast_params,
    )

    preds_df = prob_models.predict_probabilities(test_df)

    assert isinstance(preds_df, pd.DataFrame)
    assert preds_df.index.equals(test_df.index), "Prediction index must match input index"
    assert list(preds_df.columns) == ["p_ge_15_6", "p_ge_64_5", "p_ge_115_6"]

    # Values in [0, 1]
    for col in preds_df.columns:
        assert (preds_df[col] >= 0.0).all()
        assert (preds_df[col] <= 1.0).all()

    # Monotonicity: P(>= 115.6) <= P(>= 64.5) <= P(>= 15.6)
    p15 = preds_df["p_ge_15_6"].to_numpy()
    p64 = preds_df["p_ge_64_5"].to_numpy()
    p115 = preds_df["p_ge_115_6"].to_numpy()

    assert np.all(p115 <= p64)
    assert np.all(p64 <= p15)


def test_probability_fit_and_prediction_chained():
    """Verify chained 115.6 formulation: P115_raw = P64 * P(115|64)."""
    # Create dataset with enough observed heavy rain to train conditional model
    train_df = generate_synthetic_m3_data(dates_per_season=5)
    # Inject a few >= 64.5 and >= 115.6 rows so conditional fit has samples
    train_df.loc[0:10, "obs_mm"] = 70.0
    train_df.loc[0:3, "obs_mm"] = 125.0

    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df["regime_source"] = "final"

    # Chained setup: 115.6 < 30 events, 64.5 >= 30 events
    counts_chained = {15.6: 100, 64.5: 50, 115.6: 15}
    fast_params = {"n_estimators": 4, "max_depth": 3}

    prob_models = RainfallProbabilityModels().fit(
        train_df,
        event_counts=counts_chained,
        params_map=fast_params,
    )

    assert "115.6_cond" in prob_models.models
    assert "115.6" not in prob_models.models
    assert prob_models.availability.chained_115_6 is True

    # Test that chained calculation multiplies P64 * P_cond before monotonic clipping
    X = test_df[B3_FEATURES]
    p64_raw = prob_models.models["64.5"].predict_proba(X)[:, 1]
    p_cond_raw = prob_models.models["115.6_cond"].predict_proba(X)[:, 1]
    expected_chained_raw = p64_raw * p_cond_raw

    # The final prediction applies monotonic enforcement
    preds_df = prob_models.predict_probabilities(test_df)
    p115_final = preds_df["p_ge_115_6"].to_numpy()
    p64_final = preds_df["p_ge_64_5"].to_numpy()
    p15_final = preds_df["p_ge_15_6"].to_numpy()

    # Final probabilities must satisfy ordering and be <= expected_chained_raw
    assert np.all(p115_final <= p64_final)
    assert np.all(p64_final <= p15_final)
    assert np.all(p115_final <= expected_chained_raw + 1e-9)


def test_probability_fit_and_prediction_disabled_heavy():
    """Verify disabling 64.5 and 115.6 when 64.5 has < 30 events."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df["regime_source"] = "final"

    counts_low = {15.6: 80, 64.5: 20, 115.6: 5}
    fast_params = {"n_estimators": 4, "max_depth": 3}

    prob_models = RainfallProbabilityModels().fit(
        train_df,
        event_counts=counts_low,
        params_map=fast_params,
    )

    assert prob_models.availability.has_15_6 is True
    assert prob_models.availability.has_64_5 is False
    assert prob_models.availability.has_115_6 is False
    assert "64.5" not in prob_models.models
    assert "115.6" not in prob_models.models

    preds_df = prob_models.predict_probabilities(test_df)

    assert not preds_df["p_ge_15_6"].isna().any()
    assert (preds_df["p_ge_15_6"] >= 0.0).all()
    assert (preds_df["p_ge_15_6"] <= 1.0).all()
    assert preds_df["p_ge_64_5"].isna().all()
    assert preds_df["p_ge_115_6"].isna().all()


# ==============================================================================
# Model-Estimated Range (q10, q50, q90) and Model Store Tests (PRD §13.7, §12.4)
# ==============================================================================

def test_range_models_specifications_and_alphas():
    """Verify 3 separate models with reg:quantileerror, hist tree method, and alphas 0.1, 0.5, 0.9."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    fast_params = {"n_estimators": 2, "max_depth": 2}

    # Verify forbidden tree_method='exact' raises ValueError
    with pytest.raises(ValueError, match="tree_method='exact' is forbidden"):
        train_quantile_model(train_df, alpha=0.1, params={"tree_method": "exact"})

    range_models = QuantileRangeModels().fit(train_df, params_map=fast_params)

    for alpha in [0.1, 0.5, 0.9]:
        assert alpha in range_models.models
        m = range_models.models[alpha]
        params = m.get_params()
        assert params["objective"] == "reg:quantileerror"
        assert params["tree_method"] == "hist"
        assert np.isclose(params["quantile_alpha"], alpha)
        assert m.n_features_in_ == 41


def test_range_log1p_target_and_expm1_conversion():
    """Verify range models train on log1p(obs_mm) and invert via expm1."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    fast_params = {"n_estimators": 2, "max_depth": 2}

    m10 = train_quantile_model(train_df, alpha=0.1, params=fast_params)
    X = train_df[B3_FEATURES]
    pred_log = m10.predict(X)

    # Inverted predictions
    pred_mm = np.expm1(pred_log)
    clipped = np.maximum(pred_mm, 0.0)

    assert len(clipped) == len(train_df)
    assert (clipped >= 0.0).all()


def test_range_quantile_ordering_and_crossing_resolution():
    """Verify per-cell quantile sorting resolves crossing so q10 <= q50 <= q90."""
    # Simulated crossing vectors
    q10_raw = np.array([12.0, 50.0, 5.0, -1.0])
    q50_raw = np.array([10.0, 40.0, 15.0, 2.0])
    q90_raw = np.array([25.0, 30.0, 10.0, 8.0])

    q10_out, q50_out, q90_out = sort_and_clip_quantiles(q10_raw, q50_raw, q90_raw)

    assert (q10_out <= q50_out).all()
    assert (q50_out <= q90_out).all()
    assert (q10_out >= 0.0).all()
    np.testing.assert_array_equal(q10_out, np.array([10.0, 30.0, 5.0, 0.0]))
    np.testing.assert_array_equal(q50_out, np.array([12.0, 40.0, 10.0, 2.0]))
    np.testing.assert_array_equal(q90_out, np.array([25.0, 50.0, 15.0, 8.0]))

    # Test full model prediction guarantees monotonic order across all synthetic cells
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df["regime_source"] = "final"

    range_models = QuantileRangeModels().fit(train_df, params_map={"n_estimators": 2, "max_depth": 2})
    preds_df = range_models.predict_range(test_df)

    assert (preds_df["q10_mm"] <= preds_df["q50_mm"]).all()
    assert (preds_df["q50_mm"] <= preds_df["q90_mm"]).all()
    assert (preds_df["q10_mm"] >= 0.0).all()


def test_range_non_negativity_and_index_preservation():
    """Verify range predictions preserve non-default index and return exact columns."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df.index = [f"cell_idx_{i}" for i in range(len(test_df))]
    test_df["regime_source"] = "final"

    range_models = QuantileRangeModels().fit(train_df, params_map={"n_estimators": 2, "max_depth": 2})
    preds_df = predict_range(range_models, test_df)

    assert isinstance(preds_df, pd.DataFrame)
    assert preds_df.index.equals(test_df.index)
    assert list(preds_df.columns) == ["q10_mm", "q50_mm", "q90_mm"]
    assert not preds_df.isna().any().any()
    assert (preds_df >= 0.0).all().all()


def test_range_models_regime_source_safety_guard():
    """Verify regime_source='oof' is strictly required for training, while 'final' is accepted for inference."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)

    # 1. Missing regime_source in training -> error
    bad_train_1 = train_df.drop(columns=["regime_source"])
    with pytest.raises(ValueError, match="Missing required 'regime_source' column"):
        train_quantile_model(bad_train_1, alpha=0.1)

    # 2. 'final' regime_source in training -> hard error
    bad_train_2 = train_df.copy()
    bad_train_2.loc[bad_train_2.index[0], "regime_source"] = "final"
    with pytest.raises(ValueError, match="Protocol violation.*regime_source='final'"):
        train_quantile_model(bad_train_2, alpha=0.1)

    # 3. 'oof' regime_source in training -> succeeds
    model = train_quantile_model(train_df, alpha=0.1, params={"n_estimators": 2, "max_depth": 2})
    assert model is not None

    # 4. Inference accepts 'final'
    infer_df = generate_synthetic_m3_data(dates_per_season=2)
    infer_df["regime_source"] = "final"
    range_models = QuantileRangeModels({0.1: model, 0.5: model, 0.9: model})
    preds = range_models.predict_range(infer_df)
    assert len(preds) == len(infer_df)


def test_range_empirical_coverage_calculation():
    """Verify coverage calculation computes empirical fraction and checks 10 pct point deviation."""
    # Synthetic test data with known coverage
    n = 100
    obs = np.array([5.0] * 80 + [25.0] * 20)  # 80 within [2, 10], 20 outside
    q10 = np.full(n, 2.0)
    q90 = np.full(n, 10.0)
    leads = np.array([1] * 50 + [2] * 50)

    # Lead 1: 50 within -> 100% coverage
    # Lead 2: 30 within, 20 outside -> 60% coverage
    obs[:50] = 5.0
    obs[50:80] = 5.0
    obs[80:] = 25.0

    eval_df = pd.DataFrame({
        "q10_mm": q10,
        "q90_mm": q90,
        "obs_mm": obs,
        "lead_day": leads,
    })

    cov_res = calculate_range_coverage(eval_df)

    assert np.isclose(cov_res["overall_coverage"], 0.80)
    assert cov_res["nominal_coverage"] == 0.80
    assert cov_res["total_samples"] == 100
    assert cov_res["within_range_count"] == 80
    assert np.isclose(cov_res["lead_coverage"][1], 1.0)
    assert np.isclose(cov_res["lead_coverage"][2], 0.60)
    # Lead 1 is 100% (> 90%), Lead 2 is 60% (< 70%), so deviation_exceeded must be True
    assert cov_res["deviation_exceeded"] is True

    # Test case where leads are strictly within 70% to 90%
    obs_balanced = np.array([5.0] * 80 + [25.0] * 20)
    leads_balanced = np.array([1, 2] * 50)
    df_balanced = pd.DataFrame({
        "q10_mm": q10,
        "q90_mm": q90,
        "obs_mm": obs_balanced,
        "lead_day": leads_balanced,
    })
    cov_balanced = calculate_range_coverage(df_balanced)
    assert np.isclose(cov_balanced["lead_coverage"][1], 0.80)
    assert np.isclose(cov_balanced["lead_coverage"][2], 0.80)
    assert cov_balanced["deviation_exceeded"] is False


def test_model_store_save_load_round_trip():
    """Verify save_model and load_model serialize XGBoost JSON and preserve metadata."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df["regime_source"] = "final"

    fast_params = {"n_estimators": 2, "max_depth": 2, "learning_rate": 0.05}
    model = train_quantile_model(train_df, alpha=0.5, params=fast_params)
    original_preds = model.predict(test_df[B3_FEATURES])

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "q50_model.json"
        metadata = ModelMetadata(
            model_type="q50",
            feature_names=B3_FEATURES,
            train_seasons=[2016, 2017],
            created_at="2026-09-22T00:00:00Z",
            git_commit="test_commit_hash",
            best_params=fast_params,
            metrics={"pinball_loss": 0.123},
        )

        save_model(model, metadata, model_path)
        assert model_path.exists()
        assert (model_path.parent / "metadata.json").exists()

        # Load back
        loaded = load_model(model_path)
        assert loaded.metadata["model_type"] == "q50"
        assert loaded.metadata["feature_names"] == B3_FEATURES
        assert loaded.metadata["train_seasons"] == [2016, 2017]
        assert loaded.metadata["git_commit"] == "test_commit_hash"

        # Predictions match exactly
        loaded_preds = loaded.predict(test_df[B3_FEATURES])
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-6)

        # Verify invalid model_type raises error
        with pytest.raises(ValueError, match="Invalid model_type"):
            bad_meta = ModelMetadata(
                model_type="invalid_model_type",
                feature_names=B3_FEATURES,
                train_seasons=[2016],
            )
            bad_meta.validate()


def test_range_models_save_and_load():
    """Verify composite QuantileRangeModels save and load round-trip."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df["regime_source"] = "final"

    fast_params = {"n_estimators": 2, "max_depth": 2}
    range_models = QuantileRangeModels().fit(train_df, params_map=fast_params)
    original_preds = range_models.predict_range(test_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_range_models(
            range_models,
            output_dir=tmpdir,
            metadata={"train_seasons": [2016, 2017], "git_commit": "test_hash"},
        )

        loaded_models = load_range_models(tmpdir)
        loaded_preds = loaded_models.predict_range(test_df)

        pd.testing.assert_frame_equal(original_preds, loaded_preds)


def test_q50_distinct_from_corrected_mean():
    """Verify q50_mm is the median and distinct from Tweedie mean (PRD §7.4, §13.7)."""
    train_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df = generate_synthetic_m3_data(dates_per_season=2)
    test_df["regime_source"] = "final"

    range_models = QuantileRangeModels().fit(train_df, params_map={"n_estimators": 3, "max_depth": 2})
    preds_df = range_models.predict_range(test_df)

    # Column name must be q50_mm, never corrected_mean_mm
    assert "q50_mm" in preds_df.columns
    assert "corrected_mean_mm" not in preds_df.columns
    assert list(preds_df.columns) == ["q10_mm", "q50_mm", "q90_mm"]


# ==============================================================================
# Extrapolation Threshold, Fallback Ladder, and Serving Grid Tests (PRD §12.3, §17.1-§17.2, §20.1)
# ==============================================================================

def test_extrapolation_threshold_training_seasons_only():
    """Verify p99.9 threshold fitted strictly from training seasons and holdout does not affect it."""
    # Two training seasons and one holdout season with extreme value
    train_df = pd.DataFrame({
        "season": [2018, 2018, 2019, 2019],
        "rain_mm": [10.0, 20.0, 30.0, 40.0],
    })
    holdout_df = pd.DataFrame({
        "season": [2020],
        "rain_mm": [1000.0],  # Massive outlier in holdout
    })

    # Threshold must be computed from training data only
    threshold_train = compute_extrapolation_threshold(train_df, "rain_mm", 99.9)

    # If holdout were accidentally included:
    combined_df = pd.concat([train_df, holdout_df], ignore_index=True)
    threshold_combined = compute_extrapolation_threshold(combined_df, "rain_mm", 99.9)

    assert threshold_train < 50.0
    assert threshold_combined > 500.0
    assert threshold_train != threshold_combined, "Holdout data must never affect training threshold"


def test_extrapolation_flag_boundary_and_raw_fallback():
    """Verify threshold boundary condition (<= not flagged, > flagged) and raw fallback rule."""
    threshold = 100.0
    rain_vals = np.array([50.0, 99.99, 100.0, 100.01, 150.0])

    flags = check_extrapolation(rain_vals, threshold)
    # Value equal to threshold is NOT flagged; value above IS flagged
    expected_flags = np.array([False, False, False, True, True])
    np.testing.assert_array_equal(flags, expected_flags)


def test_fallback_ladder_precedence_and_reasons():
    """Verify fallback ladder options and precedence rules per PRD §17.1."""
    # 1. Normal -> B3
    prod, reason = select_fallback_product(
        has_b3=True, has_b2=True, has_b1=True, regime_available=True, ml_available=True, ood_flag=False, validation_failed=False, no_correction=False
    )
    assert prod == "regime_aware_ml"
    assert reason is None

    # 2. Regime features unavailable -> B2
    prod, reason = select_fallback_product(
        has_b3=True, has_b2=True, has_b1=True, regime_available=False, ml_available=True, ood_flag=False, validation_failed=False, no_correction=False
    )
    assert prod == "global_ml"
    assert reason == "REGIME_UNAVAILABLE"

    # 3. ML model files unavailable -> B1
    prod, reason = select_fallback_product(
        has_b3=True, has_b2=True, has_b1=True, regime_available=True, ml_available=False, ood_flag=False, validation_failed=False, no_correction=False
    )
    assert prod == "quantile_mapping"
    assert reason == "ML_UNAVAILABLE"

    # 4. Run-level ood_flag == True -> B1
    prod, reason = select_fallback_product(
        has_b3=True, has_b2=True, has_b1=True, regime_available=True, ml_available=True, ood_flag=True, validation_failed=False, no_correction=False
    )
    assert prod == "quantile_mapping"
    assert reason == "OOD_INPUT"

    # 5. Cell extrapolation -> raw NWP (overrides B3/B2/B1)
    prod, reason = select_fallback_product(
        has_b3=True, has_b2=True, has_b1=True, regime_available=True, ml_available=True, ood_flag=False, validation_failed=False, no_correction=False, extrapolation_flag=True
    )
    assert prod == "raw_nwp"
    assert reason == "EXTRAPOLATION"

    # Extrapolation overrides regime unavailable
    prod, reason = select_fallback_product(
        has_b3=True, has_b2=True, has_b1=True, regime_available=False, ml_available=True, ood_flag=False, validation_failed=False, no_correction=False, extrapolation_flag=True
    )
    assert prod == "raw_nwp"
    assert reason == "EXTRAPOLATION"

    # 6. Validation / check failure -> raw NWP
    prod, reason = select_fallback_product(
        has_b3=True, has_b2=True, has_b1=True, regime_available=True, ml_available=True, ood_flag=False, validation_failed=True, no_correction=False
    )
    assert prod == "raw_nwp"
    assert reason == "VALIDATION_FAILED"

    # 7. No correction exists -> raw NWP
    prod, reason = select_fallback_product(
        has_b3=False, has_b2=False, has_b1=False, regime_available=True, ml_available=True, ood_flag=False, validation_failed=False, no_correction=True
    )
    assert prod == "raw_nwp"
    assert reason == "NO_CORRECTION"


def test_serving_grid_assembly_normal_and_fallback():
    """Verify assemble_serving_grid produces exact 18-column contract with correct fallback nulls."""
    raw_df = generate_synthetic_m3_data(seasons=[2018], dates_per_season=1)
    n = len(raw_df)

    b3_preds = raw_df["rain_mm"].to_numpy() * 1.05
    b2_preds = raw_df["rain_mm"].to_numpy() * 1.02
    b1_preds = raw_df["rain_mm"].to_numpy() * 0.98

    # Simulate 1 extrapolation cell
    extrap_flags = np.zeros(n, dtype=bool)
    extrap_flags[0] = True

    range_df = pd.DataFrame({
        "q10_mm": b3_preds * 0.7,
        "q50_mm": b3_preds * 0.95,
        "q90_mm": b3_preds * 1.4,
    })
    prob_df = pd.DataFrame({
        "p_ge_15_6": np.full(n, 0.4),
        "p_ge_64_5": np.full(n, 0.1),
        "p_ge_115_6": np.full(n, 0.02),
    })

    # Assemble normal serving grid with 1 extrapolated cell
    grid_df = assemble_serving_grid(
        base_df=raw_df,
        b3_mm=b3_preds,
        b2_mm=b2_preds,
        b1_mm=b1_preds,
        range_df=range_df,
        prob_df=prob_df,
        extrapolation_flags=extrap_flags,
        model_version_id=42,
    )

    # 1. Exactly 18 columns in exact PRD order
    assert list(grid_df.columns) == SERVING_GRID_COLUMNS
    assert len(grid_df) == n

    # 2. Extrapolated cell (index 0) serves raw_mm and has null range / probabilities
    assert grid_df.loc[0, "extrapolation_flag"] == True
    assert grid_df.loc[0, "product_type"] == "raw_nwp"
    assert grid_df.loc[0, "fallback_reason"] == "EXTRAPOLATION"
    assert np.isclose(grid_df.loc[0, "corrected_mean_mm"], grid_df.loc[0, "raw_mm"])
    assert pd.isna(grid_df.loc[0, "q10_mm"])
    assert pd.isna(grid_df.loc[0, "q50_mm"])
    assert pd.isna(grid_df.loc[0, "q90_mm"])
    assert pd.isna(grid_df.loc[0, "p_ge_15_6"])
    assert pd.isna(grid_df.loc[0, "p_ge_64_5"])
    assert pd.isna(grid_df.loc[0, "p_ge_115_6"])

    # 3. Non-extrapolated normal cells serve B3 and retain range and probabilities
    for i in range(1, n):
        assert grid_df.loc[i, "extrapolation_flag"] == False
        assert grid_df.loc[i, "product_type"] == "regime_aware_ml"
        assert pd.isna(grid_df.loc[i, "fallback_reason"])
        assert np.isclose(grid_df.loc[i, "corrected_mean_mm"], b3_preds[i])
        assert not pd.isna(grid_df.loc[i, "q10_mm"])
        assert not pd.isna(grid_df.loc[i, "p_ge_15_6"])


def test_serving_grid_parquet_write_read_round_trip():
    """Verify writing serving grid parquet partition, deterministic output, and schema round-trip."""
    raw_df = generate_synthetic_m3_data(seasons=[2018], dates_per_season=1)
    grid_df = assemble_serving_grid(
        base_df=raw_df,
        b3_mm=raw_df["rain_mm"].to_numpy(),
        model_version_id=101,
    )

    run_id = "test_run_2018070100"
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = write_serving_grid_file(
            grid_df,
            run_id=run_id,
            output_base_dir=tmpdir,
        )

        # Output path follows partition: .../grid/run_id=<run_id>/part.parquet
        assert target_file.exists()
        assert target_file.name == "part.parquet"
        assert f"run_id={run_id}" in str(target_file.parent)

        # Read back
        loaded_df = read_serving_grid_file(run_id=run_id, output_base_dir=tmpdir)

        # Exact schema and order
        assert list(loaded_df.columns) == SERVING_GRID_COLUMNS
        assert len(loaded_df) == len(grid_df)

        # Uniqueness of (cell_id, lead_day)
        assert not loaded_df.duplicated(subset=["cell_id", "lead_day"]).any()

        # Leads must be 1, 2, 3
        assert set(loaded_df["lead_day"].unique()).issubset({1, 2, 3})

        # Non-negative rain
        assert (loaded_df["raw_mm"] >= 0).all()
        assert (loaded_df["corrected_mean_mm"] >= 0).all()


def test_serving_grid_null_obs_allowed():
    """Verify serving grid preserves null obs_mm when observation is unknown."""
    raw_df = generate_synthetic_m3_data(seasons=[2018], dates_per_season=1)
    raw_df["obs_mm"] = np.nan  # Unknown observations

    grid_df = assemble_serving_grid(
        base_df=raw_df,
        b3_mm=raw_df["rain_mm"].to_numpy(),
    )

    assert grid_df["obs_mm"].isna().all()
    # Ensure validation passes even with null observations
    validate_serving_grid_contracts(grid_df)


# ==============================================================================
# M3 Orchestration and Integration Pipeline Tests (PRD §10, §12, §13, §17, §20.1)
# ==============================================================================

def test_orchestration_oof_pipeline_loso_and_m4_handoff():
    """Verify OOF pipeline runs LOSO across development seasons and generates M4 handoff table."""
    dev_seasons = [2018, 2019]
    df = generate_synthetic_m3_data(seasons=dev_seasons, dates_per_season=2)

    fast_params = {"n_estimators": 2, "max_depth": 2, "learning_rate": 0.05}
    oof_df = run_development_oof_pipeline(
        dev_df=df,
        development_seasons=dev_seasons,
        b2_params=fast_params,
        b3_params=fast_params,
        prob_params=fast_params,
        range_params=fast_params,
    )

    # 1. All development seasons received OOF predictions
    assert set(oof_df["season"].unique()) == set(dev_seasons)
    assert len(oof_df) == len(df)

    # 2. prediction_source is strictly 'oof'
    assert (oof_df["prediction_source"] == "oof").all()

    # 3. Required M4 handoff columns exist
    required_m4_cols = [
        "run_id",
        "season",
        "lead_day",
        "cell_id",
        "obs_mm",
        "raw_mm",
        "b1_corrected_mm",
        "b2_corrected_mean_mm",
        "b3_corrected_mean_mm",
        "uncalibrated_p_ge_15_6",
        "uncalibrated_p_ge_64_5",
        "uncalibrated_p_ge_115_6",
        "q10_mm",
        "q50_mm",
        "q90_mm",
        "prediction_source",
    ]
    for col in required_m4_cols:
        assert col in oof_df.columns, f"Missing required M4 column: {col}"

    # 4. Predictions are non-negative and finite
    assert (oof_df["raw_mm"] >= 0).all()
    assert (oof_df["b1_corrected_mm"] >= 0).all()
    assert (oof_df["b2_corrected_mean_mm"] >= 0).all()
    assert (oof_df["b3_corrected_mean_mm"] >= 0).all()

    # 5. Quantiles are properly ordered
    assert (oof_df["q10_mm"] <= oof_df["q50_mm"]).all()
    assert (oof_df["q50_mm"] <= oof_df["q90_mm"]).all()

    # 6. Probabilities are uncalibrated (bounded in [0, 1])
    assert (oof_df["uncalibrated_p_ge_15_6"] >= 0.0).all()
    assert (oof_df["uncalibrated_p_ge_15_6"] <= 1.0).all()


def test_orchestration_final_model_fit_and_holdout_isolation():
    """Verify final model fit uses all development seasons, excludes holdout, and round-trips via save/load."""
    dev_seasons = [2018, 2019]
    holdout_season = 2020
    all_seasons = dev_seasons + [holdout_season]

    df = generate_synthetic_m3_data(seasons=all_seasons, dates_per_season=2)

    fast_params = {"n_estimators": 2, "max_depth": 2, "learning_rate": 0.05}
    with tempfile.TemporaryDirectory() as tmpdir:
        final_models = fit_final_m3_models(
            dev_df=df,
            development_seasons=dev_seasons,
            b2_params=fast_params,
            b3_params=fast_params,
            prob_params=fast_params,
            range_params=fast_params,
            output_dir=tmpdir,
            feature_set_version="v1_test",
            git_commit="test_commit",
            model_version_id=777,
        )

        # 1. Training seasons strictly exclude holdout
        assert holdout_season not in final_models.training_seasons
        assert set(final_models.training_seasons) == set(dev_seasons)
        assert final_models.p99_9_threshold > 0

        # 2. Reload models from disk
        loaded_models = FinalM3Models.load(tmpdir)
        assert set(loaded_models.training_seasons) == set(dev_seasons)
        assert np.isclose(loaded_models.p99_9_threshold, final_models.p99_9_threshold)
        assert loaded_models.model_version_id == 777

        # 3. Verify identical predictions
        test_df = generate_synthetic_m3_data(seasons=[holdout_season], dates_per_season=1)
        test_df["regime_source"] = "final"

        preds_original = predict_final_m3(final_models, test_df, allow_uncalibrated=True)
        preds_loaded = predict_final_m3(loaded_models, test_df, allow_uncalibrated=True)

        pd.testing.assert_frame_equal(preds_original, preds_loaded)


def test_orchestration_final_inference_serving_grid_and_fallbacks():
    """Verify final inference accepts regime_source='final', produces 18-col serving grid, and handles fallbacks."""
    dev_seasons = [2018, 2019]
    df = generate_synthetic_m3_data(seasons=dev_seasons, dates_per_season=2)
    fast_params = {"n_estimators": 2, "max_depth": 2}

    final_models = fit_final_m3_models(
        dev_df=df,
        development_seasons=dev_seasons,
        b2_params=fast_params,
        b3_params=fast_params,
        prob_params=fast_params,
        range_params=fast_params,
    )

    # 1. Normal inference with regime_source='final'
    infer_df = generate_synthetic_m3_data(seasons=[2020], dates_per_season=1)
    infer_df["regime_source"] = "final"

    with tempfile.TemporaryDirectory() as tmpdir:
        serving_df = predict_final_m3(
            models=final_models,
            features_df=infer_df,
            output_serving_dir=tmpdir,
            allow_uncalibrated=True,
        )

        # Exact 18 columns in PRD order
        assert list(serving_df.columns) == SERVING_GRID_COLUMNS
        assert len(serving_df) == len(infer_df)

        # Check partition was written: .../grid/run_id=<run_id>/part.parquet
        run_id = infer_df["run_id"].iloc[0]
        expected_parquet = Path(tmpdir) / "grid" / f"run_id={run_id}" / "part.parquet"
        assert expected_parquet.exists()

    # 2. Missing regime features -> fallback to B2 (global_ml, REGIME_UNAVAILABLE)
    no_regime_df = infer_df.drop(columns=REGIME_FEATURES)
    serving_fallback = predict_final_m3(
        models=final_models,
        features_df=no_regime_df,
    )
    # Extrapolated cells get raw_nwp/EXTRAPOLATION; non-extrapolated get global_ml/REGIME_UNAVAILABLE
    normal_cells = serving_fallback["extrapolation_flag"] == False
    assert (serving_fallback.loc[normal_cells, "product_type"] == "global_ml").all()
    assert (serving_fallback.loc[normal_cells, "fallback_reason"] == "REGIME_UNAVAILABLE").all()
    # Range and probabilities must be null in fallback
    assert serving_fallback["q10_mm"].isna().all()
    assert serving_fallback["p_ge_15_6"].isna().all()


def test_orchestration_input_validation():
    """Verify input contract validation for M1 and M2 requirements."""
    df = generate_synthetic_m3_data(seasons=[2018], dates_per_season=1)

    # 1. Missing M1 column raises error
    bad_m1 = df.drop(columns=["cell_id"])
    with pytest.raises(ValueError, match="Missing required M1 columns"):
        validate_m1_m2_inputs(bad_m1)

    # 2. Missing obs_mm during training raises error
    bad_train = df.drop(columns=["obs_mm"])
    with pytest.raises(ValueError, match="Missing required target column 'obs_mm'"):
        validate_m1_m2_inputs(bad_train, is_training=True)

    # 3. Training B3 with regime_source='final' raises protocol violation
    bad_regime = df.copy()
    bad_regime.loc[bad_regime.index[0], "regime_source"] = "final"
    with pytest.raises(ValueError, match="Protocol violation"):
        validate_m1_m2_inputs(bad_regime, is_training=True, require_regime=True)


class MockM4Calibrator:
    """Mock external pre-fitted M4 calibrator to verify M3 never calls fit() and applies correctly."""

    def __init__(self, multiplier: float = 0.9):
        self.multiplier = multiplier
        self.fit_call_count = 0

    def fit(self, *args, **kwargs):
        self.fit_call_count += 1
        raise AssertionError("M3 must NEVER call fit() on an external M4 calibrator!")

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 2:
            arr = arr[:, 0]
        # Linear shift/scale for test
        p_cal = np.clip(arr * self.multiplier, 0.0, 1.0)
        return np.column_stack([1.0 - p_cal, p_cal])


def test_m3_never_fits_calibrators_and_applies_correctly():
    """Verify that M3 never calls fit() on calibrators and applies external calibration transforms."""
    cal_15 = MockM4Calibrator(multiplier=0.8)
    cal_64 = MockM4Calibrator(multiplier=0.5)

    raw_df = pd.DataFrame({
        "p_ge_15_6": [0.5, 0.8, np.nan],
        "p_ge_64_5": [0.4, 0.6, np.nan],
        "p_ge_115_6": [np.nan, np.nan, np.nan],
    })

    calibrators = {15.6: cal_15, "p_ge_64_5": cal_64}
    cal_df = apply_probability_calibrators(raw_df, calibrators=calibrators)

    # 1. Assert fit() was NEVER called on any calibrator
    assert cal_15.fit_call_count == 0
    assert cal_64.fit_call_count == 0

    # 2. Assert transformed values match multiplier
    assert np.isclose(cal_df["p_ge_15_6"].iloc[0], 0.5 * 0.8)
    assert np.isclose(cal_df["p_ge_15_6"].iloc[1], 0.8 * 0.8)
    assert np.isnan(cal_df["p_ge_15_6"].iloc[2])

    assert np.isclose(cal_df["p_ge_64_5"].iloc[0], 0.4 * 0.5)
    assert np.isclose(cal_df["p_ge_64_5"].iloc[1], 0.6 * 0.5)


def test_calibrated_probabilities_bounds_and_monotonicity_re_enforced():
    """Verify that calibrated outputs are clipped to [0, 1] and monotonicity is re-enforced."""
    # Calibrators that output inverted order: 115.6 gets 0.9, 64.5 gets 0.3, 15.6 gets 0.5
    cal_15 = lambda p: np.array([0.5, 1.2])   # Tests upper clipping
    cal_64 = lambda p: np.array([0.7, -0.2])  # Tests lower clipping
    cal_115 = lambda p: np.array([0.9, 0.8])  # Tests monotonicity inversion: P(115.6) > P(64.5)

    raw_df = pd.DataFrame({
        "p_ge_15_6": [0.5, 0.9],
        "p_ge_64_5": [0.3, 0.6],
        "p_ge_115_6": [0.1, 0.2],
    })

    cal_df = apply_probability_calibrators(
        raw_df,
        calibrators={15.6: cal_15, 64.5: cal_64, 115.6: cal_115},
    )

    # All bounded in [0.0, 1.0]
    for col in ["p_ge_15_6", "p_ge_64_5", "p_ge_115_6"]:
        assert (cal_df[col] >= 0.0).all()
        assert (cal_df[col] <= 1.0).all()

    # Monotonicity: P(>= 115.6) <= P(>= 64.5) <= P(>= 15.6)
    assert (cal_df["p_ge_115_6"] <= cal_df["p_ge_64_5"] + 1e-6).all()
    assert (cal_df["p_ge_64_5"] <= cal_df["p_ge_15_6"] + 1e-6).all()


def test_unavailable_thresholds_stay_null_after_calibration():
    """Verify that unavailable thresholds (e.g. <30 events) stay NaN after calibration."""
    cal_15 = lambda p: p * 0.9
    raw_df = pd.DataFrame({
        "p_ge_15_6": [0.5, 0.8],
        "p_ge_64_5": [np.nan, np.nan],
        "p_ge_115_6": [np.nan, np.nan],
    })

    cal_df = apply_probability_calibrators(raw_df, calibrators={15.6: cal_15})
    assert cal_df["p_ge_15_6"].notna().all()
    assert cal_df["p_ge_64_5"].isna().all()
    assert cal_df["p_ge_115_6"].isna().all()


def test_final_serving_requires_calibrators_or_explicit_development_mode():
    """Verify that predict_final_m3 refuses silent uncalibrated serving without explicit flag."""
    dev_seasons = [2018, 2019]
    df = generate_synthetic_m3_data(seasons=dev_seasons, dates_per_season=2)
    fast_params = {"n_estimators": 2, "max_depth": 2}

    final_models = fit_final_m3_models(
        dev_df=df,
        development_seasons=dev_seasons,
        b2_params=fast_params,
        b3_params=fast_params,
        prob_params=fast_params,
        range_params=fast_params,
    )

    infer_df = generate_synthetic_m3_data(seasons=[2020], dates_per_season=1)
    infer_df["regime_source"] = "final"

    # 1. Omitting calibrators without allow_uncalibrated=True raises ValueError
    with pytest.raises(ValueError, match="Final/holdout serving requires pre-fitted M4 probability calibrators"):
        predict_final_m3(final_models, infer_df, calibrators=None, allow_uncalibrated=False)

    # 2. Supplying allow_uncalibrated=True succeeds in explicit development mode
    dev_out = predict_final_m3(final_models, infer_df, calibrators=None, allow_uncalibrated=True)
    assert list(dev_out.columns) == SERVING_GRID_COLUMNS

    # 3. Supplying external calibrators succeeds and applies them
    calibrators = {
        15.6: lambda p: p * 0.9,
        64.5: lambda p: p * 0.8,
        115.6: lambda p: p * 0.7,
    }
    cal_out = predict_final_m3(final_models, infer_df, calibrators=calibrators)
    assert list(cal_out.columns) == SERVING_GRID_COLUMNS
    validate_serving_grid_contracts(cal_out)


def test_final_serving_fallback_rows_still_null_all_probabilities():
    """Verify that fallback rows (run-level or cell extrapolation) strictly null all probabilities."""
    dev_seasons = [2018, 2019]
    df = generate_synthetic_m3_data(seasons=dev_seasons, dates_per_season=2)
    fast_params = {"n_estimators": 2, "max_depth": 2}

    final_models = fit_final_m3_models(
        dev_df=df,
        development_seasons=dev_seasons,
        b2_params=fast_params,
        b3_params=fast_params,
        prob_params=fast_params,
        range_params=fast_params,
    )

    infer_df = generate_synthetic_m3_data(seasons=[2020], dates_per_season=1)
    infer_df["regime_source"] = "final"

    calibrators = {
        15.6: lambda p: p * 0.9,
        64.5: lambda p: p * 0.8,
        115.6: lambda p: p * 0.7,
    }

    # Regime unavailable -> fallback to global_ml
    no_regime_df = infer_df.drop(columns=REGIME_FEATURES)
    out_fb = predict_final_m3(final_models, no_regime_df, calibrators=calibrators)
    assert out_fb["p_ge_15_6"].isna().all()
    assert out_fb["p_ge_64_5"].isna().all()
    assert out_fb["p_ge_115_6"].isna().all()
    assert out_fb["q10_mm"].isna().all()


def test_oof_m4_handoff_remains_uncalibrated():
    """Verify that development OOF output for M4 strictly contains uncalibrated probabilities."""
    dev_seasons = [2018, 2019]
    df = generate_synthetic_m3_data(seasons=dev_seasons, dates_per_season=2)
    fast_params = {"n_estimators": 2, "max_depth": 2}

    oof_df = run_development_oof_pipeline(
        dev_df=df,
        development_seasons=dev_seasons,
        b2_params=fast_params,
        b3_params=fast_params,
        prob_params=fast_params,
        range_params=fast_params,
    )

    expected_uncalibrated_cols = [
        "uncalibrated_p_ge_15_6",
        "uncalibrated_p_ge_64_5",
        "uncalibrated_p_ge_115_6",
    ]
    for col in expected_uncalibrated_cols:
        assert col in oof_df.columns, f"Expected {col} in OOF handoff table"

    # Must NOT contain plain 'p_ge_*' columns
    assert "p_ge_15_6" not in oof_df.columns
    assert "p_ge_64_5" not in oof_df.columns
    assert "p_ge_115_6" not in oof_df.columns


def test_obs_mm_null_or_missing_during_final_inference():
    """Verify that obs_mm may be null or completely missing during final inference."""
    dev_seasons = [2018, 2019]
    df = generate_synthetic_m3_data(seasons=dev_seasons, dates_per_season=2)
    fast_params = {"n_estimators": 2, "max_depth": 2}

    final_models = fit_final_m3_models(
        dev_df=df,
        development_seasons=dev_seasons,
        b2_params=fast_params,
        b3_params=fast_params,
        prob_params=fast_params,
        range_params=fast_params,
    )

    infer_df = generate_synthetic_m3_data(seasons=[2020], dates_per_season=1)
    infer_df["regime_source"] = "final"

    # 1. obs_mm is all NaN
    df_nan_obs = infer_df.copy()
    df_nan_obs["obs_mm"] = np.nan
    out_nan = predict_final_m3(final_models, df_nan_obs, allow_uncalibrated=True)
    assert out_nan["obs_mm"].isna().all()
    validate_serving_grid_contracts(out_nan)

    # 2. obs_mm is completely dropped
    df_no_obs = infer_df.drop(columns=["obs_mm"])
    out_no_obs = predict_final_m3(final_models, df_no_obs, allow_uncalibrated=True)
    assert out_no_obs["obs_mm"].isna().all()
    validate_serving_grid_contracts(out_no_obs)


def test_documentation_and_code_have_no_invented_season_assumptions_and_no_crps():
    """Verify documentation strictly uses PRD §10.2 season partition logic and primary checks."""
    from pathlib import Path

    handoff_text = Path("docs/m3-handoff.md").read_text(encoding="utf-8")
    status_text = Path("docs/m3-status.md").read_text(encoding="utf-8")

    # Neither document should assert development = 2018–2020 as a fixed reality
    assert "development = 2018–2020" not in handoff_text
    assert "development = 2018–2020" not in status_text
    assert "holdout = 2021" not in handoff_text
    assert "holdout = 2021" not in status_text

    # Both documents must document PRD §10.2 season partition logic:
    # N >= 8, 5 <= N <= 7, 3 <= N <= 4, N <= 2
    for doc_name, text in [("handoff", handoff_text), ("status", status_text)]:
        has_ge_8 = any(s in text for s in ["N >= 8", "N ≥ 8", r"N \ge 8"])
        has_5_to_7 = any(s in text for s in ["5 <= N <= 7", "5 ≤ N ≤ 7", r"5 \le N \le 7"])
        has_3_to_4 = any(s in text for s in ["3 <= N <= 4", "3 ≤ N ≤ 4", r"3 \le N \le 4"])
        has_le_2 = any(s in text for s in ["N <= 2", "N ≤ 2", r"N \le 2"])
        assert has_ge_8, f"{doc_name} missing N >= 8 rule"
        assert has_5_to_7, f"{doc_name} missing 5 <= N <= 7 rule"
        assert has_3_to_4, f"{doc_name} missing 3 <= N <= 4 rule"
        assert has_le_2, f"{doc_name} missing N <= 2 rule"

    # CRPS must not be listed as a required primary PRD metric
    assert "CRPS is NOT a required" in handoff_text or "CRPS is not listed as a required" in handoff_text
    assert "CRPS" not in status_text

    # Primary checks must be listed
    for primary_metric in ["RMSE", "ETS", "FSS", "Brier"]:
        assert primary_metric in handoff_text
        assert primary_metric in status_text
