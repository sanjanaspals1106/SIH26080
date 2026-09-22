"""Unit tests for verification result packaging and primary-metric threshold selection (PRD §10.8, §16.11).

Required tests:
1. 29 holdout events selects 15.6
2. 30 holdout events keeps 64.5
3. threshold selection independent of metric skill
4. basic result matches §16.11 structure
5. undefined metric preserves None/reason
6. missing bootstrap => CI remains None
7. bootstrap CI inserted correctly
8. paired difference inserted correctly
9. development/holdout preserved
10. n_samples/n_events preserved
11. exploratory 64.5 result can coexist when fallback threshold is 15.6
12. no claim wording generated
"""

import pytest

from verification.components import VerificationComponent
from verification.results import (
    MetricRecord,
    PrimaryThresholdDecision,
    VerificationResult,
    create_example_result,
    package_verification_result,
    select_primary_threshold,
)
from verification.stats.bootstrap import BootstrapResult


def test_1_29_holdout_events_selects_15_6():
    """Test 1: 29 holdout events (< 30) selects 15.6 mm fallback threshold."""
    decision = select_primary_threshold(holdout_event_count=29)
    assert decision.selected_threshold_mm == 15.6
    assert decision.default_threshold_mm == 64.5
    assert decision.fallback_used is True
    assert decision.holdout_event_count == 29
    assert "INSUFFICIENT_HOLDOUT_EVENTS" in decision.reason
    assert "29 < 30" in decision.reason

    d = decision.to_dict()
    assert d["selected_threshold_mm"] == 15.6
    assert d["fallback_used"] is True


def test_2_30_holdout_events_keeps_64_5():
    """Test 2: 30 holdout events (>= 30) retains 64.5 mm default primary threshold."""
    decision = select_primary_threshold(holdout_event_count=30)
    assert decision.selected_threshold_mm == 64.5
    assert decision.default_threshold_mm == 64.5
    assert decision.fallback_used is False
    assert decision.holdout_event_count == 30
    assert decision.reason == "SUFFICIENT_HOLDOUT_EVENTS"


def test_3_threshold_selection_independent_of_metric_skill():
    """Test 3: Threshold selection depends strictly on holdout event counts, independent of model skill."""
    # Regardless of whether a model has high skill (e.g. ETS=0.9) or low skill (ETS=0.01),
    # select_primary_threshold accepts only event count and protocol configs.
    dec_low_count = select_primary_threshold(holdout_event_count=10)
    assert dec_low_count.selected_threshold_mm == 15.6
    assert dec_low_count.fallback_used is True

    dec_high_count = select_primary_threshold(holdout_event_count=100)
    assert dec_high_count.selected_threshold_mm == 64.5
    assert dec_high_count.fallback_used is False


def test_4_basic_result_matches_section_16_11_structure():
    """Test 4: Basic result matches the output contract specified in PRD §16.11."""
    res = create_example_result()
    d = res.to_dict()

    # Exact top-level keys from PRD §16.11
    expected_keys = {
        "forecast_type",
        "comparison_to",
        "evaluation_set",
        "lead_day",
        "threshold_mm",
        "neighbourhood_cells",
        "group",
        "metrics",
        "paired_difference",
        "n_samples",
        "n_events",
        "bootstrap",
        "model_version_id",
        "primary_metric",
        "exploratory",
    }
    assert set(d.keys()) == expected_keys

    # Group structure
    assert d["group"] == {"type": "all", "value": "all"}

    # Metrics dictionary contains core §16.11 metrics
    core_metrics = ["rmse", "pod", "far", "csi", "ets", "fss", "frequency_bias"]
    for m in core_metrics:
        assert m in d["metrics"]
        assert "value" in d["metrics"][m]
        assert "ci_low" in d["metrics"][m]
        assert "ci_high" in d["metrics"][m]

    # Paired difference structure
    assert "ets" in d["paired_difference"]
    assert "value" in d["paired_difference"]["ets"]
    assert "ci_low" in d["paired_difference"]["ets"]
    assert "ci_high" in d["paired_difference"]["ets"]

    # Bootstrap metadata structure
    assert d["bootstrap"] == {"block_days": 7, "resamples": 2000}


def test_5_undefined_metric_preserves_none_and_reason():
    """Test 5: Undefined metric preserves None value, machine-readable reason, and clears CIs."""
    # Zero sample component: all metrics undefined
    comp = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        n_samples=0,
        n=0,
    )

    # Even if someone erroneously passes a CI tuple for an undefined metric, CI must remain None
    res = package_verification_result(
        component=comp,
        metric_cis={"rmse": (1.0, 2.0), "ets": (0.2, 0.4)},
    )

    rmse_rec = res.metrics["rmse"]
    assert rmse_rec.value is None
    assert rmse_rec.undefined_reason == "NO_VALID_SAMPLES"
    assert rmse_rec.ci_low is None
    assert rmse_rec.ci_high is None

    # Serialization preserves undefined_reason
    d = res.to_dict()
    assert d["metrics"]["rmse"]["value"] is None
    assert d["metrics"]["rmse"]["ci_low"] is None
    assert d["metrics"]["rmse"]["ci_high"] is None
    assert d["metrics"]["rmse"]["undefined_reason"] == "NO_VALID_SAMPLES"


def test_6_missing_bootstrap_ci_remains_none():
    """Test 6: When bootstrap is not performed, CIs remain strictly None (no fabrication)."""
    comp = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        n_samples=100,
        n=100,
        sum_squared_error=400.0,  # RMSE = 2.0
        a=20,
        b=10,
        c=10,
        d=60,
    )

    res = package_verification_result(
        component=comp,
        comparison_to="raw_nwp",
        paired_bootstrap=None,
        metric_cis=None,
    )

    for m_name, m_rec in res.metrics.items():
        assert m_rec.ci_low is None
        assert m_rec.ci_high is None

    assert res.metrics["rmse"].value == pytest.approx(2.0)


def test_7_bootstrap_ci_inserted_correctly():
    """Test 7: Bootstrap confidence intervals for metric values are inserted correctly."""
    comp = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        n_samples=100,
        n=100,
        sum_squared_error=400.0,
        a=20,
        b=10,
        c=10,
        d=60,
    )

    metric_cis = {
        "rmse": (1.8, 2.3),
        "ets": (0.30, 0.52),
    }

    res = package_verification_result(
        component=comp,
        metric_cis=metric_cis,
    )

    assert res.metrics["rmse"].value == pytest.approx(2.0)
    assert res.metrics["rmse"].ci_low == pytest.approx(1.8)
    assert res.metrics["rmse"].ci_high == pytest.approx(2.3)

    assert res.metrics["ets"].ci_low == pytest.approx(0.30)
    assert res.metrics["ets"].ci_high == pytest.approx(0.52)


def test_8_paired_difference_inserted_correctly():
    """Test 8: Paired difference point estimates and CIs from paired bootstrap are inserted correctly."""
    comp = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
        threshold_mm=64.5,
        n_samples=200,
        n=200,
        a=40,
        b=15,
        c=15,
        d=130,
    )

    paired_bs = {
        "ets": BootstrapResult(
            metric_a=0.45,
            metric_b=0.35,
            paired_difference=0.10,
            ci_low=0.02,
            ci_high=0.18,
            block_days=7,
            n_resamples=2000,
            n_valid_resamples=2000,
        )
    }

    res = package_verification_result(
        component=comp,
        comparison_to="raw_nwp",
        paired_bootstrap=paired_bs,
    )

    assert "ets" in res.paired_difference
    ets_diff = res.paired_difference["ets"]
    assert ets_diff.value == pytest.approx(0.10)
    assert ets_diff.ci_low == pytest.approx(0.02)
    assert ets_diff.ci_high == pytest.approx(0.18)

    assert res.bootstrap == {"block_days": 7, "resamples": 2000}


def test_9_development_and_holdout_preserved():
    """Test 9: DEVELOPMENT and HOLDOUT evaluation sets remain distinct and accurately preserved."""
    comp_dev = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=15.6,
        n_samples=50,
    )
    res_dev = package_verification_result(comp_dev)
    assert res_dev.evaluation_set == "development"

    comp_holdout = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
        threshold_mm=15.6,
        n_samples=50,
    )
    res_holdout = package_verification_result(comp_holdout)
    assert res_holdout.evaluation_set == "holdout"


def test_10_n_samples_and_n_events_preserved():
    """Test 10: n_samples and n_events are accurately carried through from the component."""
    comp_with_events = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
        threshold_mm=64.5,
        n_samples=1250,
        n_events=42,
    )
    res1 = package_verification_result(comp_with_events)
    assert res1.n_samples == 1250
    assert res1.n_events == 42

    comp_without_events = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="development",
        threshold_mm=64.5,
        n_samples=800,
        n_events=None,
    )
    res2 = package_verification_result(comp_without_events)
    assert res2.n_samples == 800
    assert res2.n_events is None


def test_11_exploratory_64_5_result_can_coexist_when_fallback_threshold_is_15_6():
    """Test 11: An exploratory 64.5 mm result coexists with the primary 15.6 mm fallback result."""
    # Scenario: holdout had 22 events at 64.5 mm (< 30) -> protocol selected 15.6 mm fallback
    decision = select_primary_threshold(holdout_event_count=22)
    assert decision.selected_threshold_mm == 15.6
    assert decision.fallback_used is True

    # Primary metric result at fallback threshold 15.6 mm
    comp_15 = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
        threshold_mm=15.6,
        n_samples=100,
        n_events=65,
    )
    res_15 = package_verification_result(
        component=comp_15,
        primary_threshold_mm=decision.selected_threshold_mm,
    )
    assert res_15.threshold_mm == 15.6
    assert res_15.primary_metric is True
    assert res_15.exploratory is False

    # Exploratory result at 64.5 mm
    comp_64 = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
        threshold_mm=64.5,
        n_samples=100,
        n_events=22,
    )
    res_64 = package_verification_result(
        component=comp_64,
        primary_threshold_mm=decision.selected_threshold_mm,
    )
    assert res_64.threshold_mm == 64.5
    assert res_64.primary_metric is False
    assert res_64.exploratory is True

    # Both results coexist with clear flags
    assert res_15.primary_metric != res_64.primary_metric
    assert res_15.exploratory != res_64.exploratory


def test_12_no_claim_wording_generated():
    """Test 12: Verification results contain strictly metrics and metadata without marketing/claim text."""
    comp = VerificationComponent(
        forecast_type="regime_aware_ml",
        evaluation_set="holdout",
        threshold_mm=64.5,
        n_samples=100,
        n_events=35,
    )
    res = package_verification_result(
        component=comp,
        comparison_to="raw_nwp",
        paired_bootstrap={"ets": BootstrapResult(0.5, 0.3, 0.2, 0.05, 0.35, 7, 2000, 2000)},
    )
    d = res.to_dict()

    forbidden_phrases = [
        "regime information helps",
        "demonstrated benefit",
        "no demonstrated benefit",
        "b3 is better",
        "hypothesis h1",
        "ai beats nwp",
    ]

    import json
    serialized = json.dumps(d).lower()
    for phrase in forbidden_phrases:
        assert phrase not in serialized, f"Forbidden claim phrase '{phrase}' found in result serialization"
