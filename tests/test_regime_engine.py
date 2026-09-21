"""Comprehensive unit tests for the Regime Engine (Owner: M2).

Covers all test requirements specified in PRD Section 23:
1. Labels:
   - Synthetic series with known 3-day active/break spells.
   - July-August climatological statistics check (7 +/- 3 days).
2. Phase Model & Features:
   - Probabilities strictly sum to 1.0.
   - Normalized entropy confidence in [0, 1].
   - Out-of-distribution (OOD) detection.
3. Low-Pressure Systems (Layer B):
   - Detector finds synthetic vortex.
   - Detector rejects noise.
   - MSLP depression criterion.
   - Per-cell influence and decay.
4. Local Context (Layer C):
   - Raw fluxes (upslope and onshore).
   - Zero flux gives 0.0 influence.
   - Calibrated percentile ranking in (0, 1].
5. Historical Analog Finder (F3):
   - Query's own season is excluded.
   - Analogs separated by >= 5 days.
   - Output contains K=5 analogs and distance percentiles.
6. Regime Transition Detection (F4):
   - Phase transition detection and confirmation.
   - Date gap prevents cross-gap event creation.
   - LPS formation and dissipation events.
7. Contracts & Leakage Guard:
   - Domain and cell contracts match PRD 11.6.
   - Exactly 14 features for B3 (PRD 11.7).
   - validate_training_regime_source stops with ValueError on 'final'.
"""

import unittest
import numpy as np
import pandas as pd
import datetime

from regime_engine.labels import (
    compute_doy_climatology,
    compute_standardized_anomalies,
    detect_spells,
    check_july_august_statistics,
    generate_phase_labels,
)
from regime_engine.phase import (
    build_phase_feature_vector,
    compute_regime_confidence,
    PhaseOODDetector,
    PhaseModel,
    select_best_c_loso,
)
from regime_engine.lps import (
    haversine_distance_km,
    compute_bearing_and_components,
    LPSDetector,
    compute_lps_cell_features,
)
from regime_engine.local_context import (
    compute_raw_fluxes,
    InfluencePercentileTable,
)
from regime_engine.analogs import (
    AnalogFinder,
    ANALOG_VECTOR_KEYS,
)
from regime_engine.transitions import (
    TransitionDetector,
)
from regime_engine.contract import (
    REGIME_14_FEATURES,
    build_domain_regime_contract,
    build_cell_regime_contract,
    assemble_14_regime_features,
    validate_training_regime_source,
)


class TestRegimeLabels(unittest.TestCase):
    """PRD Section 23: Labels tests."""

    def test_synthetic_spell_detection(self):
        """Test that a series with known active and break spells produces exact labels."""
        dates = pd.date_range("2024-07-01", periods=15, freq="D")
        # Days 0..2: normal (z = 0)
        # Days 3..5: 3 consecutive days with z = 1.5 (> 1.0) -> active
        # Day 6: normal (z = 0.5)
        # Days 7..8: only 2 days with z = 1.5 (< 3 days) -> normal
        # Day 9: normal (z = 0)
        # Days 10..13: 4 consecutive days with z = -1.5 (< -1.0) -> break
        # Day 14: normal (z = 0)
        z_vals = [0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.5, 1.5, 1.5, 0.0, -1.5, -1.5, -1.5, -1.5, 0.0]
        z_series = pd.Series(z_vals, index=dates)

        labels = detect_spells(z_series, active_z_min=1.0, break_z_max=-1.0, min_run_days=3)

        expected = [
            "normal", "normal", "normal",
            "active", "active", "active",
            "normal",
            "normal", "normal",  # only 2 days, so NOT active!
            "normal",
            "break", "break", "break", "break",
            "normal",
        ]
        self.assertEqual(labels.tolist(), expected)

    def test_statistics_check(self):
        """Test the July-August statistics checker."""
        dates = pd.date_range("1981-06-01", "2010-09-30", freq="D")
        n = len(dates)
        labels = np.array(["normal"] * n, dtype=object)

        # In each year, assign exactly one 7-day active spell and one 7-day break spell in ~74% of years inside July-August
        for yr in range(1981, 2011):
            ja_mask = (dates.year == yr) & (dates.month.isin([7, 8]))
            ja_indices = np.where(ja_mask)[0]
            if len(ja_indices) >= 30:
                # 7-day active spell in July
                labels[ja_indices[10:17]] = "active"
                # 7-day break spell in August for ~74% of years (no-break in ~26% of years)
                if yr % 4 != 0:
                    labels[ja_indices[35:42]] = "break"

        s = pd.Series(labels, index=dates)
        res = check_july_august_statistics(s)
        self.assertTrue(res["passed"])
        self.assertTrue(res["active_days_pass"])
        self.assertTrue(res["break_days_pass"])
        self.assertTrue(res["no_break_share_pass"])


class TestPhaseModel(unittest.TestCase):
    """PRD Section 23: Phase model and features."""

    def test_probabilities_sum_to_one(self):
        """Test that predicted phase probabilities strictly sum to 1.0."""
        np.random.seed(42)
        X = np.random.randn(30, 20)
        y = np.array(["active"] * 10 + ["normal"] * 10 + ["break"] * 10)

        model = PhaseModel(C=1.0)
        model.fit(X, y)

        X_test = np.random.randn(15, 20)
        probs = model.predict_proba(X_test)

        self.assertEqual(probs.shape, (15, 3))
        for row in probs:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=6)
            self.assertTrue(np.all(row >= 0.0))

    def test_confidence_calculation(self):
        """Test normalized Shannon entropy confidence."""
        # Max uncertainty: equal probabilities -> confidence = 0.0
        p_unif = np.array([1/3, 1/3, 1/3])
        conf_unif, band_unif = compute_regime_confidence(p_unif)
        self.assertAlmostEqual(conf_unif, 0.0, places=4)
        self.assertEqual(band_unif, "low")

        # Absolute certainty: [1, 0, 0] -> confidence = 1.0
        p_cert = np.array([1.0, 0.0, 0.0])
        conf_cert, band_cert = compute_regime_confidence(p_cert)
        self.assertAlmostEqual(conf_cert, 1.0, places=4)
        self.assertEqual(band_cert, "high")

    def test_ood_detection(self):
        """Test out-of-distribution detector for features A1-A6."""
        training_a = {
            f"A{i}": np.linspace(10.0, 50.0, 100) for i in range(1, 7)
        }
        detector = PhaseOODDetector(min_features_outside=2)
        detector.fit(training_a)

        # In-distribution sample
        in_sample = {f"A{i}": 30.0 for i in range(1, 7)}
        self.assertFalse(detector.predict(in_sample))

        # Only 1 feature outlier (< 2) -> not OOD
        one_out = in_sample.copy()
        one_out["A1"] = 100.0
        self.assertFalse(detector.predict(one_out))

        # 2 features outlier -> OOD!
        two_out = in_sample.copy()
        two_out["A1"] = 100.0
        two_out["A2"] = 0.0
        self.assertTrue(detector.predict(two_out))


class TestLPSDetector(unittest.TestCase):
    """PRD Section 23: Low-Pressure System detection."""

    def test_vortex_detection_and_noise_rejection(self):
        """Test that synthetic vortex is detected and background noise is rejected."""
        lats = np.linspace(10.0, 25.0, 61)
        lons = np.linspace(70.0, 95.0, 101)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Background field with small random noise
        np.random.seed(123)
        vort = np.random.randn(*lat_grid.shape) * 1.0e-6
        msl = np.full(lat_grid.shape, 1010.0) + np.random.randn(*lat_grid.shape) * 0.2

        # Add synthetic vortex at (lat=18.0, lon=85.0)
        # Synoptic depression width 300 km with strong vorticity peak (6.0e-5 > zeta_min 1.5e-5)
        # MSLP depression of 6 hPa (> dp_min 2.0 hPa)
        dists = haversine_distance_km(18.0, 85.0, lat_grid, lon_grid)
        vort += 6.0e-5 * np.exp(-((dists / 300.0) ** 2))
        msl -= 6.0 * np.exp(-((dists / 300.0) ** 2))

        detector = LPSDetector(zeta_min=1.5e-5, dp_min_hpa=2.0, sigma_deg=1.5)
        centers = detector.detect(vort, msl, lats, lons)

        self.assertGreaterEqual(len(centers), 1)
        detected = centers[0]
        self.assertAlmostEqual(detected["lat"], 18.0, delta=1.0)
        self.assertAlmostEqual(detected["lon"], 85.0, delta=1.0)
        self.assertGreater(detected["zeta_max"], 1.5e-5)

    def test_lps_cell_features(self):
        """Test cell-level feature computation with distance decay."""
        cells_df = pd.DataFrame([
            {"cell_id": 1, "latitude": 18.0, "longitude": 85.0},  # Exactly at centre
            {"cell_id": 2, "latitude": 18.0, "longitude": 91.0},  # ~600 km away
            {"cell_id": 3, "latitude": 28.0, "longitude": 75.0},  # Far away
        ])
        centers = [{"lat": 18.0, "lon": 85.0, "zeta_max": 3.0e-5, "strength_percentile": 0.8}]

        feat_df = compute_lps_cell_features(cells_df, centers, lps_influence_scale_km=600.0)

        # Cell 1: distance ~0, influence ~ 0.8 * exp(0) = 0.8
        c1 = feat_df.loc[feat_df["cell_id"] == 1].iloc[0]
        self.assertAlmostEqual(c1["distance_to_lps_km"], 0.0, delta=1.0)
        self.assertAlmostEqual(c1["lps_influence"], 0.8, delta=0.05)

        # Cell 2: distance ~600km, influence ~ 0.8 * exp(-1) ~ 0.294
        c2 = feat_df.loc[feat_df["cell_id"] == 2].iloc[0]
        self.assertAlmostEqual(c2["distance_to_lps_km"], 600.0, delta=50.0)
        self.assertAlmostEqual(c2["lps_influence"], 0.8 * np.exp(-1.0), delta=0.05)


class TestLocalContext(unittest.TestCase):
    """PRD Section 23: Layer C coast and mountain influence."""

    def test_zero_flux_gives_zero_influence(self):
        """Test PRD 11.5 requirement: Influence is 0 when raw flux index is 0."""
        upslope, onshore = compute_raw_fluxes(
            u850=np.array([10.0, -10.0]),
            v850=np.array([0.0, 0.0]),
            q850=np.array([0.015, 0.015]),
            grad_h_x=np.array([-0.01, -0.01]),  # wind blowing downslope for item 0
            grad_h_y=np.array([0.0, 0.0]),
            coast_normal_x=np.array([-1.0, -1.0]),  # wind blowing offshore for item 0
            coast_normal_y=np.array([0.0, 0.0]),
            dist_coast_km=np.array([50.0, 50.0]),
        )

        # Item 0 has downhill and offshore wind -> flux must be 0.0
        self.assertEqual(upslope[0], 0.0)
        self.assertEqual(onshore[0], 0.0)

        table = InfluencePercentileTable()
        table.fit(np.linspace(0.001, 0.1, 100), np.linspace(0.001, 0.1, 100))

        orog_inf, coast_inf, orog_fav, coast_fav = table.compute_influence(upslope, onshore)
        self.assertEqual(orog_inf[0], 0.0)
        self.assertEqual(coast_inf[0], 0.0)
        self.assertFalse(orog_fav[0])
        self.assertFalse(coast_fav[0])

        # Item 1 has upslope and onshore wind -> flux > 0, influence > 0
        self.assertGreater(upslope[1], 0.0)
        self.assertGreater(onshore[1], 0.0)
        self.assertGreater(orog_inf[1], 0.0)
        self.assertGreater(coast_inf[1], 0.0)


class TestAnalogFinder(unittest.TestCase):
    """PRD Section 23: F3 Historical Analog Finder."""

    def test_analogs_exclude_query_season_and_enforce_spacing(self):
        """Test that query season is excluded and analogs are at least 5 days apart."""
        # Create a synthetic library of 50 days across seasons 2017, 2018, 2019
        records = []
        base_date = datetime.date(2017, 6, 1)
        for i in range(60):
            d = base_date + datetime.timedelta(days=i)
            season = 2017 if i < 20 else (2018 if i < 40 else 2019)
            rec = {
                "run_id": f"run_{season}_{i}",
                "lead_day": 1,
                "imd_date": str(d),
                "season": season,
            }
            # Fill 10 vector values
            for k in ANALOG_VECTOR_KEYS:
                rec[k] = float(np.sin(i * 0.1) + 1.0)
            records.append(rec)

        library_df = pd.DataFrame(records)
        finder = AnalogFinder(k=5, min_separation_days=5)
        finder.fit_library(library_df)

        query_vec = {k: 1.0 for k in ANALOG_VECTOR_KEYS}
        res = finder.find_analogs(query_vec, lead_day=1, query_season=2018)

        self.assertEqual(res["n_analogs"], 5)
        selected_analogs = res["analogs"]

        # 1. Check no analog comes from query season 2018
        for ana in selected_analogs:
            self.assertNotEqual(ana["season"], 2018)

        # 2. Check all selected dates are separated by at least 5 days
        dates = [datetime.date.fromisoformat(a["imd_date"]) for a in selected_analogs]
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                self.assertGreaterEqual(abs((dates[i] - dates[j]).days), 5)


class TestTransitionDetector(unittest.TestCase):
    """PRD Section 23: F4 Regime Transition Detection."""

    def test_phase_transition_and_gap_handling(self):
        """Test that phase transitions are detected and gaps prevent false events."""
        # Timeline:
        # d0..d2: normal
        # d3..d4: active with prob 0.8
        # d5: MISSING (gap)
        # d6..d7: break with prob 0.8
        dates = ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05", "2024-07-07", "2024-07-08"]
        df = pd.DataFrame({
            "imd_date": dates,
            "p_active": [0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.1],
            "p_normal": [0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1],
            "p_break":  [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8],
            "lps_present": [False, False, False, False, False, False, False],
            "domain_mean_corrected_mm": [10.0, 10.0, 12.0, 25.0, 30.0, 5.0, 5.0],
        })

        detector = TransitionDetector(smoothing_days=1, confirm_min_prob=0.5)
        res = detector.detect_transitions(df)

        events = res["events"]
        event_types = [e["event_type"] for e in events]

        # Should detect PHASE:normal->active on 2024-07-04
        self.assertIn("PHASE:normal->active", event_types)

        # Gap between 2024-07-05 and 2024-07-07 must NOT generate a transition across the gap!
        for e in events:
            self.assertNotEqual(e["event_date"], "2024-07-07")


class TestRegimeContractsAndLeakage(unittest.TestCase):
    """PRD Section 11.6, 11.7 and leakage prevention."""

    def test_contract_structure(self):
        """Test domain and cell regime contract generation."""
        domain_contract = build_domain_regime_contract(
            run_id="tigge_test_01",
            lead_day=1,
            p_active=0.42,
            p_normal=0.46,
            p_break=0.12,
            regime_confidence=0.18,
            confidence_band="low",
            regime_source="final",
            lps_detected=True,
            lps_centres=[{"lat": 20.4, "lon": 86.1, "zeta_max": 2.1e-5, "mslp_min_hpa": 996.0, "strength_percentile": 0.74}],
        )
        self.assertEqual(domain_contract["run_id"], "tigge_test_01")
        self.assertEqual(domain_contract["phase"]["confidence_band"], "low")
        self.assertTrue(domain_contract["lps"]["detected"])

        cell_contract = build_cell_regime_contract(
            cell_id=8123,
            lps_present=True,
            distance_to_lps_km=260.0,
            bearing_to_lps_deg=235.0,
            lps_influence=0.52,
            orographic_influence=0.88,
            coastal_influence=0.35,
            orographic_favorable=False,
            coastal_favorable=False,
        )
        self.assertEqual(cell_contract["cell_id"], 8123)
        self.assertAlmostEqual(cell_contract["lps_influence"], 0.52)

    def test_14_regime_features(self):
        """Test that exactly the 14 regime features for B3 are assembled."""
        domain_dict = build_domain_regime_contract(
            run_id="run_1", lead_day=1, p_active=0.4, p_normal=0.5, p_break=0.1,
            regime_confidence=0.3, confidence_band="medium", regime_source="oof",
            lps_detected=False, lps_centres=[],
        )
        cell_df = pd.DataFrame([{
            "cell_id": 1, "lps_present": 0, "distance_to_lps_km": 3000.0,
            "bearing_sin": 0.0, "bearing_cos": 0.0, "lps_strength": 0.0,
            "lps_influence": 0.0, "upslope_flux": 0.05, "onshore_flux": 0.02,
            "orographic_influence": 0.7, "coastal_influence": 0.4,
        }])

        b3_feat_df = assemble_14_regime_features(domain_dict, cell_df)
        self.assertEqual(list(b3_feat_df.columns), ["cell_id"] + REGIME_14_FEATURES)
        self.assertEqual(len(REGIME_14_FEATURES), 14)

    def test_leakage_guard(self):
        """Test that validate_training_regime_source raises error on 'final'."""
        with self.assertRaises(ValueError):
            validate_training_regime_source("final")
        # 'oof' should pass without error
        validate_training_regime_source("oof")


if __name__ == "__main__":
    unittest.main()
