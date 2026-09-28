"""`/verification`, `/forecasts/{forecast_id}/audit` and `/model-info`. Additive to M1's API.

API-level tests against a small, self-contained synthetic world (its own tmp data dir and database), like
`tests/test_backend_regime.py`: nothing here reads or writes the real `data/` tree.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_golden_dir
from backend.app.config import get_settings, settings_from_config
from backend.app.db.loader import ensure_schema
from backend.app.db.session import engine_for
from backend.app.db.tables import district_forecasts, district_history, districts, nwp_runs
from backend.app.main import create_app
from data_pipeline.ingestion import load_config

API = "/api/v1"
TARGET = "tigge_ecmwf_cf_2024070100"  # season 2024 (development)
H2022, H2023, H2025 = "tigge_ecmwf_cf_2022070100", "tigge_ecmwf_cf_2023070100", "tigge_ecmwf_cf_2025070100"
FID = f"{TARGET}_L1_D001"
CELL = 7


# ---- synthetic files -----------------------------------------------------------------------------------------


def _model(rmse: float, bias_events: float = 1.0) -> dict:
    m = {"bias": 0.1, "mae": rmse / 2, "rmse": rmse}
    for t, ets in (("15.6", 0.25), ("64.5", 0.10)):
        m.update({
            f"n_obs_events_{t}": 1000, f"pod_{t}": 0.5, f"far_{t}": 0.4, f"csi_{t}": 0.3,
            f"frequency_bias_{t}": bias_events, f"ets_{t}": ets,
        })  # fmt: skip
    m.update({"fss5_15.6": 0.6, "fss5_64.5": 0.4})
    return m


def _dev_metrics() -> dict:
    return {
        "correction": {
            "raw_nwp": _model(15.0), "b1_quantile_mapping": _model(16.0), "b2_global_ml": _model(13.0),
            "b3_regime_aware": _model(12.5, bias_events=0.3),
        },
        "probability": {
            "15.6": {"brier_uncalibrated": 0.09, "brier_calibrated": 0.088, "bss_vs_climatology": 0.25, "climatology_rate": 0.14},
            "64.5": {"brier_uncalibrated": 0.013, "brier_calibrated": 0.0129, "bss_vs_climatology": 0.17, "climatology_rate": 0.016},
        },
        "range_coverage": {
            "1": {"lead_day": 1, "coverage": 0.89, "n_samples": 100, "target": 0.8, "tolerance": 0.1, "needs_display_warning": False},
            "2": {"lead_day": 2, "coverage": 0.6, "n_samples": 100, "target": 0.8, "tolerance": 0.1, "needs_display_warning": True},
        },
        "settings": {"B2": {}, "B3": {}},
        "event_counts": {"15.6": 300, "64.5": 50},
        "development_seasons": [2021, 2022, 2023, 2024],
        "holdout_seasons": [2025],
    }


def _holdout_metrics() -> dict:
    corrected = _model(14.0, bias_events=0.05)
    corrected["ets_64.5"] = 0.02
    return {
        "correction": {"raw_nwp": _model(15.0), "final_corrected": corrected},
        "holdout_seasons": [2025],
        "lock": {"git_commit": "abc1234def", "timestamp": "2026-09-27T22:52:02+00:00", "forced_rerun": False, "run_count": 1},
    }


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _domain_row(run_id: str, season: int, **overrides) -> dict:
    row = {
        "run_id": run_id, "lead_day": 1, "season": season, "imd_date": pd.Timestamp(f"{season}-07-02"),
        "p_active": 0.1, "p_normal": 0.8, "p_break": 0.1, "regime_confidence": 0.8, "confidence_band": "high",
        "lps_detected": False, "lps_settings": "untuned", "regime_available": True, "ood_flag": False,
    }  # fmt: skip
    row.update(overrides)
    return row


def _write_regime(cfg) -> None:
    regime_dir = cfg.data_dir / "regime"
    (regime_dir / "regime_features").mkdir(parents=True, exist_ok=True)
    rows = [_domain_row(TARGET, 2024), _domain_row(H2022, 2022), _domain_row(H2023, 2023), _domain_row(H2025, 2025)]
    pd.DataFrame(rows).to_parquet(regime_dir / "regime_domain.parquet")
    pd.DataFrame([{
        "run_id": TARGET, "lead_day": 1, "cell_id": CELL, "distance_to_lps_km": 120.0, "bearing_sin": 1.0,
        "bearing_cos": 0.0, "lps_influence": 0.5, "orographic_influence": 0.8, "coastal_influence": 0.3,
    }]).to_parquet(regime_dir / "regime_features" / "season_2024.parquet")  # fmt: skip


def _write_golden(cfg) -> None:
    path = cfg.alignment.golden_dir / "season_2024" / "golden_202407.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"run_id": TARGET, "lead_day": 1, "cell_id": CELL, "rain_mm": 5.0}]).to_parquet(path)


def _write_models(cfg) -> None:
    final = cfg.data_dir / "models" / "m3" / "final"
    for name, n in (("b2", 27), ("b3", 41), ("probability", 41), ("range", 41)):
        _write_json(final / name / "metadata.json", {
            "feature_names": [f"f{i}" for i in range(n)], "train_seasons": [2021, 2022, 2023, 2024],
            "git_commit": "816fd3aaea7b27650505798a7b0f2f6f5678ff5c",
        })  # fmt: skip


# ---- the world -----------------------------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data")
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    engine = engine_for(url)
    ensure_schema(engine)
    with engine.begin() as conn:
        for run_id, season, evaluation_set in (
            (TARGET, 2024, "development"), (H2022, 2022, "development"), (H2023, 2023, "development"), (H2025, 2025, "holdout"),
        ):  # fmt: skip
            conn.execute(nwp_runs.insert(), {
                "run_id": run_id, "source": "ecmwf", "initialization_time": datetime(season, 7, 1, tzinfo=UTC),
                "season": season, "evaluation_set": evaluation_set, "mode": "replay", "alignment_method": "C1",
                "alignment_offset_hours": 0, "imd_stamp": "end_date", "status": "complete",
            })  # fmt: skip
        conn.execute(districts.insert(), {"district_id": "D001", "name": "Alpha", "state": "State A", "is_small": False})
        conn.execute(district_forecasts.insert(), {
            "run_id": TARGET, "lead_day": 1, "district_id": "D001", "imd_date": date(2024, 7, 2),
            "raw_mean_mm": 10.0, "corrected_mean_mm": 14.0, "observed_mean_mm": 13.0, "wettest_cell_id": CELL,
            "wettest_cell_mean_mm": 22.0, "wettest_cell_q10_mm": 5.0, "wettest_cell_q50_mm": 20.0,
            "wettest_cell_q90_mm": 60.0, "heavy_prob_max_cell": 0.31, "very_heavy_prob_max_cell": 0.04,
        })  # fmt: skip
        # Past dates for the same district and lead. observed minus raw: 2022 -> +2, 2023 -> +4 (median 3),
        # 2025 -> +100 (holdout: must never enter the history), 2024 (own season) -> +50 (must be excluded too).
        for run_id, season, diff in ((H2022, 2022, 2.0), (H2023, 2023, 4.0), (H2025, 2025, 100.0), (TARGET, 2024, 50.0)):
            conn.execute(district_history.insert(), {
                "run_id": run_id, "lead_day": 1, "district_id": "D001", "imd_date": date(season, 7, 2), "season": season,
                "observed_mean_mm": 10.0 + diff, "raw_mean_mm": 10.0,
            })  # fmt: skip
    _write_json(cfg.data_dir / "verification" / "m3_dev" / "metrics.json", _dev_metrics())
    _write_json(cfg.data_dir / "verification" / "m3_holdout" / "metrics.json", _holdout_metrics())
    _write_regime(cfg)
    _write_golden(cfg)
    _write_models(cfg)
    yield SimpleNamespace(cfg=cfg, url=url, engine=engine)
    engine.dispose()


def _client(world) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings_from_config(world.cfg, world.url)
    app.dependency_overrides[get_golden_dir] = lambda: world.cfg.alignment.golden_dir
    return TestClient(app, raise_server_exceptions=False)


def error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and body["error"]["message"]


# ---- /verification -------------------------------------------------------------------------------------------


def test_verification_unavailable_without_metrics(world):
    (world.cfg.data_dir / "verification" / "m3_dev" / "metrics.json").unlink()
    error(_client(world).get(f"{API}/verification"), 503, "VERIFICATION_DATA_UNAVAILABLE")


def test_verification_reports_both_evaluations_and_real_numbers(world):
    r = _client(world).get(f"{API}/verification")
    assert r.status_code == 200, r.text
    body = r.json()
    dev, hold = body["development"], body["holdout"]
    assert dev["evaluation_set"] == "development" and dev["seasons"] == [2021, 2022, 2023, 2024]
    assert [m["key"] for m in dev["models"]] == ["raw_nwp", "b1_quantile_mapping", "b2_global_ml", "b3_regime_aware"]
    assert dev["corrected_key"] == "b3_regime_aware" and dev["thresholds_mm"] == [15.6, 64.5]
    raw = dev["models"][0]
    assert raw["scalars"]["rmse"] == 15.0
    assert raw["thresholds"]["64.5"]["ets"] == 0.10 and raw["thresholds"]["64.5"]["fss5"] == 0.4
    assert raw["thresholds"]["64.5"]["n_obs_events"] == 1000
    assert hold["evaluation_set"] == "holdout" and hold["seasons"] == [2025]
    assert [m["key"] for m in hold["models"]] == ["raw_nwp", "final_corrected"]
    assert hold["lock"] == {
        "timestamp": "2026-09-27T22:52:02+00:00", "git_commit": None, "run_count": 1, "forced_rerun": False,
    }  # fmt: skip  (commit hashes are not exposed)
    assert dev["lock"] is None


def test_verification_probability_and_range_are_development_only(world):
    body = _client(world).get(f"{API}/verification").json()
    assert body["probability"]["evaluation_set"] == "development"
    assert [r["threshold_mm"] for r in body["probability"]["rows"]] == [15.6, 64.5]
    assert body["probability"]["rows"][1]["brier_skill_score"] == 0.17 and body["probability"]["rows"][1]["n_events"] == 50
    assert body["range_coverage"]["evaluation_set"] == "development"
    assert [(r["lead_day"], r["within_tolerance"]) for r in body["range_coverage"]["rows"]] == [(1, True), (2, False)]


def test_verification_highlights_are_generated_from_the_numbers(world):
    hold = _client(world).get(f"{API}/verification").json()["holdout"]
    kinds = {h["text"]: h["kind"] for h in hold["highlights"]}
    assert kinds["RMSE 15.00 to 14.00 mm (6.7% lower than raw NWP)."] == "improved"
    assert kinds["Equitable threat score at 64.5 mm: 0.100 to 0.020."] == "declined"
    assert any(h["kind"] == "note" and "frequency bias 0.05" in h["text"] for h in hold["highlights"])


def test_verification_never_returns_intervals_or_fabricated_breakdowns(world):
    text = _client(world).get(f"{API}/verification").text
    for forbidden in ("ci_low", "ci_high", "confidence_interval", "reliability", "region"):
        assert forbidden not in text


def test_verification_without_holdout_file_returns_development_only(world):
    (world.cfg.data_dir / "verification" / "m3_holdout" / "metrics.json").unlink()
    body = _client(world).get(f"{API}/verification").json()
    assert body["holdout"] is None and body["development"]["models"]


# ---- /forecasts/{forecast_id}/audit --------------------------------------------------------------------------


def test_audit_happy_path(world):
    r = _client(world).get(f"{API}/forecasts/{FID}/audit")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["forecast_id"] == FID and body["evaluation_set"] == "development"
    assert body["district_name"] == "Alpha" and body["imd_date"] == "2024-07-02" and body["lead_day"] == 1
    s = body["steps"]
    assert s["raw"] == {"district_mean_mm": 10.0, "wettest_cell_mm": 5.0}  # wettest-cell raw comes from the golden file
    assert s["corrected"] == {"district_mean_mm": 14.0, "wettest_cell_mm": 22.0}
    assert s["correction"] == {"district_mean_mm": 4.0, "wettest_cell_mm": 17.0}
    assert s["confidence"]["range_wettest_cell_mm"] == {"q10": 5.0, "q50": 20.0, "q90": 60.0}
    assert s["confidence"]["heavy_prob_max_cell"] == 0.31 and s["confidence"]["very_heavy_prob_max_cell"] == 0.04
    assert s["confidence"]["measured_coverage_q10_q90"] == 0.89  # from m3_dev range_coverage, lead 1
    assert s["record"]["alignment_method"] == "C1" and s["record"]["fallback_used"] is False
    assert s["record"]["feature_set_version"] == "41 regime-aware features" and "816fd3a" not in s["record"]["model_version"]
    assert "raised the district mean from 10.0 to 14.0 mm" in body["summary"]


def test_audit_regime_step_uses_the_wettest_cell(world):
    regime = _client(world).get(f"{API}/forecasts/{FID}/audit").json()["steps"]["regime"]
    assert regime["phase"] == {"active": 0.1, "normal": 0.8, "break": 0.1, "confidence_band": "high"}
    assert regime["nearest_lps"] == {
        "present": True, "settings": "untuned", "distance_km": 120.0, "bearing_deg": 90.0, "influence": 0.5,
    }  # fmt: skip
    assert regime["orographic_influence"] == 0.8 and regime["coastal_influence"] == 0.3
    assert regime["regime_available"] is True and regime["ood_flag"] is False


def test_audit_history_uses_only_other_development_seasons(world):
    h = _client(world).get(f"{API}/forecasts/{FID}/audit").json()["steps"]["history"]
    # 2022 (+2) and 2023 (+4) only: the holdout season (+100) and the forecast's own season (+50) are excluded.
    assert h["n_dates"] == 2 and h["median_diff_mm"] == 3.0
    assert h["phase"] == "normal" and h["lps_near"] is False
    assert "few past cases" in h["note"] and "2022, 2023" in h["note"] and "2025" not in h["note"]


def test_audit_without_regime_data_omits_regime_and_history(world):
    (world.cfg.data_dir / "regime" / "regime_domain.parquet").unlink()
    from backend.app.services import regime as regime_service

    regime_service._domain_table.cache_clear()
    body = _client(world).get(f"{API}/forecasts/{FID}/audit").json()
    assert body["steps"]["regime"] is None and body["steps"]["history"] is None
    assert body["steps"]["corrected"]["district_mean_mm"] == 14.0  # the rest is still real
    assert "Most likely phase" not in body["summary"]


def test_audit_without_a_wettest_cell_leaves_cell_values_null(world):
    with world.engine.begin() as conn:
        conn.execute(district_forecasts.update().values(
            wettest_cell_id=None, wettest_cell_mean_mm=None, wettest_cell_q10_mm=None, wettest_cell_q50_mm=None,
            wettest_cell_q90_mm=None, heavy_prob_max_cell=None, very_heavy_prob_max_cell=None,
        ))  # fmt: skip
    s = _client(world).get(f"{API}/forecasts/{FID}/audit").json()["steps"]
    assert s["raw"]["wettest_cell_mm"] is None and s["corrected"]["wettest_cell_mm"] is None
    assert s["correction"]["wettest_cell_mm"] is None
    assert s["regime"]["orographic_influence"] is None  # no cell, so no cell-level terrain values either
    assert s["confidence"]["heavy_prob_max_cell"] is None and s["confidence"]["range_wettest_cell_mm"]["q10"] is None


def test_audit_errors(world):
    client = _client(world)
    error(client.get(f"{API}/forecasts/not-an-id/audit"), 422, "INVALID_FORECAST_ID")
    error(client.get(f"{API}/forecasts/{TARGET}_L4_D001/audit"), 422, "INVALID_FORECAST_ID")
    error(client.get(f"{API}/forecasts/tigge_ecmwf_cf_2099070100_L1_D001/audit"), 404, "RUN_NOT_FOUND")
    error(client.get(f"{API}/forecasts/{TARGET}_L1_D999/audit"), 404, "FORECAST_NOT_FOUND")
    error(client.get(f"{API}/forecasts/{TARGET}_L2_D001/audit"), 404, "FORECAST_NOT_FOUND")


# ---- /model-info ---------------------------------------------------------------------------------------------


def test_model_info_from_metadata_and_protocol(world):
    r = _client(world).get(f"{API}/model-info")
    assert r.status_code == 200, r.text
    body = r.json()
    counts = {m["role"]: m["n_features"] for m in body["models"]}
    assert counts["B2: Global ML"] == 27 and counts["B3: Regime-Aware ML"] == 41
    b3 = next(m for m in body["models"] if m["role"].startswith("B3"))
    assert b3["training_seasons"] == [2021, 2022, 2023, 2024] and b3.get("git_commit") is None and b3["version"] == "v1"
    assert body["protocol"] == {
        "n_seasons": 5, "development_seasons": [2021, 2022, 2023, 2024], "holdout_seasons": [2025],
        "holdout_locked": True, "status": "LOCKED EVALUATION",
    }  # fmt: skip
    assert body["data"]["alignment_method"] == "C1" and body["data"]["lead_days"] == [1, 2, 3]
    assert body["fallback_share_recent"] is None
    text = " ".join(body["limitations"]).lower()
    for phrase in ("replay", "0.25 degree", "2011", "defaults", "heavy-rain"):
        assert phrase in text


def test_model_info_without_metrics_or_models_still_answers(world):
    (world.cfg.data_dir / "verification" / "m3_dev" / "metrics.json").unlink()
    import shutil

    shutil.rmtree(world.cfg.data_dir / "models")
    body = _client(world).get(f"{API}/model-info").json()
    assert body["protocol"]["status"] == "DEVELOPMENT ONLY" and body["protocol"]["holdout_locked"] is False
    assert all(m["n_features"] in (0, 1) for m in body["models"])  # nothing invented when metadata is absent


# ---- read-only -----------------------------------------------------------------------------------------------


def test_endpoints_never_write_anything(world):
    def digest() -> dict[str, str]:
        return {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(world.cfg.data_dir.rglob("*")) if p.is_file()
        }  # fmt: skip

    before = digest()
    client = _client(world)
    for path in (f"{API}/verification", f"{API}/forecasts/{FID}/audit", f"{API}/model-info"):
        assert client.get(path).status_code == 200
    assert digest() == before
    assert client.post(f"{API}/verification").status_code == 405
