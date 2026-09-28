"""The regime endpoints (`/regime`, `/regime/transitions`, PRD 15 F1-F4). Owner: this task, additive to M1's API.

Two kinds of coverage:
- A parity test proving `services/regime.py::_detect_transitions` (a local, backend-owned port -- see that
  module's docstring for why it does not import `regime_engine`) produces the same output as the real
  `regime_engine.transitions.detector.TransitionDetector` on the same input. Importing `regime_engine` here, in
  the *test* process, is fine; it must never happen in the API process (see `test_no_heavy_imports` below and
  `tests/test_backend_integration.py`).
- API-level tests against a small, self-contained synthetic world (its own tmp data dir and database row), not
  the shared `m1_outputs`/`client` fixtures, so writing `data/regime/` here can never leak into other modules'
  "no regime data yet" tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_golden_dir
from backend.app.config import get_settings, settings_from_config
from backend.app.db.loader import ensure_schema
from backend.app.db.session import engine_for
from backend.app.db.tables import nwp_runs
from backend.app.main import create_app
from backend.app.services.regime import _detect_transitions
from data_pipeline.ingestion import load_config

API = "/api/v1"
RUN = "tigge_ecmwf_cf_2024070100"
UNKNOWN_RUN = "tigge_ecmwf_cf_2099070100"


# ---- parity with regime_engine's own detector ----------------------------------------------------------------


def _synthetic_series() -> pd.DataFrame:
    # d0..d2 normal, d3..d4 active (confirmed d3), d5 MISSING (gap), d6..d7 break (unconfirmed: series ends on d7).
    dates = ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05", "2024-07-07", "2024-07-08"]
    return pd.DataFrame({
        "imd_date": dates,
        "p_active": [0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.1],
        "p_normal": [0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1],
        "p_break": [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8],
        "lps_present": [False, False, True, True, True, False, False],
    })  # fmt: skip


def test_local_port_matches_regime_engine_detector():
    from regime_engine.transitions.detector import TransitionDetector

    df = _synthetic_series()
    expected = TransitionDetector(smoothing_days=3, confirm_min_prob=0.50).detect_transitions(df)
    actual = _detect_transitions(df)
    assert actual["series"] == expected["series"]
    # The real detector also computes `domain_mean_corrected_change_mm`, which the backend never has for this
    # series (see the module docstring); strip it before comparing the rest of each event.
    for ev in expected["events"]:
        ev.pop("domain_mean_corrected_change_mm", None)
    for ev in actual["events"]:
        ev.pop("domain_mean_corrected_change_mm", None)
    assert actual["events"] == expected["events"]
    event_types = [e["event_type"] for e in actual["events"]]
    assert "PHASE:normal->active" in event_types
    assert all(e["event_date"] != "2024-07-07" for e in actual["events"])  # nothing detected across the gap


# Importing `backend.app.main` (and therefore this module's own routes/service) must never pull in
# `regime_engine` or `sklearn` -- covered process-wide, in a subprocess, by
# `test_backend_integration.py::test_the_api_process_never_imports_pipeline_or_model_code`. Not repeated here:
# checking `sys.modules` in-process would just observe whatever the parity test above already imported.


# ---- API: a small, self-contained synthetic world ------------------------------------------------------------


@pytest.fixture
def regime_world(tmp_path):
    """One forecast run in the database, and (unless a test opts out) matching regime files on disk."""
    cfg = load_config(data_dir=tmp_path / "data")
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    engine = engine_for(url)
    ensure_schema(engine)
    with engine.begin() as conn:
        conn.execute(
            nwp_runs.insert(),
            {
                "run_id": RUN, "source": "ecmwf", "initialization_time": datetime(2024, 7, 1, tzinfo=UTC),
                "season": 2024, "evaluation_set": None, "mode": "replay", "alignment_method": "C1",
                "alignment_offset_hours": 0, "imd_stamp": "end_date", "status": "complete",
            },
        )  # fmt: skip
    yield SimpleNamespace(cfg=cfg, url=url, engine=engine)
    engine.dispose()


def _write_regime_domain(cfg, rows: list[dict]) -> None:
    regime_dir = cfg.data_dir / "regime"
    regime_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(regime_dir / "regime_domain.parquet")


def _write_labels(cfg, season: int) -> None:
    dates = pd.date_range("2024-07-01", periods=5, freq="D")
    lab = pd.DataFrame(
        {"year": [season] * 5, "is_base": [False] * 5, "phase_label": ["normal", "normal", "active", "active", "active"]},
        index=dates,
    )
    (cfg.data_dir / "regime").mkdir(parents=True, exist_ok=True)
    lab.to_parquet(cfg.data_dir / "regime" / "labels.parquet")


def _domain_row(**overrides) -> dict:
    row = {
        "run_id": RUN, "lead_day": 1, "season": 2024, "imd_date": pd.Timestamp("2024-07-02"),
        "p_active": 0.2, "p_normal": 0.7, "p_break": 0.1, "regime_confidence": 0.7,
        "confidence_band": "high", "lps_detected": False, "lps_settings": "v1",
        "regime_available": True, "ood_flag": False,
    }  # fmt: skip
    row.update(overrides)
    return row


def _client(world) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings_from_config(world.cfg, world.url)
    app.dependency_overrides[get_golden_dir] = lambda: world.cfg.alignment.golden_dir
    return TestClient(app, raise_server_exceptions=False)


def error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and body["error"]["message"]
    return body["error"]["message"]


def test_regime_unavailable_before_any_regime_data_exists(regime_world):
    r = _client(regime_world).get(f"{API}/regime", params={"run_id": RUN, "lead_day": 1})
    error(r, 503, "REGIME_DATA_UNAVAILABLE")


def test_regime_unknown_run_is_404_even_without_regime_data(regime_world):
    r = _client(regime_world).get(f"{API}/regime", params={"run_id": UNKNOWN_RUN, "lead_day": 1})
    error(r, 404, "RUN_NOT_FOUND")


def test_regime_bad_params_are_422(regime_world):
    client = _client(regime_world)
    assert client.get(f"{API}/regime", params={"run_id": "not-a-run-id", "lead_day": 1}).status_code == 422
    assert client.get(f"{API}/regime", params={"run_id": RUN, "lead_day": 9}).status_code == 422
    assert client.get(f"{API}/regime", params={"run_id": RUN}).status_code == 422  # lead_day required


def test_regime_happy_path(regime_world):
    _write_regime_domain(regime_world.cfg, [_domain_row()])
    r = _client(regime_world).get(f"{API}/regime", params={"run_id": RUN, "lead_day": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == RUN and body["mode"] == "replay" and body["evaluation_set"] is None
    assert body["imd_date"] == "2024-07-02"
    assert body["phase"] == {
        "active": 0.2, "normal": 0.7, "break": 0.1, "confidence": 0.7,
        "confidence_band": "high", "dominant_phase": "normal",
    }  # fmt: skip
    assert body["nearest_lps"] == {
        "present": False, "latitude": None, "longitude": None,
        "distance_km": None, "bearing_deg": None, "influence": None, "settings": "v1",
    }  # fmt: skip
    assert body["indicators"] == []
    assert body["regime_available"] is True and body["ood_flag"] is False and body["note"] is None
    # No golden/corrected files were written for this world: both means are honestly None, not fabricated.
    assert body["domain_mean_raw_mm"] is None and body["domain_mean_corrected_mm"] is None


def test_regime_unavailable_flag_produces_a_note(regime_world):
    _write_regime_domain(regime_world.cfg, [_domain_row(regime_available=False)])
    body = _client(regime_world).get(f"{API}/regime", params={"run_id": RUN, "lead_day": 1}).json()
    assert body["regime_available"] is False
    assert "fallback" in body["note"]


def test_regime_no_row_for_run_and_lead_is_404(regime_world):
    _write_regime_domain(regime_world.cfg, [_domain_row()])
    r = _client(regime_world).get(f"{API}/regime", params={"run_id": RUN, "lead_day": 2})
    error(r, 404, "REGIME_NOT_FOUND")


def test_transitions_primary_series_is_the_forecast_not_imd_labels(regime_world):
    # Forecast (lead_day=1) chain says normal throughout; IMD labels say normal->active. The two must not mix.
    rows = [
        _domain_row(lead_day=1, imd_date=pd.Timestamp(d), p_active=0.1, p_normal=0.8, p_break=0.1)
        for d in ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05"]
    ]
    _write_regime_domain(regime_world.cfg, rows)
    _write_labels(regime_world.cfg, 2024)
    body = _client(regime_world).get(f"{API}/regime/transitions", params={"run_id": RUN}).json()
    assert body["scope"] == "domain" and body["district_id"] is None
    assert all(pt["p_normal"] == 0.8 for pt in body["series"])  # primary = the model's own forecast
    assert body["events"] == []  # no transition in the forecast chain
    assert any(e["event_type"] == "PHASE:normal->active" for e in body["observed_imd"]["events"])  # IMD, separate


def test_transitions_refuses_gappy_forecast_chain(regime_world):
    rows = [
        _domain_row(lead_day=1, imd_date=pd.Timestamp(d))
        for d in ["2024-07-01", "2024-07-02", "2024-07-04"]  # 07-03 missing
    ]
    _write_regime_domain(regime_world.cfg, rows)
    _write_labels(regime_world.cfg, 2024)
    r = _client(regime_world).get(f"{API}/regime/transitions", params={"run_id": RUN})
    error(r, 503, "REGIME_SERIES_GAP")


def test_transitions_district_id_accepted_but_not_filtered_on(regime_world):
    _write_regime_domain(regime_world.cfg, [_domain_row(lead_day=1)])
    _write_labels(regime_world.cfg, 2024)
    body = _client(regime_world).get(
        f"{API}/regime/transitions", params={"run_id": RUN, "district_id": "D001"}
    ).json()
    assert body["scope"] == "domain" and body["district_id"] == "D001"


def test_regime_endpoints_are_read_only(regime_world):
    _write_regime_domain(regime_world.cfg, [_domain_row()])
    client = _client(regime_world)
    for method, path, params in (
        ("post", "/regime", {"run_id": RUN, "lead_day": 1}),
        ("post", "/regime/transitions", {"run_id": RUN}),
    ):
        assert getattr(client, method)(f"{API}{path}", params=params).status_code == 405
