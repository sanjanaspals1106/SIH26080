"""`/analogs` (PRD 15 F3, reduced vector).

- Parity: `services/analogs.py::find_analogs` (a local port -- the API process must not import `regime_engine`)
  picks the same cases, distances and percentiles as the real `regime_engine.analogs.AnalogFinder` when the six
  A1-A6 columns carry no information (constant), i.e. when its 10-number vector collapses to our 4.
- Behaviour: own season excluded, cases at least 5 days apart, nearest first.
- API: a small self-contained synthetic world (own tmp data dir and database), same pattern as
  `test_backend_regime.py`.
- Guard: importing the API never pulls in `regime_engine` or scikit-learn.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_golden_dir
from backend.app.config import get_settings, settings_from_config
from backend.app.db.loader import ensure_schema
from backend.app.db.session import engine_for
from backend.app.db.tables import nwp_runs
from backend.app.main import create_app
from backend.app.services.analogs import VECTOR_KEYS, find_analogs
from data_pipeline.ingestion import load_config

API = "/api/v1"


def _library(seed: int = 0, n_per_season: int = 40, seasons=(2021, 2022, 2023, 2024)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i, day in enumerate(pd.date_range(f"{season}-06-01", periods=n_per_season, freq="D")):
            p_active = float(rng.uniform(0, 0.6))
            p_break = float(rng.uniform(0, 0.4))
            lps = float(rng.integers(0, 2))
            rows.append({
                "run_id": f"tigge_ecmwf_cf_{day:%Y%m%d}00", "lead_day": 1, "season": season, "imd_date": day,
                "p_active": p_active, "p_break": p_break, "lps_present": lps,
                "lps_strength": float(rng.uniform(0.1, 0.9)) * lps,
            })  # fmt: skip
    return pd.DataFrame(rows)


QUERY = {"p_active": 0.3, "p_break": 0.05, "lps_present": 1.0, "lps_strength": 0.5}


def test_local_port_matches_regime_engine_finder():
    from regime_engine.analogs.finder import AnalogFinder

    lib = _library()
    expected_lib = lib.copy()
    for key in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        expected_lib[key] = 0.0  # no information: z-score is 0 for library and query alike
    finder = AnalogFinder(k=5, min_separation_days=5).fit_library(expected_lib)
    expected = finder.find_analogs({**QUERY, **{f"A{i}": 0.0 for i in range(1, 7)}}, lead_day=1, query_season=2024)

    actual = find_analogs(lib, QUERY, lead_day=1, query_season=2024)

    assert [a["analog_run_id"] for a in expected["analogs"]] == list(actual["run_id"])
    assert [a["distance"] for a in expected["analogs"]] == [round(d, 3) for d in actual["distance"]]
    assert [a["distance_percentile"] for a in expected["analogs"]] == [
        round(p, 1) for p in actual["distance_percentile"]
    ]


def test_excludes_own_season_and_enforces_separation():
    lib = _library()
    chosen = find_analogs(lib, QUERY, lead_day=1, query_season=2023)
    assert len(chosen) == 5
    assert 2023 not in set(chosen["season"])
    dates = sorted(chosen["imd_date"])
    assert all((b - a).days >= 5 for a, b in zip(dates, dates[1:]))
    assert list(chosen["distance"]) == sorted(chosen["distance"])  # nearest first


def test_identical_situation_ranks_first_and_lead_is_respected():
    lib = _library()
    target = lib[lib["season"] == 2022].iloc[7]
    q = {k: float(target[k]) for k in VECTOR_KEYS}
    chosen = find_analogs(lib, q, lead_day=1, query_season=2024)
    assert chosen.iloc[0]["run_id"] == target["run_id"] and chosen.iloc[0]["distance"] == pytest.approx(0.0)
    assert find_analogs(lib, q, lead_day=2, query_season=2024).empty  # library only has lead 1


def test_fewer_candidates_than_k_returns_what_exists():
    lib = _library(n_per_season=3, seasons=(2021, 2022))
    assert len(find_analogs(lib, QUERY, lead_day=1, query_season=2024)) <= 2


# ---- API -----------------------------------------------------------------------------------------------------

DISTRICT = "D010"
QUERY_RUN = "tigge_ecmwf_cf_2024070100"


@pytest.fixture
def analog_world(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data")
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    engine = engine_for(url)
    ensure_schema(engine)

    lib = _library(seasons=(2021, 2022, 2023, 2024))
    regime_dir = cfg.data_dir / "regime"
    (regime_dir / "regime_features").mkdir(parents=True)
    domain = lib.assign(
        p_normal=1 - lib["p_active"] - lib["p_break"], regime_confidence=0.7, confidence_band="high",
        regime_source="oof", ood_flag=False, regime_available=True, lps_detected=lib["lps_present"] > 0,
        n_lps=lib["lps_present"].astype(int), lps_settings="v1",
    )  # fmt: skip
    domain.to_parquet(regime_dir / "regime_domain.parquet")
    for season, part in lib.groupby("season"):
        pd.DataFrame({
            "run_id": part["run_id"], "lead_day": part["lead_day"], "cell_id": 1, "lps_strength": part["lps_strength"],
        }).to_parquet(regime_dir / "regime_features" / f"season_{season}.parquet")  # fmt: skip

    hist_dir = cfg.data_dir / "features" / "district_history"
    for season, part in lib.groupby("season"):
        (hist_dir / f"season_{season}").mkdir(parents=True)
        hist = pd.DataFrame({
            "run_id": part["run_id"], "lead_day": part["lead_day"], "district_id": DISTRICT,
            "imd_date": part["imd_date"].dt.strftime("%Y-%m-%d"), "season": season,
            "observed_mean_mm": 10.0, "observed_max_cell_mm": np.nan, "raw_mean_mm": 7.5,
        })  # fmt: skip
        other = hist.assign(district_id="D011")
        pd.concat([hist, other]).to_parquet(hist_dir / f"season_{season}" / f"district_history_{season}06.parquet")

    with engine.begin() as conn:
        for season, part in lib.groupby("season"):
            for run_id in part["run_id"]:
                conn.execute(
                    nwp_runs.insert(),
                    {
                        "run_id": run_id, "source": "ecmwf", "initialization_time": datetime(season, 6, 1, tzinfo=UTC),
                        "season": int(season), "evaluation_set": "development", "mode": "replay",
                        "alignment_method": "C1", "alignment_offset_hours": 0, "imd_stamp": "end_date",
                        "status": "complete",
                    },
                )  # fmt: skip
    yield SimpleNamespace(cfg=cfg, url=url, engine=engine, lib=lib)
    engine.dispose()


def _client(world) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings_from_config(world.cfg, world.url)
    app.dependency_overrides[get_golden_dir] = lambda: world.cfg.alignment.golden_dir
    return TestClient(app, raise_server_exceptions=False)


def _a_2024_run(world) -> str:
    return world.lib[world.lib["season"] == 2024].iloc[3]["run_id"]


def test_api_happy_path(analog_world):
    run = _a_2024_run(analog_world)
    r = _client(analog_world).get(f"{API}/analogs", params={"run_id": run, "lead_day": 1, "district_id": DISTRICT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == run and body["district_id"] == DISTRICT and body["lead_day"] == 1
    assert body["n_analogs"] == 5 and len(body["analogs"]) == 5
    assert 2024 not in {a["season"] for a in body["analogs"]}  # query's own season excluded
    assert body["library_size"] == 120
    first = body["analogs"][0]
    assert first["rank"] == 1 and first["observed_mean_mm"] == 10.0 and first["raw_mean_mm"] == 7.5
    assert first["error_observed_minus_raw_mm"] == 2.5 and body["median_error_observed_minus_raw_mm"] == 2.5
    assert first["observed_wettest_cell_mm"] is None  # honest null, never 0
    assert "corrected_mean_mm" not in first  # in-sample for development seasons: not served
    assert "A1" not in body["method"] and "phase" in body["method"]


def test_api_unknown_district_is_404(analog_world):
    r = _client(analog_world).get(
        f"{API}/analogs", params={"run_id": _a_2024_run(analog_world), "lead_day": 1, "district_id": "D999"}
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "DISTRICT_NOT_FOUND"


def test_api_unknown_run_and_bad_params(analog_world):
    client = _client(analog_world)
    unknown = client.get(
        f"{API}/analogs", params={"run_id": "tigge_ecmwf_cf_2099070100", "lead_day": 1, "district_id": DISTRICT}
    )
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "RUN_NOT_FOUND"
    run = _a_2024_run(analog_world)
    assert client.get(f"{API}/analogs", params={"run_id": run, "lead_day": 1}).status_code == 422  # district required
    assert client.get(f"{API}/analogs", params={"run_id": run, "lead_day": 7, "district_id": DISTRICT}).status_code == 422


def test_api_no_regime_row_for_lead_is_404(analog_world):
    r = _client(analog_world).get(
        f"{API}/analogs", params={"run_id": _a_2024_run(analog_world), "lead_day": 2, "district_id": DISTRICT}
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "REGIME_NOT_FOUND"


def test_api_process_never_imports_regime_engine_or_sklearn():
    code = (
        "import sys, backend.app.main, backend.app.services.analogs\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('regime_engine', 'sklearn', 'xgboost')]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
