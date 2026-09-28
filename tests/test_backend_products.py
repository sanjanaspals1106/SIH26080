"""F1 improvement summary, F5 priority table and F6 hotspots: the three endpoints, the builder rules behind the
district products, and the loader path that puts them in the database.

Self-contained: a tiny hand-made world (six cells, three districts, one run) written to a tmp data dir and a tmp
SQLite database, so nothing here depends on the shared `m1_outputs` fixtures or on real data.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_golden_dir
from backend.app.config import get_settings, settings_from_config
from backend.app.db.loader import ensure_schema, load_district_corrected, load_district_forecasts
from backend.app.db.session import engine_for
from backend.app.db.tables import cell_district_weights, district_forecasts, districts, grid_cells, nwp_runs
from backend.app.main import create_app
from data_pipeline.districts.products import FORECAST_COLUMNS, FORECAST_SCHEMA, KEY, NULLABLE_INTS, _dtypes
from data_pipeline.features.io import write_month_tables
from data_pipeline.ingestion import load_config

API = "/api/v1"
RUN = "tigge_ecmwf_cf_2024070100"
UNKNOWN_RUN = "tigge_ecmwf_cf_2099070100"

# id: (lat, lon, raw, observed, corrected, p(>=64.5))
CELLS = {
    100: (10.00, 76.00, 10.0, 30.0, 28.0, 0.8),   # hotspot cell; corrected closer
    101: (10.00, 76.25, 10.0, 20.0, 12.0, 0.6),   # hotspot cell for lead 1 (its hotspot mean is HOTSPOT_MEANS); corrected closer
    102: (10.00, 76.50, 30.0, 30.0, 20.0, 0.4),   # P below 0.5: not a hotspot; raw closer
    103: (10.25, 76.00, 5.0, 5.0, 5.0, 0.9),      # corrected mean below 15.6: not a hotspot; tie
    104: (10.25, 76.25, 5.0, np.nan, 6.0, np.nan),  # no probability, no observation
    105: (10.25, 76.50, 20.0, 40.0, 40.0, 0.9),   # hotspot cell, touches 101 diagonally; corrected closer
}
# Corrected means used for the hotspot rule at lead 1 (the improvement summary at lead 2 uses the CELLS values).
HOTSPOT_MEANS = {100: 30.0, 101: 20.0, 102: 30.0, 103: 10.0, 104: 6.0, 105: 40.0}


def _write_cell_files(cfg, lead_day_rows: dict[int, dict[int, tuple]]) -> None:
    """lead -> cell -> (raw, obs, corrected_mean, p64). Golden and corrected parquet for RUN."""
    gold, cor = [], []
    for lead, cells in lead_day_rows.items():
        for cid, (raw, obs, mean, p) in cells.items():
            lat, lon = CELLS[cid][0], CELLS[cid][1]
            gold.append({
                "run_id": RUN, "lead_day": lead, "cell_id": cid, "latitude": lat, "longitude": lon,
                "imd_date": date(2024, 7, 1 + lead), "rain_mm": raw, "obs_mm": obs,
            })  # fmt: skip
            cor.append({
                "run_id": RUN, "lead_day": lead, "cell_id": cid, "corrected_mean_mm": mean,
                "q10_mm": mean * 0.5, "q50_mm": mean, "q90_mm": mean * 1.5, "p_ge_64_5": p, "p_ge_115_6": 0.01,
            })  # fmt: skip
    gpath = cfg.alignment.golden_dir / "season_2024" / "golden_202407.parquet"
    cpath = cfg.data_dir / "serving" / "corrected" / "season_2024" / "corrected_202407.parquet"
    for path, rows in ((gpath, gold), (cpath, cor)):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path)


@pytest.fixture
def world(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data")
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    engine = engine_for(url)
    ensure_schema(engine)
    lead1 = {cid: (c[2], c[3], HOTSPOT_MEANS[cid], c[5]) for cid, c in CELLS.items()}
    # Improvement uses the CELLS corrected values; hotspots use HOTSPOT_MEANS. Two leads keep them independent.
    lead2 = {cid: (c[2], c[3], c[4], 0.1) for cid, c in CELLS.items()}  # every P low: no hotspot at lead 2
    _write_cell_files(cfg, {1: lead1, 2: lead2})
    with engine.begin() as conn:
        conn.execute(nwp_runs.insert(), {
            "run_id": RUN, "source": "ecmwf", "initialization_time": datetime(2024, 7, 1, tzinfo=UTC), "season": 2024,
            "evaluation_set": "development", "mode": "replay", "alignment_method": "C1", "alignment_offset_hours": 0,
            "imd_stamp": "end_date", "status": "complete",
        })  # fmt: skip
        conn.execute(grid_cells.insert(), [
            {"cell_id": cid, "latitude": c[0], "longitude": c[1], "is_valid": True} for cid, c in CELLS.items()
        ])  # fmt: skip
        conn.execute(districts.insert(), [
            {"district_id": "D1", "name": "Alpha", "state": "S1"},
            {"district_id": "D2", "name": "Beta", "state": "S1"},
            {"district_id": "D3", "name": "Gamma", "state": "S2"},
        ])  # fmt: skip
        # D1: 100 and 101 main. D2: 105 and 103 main. D3 has NO main cell (every weight < 0.05): all its cells count.
        conn.execute(cell_district_weights.insert(), [
            {"cell_id": 100, "district_id": "D1", "area_weight": 0.5, "is_main": True},
            {"cell_id": 101, "district_id": "D1", "area_weight": 0.5, "is_main": True},
            {"cell_id": 105, "district_id": "D2", "area_weight": 0.5, "is_main": True},
            {"cell_id": 103, "district_id": "D2", "area_weight": 0.5, "is_main": True},
            {"cell_id": 104, "district_id": "D3", "area_weight": 0.04, "is_main": False},
            {"cell_id": 105, "district_id": "D3", "area_weight": 0.03, "is_main": False},
        ])  # fmt: skip
        base = {"run_id": RUN, "lead_day": 1, "imd_date": date(2024, 7, 2)}
        conn.execute(district_forecasts.insert(), [
            {**base, "district_id": "D1", "raw_mean_mm": 10.0, "corrected_mean_mm": 20.0, "observed_mean_mm": 25.0,
             "attention_level": "WATCH", "priority_rank": 2, "product_type": "regime_aware_ml"},
            {**base, "district_id": "D2", "raw_mean_mm": 12.0, "corrected_mean_mm": 15.0, "observed_mean_mm": 12.0,
             "attention_level": "HIGH", "priority_rank": 1, "product_type": "regime_aware_ml"},
            {**base, "district_id": "D3", "raw_mean_mm": 5.0, "corrected_mean_mm": 6.0, "observed_mean_mm": None,
             "attention_level": "NORMAL", "priority_rank": 3, "product_type": "regime_aware_ml"},
        ])  # fmt: skip
    yield SimpleNamespace(cfg=cfg, url=url, engine=engine, tmp=tmp_path)
    engine.dispose()


def _client(world) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings_from_config(world.cfg, world.url)
    app.dependency_overrides[get_golden_dir] = lambda: world.cfg.alignment.golden_dir
    return TestClient(app, raise_server_exceptions=False)


def error(response, status, code):
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code


# ---- F6 hotspots -----------------------------------------------------------------------------------------------


def test_hotspots_group_touching_cells_and_apply_the_prd_rule(world):
    r = _client(world).get(f"{API}/hotspots", params={"run_id": RUN, "lead_day": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["threshold_mm"] == 64.5 and body["probability_min"] == 0.5 and body["corrected_mean_min_mm"] == 15.6
    assert body["evaluation_set"] == "development" and body["imd_date"] == "2024-07-02"
    assert len(body["hotspots"]) == 1  # 100, 101 side by side + 105 touching 101 diagonally = one group
    h = body["hotspots"][0]
    # excluded: 102 (P 0.4), 103 (mean 10 < 15.6), 104 (no probability)
    assert h["hotspot_id"] == 1 and h["n_cells"] == 3
    assert h["max_probability"] == 0.9 and h["max_corrected_mean_mm"] == 40.0
    assert h["centroid"] == {"lat": 10.0833, "lon": 76.25}
    assert h["outline"]["type"] in ("Polygon", "MultiPolygon") and h["outline"]["coordinates"]


def test_hotspot_districts_are_those_with_a_main_cell_in_the_group_including_the_no_main_cell_fallback(world):
    h = _client(world).get(f"{API}/hotspots", params={"run_id": RUN, "lead_day": 1}).json()["hotspots"][0]
    # D1 (main 100/101), D2 (main 105); D3 has no main cell at all so its cell 105 counts. 103 and 104 are not in the group.
    assert h["district_ids"] == ["D1", "D2", "D3"]


def test_no_hotspot_when_no_cell_qualifies(world):
    body = _client(world).get(f"{API}/hotspots", params={"run_id": RUN, "lead_day": 2}).json()
    assert body["hotspots"] == [] and body["lead_day"] == 2


def test_hotspot_errors(world):
    c = _client(world)
    error(c.get(f"{API}/hotspots", params={"run_id": UNKNOWN_RUN, "lead_day": 1}), 404, "RUN_NOT_FOUND")
    error(c.get(f"{API}/hotspots", params={"run_id": RUN, "lead_day": 3}), 503, "GRID_DATA_UNAVAILABLE")
    assert c.get(f"{API}/hotspots", params={"run_id": RUN}).status_code == 422
    assert c.get(f"{API}/hotspots", params={"run_id": "bad", "lead_day": 1}).status_code == 422


def test_hotspots_report_unavailable_when_the_64_5_model_is_missing(world):
    path = world.cfg.data_dir / "verification" / "m3_dev" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"availability": {"p_64_5_available": false}}')
    error(_client(world).get(f"{API}/hotspots", params={"run_id": RUN, "lead_day": 1}), 503, "HOTSPOTS_UNAVAILABLE")


# ---- F1 improvement summary ------------------------------------------------------------------------------------


def test_improvement_summary_counts_only_rows_with_all_three_values(world):
    r = _client(world).get(f"{API}/forecasts/improvement-summary", params={"run_id": RUN, "lead_day": 2})
    assert r.status_code == 200, r.text
    b = r.json()
    # lead 2 carries the CELLS corrected values. 104 has no observation and is excluded; 103 is a tie.
    assert b["total_valid_cells"] == 5
    assert (b["cells_corrected_closer"], b["cells_raw_closer"], b["cells_no_change"]) == (3, 1, 1)
    assert b["imd_date"] == "2024-07-03" and b["note"] == "One day only, not evidence."
    assert b["evaluation_set"] == "development"


def test_improvement_summary_districts_come_from_the_database(world):
    b = _client(world).get(f"{API}/forecasts/improvement-summary", params={"run_id": RUN, "lead_day": 1}).json()
    # D1 corrected closer (|20-25| < |10-25|), D2 raw closer (|12-12| < |15-12|), D3 has no observation: excluded.
    assert (b["total_districts"], b["districts_corrected_closer"], b["districts_raw_closer"]) == (2, 1, 1)


def test_improvement_summary_errors(world):
    c = _client(world)
    error(c.get(f"{API}/forecasts/improvement-summary", params={"run_id": UNKNOWN_RUN, "lead_day": 1}), 404, "RUN_NOT_FOUND")
    error(c.get(f"{API}/forecasts/improvement-summary", params={"run_id": RUN, "lead_day": 3}), 503, "GRID_DATA_UNAVAILABLE")


# ---- F5 priority table -----------------------------------------------------------------------------------------


def test_priority_table_is_in_rank_order_with_the_note(world):
    r = _client(world).get(f"{API}/districts/priority", params={"run_id": RUN, "lead_day": 1})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["note"] == "Model-based attention level. Not an official warning."
    assert b["total"] == 3 and b["imd_date"] == "2024-07-02" and b["lead_day"] == 1
    assert [(d["priority_rank"], d["district_id"], d["attention_level"]) for d in b["districts"]] == [
        (1, "D2", "HIGH"), (2, "D1", "WATCH"), (3, "D3", "NORMAL"),
    ]  # fmt: skip
    assert {d["product_type"] for d in b["districts"]} == {"regime_aware_ml"}


def test_priority_table_state_filter_and_errors(world):
    c = _client(world)
    b = c.get(f"{API}/districts/priority", params={"run_id": RUN, "lead_day": 1, "state": "S2"}).json()
    assert [d["district_id"] for d in b["districts"]] == ["D3"]
    error(c.get(f"{API}/districts/priority", params={"run_id": UNKNOWN_RUN, "lead_day": 1}), 404, "RUN_NOT_FOUND")
    error(c.get(f"{API}/districts/priority", params={"run_id": RUN, "lead_day": 2}), 404, "FORECAST_NOT_FOUND")


# ---- the builder: no-main-cell districts and the priority frame ------------------------------------------------


@pytest.fixture(scope="module")
def builder():
    """scripts/ is not a package: load the module by path (its top-level imports pull in the model code, which is
    fine in the test process)."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_district_corrected.py"
    spec = importlib.util.spec_from_file_location("build_district_corrected", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_forecasts(rows: list[dict]) -> pd.DataFrame:
    """A district_forecasts frame in the M1 schema with the 'later stage' columns empty, as M1 writes it."""
    df = pd.DataFrame(rows)
    for col in FORECAST_COLUMNS:
        if col not in df:
            df[col] = np.nan
    df["imd_date"] = pd.Timestamp("2024-07-02")
    df["season"], df["alignment_offset_hours"], df["imd_stamp"] = 2024, 0, "end_date"
    df["is_small"], df["n_main_cells"], df["n_effective_cells"] = False, 0, 4.0
    df["raw_mean_mm"] = df["raw_mean_mm"].fillna(1.0)
    return df[FORECAST_COLUMNS].astype(_dtypes(FORECAST_SCHEMA)).astype(dict.fromkeys(NULLABLE_INTS, "Int32"))


def test_districts_without_a_main_cell_still_get_a_wettest_cell_and_probabilities(builder):
    weights = pd.DataFrame({
        "district_id": ["D1", "D1", "D3", "D3"], "cell_id": [1, 2, 3, 4],
        "area_weight": [0.5, 0.5, 0.5, 0.5], "is_main": [True, True, False, False],
    })  # fmt: skip
    corrected = pd.DataFrame({
        "run_id": RUN, "lead_day": 1, "cell_id": [1, 2, 3, 4], "corrected_mean_mm": [5.0, 9.0, 4.0, 30.0],
        "q10_mm": [1, 2, 1, 8.0], "q50_mm": [4, 8, 3, 25.0], "q90_mm": [9, 15, 7, 50.0],
        "p_ge_64_5": [0.1, 0.2, 0.3, 0.7], "p_ge_115_6": [0.0, 0.0, 0.0, 0.2],
    })  # fmt: skip
    base = _base_forecasts([{"run_id": RUN, "lead_day": 1, "district_id": d} for d in ("D1", "D3")])
    out = builder.fill_forecasts(base, corrected, weights).set_index("district_id")
    assert out.loc["D1", "wettest_cell_id"] == 2 and out.loc["D1", "heavy_prob_max_cell"] == pytest.approx(0.2)
    # D3 has no main cell: previously NULL, now the wettest of all its cells
    assert out.loc["D3", "wettest_cell_id"] == 4
    assert out.loc["D3", "wettest_cell_mean_mm"] == pytest.approx(30.0)
    assert out.loc["D3", "wettest_cell_q90_mm"] == pytest.approx(50.0)
    assert out.loc["D3", "heavy_prob_max_cell"] == pytest.approx(0.7)
    assert out.loc["D3", "very_heavy_prob_max_cell"] == pytest.approx(0.2)


def test_priority_frame_ranks_every_run_and_lead_by_the_prd_rule(builder):
    fc = pd.DataFrame({
        "run_id": RUN, "lead_day": [1, 1, 1, 2],
        "district_id": ["D1", "D2", "D3", "D1"],
        "corrected_mean_mm": [5.0, 5.0, 20.0, 5.0], "wettest_cell_mean_mm": [8.0, 9.0, 30.0, 8.0],
        "heavy_prob_max_cell": [0.10, 0.55, 0.25, 0.9], "very_heavy_prob_max_cell": [0.0, 0.0, 0.0, 0.0],
        "heavy_area_fraction_expected": [0.0, 0.1, 0.05, 0.3], "raw_mean_mm": [4.0, 4.0, 4.0, 4.0],
        "is_small": [False] * 4,
    })  # fmt: skip
    names = pd.DataFrame({"district_id": ["D1", "D2", "D3"], "name": ["Alpha", "Beta", "Gamma"], "state": ["S"] * 3})
    out = builder.priority_frame(fc, names, is_64_5_available=True)
    one = out[out["lead_day"] == 1].set_index("district_id")
    assert one.loc["D2", "attention_level"] == "HIGH" and one.loc["D2", "priority_rank"] == 1  # P >= 0.50
    assert one.loc["D3", "attention_level"] == "WATCH" and one.loc["D3", "priority_rank"] == 2  # P >= 0.20
    assert one.loc["D1", "attention_level"] == "NORMAL" and one.loc["D1", "priority_rank"] == 3
    assert out[out["lead_day"] == 2]["priority_rank"].tolist() == [1]  # ranks restart per (run, lead)
    assert set(out["product_type"]) == {"regime_aware_ml"}


def test_priority_frame_refuses_to_rank_districts_with_gaps(builder):
    fc = pd.DataFrame({
        "run_id": [RUN], "lead_day": [1], "district_id": ["D1"], "corrected_mean_mm": [np.nan],
        "wettest_cell_mean_mm": [1.0], "heavy_prob_max_cell": [0.1], "very_heavy_prob_max_cell": [0.0],
        "heavy_area_fraction_expected": [0.0], "raw_mean_mm": [1.0], "is_small": [False],
    })  # fmt: skip
    names = pd.DataFrame({"district_id": ["D1"], "name": ["Alpha"], "state": ["S"]})
    with pytest.raises(SystemExit):
        builder.priority_frame(fc, names, is_64_5_available=True)


# ---- the loader: priority side table -> database ---------------------------------------------------------------


def test_load_district_corrected_writes_priority_columns_from_the_side_table(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data")
    engine = engine_for(f"sqlite:///{tmp_path / 'db.sqlite'}")
    ensure_schema(engine)
    with engine.begin() as conn:
        conn.execute(nwp_runs.insert(), {
            "run_id": RUN, "source": "ecmwf", "initialization_time": datetime(2024, 7, 1, tzinfo=UTC), "season": 2024,
            "mode": "replay",
        })  # fmt: skip
        conn.execute(districts.insert(), [
            {"district_id": "D1", "name": "Alpha", "state": "S1"}, {"district_id": "D2", "name": "Beta", "state": "S1"},
        ])  # fmt: skip
    rows = [{"run_id": RUN, "lead_day": 1, "district_id": d, "corrected_mean_mm": 7.0, "wettest_cell_id": 9,
             "wettest_cell_mean_mm": 8.0, "heavy_prob_max_cell": 0.3} for d in ("D1", "D2")]  # fmt: skip
    df = _base_forecasts(rows)
    write_month_tables(df, "district_forecasts", FORECAST_SCHEMA, KEY, df["run_id"].str[-10:-4], cfg)
    load_district_forecasts(engine, cfg)

    # Without the side table the priority columns are left alone (nothing is blanked).
    load_district_corrected(engine, cfg)
    with engine.connect() as conn:
        got = conn.execute(district_forecasts.select().order_by(district_forecasts.c.district_id)).all()
    assert [r.attention_level for r in got] == [None, None] and [r.wettest_cell_id for r in got] == [9, 9]

    side = pd.DataFrame({
        "run_id": RUN, "lead_day": np.int8(1), "district_id": ["D1", "D2"], "attention_level": ["HIGH", "NORMAL"],
        "priority_rank": np.array([1, 2], dtype="int32"), "product_type": "regime_aware_ml",
    })  # fmt: skip
    path = cfg.data_dir / "serving" / "priority" / "season_2024" / "priority_202407.parquet"
    path.parent.mkdir(parents=True)
    side.to_parquet(path, index=False)
    assert load_district_corrected(engine, cfg) == 2
    with engine.connect() as conn:
        got = conn.execute(district_forecasts.select().order_by(district_forecasts.c.district_id)).all()
    assert [(r.attention_level, r.priority_rank, r.product_type) for r in got] == [
        ("HIGH", 1, "regime_aware_ml"), ("NORMAL", 2, "regime_aware_ml"),
    ]  # fmt: skip
    engine.dispose()
