"""The offline loader: Stage 2/3 Parquet -> M1 tables. Rerunnable, upsert-based, schema-faithful."""

import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from backend.app.db import tables as t
from backend.app.db.loader import (
    LoaderError,
    ensure_schema,
    load_all,
    load_district_rows,
    load_weights,
    outline_feature,
)
from backend.app.db.session import make_engine
from backend.app.services.runs import as_utc
from data_pipeline.features import read_district_forecasts, read_district_history
from data_pipeline.ingestion import MissingInputError

SCHEMA_SQL = Path(__file__).resolve().parent.parent / "database" / "schema.sql"


def count(engine, table):
    with engine.connect() as c:
        return c.execute(select(func.count()).select_from(table)).scalar_one()


@pytest.fixture
def loaded(empty_db, m1_outputs):
    url, engine = empty_db
    report = load_all(engine, m1_outputs.cfg, districts=m1_outputs.districts)
    return engine, report


# ---- schema.sql and the table definitions agree ---------------------------------------------------------


def columns_from_ddl(sql: str, table: str) -> tuple[list[str], list[str]]:
    body = re.search(rf"CREATE TABLE {table} \((.*?)\n\);", sql, re.S).group(1)
    body = re.sub(r"--[^\n]*", "", body)
    items, depth, cur = [], 0, ""
    for ch in body:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    items.append(cur.strip())
    cols, pk = [], []
    for item in items:
        if item.upper().startswith(("PRIMARY KEY", "UNIQUE")):
            if item.upper().startswith("PRIMARY KEY"):
                pk = [c.strip() for c in item[item.index("(") + 1 : item.rindex(")")].split(",")]
            continue
        name = item.split()[0]
        cols.append(name)
        if "PRIMARY KEY" in item.upper():
            pk = [name]
    return cols, pk


@pytest.mark.parametrize("table", t.M1_TABLES, ids=lambda x: x.name)
def test_table_definitions_match_schema_sql(table):
    cols, pk = columns_from_ddl(SCHEMA_SQL.read_text(), table.name)
    assert [c.name for c in table.columns] == cols
    assert [c.name for c in table.primary_key.columns] == pk


def test_schema_sql_adds_the_run_foreign_keys_and_indexes():
    sql = SCHEMA_SQL.read_text()
    for fragment in (
        "fk_district_forecasts_run",
        "fk_district_history_run",
        "idx_district_forecasts_imd_date",
        "idx_district_forecasts_district_date",
    ):
        assert fragment in sql
    assert {i.name for i in t.district_forecasts.indexes} >= {
        "idx_district_forecasts_imd_date",
        "idx_district_forecasts_run_district",
    }


# ---- loading ---------------------------------------------------------------------------------------------


def test_load_writes_every_m1_table(loaded, m1_outputs):
    engine, report = loaded
    cfg = m1_outputs.cfg
    weights = pd.read_parquet(cfg.features.dir / "districts" / "cell_district_weights.parquet")
    fc = read_district_forecasts(cfg)
    assert report.rows == {
        "grid_cells": 129 * 135, "districts": 4, "cell_district_weights": len(weights), "nwp_runs": 2,
        "district_forecasts": len(fc), "district_history": len(read_district_history(cfg)),
    }  # fmt: skip
    for table, n in (
        (t.grid_cells, 129 * 135),
        (t.districts, 4),
        (t.nwp_runs, 2),
        (t.cell_district_weights, len(weights)),
        (t.district_forecasts, len(fc)),
    ):
        assert count(engine, table) == n
    assert len(fc) == 2 * 3 * 3  # D004 has no valid cell, so no forecasts


def test_grid_cells_carry_validity_and_static_geography(loaded, m1_outputs):
    engine, _ = loaded
    with engine.connect() as c:
        rows = {r.cell_id: r for r in c.execute(select(t.grid_cells))}
    grid = m1_outputs.grid.set_index("cell_id")
    assert sum(r.is_valid for r in rows.values()) == int(grid["is_valid"].sum())
    cell = int(grid.index[grid["is_valid"]][10])
    assert (rows[cell].latitude, rows[cell].longitude) == (
        grid.loc[cell, "latitude"],
        grid.loc[cell, "longitude"],
    )  # T7: same cell, same place
    assert (
        rows[cell].elevation_m is not None and rows[cell].slope > 0 and rows[cell].dist_coast_km is not None
    )
    assert rows[cell].region_code is None  # not produced by M1's pipeline


def test_districts_have_simplified_outlines_and_summary_numbers(loaded):
    engine, _ = loaded
    with engine.connect() as c:
        rows = {r.district_id: r for r in c.execute(select(t.districts))}
    assert set(rows) == {"D001", "D002", "D003", "D004"}
    a = rows["D001"]
    assert a.name == "Alpha" and a.state == "State A" and a.source_year == 2011
    assert a.geojson["type"] == "Feature" and a.geojson["geometry"]["type"] == "Polygon"
    assert a.geojson["properties"] == {"source_id": "101"}
    assert (
        a.is_small is False
        and a.n_effective_cells == pytest.approx(256, rel=0.02)
        and a.centroid_lat == pytest.approx(16.9, abs=0.1)
    )
    assert rows["D002"].is_small is True
    d4 = rows["D004"]  # loaded for the map even though no valid cell belongs to it
    assert d4.n_effective_cells is None and d4.is_small is None and d4.geojson["properties"] == {}


def test_weights_match_the_stage3_table(loaded, m1_outputs):
    engine, _ = loaded
    weights = pd.read_parquet(m1_outputs.cfg.features.dir / "districts" / "cell_district_weights.parquet")
    with engine.connect() as c:
        db = pd.DataFrame(
            c.execute(select(t.cell_district_weights)).all(),
            columns=["cell_id", "district_id", "area_weight", "is_main"],
        )
    merged = weights.merge(db, on=["district_id", "cell_id"], suffixes=("", "_db"))
    assert len(merged) == len(weights) == len(db)
    assert merged["area_weight_db"].to_numpy() == pytest.approx(merged["area_weight"].to_numpy(), rel=1e-5)
    assert (merged["is_main"] == merged["is_main_db"].astype(bool)).all()
    assert db.groupby("district_id")["area_weight"].sum().to_numpy() == pytest.approx(1.0, abs=1e-5)


def test_runs_are_described_from_the_golden_dataset(loaded):
    engine, _ = loaded
    with engine.connect() as c:
        runs = {r.run_id: r for r in c.execute(select(t.nwp_runs))}
    r = runs["tigge_ecmwf_cf_2024070100"]
    assert r.source == "ECMWF-TIGGE-control" and r.season == 2024 and r.mode == "replay"
    assert (r.alignment_method, r.alignment_offset_hours, r.imd_stamp) == ("C1", 0, "end_date")
    assert as_utc(r.initialization_time) == datetime(
        2024, 7, 1, 0, 0, tzinfo=UTC
    )  # the instant is right, whatever the session zone
    assert (
        r.evaluation_set is None and r.status is None
    )  # set by the protocol / later stages, not invented here


def test_district_forecasts_match_the_parquet_and_keep_later_stage_fields_null(loaded, m1_outputs):
    engine, _ = loaded
    fc = read_district_forecasts(m1_outputs.cfg).set_index(["run_id", "lead_day", "district_id"])
    with engine.connect() as c:
        rows = c.execute(select(t.district_forecasts)).all()
    assert len(rows) == len(fc)
    for r in rows:
        src = fc.loc[(r.run_id, r.lead_day, r.district_id)]
        assert r.raw_mean_mm == pytest.approx(float(src["raw_mean_mm"]), rel=1e-6)
        assert (r.observed_mean_mm is None) == bool(
            pd.isna(src["observed_mean_mm"])
        )  # NaN became NULL, not 0
        assert bool(r.is_small) == bool(src["is_small"]) and str(r.imd_date) == str(src["imd_date"].date())
        for later in ["corrected_mean_mm", "wettest_cell_id", "wettest_cell_mean_mm", "wettest_cell_q10_mm", "wettest_cell_q50_mm",
                      "wettest_cell_q90_mm", "heavy_prob_max_cell", "very_heavy_prob_max_cell", "heavy_area_fraction_expected",
                      "very_heavy_area_fraction_expected", "attention_level", "priority_rank", "product_type", "fallback_used",
                      "fallback_reason", "model_version_id"]:  # fmt: skip
            assert getattr(r, later) is None, later
    assert any(r.observed_mean_mm is None for r in rows)  # the fixture has missing observations


def test_district_history_keeps_only_observed_rows_and_null_later_fields(loaded, m1_outputs):
    engine, _ = loaded
    expected = read_district_history(m1_outputs.cfg)
    with engine.connect() as c:
        rows = c.execute(select(t.district_history)).all()
    assert len(rows) == len(expected) > 0 and all(r.observed_mean_mm is not None for r in rows)
    assert all(
        r.corrected_mean_mm is None and r.prediction_source is None and r.phase is None and r.lps_near is None
        for r in rows
    )


# ---- idempotency and ownership -----------------------------------------------------------------------------


def test_loading_twice_gives_the_same_database(empty_db, m1_outputs):
    _, engine = empty_db
    first = load_all(engine, m1_outputs.cfg, districts=m1_outputs.districts)
    snapshot = {tb.name: count(engine, tb) for tb in t.M1_TABLES}
    second = load_all(engine, m1_outputs.cfg, districts=m1_outputs.districts)
    assert first.rows == second.rows
    assert {tb.name: count(engine, tb) for tb in t.M1_TABLES} == snapshot  # no duplicates, no growth


def test_reload_repairs_m1_columns_but_never_overwrites_later_stage_values(loaded, m1_outputs):
    engine, _ = loaded
    key = (
        (t.district_forecasts.c.run_id == "tigge_ecmwf_cf_2024070100")
        & (t.district_forecasts.c.district_id == "D002")
        & (t.district_forecasts.c.lead_day == 1)
    )
    with engine.begin() as c:  # a later stage fills its columns; someone also damages an M1 column
        c.execute(
            update(t.district_forecasts)
            .where(key)
            .values(corrected_mean_mm=99.0, attention_level="HIGH", raw_mean_mm=-1.0)
        )
        c.execute(
            update(t.district_history)
            .where(t.district_history.c.district_id == "D002")
            .values(phase="active", lps_near=True)
        )
    load_all(engine, m1_outputs.cfg, districts=m1_outputs.districts)
    with engine.connect() as c:
        row = c.execute(select(t.district_forecasts).where(key)).one()
        hist = c.execute(select(t.district_history).where(t.district_history.c.district_id == "D002")).all()
    assert (
        row.corrected_mean_mm == 99.0 and row.attention_level == "HIGH"
    )  # later-stage values survive the reload
    assert row.raw_mean_mm > 0  # the M1 column was restored
    assert hist and all(h.phase == "active" and h.lps_near is True for h in hist)


def test_weights_are_replaced_as_a_whole(loaded, m1_outputs):
    engine, _ = loaded
    with engine.begin() as c:
        c.execute(
            insert(t.cell_district_weights).values(
                cell_id=0, district_id="D004", area_weight=1.0, is_main=True
            )
        )
    before = count(engine, t.cell_district_weights)
    load_weights(engine, m1_outputs.cfg)
    assert count(engine, t.cell_district_weights) == before - 1  # the stale row is gone


def test_database_refuses_forecasts_for_unknown_runs_or_districts(loaded):
    engine, _ = loaded
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(
            insert(t.district_forecasts).values(
                run_id="tigge_ecmwf_cf_1999010100", lead_day=1, district_id="D001"
            )
        )
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(
            insert(t.district_forecasts).values(
                run_id="tigge_ecmwf_cf_2024070100", lead_day=1, district_id="NOPE"
            )
        )


def test_the_identity_of_a_district_forecast_is_unique(loaded):
    engine, _ = loaded
    with engine.connect() as c:
        row = c.execute(
            select(
                t.district_forecasts.c.run_id,
                t.district_forecasts.c.lead_day,
                t.district_forecasts.c.district_id,
            ).limit(1)
        ).one()
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(
            insert(t.district_forecasts).values(
                run_id=row.run_id, lead_day=row.lead_day, district_id=row.district_id
            )
        )


# ---- failures and helpers ---------------------------------------------------------------------------------------


def test_missing_inputs_fail_clearly(empty_db, m1_outputs, tmp_path):
    from data_pipeline.ingestion import load_config

    _, engine = empty_db
    bare = load_config(data_dir=tmp_path / "nothing")
    with pytest.raises(MissingInputError, match="Valid-cell mask not found"):
        load_all(engine, bare, districts=m1_outputs.districts)
    with pytest.raises(MissingInputError, match="No district file is configured"):  # nothing is substituted
        load_district_rows(engine, m1_outputs.cfg)


def test_missing_schema_is_reported_with_the_fix(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    with pytest.raises(LoaderError, match="schema.sql"):
        ensure_schema(engine, create_if_sqlite=False)
    ensure_schema(engine)  # SQLite (tests, demos): created from the table definitions
    assert count(engine, t.nwp_runs) == 0
    with engine.connect() as c:
        assert c.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_outline_is_simplified_and_rounded():
    from shapely.geometry import Polygon

    ring = [(70 + i * 0.001, 10 + (0.0000123 if i % 2 else 0.0)) for i in range(50)] + [
        (72.0, 12.123456789),
        (70.0, 12.0),
    ]
    feature = outline_feature(Polygon(ring), "X1", 0.005)
    coords = feature["geometry"]["coordinates"][0]
    assert len(coords) < len(ring) and feature["properties"] == {"source_id": "X1"}
    assert all(len(str(v).split(".")[-1]) <= 4 for pt in coords for v in pt)
    assert outline_feature(Polygon(ring), None, 0.005)["properties"] == {}
