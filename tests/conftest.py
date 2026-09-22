"""Shared fixtures for the Stage 1 ingestion tests. Everything is synthetic; nothing is downloaded."""

import pytest
from fastapi.testclient import TestClient

from data_pipeline.ingestion import load_config


@pytest.fixture
def cfg(tmp_path):
    """Real repository config, but with DATA_DIR pointing at a temporary folder."""
    return load_config(data_dir=tmp_path / "data")


# ---- M1 persistence / API fixtures (Stage 4) --------------------------------------------------------
# A small synthetic Stage 2/3 output (NOT real data), a database, and an API client on top of it.
# The database is SQLite by default. Set TEST_DATABASE_URL to a PostgreSQL URL to run the same tests against a
# real PostgreSQL built from database/schema.sql (see backend/README.md).

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
from shapely.geometry import box
from sqlalchemy import create_engine, text
from stage3_world import make_golden, make_grid, make_tigge_static

from backend.app.config import get_settings, settings_from_config
from backend.app.db.loader import load_all
from backend.app.db.session import engine_for
from backend.app.main import create_app

SCHEMA_SQL = Path(__file__).resolve().parent.parent / "database" / "schema.sql"
DB_KINDS = ["sqlite"] + (["postgres"] if os.environ.get("TEST_DATABASE_URL") else [])


@pytest.fixture(scope="session")
def m1_outputs(tmp_path_factory):
    """Synthetic Golden Dataset + Stage 3 outputs on disk, exactly as the real pipeline would write them."""
    from data_pipeline.alignment import save_valid_cells, write_golden
    from data_pipeline.features import build_stage3_season
    from data_pipeline.ingestion.synthetic import write_synthetic_imd_year

    cfg = load_config(data_dir=tmp_path_factory.mktemp("m1data"))
    grid = make_grid(cfg)
    golden = make_golden(cfg, grid, obs_nan=lambda run_i, cell: run_i == 1 and cell % 50 == 0)
    write_golden(golden, cfg)
    save_valid_cells(grid, cfg)
    for year in (2019, 2020):
        write_synthetic_imd_year(cfg, year, missing_fraction=0.02, seed=year)
    districts = gpd.GeoDataFrame(
        {
            "district_id": ["D001", "D002", "D003", "D004"],
            "name": ["Alpha", "Beta", "Gamma", "Delta"],
            "state": ["State A", "State A", "State B", "State C"],
            "source_id": ["101", "102", "201", None],
        },
        geometry=[
            box(74.875, 14.875, 78.875, 18.875),  # 16 x 16 valid cells
            box(69.875, 9.875, 70.375, 10.125),  # 2 cells
            box(79.6, 19.6, 81.0, 21.0),  # straddles the hole in the valid area
            box(74.875, 34.875, 75.375, 35.375),  # outside the valid area: no weights, no forecasts
        ],
        crs="EPSG:4326",
    )
    build_stage3_season(
        2024,
        [2019, 2020],
        cfg,
        holdout_seasons=[2024],
        static=make_tigge_static(),
        districts=districts,
        grid=grid,
    )
    return SimpleNamespace(cfg=cfg, grid=grid, golden=golden, districts=districts)


def _new_database(kind: str, tmp_path: Path):
    """(url, cleanup) for an empty database. PostgreSQL gets a private schema built from the real schema.sql."""
    if kind == "sqlite":
        return f"sqlite:///{tmp_path / f'{uuid.uuid4().hex}.db'}", lambda: None
    base = os.environ["TEST_DATABASE_URL"]
    schema = f"t_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}options=-csearch_path%3D{schema}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL.read_text())
    engine.dispose()

    def cleanup():
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()

    return url, cleanup


@pytest.fixture(params=DB_KINDS)
def empty_db(request, tmp_path):
    """An empty database with the M1 schema, per test. Yields (url, engine)."""
    url, cleanup = _new_database(request.param, tmp_path)
    engine = engine_for(url)
    if request.param == "sqlite":
        from backend.app.db.loader import ensure_schema

        ensure_schema(engine)
    yield url, engine
    engine.dispose()
    cleanup()


@pytest.fixture(scope="module", params=DB_KINDS)
def loaded_db(request, m1_outputs, tmp_path_factory):
    """A database loaded once with the synthetic M1 outputs and shared by the read-only API tests of a module."""
    url, cleanup = _new_database(request.param, tmp_path_factory.mktemp("db"))
    engine = engine_for(url)
    if request.param == "sqlite":
        from backend.app.db.loader import ensure_schema

        ensure_schema(engine)
    load_all(engine, m1_outputs.cfg, districts=m1_outputs.districts)
    yield SimpleNamespace(url=url, engine=engine, kind=request.param)
    engine.dispose()
    cleanup()


def make_client(url: str | None, cfg) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings_from_config(cfg, url)
    from backend.app.api.deps import get_golden_dir

    app.dependency_overrides[get_golden_dir] = lambda: cfg.alignment.golden_dir
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(loaded_db, m1_outputs):
    return make_client(loaded_db.url, m1_outputs.cfg)
