"""M1 handoff, end to end: synthetic Stage 3 output on disk -> database loader -> FastAPI -> JSON.

Every number the API returns is compared with the Parquet file that Stage 3 wrote, so this proves the actual
handoff and not just the individual layers. The data is synthetic (see tests/stage3_world.py), not real."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_client

from backend.app.db.loader import load_all
from data_pipeline.features import read_district_forecasts, read_district_history

API = "/api/v1"


def fetch_all(client, path, key, **params):
    rows, offset = [], 0
    while True:
        body = client.get(f"{API}{path}", params={**params, "limit": 5, "offset": offset}).json()
        rows += body[key]
        offset += 5
        if offset >= body["total"]:
            return rows


def test_stage3_output_reaches_the_api_unchanged(empty_db, m1_outputs):
    url, engine = empty_db
    cfg = m1_outputs.cfg
    load_all(engine, cfg, districts=m1_outputs.districts)  # the offline step
    client = make_client(url, cfg)  # the API reads what was stored

    stage3 = read_district_forecasts(cfg)
    api = pd.DataFrame(fetch_all(client, "/forecasts/districts", "districts"))  # paged, like a real client
    assert len(api) == len(stage3) == 18
    merged = stage3.merge(api, on=["run_id", "lead_day", "district_id"], suffixes=("_s3", "_api"))
    assert len(merged) == 18
    for col in ("raw_mean_mm", "observed_mean_mm"):
        s3, got = merged[f"{col}_s3"].astype(float), merged[f"{col}_api"].astype(float)
        assert np.array_equal(s3.isna(), got.isna())  # NULL stays NULL
        assert got.dropna().to_numpy() == pytest.approx(s3.dropna().to_numpy(), rel=1e-5)
    assert (merged["is_small_s3"].astype(bool) == merged["is_small_api"].astype(bool)).all()
    assert (pd.to_datetime(merged["imd_date_s3"]) == pd.to_datetime(merged["imd_date_api"])).all()

    # the same district through the per-district route, and the run list that the frontend starts from
    one = client.get(
        f"{API}/forecasts/districts/D001", params={"run_id": "tigge_ecmwf_cf_2024070100", "lead_day": 3}
    ).json()
    row = stage3[
        (stage3.run_id == "tigge_ecmwf_cf_2024070100")
        & (stage3.lead_day == 3)
        & (stage3.district_id == "D001")
    ].iloc[0]
    assert one["forecasts"][0]["raw_mean_mm"] == pytest.approx(float(row["raw_mean_mm"]), rel=1e-5)
    runs = client.get(f"{API}/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == ["tigge_ecmwf_cf_2024070100", "tigge_ecmwf_cf_2024070200"]

    # cell level comes from the Golden Parquet, district outlines and metadata from the database
    grid = client.get(f"{API}/forecasts/grid", params={"run_id": runs[0]["run_id"], "lead_day": 1}).json()
    assert grid["total"] == int(m1_outputs.grid["is_valid"].sum())
    assert len(client.get(f"{API}/map/districts").json()["features"]) == 4
    assert client.get(f"{API}/map/metadata").json()["runs"]["total"] == 2
    assert len(read_district_history(cfg)) > 0  # history is in the database too (no public endpoint at M1)


def test_a_second_load_changes_nothing_the_api_can_see(empty_db, m1_outputs):
    url, engine = empty_db
    cfg = m1_outputs.cfg
    load_all(engine, cfg, districts=m1_outputs.districts)
    client = make_client(url, cfg)
    before = [
        client.get(f"{API}{p}").json()
        for p in ("/runs", "/forecasts/districts?limit=1000", "/map/metadata", "/map/districts")
    ]
    load_all(engine, cfg, districts=m1_outputs.districts)
    after = [
        client.get(f"{API}{p}").json()
        for p in ("/runs", "/forecasts/districts?limit=1000", "/map/metadata", "/map/districts")
    ]
    assert before == after


def test_the_api_process_never_imports_pipeline_or_model_code():
    """The API only reads stored results: importing it must not pull in feature building, weights, the loader or ML."""
    code = (
        "import sys, backend.app.main\n"
        "bad = [m for m in sys.modules if m.startswith(('data_pipeline.alignment', 'data_pipeline.features', "
        "'data_pipeline.districts', 'backend.app.db.loader', 'xgboost', 'sklearn', 'geopandas'))]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
