"""The M1 API: every endpoint, filters, pagination, 404s, bad parameters, read-only, no leaking of DB errors."""

import re

import numpy as np
import pytest
from tests.conftest import make_client

from data_pipeline.features import read_district_forecasts

RUN1, RUN2 = "tigge_ecmwf_cf_2024070100", "tigge_ecmwf_cf_2024070200"
API = "/api/v1"
LATER_STAGE_FIELDS = [
    "corrected_mean_mm", "wettest_cell_id", "wettest_cell_mean_mm", "wettest_cell_q10_mm", "wettest_cell_q50_mm",
    "wettest_cell_q90_mm", "heavy_prob_max_cell", "very_heavy_prob_max_cell", "heavy_area_fraction_expected",
    "very_heavy_area_fraction_expected", "attention_level", "priority_rank", "product_type",
]  # fmt: skip


def error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert (
        set(body) == {"error"} and body["error"]["code"] == code and body["error"]["message"]
    )  # PRD 18.3 error format
    return body["error"]["message"]


# ---- health ------------------------------------------------------------------------------------------------


def test_health_reports_the_database(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "mode": "replay", "version": "0.1.0", "database": "ok"}


def test_health_without_a_database_url(m1_outputs):
    r = make_client(None, m1_outputs.cfg).get(f"{API}/health")
    assert (
        r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["database"] == "not_configured"
    )


def test_health_when_the_database_is_down(m1_outputs, tmp_path):
    r = make_client(f"sqlite:///{tmp_path / 'no' / 'such' / 'dir.db'}", m1_outputs.cfg).get(f"{API}/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["database"] == "unavailable"


# ---- runs --------------------------------------------------------------------------------------------------


def test_runs_list(client):
    body = client.get(f"{API}/runs").json()
    assert (body["total"], body["limit"], body["offset"], body["mode"]) == (2, 100, 0, "replay")
    first, second = body["runs"]
    assert (first["run_id"], second["run_id"]) == (RUN1, RUN2)  # oldest first
    assert first["initialization_time"].startswith("2024-07-01T00:00:00") and first["season"] == 2024
    assert (
        first["lead_days"] == [1, 2, 3]
        and first["first_imd_date"] == "2024-07-02"
        and first["last_imd_date"] == "2024-07-04"
    )
    assert (first["alignment_method"], first["alignment_offset_hours"], first["imd_stamp"]) == (
        "C1",
        0,
        "end_date",
    )
    assert first["evaluation_set"] is None and first["n_districts"] is None


def test_runs_filter_and_pagination(client):
    assert client.get(f"{API}/runs?season=2024").json()["total"] == 2
    assert client.get(f"{API}/runs?season=1999").json() == {
        "total": 0,
        "limit": 100,
        "offset": 0,
        "mode": "replay",
        "runs": [],
    }
    page = client.get(f"{API}/runs?limit=1&offset=1").json()
    assert page["total"] == 2 and [r["run_id"] for r in page["runs"]] == [RUN2]


def test_run_detail(client):
    body = client.get(f"{API}/runs/{RUN1}").json()
    assert body["run_id"] == RUN1 and body["n_districts"] == 3 and body["lead_days"] == [1, 2, 3]
    assert body["mode"] == "replay" and "created_at" not in body


def test_unknown_run_is_404_and_malformed_run_is_422(client):
    error(client.get(f"{API}/runs/tigge_ecmwf_cf_1999010100"), 404, "RUN_NOT_FOUND")
    error(client.get(f"{API}/runs/not-a-run"), 422, "INVALID_PARAMETER")


# ---- district forecasts ------------------------------------------------------------------------------------


def test_district_forecasts_for_a_run_and_lead(client, m1_outputs):
    body = client.get(f"{API}/forecasts/districts", params={"run_id": RUN1, "lead_day": 1}).json()
    assert (
        body["total"] == 3
        and body["run_id"] == RUN1
        and body["mode"] == "replay"
        and body["evaluation_set"] is None
    )
    fc = read_district_forecasts(m1_outputs.cfg).set_index(["run_id", "lead_day", "district_id"])
    assert [d["district_id"] for d in body["districts"]] == ["D001", "D002", "D003"]
    for d in body["districts"]:
        src = fc.loc[(RUN1, 1, d["district_id"])]
        assert d["raw_mean_mm"] == pytest.approx(float(src["raw_mean_mm"]), rel=1e-5)
        assert d["observed_mean_mm"] == pytest.approx(float(src["observed_mean_mm"]), rel=1e-5)
        assert d["is_small"] == bool(src["is_small"]) and d["imd_date"] == "2024-07-02" and d["lead_day"] == 1
    alpha = body["districts"][0]
    assert (alpha["district_name"], alpha["state"]) == ("Alpha", "State A")


def test_fields_owned_by_later_stages_are_null_not_invented(client):
    for d in client.get(f"{API}/forecasts/districts", params={"run_id": RUN1}).json()["districts"]:
        assert all(d[f] is None for f in LATER_STAGE_FIELDS), d
        assert d["flags"] == {"fallback_used": None, "fallback_reason": None}


def test_district_forecast_filters(client):
    get = lambda **p: client.get(f"{API}/forecasts/districts", params=p).json()  # noqa: E731
    assert get()["total"] == 18  # 2 runs x 3 leads x 3 districts
    assert get(lead_day=2)["total"] == 6
    assert {d["district_id"] for d in get(district_id="D002")["districts"]} == {"D002"} and get(
        district_id="D002"
    )["total"] == 6
    assert {d["state"] for d in get(state="State A")["districts"]} == {"State A"} and get(state="State A")[
        "total"
    ] == 12
    assert get(imd_date="2024-07-03")["total"] == 3 + 3  # run 1 lead 2 and run 2 lead 1
    assert {(d["run_id"], d["lead_day"]) for d in get(imd_date="2024-07-03")["districts"]} == {
        (RUN1, 2),
        (RUN2, 1),
    }
    assert get(season=2024)["total"] == 18 and get(season=2000) == {
        **get(season=2000),
        "total": 0,
        "districts": [],
    }
    assert get(run_id=RUN2, lead_day=3, district_id="D003")["total"] == 1


def test_district_forecast_pagination_and_stable_order(client):
    p1 = client.get(f"{API}/forecasts/districts", params={"limit": 7, "offset": 0}).json()
    p2 = client.get(f"{API}/forecasts/districts", params={"limit": 7, "offset": 7}).json()
    p3 = client.get(f"{API}/forecasts/districts", params={"limit": 7, "offset": 14}).json()
    keys = [(d["run_id"], d["lead_day"], d["district_id"]) for p in (p1, p2, p3) for d in p["districts"]]
    assert len(keys) == 18 == len(set(keys)) and keys == sorted(keys)
    assert p1["total"] == p2["total"] == p3["total"] == 18 and len(p3["districts"]) == 4


def test_unknown_run_or_district_is_404(client):
    error(
        client.get(f"{API}/forecasts/districts", params={"run_id": "tigge_ecmwf_cf_1999010100"}),
        404,
        "RUN_NOT_FOUND",
    )
    error(client.get(f"{API}/forecasts/districts", params={"district_id": "D999"}), 404, "DISTRICT_NOT_FOUND")


@pytest.mark.parametrize(
    "params",
    [
        {"lead_day": 4},
        {"lead_day": 0},
        {"lead_day": "x"},
        {"imd_date": "yesterday"},
        {"limit": 0},
        {"limit": 5000},
        {"offset": -1},
        {"run_id": "abc"},
        {"district_id": "D 1;drop"},
        {"season": "x"},
    ],
)
def test_bad_query_parameters_are_422(client, params):
    error(client.get(f"{API}/forecasts/districts", params=params), 422, "INVALID_PARAMETER")


def test_one_district(client):
    body = client.get(f"{API}/forecasts/districts/D002").json()
    assert body["district"]["name"] == "Beta" and body["district"]["is_small"] is True
    assert (
        body["district"]["n_effective_cells"] == pytest.approx(2.0, rel=0.02)
        and body["district"]["source_year"] == 2011
    )
    assert body["total"] == 6 and {f["district_id"] for f in body["forecasts"]} == {"D002"}
    one = client.get(f"{API}/forecasts/districts/D002", params={"run_id": RUN1, "lead_day": 2}).json()
    assert one["total"] == 1 and one["forecasts"][0]["imd_date"] == "2024-07-03"


def test_one_district_404s(client):
    error(client.get(f"{API}/forecasts/districts/D999"), 404, "DISTRICT_NOT_FOUND")
    error(
        client.get(f"{API}/forecasts/districts/D004"), 404, "FORECAST_NOT_FOUND"
    )  # exists (for the map) but has no forecast
    error(
        client.get(f"{API}/forecasts/districts/D001", params={"run_id": "tigge_ecmwf_cf_1999010100"}),
        404,
        "RUN_NOT_FOUND",
    )
    error(
        client.get(f"{API}/forecasts/districts/D001", params={"imd_date": "1999-01-01"}),
        404,
        "FORECAST_NOT_FOUND",
    )
    error(client.get(f"{API}/forecasts/districts/bad id"), 422, "INVALID_PARAMETER")


# ---- grid (cell level, from Parquet) -------------------------------------------------------------------------


def test_grid_raw_layer(client, m1_outputs):
    body = client.get(f"{API}/forecasts/grid", params={"run_id": RUN1, "lead_day": 2}).json()
    g = m1_outputs.golden
    rows = g[(g.run_id == RUN1) & (g.lead_day == 2)].sort_values("cell_id")
    assert (body["variable"], body["unit"], body["mode"], body["imd_date"]) == (
        "raw",
        "mm",
        "replay",
        "2024-07-03",
    )
    assert body["total"] == len(body["cells"]) == len(rows) == int(m1_outputs.grid["is_valid"].sum())
    assert [c["cell_id"] for c in body["cells"]] == list(rows["cell_id"])
    assert [c["value"] for c in body["cells"]] == pytest.approx(list(rows["rain_mm"]), abs=1e-3)
    first = body["cells"][0]
    assert (first["latitude"], first["longitude"]) == (rows.iloc[0]["latitude"], rows.iloc[0]["longitude"])


def test_grid_observed_layer_has_nulls_where_imd_is_missing(client, m1_outputs):
    body = client.get(
        f"{API}/forecasts/grid", params={"run_id": RUN2, "lead_day": 1, "variable": "observed"}
    ).json()
    g = m1_outputs.golden
    rows = g[(g.run_id == RUN2) & (g.lead_day == 1)].sort_values("cell_id")
    values = [c["value"] for c in body["cells"]]
    assert sum(v is None for v in values) == int(rows["obs_mm"].isna().sum()) > 0
    assert not any(isinstance(v, float) and np.isnan(v) for v in values if v is not None)
    ok = rows["obs_mm"].notna().to_numpy()
    assert [v for v in values if v is not None] == pytest.approx(list(rows["obs_mm"][ok]), abs=1e-3)


def test_grid_pagination(client):
    full = client.get(f"{API}/forecasts/grid", params={"run_id": RUN1, "lead_day": 1}).json()
    page = client.get(
        f"{API}/forecasts/grid", params={"run_id": RUN1, "lead_day": 1, "limit": 100, "offset": 50}
    ).json()
    assert page["total"] == full["total"] and page["cells"] == full["cells"][50:150]


def test_grid_layers_that_need_later_stages_are_503_not_invented(client):
    for variable in ("corrected", "difference", "improvement", "q50", "p_ge_64_5", "lps_influence"):
        msg = error(
            client.get(f"{API}/forecasts/grid", params={"run_id": RUN1, "lead_day": 1, "variable": variable}),
            503,
            "VARIABLE_NOT_AVAILABLE",
        )
        assert variable in msg


def test_grid_errors(client, m1_outputs, tmp_path):
    error(
        client.get(f"{API}/forecasts/grid", params={"run_id": RUN1, "lead_day": 1, "variable": "banana"}),
        422,
        "INVALID_PARAMETER",
    )
    error(
        client.get(f"{API}/forecasts/grid", params={"lead_day": 1}), 422, "INVALID_PARAMETER"
    )  # run_id is required
    error(
        client.get(f"{API}/forecasts/grid", params={"run_id": RUN1, "lead_day": 9}), 422, "INVALID_PARAMETER"
    )
    error(
        client.get(f"{API}/forecasts/grid", params={"run_id": "tigge_ecmwf_cf_1999010100", "lead_day": 1}),
        404,
        "RUN_NOT_FOUND",
    )


def test_grid_without_the_parquet_file_is_503(loaded_db, m1_outputs, tmp_path):
    import dataclasses

    cfg = m1_outputs.cfg
    empty = dataclasses.replace(
        cfg, alignment=dataclasses.replace(cfg.alignment, golden_dir=tmp_path / "no_golden")
    )
    error(
        make_client(loaded_db.url, empty).get(
            f"{API}/forecasts/grid", params={"run_id": RUN1, "lead_day": 1}
        ),
        503,
        "GRID_DATA_UNAVAILABLE",
    )


# ---- map ---------------------------------------------------------------------------------------------------


def test_map_metadata(client, m1_outputs):
    body = client.get(f"{API}/map/metadata").json()
    grid = body["grid"]
    assert (grid["n_lat"], grid["n_lon"], grid["spacing_deg"]) == (129, 135, 0.25)
    assert (grid["lat_min"], grid["lat_max"], grid["lon_min"], grid["lon_max"]) == (6.5, 38.5, 66.5, 100.0)
    assert grid["n_cells"] == 129 * 135 and grid["n_valid_cells"] == int(m1_outputs.grid["is_valid"].sum())
    assert body["lead_days"] == [1, 2, 3] and body["thresholds_mm"] == [15.6, 64.5, 115.6]
    assert body["runs"]["total"] == 2 and body["runs"]["seasons"] == [2024]
    assert body["runs"]["first_initialization_time"].startswith("2024-07-01") and body["runs"][
        "last_initialization_time"
    ].startswith("2024-07-02")
    assert body["imd_dates"] == {"first": "2024-07-02", "last": "2024-07-05"}
    assert body["districts"]["count"] == 4 and body["districts"]["census_basis_year"] == 2011
    assert "after 2011" in body["districts"]["note"]
    layers = {lay["variable"]: lay for lay in body["layers"]}
    assert layers["raw"]["available"] and layers["observed"]["available"] and len(layers) == 14
    assert not layers["corrected"]["available"] and "later pipeline stage" in layers["corrected"]["note"]


def test_map_districts_geojson(client):
    r = client.get(f"{API}/map/districts")
    assert r.status_code == 200 and "max-age" in r.headers["cache-control"]
    body = r.json()
    assert (
        body["type"] == "FeatureCollection"
        and body["census_basis_year"] == 2011
        and "DataMeet" in body["district_file"]
    )
    assert [f["id"] for f in body["features"]] == ["D001", "D002", "D003", "D004"]
    a = body["features"][0]
    assert a["type"] == "Feature" and a["geometry"]["type"] == "Polygon"
    assert a["properties"] == {
        "source_id": "101",
        "district_id": "D001",
        "name": "Alpha",
        "state": "State A",
        "is_small": False,
        "n_effective_cells": pytest.approx(256, rel=0.02),
        "source_year": 2011,
    }
    ring = a["geometry"]["coordinates"][0]
    assert min(p[0] for p in ring) == pytest.approx(74.875) and max(p[1] for p in ring) == pytest.approx(
        18.875
    )
    assert body["features"][3]["properties"]["is_small"] is None  # no weights for D004


# ---- errors and safety ---------------------------------------------------------------------------------------


def test_api_is_read_only(client):
    for method in ("post", "put", "patch", "delete"):
        for path in ("/runs", "/forecasts/districts", "/map/districts"):
            assert getattr(client, method)(f"{API}{path}").status_code == 405


def test_unknown_route_uses_the_error_format(client):
    error(client.get(f"{API}/nope"), 404, "NOT_FOUND")


def test_database_not_configured_is_503(m1_outputs):
    error(make_client(None, m1_outputs.cfg).get(f"{API}/runs"), 503, "DATABASE_NOT_CONFIGURED")


def test_database_errors_are_not_exposed(m1_outputs, tmp_path):
    from backend.app.db.session import engine_for

    url = f"sqlite:///{tmp_path / 'blank.db'}"
    engine_for(url).connect().close()  # an empty database: the tables do not exist
    r = make_client(url, m1_outputs.cfg).get(f"{API}/runs")
    error(r, 503, "DATABASE_UNAVAILABLE")
    assert not re.search(r"no such table|SELECT|nwp_runs|sqlite", r.text, re.I)


def test_openapi_lists_only_get_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {p for p in paths if p.startswith(API)} == {
        f"{API}/health", f"{API}/runs", f"{API}/runs/{{run_id}}", f"{API}/forecasts/districts",
        f"{API}/forecasts/districts/{{district_id}}", f"{API}/forecasts/grid", f"{API}/map/metadata", f"{API}/map/districts",
    }  # fmt: skip
    assert all(set(ops) == {"get"} for p, ops in paths.items() if p.startswith(API))
