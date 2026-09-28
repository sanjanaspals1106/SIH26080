"""District forecast rows carry the district centroid and effective-cell count, so the map and globe can place
every district without a hardcoded coordinate table (Track A: dashboard)."""

import pytest

RUN1 = "tigge_ecmwf_cf_2024070100"
API = "/api/v1"


def test_forecast_rows_carry_centroid_and_effective_cells(client):
    rows = client.get(f"{API}/forecasts/districts", params={"run_id": RUN1, "lead_day": 1, "limit": 1000}).json()[
        "districts"
    ]
    assert rows
    for d in rows:
        detail = client.get(f"{API}/forecasts/districts/{d['district_id']}", params={"run_id": RUN1}).json()["district"]
        assert d["centroid_lat"] == pytest.approx(detail["centroid_lat"])
        assert d["centroid_lon"] == pytest.approx(detail["centroid_lon"])
        assert d["n_effective_cells"] == pytest.approx(detail["n_effective_cells"])


def test_centroid_lies_inside_its_outline_bounding_box(client):
    features = {f["id"]: f for f in client.get(f"{API}/map/districts").json()["features"]}
    rows = client.get(f"{API}/forecasts/districts", params={"run_id": RUN1, "lead_day": 1}).json()["districts"]
    for d in rows:
        ring = features[d["district_id"]]["geometry"]["coordinates"][0]
        lons, lats = [p[0] for p in ring], [p[1] for p in ring]
        assert min(lons) <= d["centroid_lon"] <= max(lons)
        assert min(lats) <= d["centroid_lat"] <= max(lats)


def test_effective_cells_match_the_map_outline_properties(client):
    features = {f["id"]: f for f in client.get(f"{API}/map/districts").json()["features"]}
    rows = client.get(f"{API}/forecasts/districts", params={"run_id": RUN1, "lead_day": 1}).json()["districts"]
    for d in rows:
        props = features[d["district_id"]]["properties"]
        assert d["n_effective_cells"] == pytest.approx(props["n_effective_cells"])
        assert d["is_small"] == props["is_small"]
