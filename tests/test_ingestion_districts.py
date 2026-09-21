"""District boundary loader: schema, CRS, geometry repair, bad input. No weighting here (Stage 3)."""

import dataclasses

import geopandas as gpd
import pytest
from shapely.geometry import box

from data_pipeline.districts import load_districts
from data_pipeline.ingestion import IngestionError, MissingInputError, ValidationError
from data_pipeline.ingestion.synthetic import write_synthetic_districts


def with_districts(cfg, **changes):
    return dataclasses.replace(cfg, districts=dataclasses.replace(cfg.districts, **changes))


def test_load_districts(cfg, tmp_path):
    path = write_synthetic_districts(tmp_path / "d.geojson")
    gdf = load_districts(path, cfg)
    assert len(gdf) == 4
    assert list(gdf.columns) == ["district_id", "name", "state", "source_id", "geometry"]
    assert gdf.crs.to_epsg() == 4326
    # stable IDs, sorted by (state, name)
    assert list(gdf["district_id"]) == ["D001", "D002", "D003", "D004"]
    assert list(gdf["name"]) == ["Alpha", "Beta", "Gamma", "Delta"]
    assert gdf["district_id"].is_unique
    assert gdf.geometry.is_valid.all()
    assert set(gdf["source_id"]) == {"101", "102", "201", "301"}


def test_same_file_gives_the_same_ids(cfg, tmp_path):
    path = write_synthetic_districts(tmp_path / "d.geojson")
    assert load_districts(path, cfg)[["district_id", "name"]].equals(
        load_districts(path, cfg)[["district_id", "name"]]
    )


def test_file_is_found_through_config(cfg):
    write_synthetic_districts(cfg.districts.raw_dir / "d.geojson")
    assert len(load_districts(config=with_districts(cfg, file="d.geojson"))) == 4


def test_no_file_configured(cfg):
    with pytest.raises(MissingInputError, match="districts.yaml"):
        load_districts(config=cfg)


def test_missing_file(cfg, tmp_path):
    with pytest.raises(MissingInputError, match="not found"):
        load_districts(tmp_path / "nope.geojson", cfg)


def test_projected_input_is_reprojected_to_lat_lon(cfg, tmp_path):
    gdf = gpd.read_file(write_synthetic_districts(tmp_path / "d.geojson")).to_crs("EPSG:3857")
    path = tmp_path / "proj.geojson"
    gdf.to_file(path, driver="GeoJSON")
    out = load_districts(path, cfg)
    assert out.crs.to_epsg() == 4326
    assert 70 < out.total_bounds[0] < 90 and 5 < out.total_bounds[1] < 40


def test_invalid_geometry_is_repaired(cfg, tmp_path):
    path = write_synthetic_districts(tmp_path / "bad.geojson", invalid=True)
    assert not gpd.read_file(path).geometry.is_valid.all()  # the fixture really is invalid
    out = load_districts(path, cfg)
    assert out.geometry.is_valid.all() and not out.geometry.is_empty.any()
    assert len(out) == 4


@pytest.mark.filterwarnings("ignore:'crs' was not provided")
def test_missing_crs_is_an_error_unless_assumed(cfg, tmp_path):
    path = write_synthetic_districts(tmp_path / "nocrs.shp", with_crs=False)
    assert gpd.read_file(path).crs is None
    with pytest.raises(ValidationError, match="no CRS"):
        load_districts(path, cfg)
    assert load_districts(path, with_districts(cfg, assume_crs="EPSG:4326")).crs.to_epsg() == 4326


def test_missing_name_column(cfg, tmp_path):
    path = write_synthetic_districts(tmp_path / "d.geojson")
    with pytest.raises(ValidationError, match="Columns found"):
        load_districts(path, with_districts(cfg, name_column="NOT_THERE"))


def test_optional_columns_may_be_absent(cfg, tmp_path):
    path = write_synthetic_districts(tmp_path / "d.geojson")
    out = load_districts(path, with_districts(cfg, state_column=None, source_id_column=None))
    assert "source_id" not in out.columns and (out["state"] == "").all()


def test_empty_dataset(cfg, tmp_path):
    empty = gpd.GeoDataFrame({"DISTRICT": [], "ST_NM": []}, geometry=[], crs="EPSG:4326")
    path = tmp_path / "empty.geojson"
    empty.to_file(path, driver="GeoJSON")
    with pytest.raises(ValidationError, match="no districts"):
        load_districts(path, cfg)


def test_empty_geometry(cfg, tmp_path):
    gdf = gpd.GeoDataFrame(
        {"DISTRICT": ["A", "B"], "ST_NM": ["S", "S"]}, geometry=[box(75, 10, 76, 11), None], crs="EPSG:4326"
    )
    path = tmp_path / "nogeom.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ValidationError, match="no geometry"):
        load_districts(path, cfg)


def test_bounds_outside_india(cfg, tmp_path):
    gdf = gpd.GeoDataFrame(
        {"DISTRICT": ["A"], "ST_NM": ["S"]}, geometry=[box(-10, 50, -9, 51)], crs="EPSG:4326"
    )
    path = tmp_path / "far.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ValidationError, match="outside India"):
        load_districts(path, cfg)


def test_garbage_file(cfg, tmp_path):
    path = tmp_path / "junk.geojson"
    path.write_text("this is not geojson")
    with pytest.raises(IngestionError, match="Could not read"):
        load_districts(path, cfg)
