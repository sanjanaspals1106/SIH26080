"""Stage 1 configuration: values come from config/*.yaml and match the PRD."""

import shutil

import pytest
import yaml

from data_pipeline.ingestion import IngestionError, load_config
from data_pipeline.ingestion.config import CONFIG_DIR


def test_tigge_settings_match_prd(cfg):
    t = cfg.tigge
    assert t.forecast_type == "cf"
    assert t.init_hour_utc == 0
    assert t.area.as_list() == [40, 55, 0, 100]  # north, west, south, east
    assert t.grid_spacing_deg == 0.25
    assert list(t.variables) == ["tp", "msl", "u850", "v850", "u200", "v200", "q850", "orog", "lsm"]
    assert t.steps_hours["rain"] == list(range(0, 79, 6))
    assert t.steps_hours["atmosphere"] == list(range(6, 73, 6))
    assert t.steps_hours["static"] == [0]
    assert t.season_months == [6, 7, 8, 9]


def test_imd_settings_match_prd(cfg):
    i = cfg.imd
    assert (i.n_lat, i.n_lon) == (129, 135)
    assert i.missing_value == -999
    assert (i.lat_origin, i.lon_origin, i.lat_end, i.lon_end) == (6.5, 66.5, 38.5, 100.0)


def test_raw_data_is_kept_under_data_dir_raw(cfg):
    for d in (cfg.tigge.raw_dir, cfg.imd.raw_dir, cfg.districts.raw_dir):
        assert d.is_relative_to(cfg.data_dir / "raw")


def test_data_dir_comes_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "elsewhere"))
    assert load_config().data_dir == (tmp_path / "elsewhere").resolve()


def test_default_data_dir_is_inside_the_repository(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr("data_pipeline.ingestion.config._dotenv_value", lambda name, root: None)
    assert load_config().data_dir == (CONFIG_DIR.parent / "data").resolve()


def test_missing_config_key_gives_a_useful_error(tmp_path):
    shutil.copytree(CONFIG_DIR, tmp_path / "config")
    (tmp_path / "config" / "alignment.yaml").write_text("forecast: {}\ntruth: {}\ningestion: {}\n")
    with pytest.raises(IngestionError, match="missing the key"):
        load_config(config_dir=tmp_path / "config")


def test_no_secret_keys_in_config_files():
    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k).lower()
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    for name in ("alignment.yaml", "districts.yaml"):
        found = set(keys(yaml.safe_load((CONFIG_DIR / name).read_text())))
        assert not {k for k in found if any(s in k for s in ("key", "password", "token", "secret"))} - {
            "request_keys"
        }
