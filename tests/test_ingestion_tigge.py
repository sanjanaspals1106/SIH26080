"""TIGGE ingestion: request building, download (network mocked), GRIB parsing, validation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data_pipeline.ingestion import (
    DownloadError,
    IngestionError,
    MissingCredentialsError,
    MissingInputError,
    ValidationError,
    download_tigge_month,
    download_tigge_static,
    load_config,
    read_tigge,
    run_id,
    validate_tigge,
)
from data_pipeline.ingestion.synthetic import SyntheticClient


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One synthetic 'download' of July 2024 (2 dates) plus static fields, shared by the tests."""
    cfg = load_config(data_dir=tmp_path_factory.mktemp("data"))
    client = SyntheticClient(cfg, max_dates=2)
    files = download_tigge_month(2024, 7, client=client, config=cfg)
    static = download_tigge_static(date(2024, 6, 1), config=cfg, client=client)
    return cfg, client, files, static


@pytest.fixture(scope="module")
def rain(world):
    cfg, _, files, _ = world
    return read_tigge(files["rain"], "rain", cfg)


# ---- download ------------------------------------------------------------------------------


def test_request_follows_the_prd(world):
    cfg, client, _, _ = world
    dataset, req = client.calls[0]
    assert dataset == cfg.tigge.dataset
    assert (req["origin"], req["type"], req["time"]) == ("ecmwf", "cf", "00:00")
    assert req["area"] == [40, 55, 0, 100]
    assert req["grid"] == [0.25, 0.25]
    assert req["step"] == list(range(0, 79, 6))
    assert req["date"][0] == "2024-07-01" and len(req["date"]) == 31
    assert req["param"] == ["tp"]


def test_atmosphere_is_split_by_level_and_asks_for_the_prd_fields(world):
    _, client, _, _ = world
    by_level = {
        (r["levtype"], tuple(r.get("levelist", []))): sorted(r["param"]) for _, r in client.calls[1:4]
    }
    assert by_level == {
        ("sfc", ()): ["msl"],
        ("pl", (850,)): ["q", "u", "v"],
        ("pl", (200,)): ["u", "v"],
    }
    assert client.calls[1][1]["step"] == list(range(6, 73, 6))


def test_raw_files_have_deterministic_names(world):
    cfg, _, files, static = world
    assert [p.name for p in files["rain"]] == ["tigge_ecmwf_cf_202407_rain_sfc.grib"]
    assert {p.name for p in files["atmosphere"]} == {
        "tigge_ecmwf_cf_202407_atmosphere_sfc.grib",
        "tigge_ecmwf_cf_202407_atmosphere_pl850.grib",
        "tigge_ecmwf_cf_202407_atmosphere_pl200.grib",
    }
    assert static[0].name == "tigge_ecmwf_cf_20240601_static_sfc.grib"
    assert all(p.is_relative_to(cfg.data_dir / "raw" / "tigge") for p in files["rain"] + static)


def test_existing_files_are_not_downloaded_again(world):
    cfg, client, _, _ = world
    n = len(client.calls)
    download_tigge_month(2024, 7, client=client, config=cfg)
    assert len(client.calls) == n
    download_tigge_month(2024, 7, groups=["rain"], client=client, config=cfg, force=True)
    assert len(client.calls) == n + 1


def test_failed_download_leaves_no_file(cfg):
    class Broken:
        def retrieve(self, dataset, request, target):
            open(target, "wb").write(b"partial")
            raise RuntimeError("queue timeout")

    with pytest.raises(DownloadError, match="queue timeout"):
        download_tigge_month(2024, 7, groups=["rain"], client=Broken(), config=cfg)
    assert not any(p.is_file() for p in cfg.tigge.raw_dir.rglob("*"))  # no .part, no final file


def test_empty_download_is_an_error(cfg):
    class Empty:
        def retrieve(self, dataset, request, target):
            open(target, "wb").close()

    with pytest.raises(DownloadError, match="no data"):
        download_tigge_month(2024, 7, groups=["rain"], client=Empty(), config=cfg)


def test_missing_credentials_message(cfg, monkeypatch, tmp_path):
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "no_such_cdsapirc"))
    with pytest.raises(MissingCredentialsError, match="cdsapirc"):
        download_tigge_month(2024, 7, config=cfg)


def test_month_outside_jjas_is_rejected(cfg):
    with pytest.raises(IngestionError, match="JJAS"):
        download_tigge_month(2024, 3, config=cfg, client=object())


# ---- reading -------------------------------------------------------------------------------


def test_read_rain(rain):
    assert list(rain.data_vars) == ["tp"]
    assert rain["tp"].dims == ("init_time", "lead_hours", "lat", "lon")
    assert rain.sizes == {"init_time": 2, "lead_hours": 14, "lat": 161, "lon": 181}
    assert list(rain["lead_hours"].values) == list(range(0, 79, 6))
    assert list(rain["init_time"].values) == list(pd.to_datetime(["2024-07-01", "2024-07-02"]))
    # lat and lon ascending, covering the PRD area
    assert rain["lat"].values[0] == 0 and rain["lat"].values[-1] == 40
    assert rain["lon"].values[0] == 55 and rain["lon"].values[-1] == 100
    # valid_time = init_time + lead
    assert rain["valid_time"].sel(init_time="2024-07-01", lead_hours=48).values == np.datetime64(
        "2024-07-03T00:00"
    )
    # tp stays cumulative: nothing here differences it (Stage 2)
    assert (rain["tp"].diff("lead_hours") >= 0).all()


def test_read_atmosphere_selects_the_right_levels(world):
    cfg, _, files, _ = world
    atm = read_tigge(files["atmosphere"], "atmosphere", cfg)
    assert set(atm.data_vars) == {"msl", "u850", "v850", "u200", "v200", "q850"}
    assert list(atm["lead_hours"].values) == list(range(6, 73, 6))
    assert not np.allclose(atm["u850"].values, atm["u200"].values)


def test_read_static_is_two_dimensional(world):
    cfg, _, _, static = world
    st = read_tigge(static, "static", cfg)
    assert set(st.data_vars) == {"orog", "lsm"}
    assert st["orog"].dims == ("lat", "lon")


def test_read_several_months_concatenates_start_times(world):
    cfg, client, files, _ = world
    june = download_tigge_month(2024, 6, groups=["rain"], client=client, config=cfg)["rain"]
    both = read_tigge(june + files["rain"], "rain", cfg)
    assert both.sizes["init_time"] == 4
    assert both["init_time"].to_index().is_monotonic_increasing


def test_same_file_twice_is_reported_as_duplicate(world):
    cfg, _, files, _ = world
    with pytest.raises(ValidationError, match="more than one file"):
        read_tigge(files["rain"] * 2, "rain", cfg)


def test_run_id_format():
    assert run_id("2024-07-15T00:00:00Z") == "tigge_ecmwf_cf_2024071500"


# ---- malformed / missing input -------------------------------------------------------------


def test_missing_file(cfg, tmp_path):
    with pytest.raises(MissingInputError, match="not found"):
        read_tigge(tmp_path / "nope.grib", "rain", cfg)


def test_empty_file(cfg, tmp_path):
    f = tmp_path / "empty.grib"
    f.write_bytes(b"")
    with pytest.raises(MissingInputError, match="empty"):
        read_tigge(f, "rain", cfg)


def test_garbage_file_is_not_an_obscure_error(cfg, tmp_path):
    f = tmp_path / "junk.grib"
    f.write_text("<html>Service unavailable</html>")
    with pytest.raises(IngestionError):
        read_tigge(f, "rain", cfg)


def test_wrong_file_type(cfg, tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b")
    with pytest.raises(IngestionError, match="GRIB"):
        read_tigge(f, "rain", cfg)


def test_reading_the_wrong_group(world):
    cfg, _, files, _ = world
    with pytest.raises(ValidationError, match="None of the atmosphere variables"):
        read_tigge(files["rain"], "atmosphere", cfg)


def test_unknown_group(cfg, tmp_path):
    with pytest.raises(IngestionError, match="Unknown TIGGE group"):
        read_tigge(tmp_path / "x.grib", "wind", cfg)


# ---- validation ----------------------------------------------------------------------------


def test_valid_dataset_passes(world, rain):
    validate_tigge(rain, "rain", world[0])


def test_missing_variable(world, rain):
    with pytest.raises(ValidationError, match=r"missing variables \['tp'\]"):
        validate_tigge(rain.drop_vars("tp"), "rain", world[0])


def test_missing_coordinates(world, rain):
    with pytest.raises(ValidationError, match="missing coordinates"):
        validate_tigge(
            rain.drop_vars("valid_time").reset_coords(drop=True).drop_vars("lead_hours"), "rain", world[0]
        )


def test_empty_data(world, rain):
    with pytest.raises(ValidationError, match="empty"):
        validate_tigge(rain.where(False), "rain", world[0])


def test_wrong_domain(world, rain):
    small = rain.sel(lat=slice(10, 30))
    with pytest.raises(ValidationError, match="domain south edge"):
        validate_tigge(small, "rain", world[0])


def test_wrong_start_hour(world, rain):
    shifted = rain.assign_coords(init_time=rain["init_time"] + np.timedelta64(12, "h"))
    with pytest.raises(ValidationError, match="00:00 UTC"):
        validate_tigge(shifted, "rain", world[0])


def test_missing_steps(world, rain):
    with pytest.raises(ValidationError, match="missing.*78"):
        validate_tigge(rain.isel(lead_hours=slice(0, -1)), "rain", world[0])
