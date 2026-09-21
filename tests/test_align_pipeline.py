"""Stage 1 raw files -> valid mask -> Golden Dataset, on synthetic GRIB and IMD files (no network)."""

import dataclasses
from datetime import date

import pytest

from data_pipeline.alignment import build_golden_season, get_valid_cells, read_golden
from data_pipeline.ingestion import (
    MissingInputError,
    download_tigge_month,
    download_tigge_static,
    load_config,
)
from data_pipeline.ingestion.synthetic import SyntheticClient, write_synthetic_imd_year


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    cfg = load_config(data_dir=tmp_path_factory.mktemp("data"))
    cfg = dataclasses.replace(cfg, alignment=dataclasses.replace(cfg.alignment, static_date="2024-06-01"))
    client = SyntheticClient(cfg, max_dates=2)
    download_tigge_month(2024, 7, client=client, config=cfg)
    download_tigge_static(date(2024, 6, 1), config=cfg, client=client)
    write_synthetic_imd_year(
        cfg, 2024, missing_fraction=0.0, missing_cells_fraction=0.4
    )  # 40% of cells are "ocean"
    return cfg


def test_end_to_end_from_raw_files(raw):
    cfg = raw
    grid = get_valid_cells(cfg, years=[2024])  # synthetic stand-in for the 1981-2010 base period
    n_valid = int(grid["is_valid"].sum())
    assert 0.5 * len(grid) < n_valid < 0.7 * len(grid)  # the ~40% dead cells are excluded

    paths = build_golden_season(2024, cfg, months=[7])
    assert [p.name for p in paths] == ["golden_202407.parquet"]
    df = read_golden(cfg, seasons=[2024])
    assert len(df) == 2 * 3 * n_valid  # 2 synthetic start dates x 3 leads x valid cells
    assert not df.duplicated(["run_id", "lead_day", "cell_id"]).any()
    assert set(df["cell_id"]) == set(grid.loc[grid["is_valid"], "cell_id"])
    assert (
        df["obs_mm"].notna().all() and not df["obs_missing"].any()
    )  # synthetic IMD has no gaps in valid cells
    assert df["rain_mm"].between(0, 1000).all()
    assert sorted(df["imd_date"].dt.strftime("%Y-%m-%d").unique()) == [
        "2024-07-02",
        "2024-07-03",
        "2024-07-04",
        "2024-07-05",
    ]

    build_golden_season(2024, cfg, months=[7])  # running it again does not duplicate anything
    assert len(read_golden(cfg, seasons=[2024])) == len(df)


def test_missing_inputs_give_actionable_errors(raw, tmp_path):
    fresh = load_config(data_dir=tmp_path / "empty")
    with pytest.raises(MissingInputError, match="Valid-cell mask not found"):
        build_golden_season(2024, fresh, months=[7])
    with pytest.raises(MissingInputError, match="static_date"):
        build_golden_season(
            2024,
            dataclasses.replace(raw, alignment=dataclasses.replace(raw.alignment, static_date=None)),
            months=[7],
        )
    with pytest.raises(MissingInputError, match="not found"):  # no TIGGE file for June in this data dir
        build_golden_season(2024, raw, months=[6])
