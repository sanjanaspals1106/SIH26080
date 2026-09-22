#!/usr/bin/env python
"""Stage 3 end-to-end example on synthetic data (no credentials, no network, nothing large).

Stage 1 synthetic files -> Stage 2 Golden Dataset -> Stage 3: static geography, climatology (from synthetic
"training" years), district weights (from synthetic district squares), the 27 features, district forecasts and
district_history. Everything is written to a temporary folder. The data is random: it shows that the pipeline
runs, not that anything is meaningful.

Run from the repository root: python scripts/features_example.py
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment import build_golden_season, get_valid_cells  # noqa: E402
from data_pipeline.districts import load_districts  # noqa: E402
from data_pipeline.features import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    build_stage3_season,
    feature_matrix,
    read_district_forecasts,
    read_district_history,
    read_features,
)
from data_pipeline.ingestion import (  # noqa: E402
    download_tigge_month,
    download_tigge_static,
    load_config,
)
from data_pipeline.ingestion.synthetic import (  # noqa: E402
    SyntheticClient,
    write_synthetic_districts,
    write_synthetic_imd_year,
)


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory(prefix="sih_stage3_") as tmp:
        cfg = load_config(data_dir=tmp)
        cfg = dataclasses.replace(cfg, alignment=dataclasses.replace(cfg.alignment, static_date="2024-06-01"))

        # Stage 1 and 2 (synthetic): raw files, valid-cell mask, Golden Dataset for July 2024
        client = SyntheticClient(cfg, max_dates=3)
        download_tigge_month(2024, 7, client=client, config=cfg)
        download_tigge_static(date(2024, 6, 1), config=cfg, client=client)
        write_synthetic_imd_year(cfg, 2024, missing_fraction=0.0, missing_cells_fraction=0.4)
        grid = get_valid_cells(cfg, years=[2024])
        build_golden_season(2024, cfg, months=[7])

        # Stage 3 inputs: two synthetic "training" years for the climatology, synthetic district squares
        for year in (2022, 2023):
            write_synthetic_imd_year(cfg, year, missing_fraction=0.0, missing_cells_fraction=0.4, seed=year)
        districts = load_districts(
            write_synthetic_districts(cfg.districts.raw_dir / "synthetic.geojson"), cfg
        )

        out = build_stage3_season(
            2024, [2022, 2023], cfg, holdout_seasons=[2024], districts=districts, grid=grid
        )
        print(f"feature version: {FEATURE_VERSION}")
        for kind, paths in out.items():
            print(f"{kind}: {[str(p.relative_to(cfg.data_dir)) for p in paths]}")

        feats = read_features(cfg)
        print(
            f"\ncell features: {len(feats)} rows, {len(FEATURE_COLUMNS)} model inputs: {list(feature_matrix(feats).columns)}"
        )
        print(
            feats[
                [
                    "run_id",
                    "lead_day",
                    "cell_id",
                    "rain_mm",
                    "nbr_mean_3",
                    "rain_grad",
                    "wspd850",
                    "vort850",
                    "slope",
                    "dist_coast_km",
                    "clim_mean",
                ]
            ]
            .head(3)
            .to_string(index=False)
        )
        fc = read_district_forecasts(cfg)
        print(f"\ndistrict forecasts: {len(fc)} rows ({fc['district_id'].nunique()} districts)")
        print(
            fc[
                [
                    "run_id",
                    "lead_day",
                    "district_id",
                    "raw_mean_mm",
                    "observed_mean_mm",
                    "n_effective_cells",
                    "is_small",
                    "corrected_mean_mm",
                ]
            ]
            .head(4)
            .to_string(index=False)
        )
        print(f"\ndistrict history: {len(read_district_history(cfg))} rows")
    print("\nStage 3 example finished OK.")


if __name__ == "__main__":
    main()
