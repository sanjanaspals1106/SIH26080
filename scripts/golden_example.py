#!/usr/bin/env python
"""Stage 2 end-to-end example on synthetic data (no credentials, no network, nothing large).

Stage 1 synthetic files (GRIB + IMD binary) -> Stage 1 readers -> valid-cell mask -> C1 temporal alignment
-> spatial alignment to the IMD grid -> Golden Dataset (Parquet) -> read it back. Everything is written to a
temporary folder. The data is random: it proves the pipeline runs, not that any forecast is good.

Run from the repository root: python scripts/golden_example.py
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment import (  # noqa: E402
    build_golden_season,
    get_valid_cells,
    read_golden,
)
from data_pipeline.ingestion import (  # noqa: E402
    download_tigge_month,
    download_tigge_static,
    load_config,
)
from data_pipeline.ingestion.synthetic import (  # noqa: E402
    SyntheticClient,
    write_synthetic_imd_year,
)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory(prefix="sih_golden_") as tmp:
        cfg = load_config(data_dir=tmp)
        cfg = dataclasses.replace(cfg, alignment=dataclasses.replace(cfg.alignment, static_date="2024-06-01"))

        # Stage 1: raw inputs (synthetic client instead of ECDS; synthetic IMD year with 40% "ocean" cells)
        client = SyntheticClient(cfg, max_dates=3)
        download_tigge_month(2024, 7, client=client, config=cfg)
        download_tigge_static(date(2024, 6, 1), config=cfg, client=client)
        write_synthetic_imd_year(cfg, 2024, missing_fraction=0.0, missing_cells_fraction=0.4)

        # Stage 2
        grid = get_valid_cells(cfg, years=[2024])  # the real run uses the 1981-2010 base period
        print(f"valid cells: {int(grid['is_valid'].sum())} of {len(grid)} (mask saved and reused)")
        paths = build_golden_season(2024, cfg, months=[7])
        df = read_golden(cfg, seasons=[2024])
        print(f"golden files: {[str(p.relative_to(cfg.data_dir)) for p in paths]}")
        print(
            f"rows: {len(df)} = {df['run_id'].nunique()} runs x {df['lead_day'].nunique()} leads x {df['cell_id'].nunique()} cells"
        )
        print(
            f"duplicate (run_id, lead_day, cell_id): {int(df.duplicated(['run_id', 'lead_day', 'cell_id']).sum())}"
        )
        print(f"columns ({len(df.columns)}): {list(df.columns)}\n")
        cols = ["run_id", "lead_day", "cell_id", "imd_date", "rain_mm", "obs_mm", "u850", "orog", "lsm"]
        print(df[cols].head(4).to_string(index=False))
    print("\nGolden Dataset example finished OK.")


if __name__ == "__main__":
    main()
