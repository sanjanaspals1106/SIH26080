#!/usr/bin/env python
"""Stage 1 end-to-end example on synthetic data (no credentials, no network, nothing large).

Writes small synthetic TIGGE GRIB, IMD binary and district files into a temporary folder, then runs
the real ingestion code on them: download (with the network replaced by a synthetic client, run
twice to show that nothing is downloaded again), read, validate. Prints what Stage 2 receives.

Run from the repository root: python scripts/ingest_example.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.districts import load_districts  # noqa: E402
from data_pipeline.ingestion import (  # noqa: E402
    download_imd,
    download_tigge_month,
    download_tigge_static,
    load_config,
    read_imd,
    read_tigge,
    run_id,
)
from data_pipeline.ingestion.synthetic import (  # noqa: E402
    SyntheticClient,
    write_synthetic_districts,
    write_synthetic_imd_year,
)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory(prefix="sih_ingest_") as tmp:
        cfg = load_config(data_dir=tmp)
        print(f"Synthetic data dir: {cfg.data_dir}\n")

        # TIGGE: 2 forecast dates of July 2024, plus static fields.
        client = SyntheticClient(cfg, max_dates=2)
        files = download_tigge_month(2024, 7, client=client, config=cfg)
        static_files = download_tigge_static(date(2024, 6, 1), config=cfg, client=client)
        n = len(client.calls)
        download_tigge_month(2024, 7, client=client, config=cfg)  # already on disk
        print(f"TIGGE requests: {n}; second run made {len(client.calls) - n} new requests")
        for group in ("rain", "atmosphere"):
            ds = read_tigge(files[group], group, cfg)
            print(f"TIGGE {group}: {dict(ds.sizes)} vars={list(ds.data_vars)}")
        static = read_tigge(static_files, "static", cfg)
        print(f"TIGGE static: {dict(static.sizes)} vars={list(static.data_vars)}")
        print(f"Run IDs: {[run_id(t) for t in ds['init_time'].values]}\n")

        # IMD: one synthetic year, with the missing value -999 in the raw file.
        write_synthetic_imd_year(cfg, 2023)
        download_imd([2023], cfg, fetch=lambda year, c: None)  # file exists: fetch is never called
        imd = read_imd([2023], cfg)
        print(f"IMD rain: {dict(imd.sizes)}, {int(imd['rain'].isnull().sum())} missing cells now NaN\n")

        # Districts
        path = write_synthetic_districts(cfg.districts.raw_dir / "synthetic.geojson")
        districts = load_districts(path, cfg)
        print(districts[["district_id", "name", "state"]].to_string(index=False))
        print(f"CRS: {districts.crs}")
    print("\nStage 1 example finished OK.")


if __name__ == "__main__":
    main()
