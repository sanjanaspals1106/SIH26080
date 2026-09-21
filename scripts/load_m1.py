#!/usr/bin/env python
"""Load the M1 data products into the database (PRD 18.2, 20.2). An offline batch job, never run by the API.

  python scripts/load_m1.py                       # everything, all seasons found on disk
  python scripts/load_m1.py --seasons 2024 2025   # only these seasons' runs and district products
  python scripts/load_m1.py --districts data/raw/districts/2011_Dist.shp   # the district file, if not set in config

Reads what the pipeline already wrote (Golden Dataset, Stage 3 tables and caches); it does not recompute anything.
It is safe to run again: rows are upserted, so nothing is duplicated, and values that a later stage has written
(corrected forecast, probabilities, phase, ...) are not overwritten.

`DATABASE_URL` (environment or .env) says where to load. PostgreSQL needs the schema first:
psql -U sih -h localhost -d sih_rain -f database/schema.sql   (see database/README.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.db.loader import load_all  # noqa: E402
from backend.app.db.session import make_engine  # noqa: E402
from data_pipeline.districts import load_districts  # noqa: E402
from data_pipeline.ingestion import IngestionError, load_config  # noqa: E402
from data_pipeline.ingestion.config import env_value  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", help="load only these seasons (default: all on disk)"
    )
    parser.add_argument(
        "--districts",
        type=Path,
        help="district boundary file (default: source.file in config/districts.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    url = env_value("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL is not set (environment or .env). See backend/README.md.", file=sys.stderr)
        return 1
    config = load_config()
    try:
        districts = load_districts(args.districts, config) if args.districts else None
        report = load_all(make_engine(url), config, seasons=args.seasons, districts=districts)
    except IngestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"loaded: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
