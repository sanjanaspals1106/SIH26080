#!/usr/bin/env python
"""Stage 2 on real data (PRD 8, 9.3). Reads the raw files that Stage 1 downloaded under <DATA_DIR>/raw.

  python scripts/build_golden.py mask             # valid cells from IMD 1981-2010 (once; needs those IMD years)
  python scripts/build_golden.py season 2024      # Golden Dataset of one season (needs TIGGE + IMD + static files)
  python scripts/build_golden.py lag 2024         # T4 lag test on a built season (prints the three averages)

Run from the repository root. Nothing here downloads data; use the Stage 1 download functions first.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment import build_golden_season, get_valid_cells  # noqa: E402
from data_pipeline.alignment.lag import lag_test_season  # noqa: E402
from data_pipeline.ingestion import IngestionError, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mask", help="compute and save the valid-cell mask")
    p_season = sub.add_parser("season", help="build the Golden Dataset for one season")
    p_season.add_argument("year", type=int)
    p_season.add_argument("--months", type=int, nargs="+", help="start months (default: JJAS)")
    p_lag = sub.add_parser("lag", help="T4 lag test on one built season")
    p_lag.add_argument("year", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    try:
        if args.command == "mask":
            grid = get_valid_cells(config, recompute=True)
            print(
                f"{int(grid['is_valid'].sum())} valid cells of {len(grid)}; written to {config.alignment.golden_dir}"
            )
        elif args.command == "season":
            for path in build_golden_season(args.year, config, months=args.months):
                print(path)
        else:
            print(lag_test_season(args.year, config).summary())
    except IngestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
