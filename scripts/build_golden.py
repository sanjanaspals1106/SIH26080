#!/usr/bin/env python
"""Stage 2 on real data (PRD 8, 9.3). Reads the raw files that Stage 1 downloaded under <DATA_DIR>/raw.

  python scripts/build_golden.py mask             # valid cells from IMD 1981-2010 (once; needs those IMD years)
  python scripts/build_golden.py season 2024      # Golden Dataset of one season (needs TIGGE + IMD + static files)
  python scripts/build_golden.py lag 2024         # T4 lag test on a built season (prints the three averages)
  python scripts/build_golden.py check            # every built season uses exactly the cells of the PRD mask

`season` refuses a mask that was not computed from the PRD base period (1981-2010) unless --allow-dev-mask is given.
`season` also takes --tigge-dir (hand-downloaded layout <dir>/<year>/{tp_YYYYMM,single_MM,pressure_MM}.grib instead
of <DATA_DIR>/raw/tigge), --static-grib (a GRIB holding orog and lsm, instead of golden.static_date) and, for the
GRIB packing noise of TIGGE tp, --tp-tolerance-mm / --clip-max-share (default: the values of alignment.yaml).
IMD years are read from .grd or NetCDF files, whichever is on disk.

Run from the repository root. Nothing here downloads data; use the Stage 1 download functions first.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment import build_golden_season, check_mask_consistency, get_valid_cells  # noqa: E402
from data_pipeline.alignment.lag import lag_test_season  # noqa: E402
from data_pipeline.alignment.cells import GRID_FILE  # noqa: E402
from data_pipeline.ingestion import IngestionError, load_config, read_static_from_grib  # noqa: E402


def _keep_other_mask(config) -> None:
    """Recomputing the mask replaces `grid_cells.parquet`. If the file there was made from other base years (a
    development mask), keep a copy next to it first, so nothing built on it is orphaned."""
    path = config.alignment.golden_dir / GRID_FILE
    if not path.is_file():
        return
    from data_pipeline.alignment import load_valid_cells

    base = str(load_valid_cells(config).attrs.get("base_years"))
    want = str([config.alignment.base_start_year, config.alignment.base_end_year])
    if base != want:
        keep = path.with_name("grid_cells_base_" + "-".join(re.findall(r"\d+", base)) + ".parquet")
        shutil.copy2(path, keep)
        print(f"NOTE: existing mask (base years {base}) copied to {keep.name} before it is replaced")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mask", help="compute and save the valid-cell mask")
    p_season = sub.add_parser("season", help="build the Golden Dataset for one season")
    p_season.add_argument("year", type=int)
    p_season.add_argument("--months", type=int, nargs="+", help="start months (default: JJAS)")
    p_season.add_argument("--allow-dev-mask", action="store_true", help="accept a mask made from other base years")
    p_season.add_argument("--tigge-dir", help="hand-downloaded TIGGE layout (see above)")
    p_season.add_argument("--static-grib", help="GRIB with orog and lsm")
    p_season.add_argument("--tp-tolerance-mm", type=float, help="T1: largest tp decrease that counts as noise")
    p_season.add_argument("--clip-max-share", type=float, help="T2: largest share of negative windows to clip")
    sub.add_parser("check", help="all built seasons use the cells of the saved PRD valid-cell mask")
    p_lag = sub.add_parser("lag", help="T4 lag test on one built season")
    p_lag.add_argument("year", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    try:
        if args.command == "mask":
            _keep_other_mask(config)
            grid = get_valid_cells(config, recompute=True)
            print(
                f"{int(grid['is_valid'].sum())} valid cells of {len(grid)}; written to {config.alignment.golden_dir}"
            )
        elif args.command == "season":
            overrides = {
                k: v
                for k, v in (
                    ("tp_decrease_tolerance_mm", args.tp_tolerance_mm),
                    ("clip_max_share", args.clip_max_share),
                )
                if v is not None
            }
            if overrides:
                print(f"NOTE: alignment checks overridden on the command line: {overrides}")
                config = dataclasses.replace(config, alignment=dataclasses.replace(config.alignment, **overrides))
            static = read_static_from_grib(args.static_grib) if args.static_grib else None
            for path in build_golden_season(
                args.year,
                config,
                months=args.months,
                static=static,
                tigge_dir=args.tigge_dir,
                require_base_period=not args.allow_dev_mask,
            ):
                print(path)
        elif args.command == "check":
            print(check_mask_consistency(config))
        else:
            print(lag_test_season(args.year, config).summary())
    except IngestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
