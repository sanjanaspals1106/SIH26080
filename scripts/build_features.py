#!/usr/bin/env python
"""Stage 3 on real data (PRD 8.6, 9.4 to 9.6, 14). Needs the Stage 2 outputs (Golden Dataset, valid-cell mask).

  python scripts/build_features.py static                          # slope, aspect, coast distance (needs golden.static_date)
  python scripts/build_features.py climatology 2016 2017 2018      # clim_mean / clim_p95 from TRAINING seasons only
  python scripts/build_features.py weights                         # district-cell weights (needs the district file, CK9)
  python scripts/build_features.py season 2025 --train 2016 2017 2018 [--holdout 2024 2025]

Every step caches its result under <DATA_DIR>/features and reuses it. Nothing here downloads data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.districts import get_district_weights  # noqa: E402
from data_pipeline.features import (  # noqa: E402
    FEATURE_VERSION,
    build_stage3_season,
    get_climatology,
    get_static_geography,
)
from data_pipeline.ingestion import IngestionError, load_config, read_static_from_grib  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_static = sub.add_parser("static")
    p_clim = sub.add_parser("climatology")
    p_clim.add_argument("seasons", type=int, nargs="+", help="training seasons")
    sub.add_parser("weights")
    p_season = sub.add_parser("season")
    p_season.add_argument("year", type=int)
    p_season.add_argument(
        "--train", type=int, nargs="+", required=True, help="training seasons for the climatology"
    )
    p_season.add_argument(
        "--holdout", type=int, nargs="*", default=[], help="seasons the climatology must not use"
    )
    for p_ in (p_static, p_season):
        p_.add_argument(
            "--static-grib",
            help="a GRIB with orog and lsm (for example data/tigge/2025/single_06.grib) instead of golden.static_date",
        )
    p_season.add_argument(
        "--skip-districts",
        action="store_true",
        help="write only the cell features (no district file needed yet)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    static = read_static_from_grib(args.static_grib) if getattr(args, "static_grib", None) else None
    try:
        if args.command == "static":
            print(f"static geography: {len(get_static_geography(config, static=static))} cells")
        elif args.command == "climatology":
            print(
                f"climatology from seasons {args.seasons}: {len(get_climatology(args.seasons, config))} cells"
            )
        elif args.command == "weights":
            w, s = get_district_weights(config=config)
            print(
                f"{len(s)} districts, {len(w)} district-cell weights; without valid cells: {s.attrs['districts_without_cells']}"
            )
        else:
            out = build_stage3_season(
                args.year,
                args.train,
                config,
                holdout_seasons=args.holdout,
                skip_districts=args.skip_districts,
                static=static,
            )
            print(f"feature version {FEATURE_VERSION}")
            for kind, paths in out.items():
                for path in paths:
                    print(f"{kind}: {path}")
    except IngestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
