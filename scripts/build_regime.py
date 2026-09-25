#!/usr/bin/env python
"""Regime engine on real data (PRD 10.3, 11). Three steps, each cached under <DATA_DIR>/regime.

  python scripts/build_regime.py labels --base 1981 2010 --seasons 2021 2022 2023 2024 2025
      IMD-only phase labels + the July-August check of PRD 11.2 (needs IMD 1981-2010 and the season years, and the
      valid-cell mask of `scripts/build_golden.py mask`). Writes regime/labels.parquet and regime/label_check.json.
  python scripts/build_regime.py fields 2023 [--tigge-dir data/tigge]
      Full-grid C1 atmosphere windows of one season (atmosphere GRIB files only) -> regime/fields/fields_2023.nc
  python scripts/build_regime.py oof --dev 2021 2022 2023 2024 --holdout 2025 --static-grib data/tigge/2025/single_06.grib
      Leave-one-season-out regime probabilities and the 14 regime features. Development seasons are written with
      regime_source="oof", holdout seasons with "final". Every (run, lead) goes through RegimeEngine.process_run_lead.
      Writes regime/regime_features/season_<Y>.parquet, regime/regime_domain.parquet and the final engine to
      ml/models/regime/.

Run from the repository root. Nothing here downloads data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment import check_mask_consistency  # noqa: E402
from data_pipeline.ingestion import IngestionError, load_config, read_static_from_grib  # noqa: E402
from regime_engine import pipeline as rp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_lab = sub.add_parser("labels")
    p_lab.add_argument("--base", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                       help="label base years (default: base_period of config/regime.yaml)")
    p_lab.add_argument("--seasons", type=int, nargs="+", required=True)
    p_fld = sub.add_parser("fields")
    p_fld.add_argument("year", type=int)
    p_fld.add_argument("--tigge-dir", help="hand-downloaded layout <dir>/<year>/{single_MM,pressure_MM}.grib")
    p_fld.add_argument("--months", type=int, nargs="+")
    p_oof = sub.add_parser("oof")
    p_oof.add_argument("--dev", type=int, nargs="+", required=True)
    p_oof.add_argument("--holdout", type=int, nargs="*", default=[])
    p_oof.add_argument("--static-grib", required=True, help="GRIB with orog and lsm")
    p_oof.add_argument("--out", default="ml/models/regime")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    try:
        if args.command == "labels":
            rcfg = rp.load_regime_config()
            base = args.base or [rcfg["layer_a"]["label_base_period"][k] for k in ("start_year", "end_year")]
            labels, report = rp.build_phase_labels(config, range(base[0], base[1] + 1), args.seasons, rcfg)
            print(rp.write_phase_labels(labels, report, config))
            print(json.dumps(report, indent=2, default=float))
        elif args.command == "fields":
            print(rp.build_season_fields(args.year, config, args.tigge_dir, args.months))
        else:
            dev, hold = sorted(args.dev), sorted(args.holdout)
            print("mask check:", check_mask_consistency(config, seasons=dev + hold))  # one PRD mask for all seasons
            labels = rp.read_phase_labels(config)
            cells = rp.build_cell_inputs(config, read_static_from_grib(args.static_grib))
            cell_ids = cells["cell_id"].to_numpy()
            a_tables, inputs = [], {}
            for s in dev + hold:
                fields = rp.read_season_fields(s, config)
                a_tables.append(rp.lead_a_table(fields, config))
                inputs[s] = rp.season_inputs(fields, cell_ids, config)
                del fields
            out_dir = rp.regime_dir(config) / "regime_features"
            out_dir.mkdir(parents=True, exist_ok=True)

            def write(season: int, table: pd.DataFrame) -> None:
                path = out_dir / f"season_{season}.parquet"
                table.to_parquet(path, index=False, compression="zstd")
                print(f"season {season}: {len(table):,} rows, regime_source={table['regime_source'].iloc[0]} -> {path}")

            result = rp.run_regime_loso(
                pd.concat(a_tables, ignore_index=True), inputs, labels, cells, dev, hold, config, on_season_done=write
            )
            result.domain.to_parquet(rp.regime_dir(config) / "regime_domain.parquet", index=False)
            print("artifacts:", rp.save_regime_artifacts(result, args.out))
            print(f"phase model C={result.best_c}; folds: {result.folds}")
    except IngestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
