#!/usr/bin/env python
"""Development-only feature table for 2025 (one season, chronological train / holdout split).

Uses the repo's own build_features / write_month_tables. With a single season the PRD season-based split is
impossible, so the split is by forecast start date (see protocol constants below):

  train    : init 2025-06-01 .. 2025-08-10
  embargo  : init 2025-08-11 .. 2025-08-14   (dropped; > max lead, so no observation day is shared)
  holdout  : init 2025-08-15 .. 2025-09-30

`clim_mean` / `clim_p95` are computed from IMD rain on TRAINING observation days only (2025-06-01 ..
2025-08-13, the last obs day any training row uses), never from holdout days (PRD L4). Rows of the training
set still contain their own day in the climatology (unavoidable with one season); the holdout does not.

  python scripts/build_features_2025.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_golden_2025 import load_imd, load_static  # noqa: E402

from data_pipeline.alignment.cells import imd_grid  # noqa: E402
from data_pipeline.features.io import write_month_tables  # noqa: E402
from data_pipeline.features.library import FEATURE_SCHEMA, KEY_COLUMNS, build_features  # noqa: E402
from data_pipeline.features.pipeline import TABLES, _run_month, golden_month_files  # noqa: E402
from data_pipeline.features.static import get_static_geography  # noqa: E402
from data_pipeline.ingestion.config import load_config  # noqa: E402

YEAR = 2025
TRAIN_LAST_INIT = "2025-08-10"
HOLDOUT_FIRST_INIT = "2025-08-15"
CLIM_LAST_OBS_DAY = "2025-08-13"  # TRAIN_LAST_INIT + 3 leads
SPLIT_FILE = Path("data/features/split_2025.json")


def train_climatology(config) -> pd.DataFrame:
    rain = load_imd(config)["rain"].sel(time=slice(f"{YEAR}-06-01", CLIM_LAST_OBS_DAY))
    assert rain["time"].max() <= pd.Timestamp(CLIM_LAST_OBS_DAY) < pd.Timestamp(HOLDOUT_FIRST_INIT)
    block = rain.values.reshape(rain.sizes["time"], -1)
    has = np.isfinite(block).any(axis=0)
    with np.errstate(all="ignore"):
        mean, p95 = np.nanmean(block, axis=0), np.nanpercentile(block, 95, axis=0)
    table = pd.DataFrame(
        {
            "cell_id": imd_grid(config)["cell_id"].to_numpy(),
            "clim_mean": np.where(has, mean, np.nan).astype("float32"),
            "clim_p95": np.where(has, p95, np.nan).astype("float32"),
        }
    )
    table.attrs["seasons"] = [YEAR]
    return table


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    static_geo = get_static_geography(config, static=load_static())
    clim = train_climatology(config)
    print("climatology from IMD 2025-06-01..%s; cells with clim: %d" % (CLIM_LAST_OBS_DAY, clim["clim_mean"].notna().sum()))
    n = 0
    for file in golden_month_files(config, YEAR):
        golden = pq.read_table(file).to_pandas(date_as_object=False)
        feats = build_features(golden, static_geo, clim, config)
        write_month_tables(
            feats, TABLES["features"], FEATURE_SCHEMA, KEY_COLUMNS, _run_month(feats["run_id"]), config
        )
        n += len(feats)
        print(f"{file.name}: {len(feats)} feature rows")
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({
        "train_last_init": TRAIN_LAST_INIT, "holdout_first_init": HOLDOUT_FIRST_INIT,
        "climatology_obs_days": ["2025-06-01", CLIM_LAST_OBS_DAY],
    }, indent=2))
    print("TOTAL feature rows:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
