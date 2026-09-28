#!/usr/bin/env python
"""Fill the 'later stage' columns of district_forecasts/district_history from the trained M3 models.

data_pipeline/districts/products.py deliberately leaves corrected_mean_mm, the corrected wettest cell and the
heavy-rain probability fields NULL ("Stage 4 fills them; nothing is estimated here"). This script is that stage:
for every run on disk, it predicts with the final M3 models (same predict_final_m3 path as scripts/run_m3.py's
holdout step), aggregates the cell-level output to district level with the same area-weight formulas M1 already
uses for raw/obs, and writes the same district_forecasts / district_history Parquet files M1's own writer uses --
the backend loader (scripts/load_m1.py) then picks them up unchanged.

  python scripts/build_district_corrected.py --seasons 2021 2022 2023 2024 2025 --models-dir data/models/m3/final
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment.golden import read_golden  # noqa: E402
from data_pipeline.districts.products import (  # noqa: E402
    FORECAST_COLUMNS,
    FORECAST_SCHEMA,
    HISTORY_COLUMNS,
    HISTORY_SCHEMA,
    KEY,
    NULLABLE_INTS,
    _dtypes,
    _NULL_FLOATS,
    check_forecasts,
    district_forecasts,
)
from data_pipeline.districts.priority import assign_district_priorities, load_priority_thresholds  # noqa: E402
from data_pipeline.districts.weights import get_district_weights  # noqa: E402
from data_pipeline.features.io import write_month_tables  # noqa: E402
from data_pipeline.ingestion import load_config  # noqa: E402
from ml.orchestration import FinalM3Models, predict_final_m3  # noqa: E402
from regime_engine.pipeline import load_m3_frame  # noqa: E402

HEAVY_MM = 64.5
PRODUCT_TYPE = "regime_aware_ml"  # B3 serves every row here (docs/data-contracts.md)
PRIORITY_COLUMNS = ["run_id", "lead_day", "district_id", "attention_level", "priority_rank", "product_type"]


def corrected_cell_frame(config, seasons: list[int], models_dir: Path) -> pd.DataFrame:
    """(run_id, lead_day, cell_id, corrected_mean_mm, q10/50/90_mm, p_ge_64_5, p_ge_115_6) for every row."""
    frame = load_m3_frame(config, seasons)
    final = FinalM3Models.load(models_dir)
    cal_path = models_dir / "calibrators.pkl"
    cals = pickle.loads(cal_path.read_bytes()) if cal_path.is_file() else None
    parts = []
    t0 = time.time()
    for i, (rid, sub) in enumerate(frame.groupby("run_id", sort=False)):
        sub = sub.reset_index(drop=True)
        s = predict_final_m3(final, sub, calibrators=cals, allow_uncalibrated=not cals)
        s["run_id"], s["lead_day"] = rid, sub["lead_day"].to_numpy()
        parts.append(s[["run_id", "lead_day", "cell_id", "corrected_mean_mm", "q10_mm", "q50_mm", "q90_mm",
                        "p_ge_64_5", "p_ge_115_6"]])
        if (i + 1) % 100 == 0:
            print(f"  predicted {i + 1:,} runs ({time.time() - t0:.0f}s)")
    print(f"predicted all {i + 1:,} runs in {time.time() - t0:.0f}s")
    return pd.concat(parts, ignore_index=True)


def fill_forecasts(base: pd.DataFrame, corrected: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Fold the corrected cell frame into the M1 raw/obs district_forecasts frame (same area-weight formulas)."""
    w = weights[["district_id", "cell_id", "area_weight", "is_main"]].copy()
    # PRD 14.4 defines the wettest cell and the max-probability fields over the main cells (weight >= w_min). 24 large,
    # fragmented districts (e.g. Adilabad: 39 cells, none >= 5%) have no main cell, which left every one of those
    # fields NULL. For those districts only, every cell counts as a candidate: no cell is an edge sliver relative
    # to the others. Districts that do have main cells are untouched.
    w["is_main"] = w["is_main"] | ~w.groupby("district_id")["is_main"].transform("any")
    c = corrected.merge(w, on="cell_id", how="inner")
    c["w_corr"] = c["area_weight"] * c["corrected_mean_mm"].clip(lower=0)
    c["w_p64"] = c["area_weight"] * c["p_ge_64_5"].fillna(0)
    c["w_p115"] = c["area_weight"] * c["p_ge_115_6"].fillna(0)
    agg = c.groupby(KEY).agg(
        corrected_mean_mm=("w_corr", "sum"),
        heavy_area_fraction_expected=("w_p64", "sum"),
        very_heavy_area_fraction_expected=("w_p115", "sum"),
        heavy_prob_max_cell=("p_ge_64_5", lambda s: s[c.loc[s.index, "is_main"]].max()),
        very_heavy_prob_max_cell=("p_ge_115_6", lambda s: s[c.loc[s.index, "is_main"]].max()),
    ).reset_index()

    main = c[c["is_main"]].sort_values(
        KEY[:2] + ["district_id", "corrected_mean_mm", "cell_id"], ascending=[True, True, True, False, True]
    )
    wettest = main.drop_duplicates(KEY)[KEY + ["cell_id", "corrected_mean_mm", "q10_mm", "q50_mm", "q90_mm"]].rename(
        columns={"cell_id": "wettest_cell_id", "corrected_mean_mm": "wettest_cell_mean_mm",
                 "q10_mm": "wettest_cell_q10_mm", "q50_mm": "wettest_cell_q50_mm", "q90_mm": "wettest_cell_q90_mm"}
    )
    fill = agg.merge(wettest, on=KEY, how="left")

    out = base.drop(columns=_NULL_FLOATS + ["wettest_cell_id"]).merge(fill, on=KEY, how="left")
    out = out[FORECAST_COLUMNS].astype(_dtypes(FORECAST_SCHEMA)).astype(dict.fromkeys(NULLABLE_INTS, "Int32"))
    check_forecasts(out, load_config())
    return out


def priority_frame(fc: pd.DataFrame, names: pd.DataFrame, is_64_5_available: bool) -> pd.DataFrame:
    """PRD F5 attention level and rank for every (run, lead), via data_pipeline.districts.priority."""
    cfg = load_priority_thresholds()
    f = fc.merge(names[["district_id", "name", "state"]], on="district_id", how="left")
    if f[["corrected_mean_mm", "wettest_cell_mean_mm"]].isna().any().any():
        raise SystemExit("cannot rank districts: corrected_mean_mm / wettest_cell_mean_mm has gaps")

    def opt(v):
        return None if pd.isna(v) else float(v)

    out = []
    for (rid, lead), g in f.groupby(["run_id", "lead_day"], sort=False):
        recs = [
            {
                "district_id": r.district_id, "district_name": r.name, "state": r.state,
                "corrected_mean_mm": float(r.corrected_mean_mm), "wettest_cell_mean_mm": float(r.wettest_cell_mean_mm),
                "heavy_prob_max_cell": opt(r.heavy_prob_max_cell), "very_heavy_prob_max_cell": opt(r.very_heavy_prob_max_cell),
                "heavy_area_fraction_expected": opt(r.heavy_area_fraction_expected),
                "raw_mean_mm": opt(r.raw_mean_mm), "is_small": bool(r.is_small),
            }
            for r in g.itertuples(index=False)
        ]  # fmt: skip
        ranked = assign_district_priorities(recs, is_64_5_available=is_64_5_available, thresholds_config=cfg)
        out.append(pd.DataFrame({
            "run_id": rid, "lead_day": int(lead), "district_id": [d.district_id for d in ranked],
            "attention_level": [d.attention_level for d in ranked], "priority_rank": [d.priority_rank for d in ranked],
        }))  # fmt: skip
    res = pd.concat(out, ignore_index=True)
    res["product_type"] = PRODUCT_TYPE
    res["lead_day"] = res["lead_day"].astype("int8")
    res["priority_rank"] = res["priority_rank"].astype("int32")
    return res[PRIORITY_COLUMNS]


def write_priority(pri: pd.DataFrame, config) -> int:
    """Side table read by backend/app/db/loader.py::load_district_corrected (the M1 district_forecasts Parquet
    schema deliberately has no attention/priority columns)."""
    month = pri["run_id"].str[-10:-4]
    n = 0
    for yyyymm, part in pri.groupby(month, sort=True):
        path = config.data_dir / "serving" / "priority" / f"season_{yyyymm[:4]}" / f"priority_{yyyymm}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False, compression="zstd")
        n += len(part)
    return n


def is_64_5_available(models_dir: Path) -> bool:
    """From the M3 dev metrics availability block (PRD 13.2); True if the file is absent (all thresholds have events)."""
    import json

    path = Path("data/verification/m3_dev/metrics.json")
    if not path.is_file():
        return True
    return bool(json.loads(path.read_text()).get("availability", {}).get("p_64_5_available", True))


def build_history(fc: pd.DataFrame) -> pd.DataFrame:
    h = fc[["run_id", "lead_day", "district_id", "imd_date", "season", "observed_mean_mm", "observed_max_cell_mm",
           "raw_mean_mm", "corrected_mean_mm"]].copy()
    h = h[h["observed_mean_mm"].notna()]
    h["prediction_source"] = np.where(h["corrected_mean_mm"].notna(), "final", None)
    h["phase"], h["lps_near"] = None, None
    return h[HISTORY_COLUMNS]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.add_argument("--models-dir", type=Path, default=Path("data/models/m3/final"))
    args = p.parse_args()

    config = load_config()
    weights, summary = get_district_weights(config=config)
    cache = Path("/tmp/corrected_cell_frame.parquet")
    served = sorted((config.data_dir / "serving" / "corrected").glob("season_*/corrected_*.parquet"))
    if cache.is_file():
        corrected = pd.read_parquet(cache)
        print(f"reusing cached corrected cell-level predictions: {len(corrected):,} rows (delete {cache} to redo)")
    elif served:  # identical to the cache above: written from the same predict_final_m3 output
        corrected = pd.concat([pd.read_parquet(f) for f in served], ignore_index=True)
        print(f"reusing the served corrected cell-level store: {len(corrected):,} rows from {len(served)} files")
    else:
        print("computing corrected cell-level predictions for", args.seasons, "...")
        corrected = corrected_cell_frame(config, args.seasons, args.models_dir)
        corrected.to_parquet(cache, index=False)

    avail_64 = is_64_5_available(args.models_dir)
    for season in args.seasons:
        golden = read_golden(config, seasons=[season])
        base = district_forecasts(golden, weights, summary, config)
        filled = fill_forecasts(base, corrected[corrected["run_id"].isin(golden["run_id"].unique())], weights)
        hist = build_history(filled)
        n3 = write_priority(priority_frame(filled, summary, avail_64), config)
        month = filled["run_id"].str[-10:-4]
        n1 = write_month_tables(filled, "district_forecasts", FORECAST_SCHEMA, KEY, month, config)
        n2 = write_month_tables(hist, "district_history", HISTORY_SCHEMA, KEY, filled["run_id"].str[-10:-4], config)
        print(f"season {season}: {len(filled):,} district-forecast rows -> {len(n1)} files, "
              f"{len(hist):,} history rows -> {len(n2)} files, {n3:,} priority rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
