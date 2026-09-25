#!/usr/bin/env python
"""Development train + verify on the 2025 feature table (one season, chronological split).

Trains B1 (quantile mapping) and B2 (global XGBoost Tweedie, 27 features) with the repo's own code on
init dates <= 2025-08-10, predicts the holdout (init >= 2025-08-15), and verifies RAW vs B1 vs B2 against IMD
with the repo's own metric functions. B3 / probability / range models need the 14 regime features (not
available: no historical climatology) and are NOT trained here.

  python scripts/train_verify_2025.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.baselines.b1_quantile_mapping import QuantileMappingModel  # noqa: E402
from ml.feature_contracts import B2_FEATURES  # noqa: E402
from ml.inference.predict_correction import predict_b2  # noqa: E402
from ml.models.model_store import save_model  # noqa: E402
from ml.training.train_correction import train_b2_model  # noqa: E402
from data_pipeline.features.pipeline import read_features  # noqa: E402
from data_pipeline.ingestion.config import load_config  # noqa: E402
from verification.metrics.contingency import compute_contingency_counts, compute_contingency_metrics  # noqa: E402
from verification.metrics.continuous import compute_continuous_metrics  # noqa: E402
from verification.metrics.spatial import aggregate_fss, compute_fss_components  # noqa: E402

SPLIT = json.loads(Path("data/features/split_2025.json").read_text())
OUT_DIR = Path("ml/models/dev2025")
VERIF_DIR = Path("data/verification/dev2025")
B2_PARAMS = {}  # repo defaults of _build_tweedie_regressor (no hyper-parameter search: one season only)
THRESHOLDS = [15.6, 64.5, 115.6]
N_LAT, N_LON = 129, 135


def region_codes(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """First matching box of config/regions.yaml (cell centre), PRD 14.3."""
    rules = sorted(yaml.safe_load(Path("config/regions.yaml").read_text())["regions"], key=lambda r: r["order"])
    out = pd.Series(index=lat.index, dtype=object)
    for r in rules:
        rule, m = r["rule"], pd.Series(True, index=lat.index)
        if "lat_gte" in rule: m &= lat >= rule["lat_gte"]
        if "lat_lt" in rule: m &= lat < rule["lat_lt"]
        if "lon_gte" in rule: m &= lon >= rule["lon_gte"]
        if "lon_lt" in rule: m &= lon < rule["lon_lt"]
        out[m & out.isna()] = r["region_code"]
    assert out.notna().all()
    return out


def metrics(df: pd.DataFrame, col: str) -> dict:
    f, o = df[col].to_numpy(), df["obs_mm"].to_numpy()
    res = {k: v.value for k, v in compute_continuous_metrics(f, o).items()}
    for t in THRESHOLDS:
        c = compute_contingency_counts(f, o, t)
        res[f"counts_{t}"] = c.to_dict()
        res.update({f"{k}_{t}": v.value for k, v in compute_contingency_metrics(c).items()})
    # FSS 5x5 (~140 km), components summed over all (init, lead) fields, never averaged per day
    grids = {}
    for t in (15.6, 64.5):
        comps = []
        for _, g in df.groupby(["run_id", "lead_day"], sort=False):
            F = np.full(N_LAT * N_LON, np.nan); O = np.full(N_LAT * N_LON, np.nan)
            F[g["cell_id"].to_numpy()], O[g["cell_id"].to_numpy()] = g[col].to_numpy(), g["obs_mm"].to_numpy()
            comps.append(compute_fss_components(F.reshape(N_LAT, N_LON), O.reshape(N_LAT, N_LON), t, 5))
        grids[f"fss5_{t}"] = aggregate_fss(comps).value
    res.update(grids)
    return res


def main() -> int:
    config = load_config()
    t0 = time.time()
    df = read_features(config, seasons=[2025])
    df["init_date"] = pd.to_datetime(df["run_id"].str[-10:-2], format="%Y%m%d")
    df["region_code"] = region_codes(df["latitude"], df["longitude"])
    train = df[df["init_date"] <= SPLIT["train_last_init"]].reset_index(drop=True)
    hold = df[df["init_date"] >= SPLIT["holdout_first_init"]].reset_index(drop=True)
    assert train["init_date"].max() < hold["init_date"].min()
    assert train["obs_mm"].notna().all() and hold["obs_mm"].notna().all()
    print(f"train rows {len(train):,} ({train.run_id.nunique()} runs, {train.init_date.min():%F}..{train.init_date.max():%F})")
    print(f"holdout rows {len(hold):,} ({hold.run_id.nunique()} runs, {hold.init_date.min():%F}..{hold.init_date.max():%F})")

    t1 = time.time()
    b1 = QuantileMappingModel().fit(train, training_seasons=[2025])
    print(f"B1 fitted in {time.time() - t1:.1f}s")
    t1 = time.time()
    b2 = train_b2_model(train, B2_PARAMS)
    train_s = time.time() - t1
    print(f"B2 fitted in {train_s:.1f}s")

    pred = hold[["run_id", "lead_day", "cell_id", "latitude", "longitude", "region_code", "rain_mm", "obs_mm"]].copy()
    pred = pred.rename(columns={"rain_mm": "raw_mm"})
    pred["b1_mm"] = b1.predict(hold.rename(columns={})).to_numpy()
    pred["b2_mm"] = predict_b2(b2, hold).to_numpy()
    assert pred[["raw_mm", "b1_mm", "b2_mm"]].notna().all().all() and (pred[["b1_mm", "b2_mm"]] >= 0).all().all()

    results = {"raw_nwp": metrics(pred, "raw_mm"), "b1_quantile_mapping": metrics(pred, "b1_mm"),
               "b2_xgboost_tweedie": metrics(pred, "b2_mm")}
    for lead in (1, 2, 3):
        sub = pred[pred["lead_day"] == lead]
        results[f"by_lead_{lead}"] = {n: {k: v.value for k, v in compute_continuous_metrics(sub[c], sub["obs_mm"]).items()}
                                     for n, c in (("raw_nwp", "raw_mm"), ("b1", "b1_mm"), ("b2", "b2_mm"))}

    # artifacts
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() + "+uncommitted"
    meta = {
        "model_type": "b2", "feature_names": B2_FEATURES, "train_seasons": [2025],
        "git_commit": commit, "best_params": {"defaults": True, **B2_PARAMS},
        "metrics": {"holdout": results["b2_xgboost_tweedie"], "raw_nwp_holdout": results["raw_nwp"],
                    "scope": "DEVELOPMENT ONLY: one season (2025), chronological split; no regime features"},
    }
    save_model(b2, meta, OUT_DIR / "b2")
    b1.save(OUT_DIR / "b1_quantile_mapping.json")
    VERIF_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(VERIF_DIR / "holdout_predictions.parquet", index=False)
    summary = {"train_rows": len(train), "holdout_rows": len(hold), "train_last_init": SPLIT["train_last_init"],
               "holdout_first_init": SPLIT["holdout_first_init"], "b2_train_seconds": train_s, "results": results}
    (VERIF_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print("artifacts:", OUT_DIR, VERIF_DIR, f"(total {time.time() - t0:.0f}s)")

    def row(name, key):
        m = results[key]
        return (f"{name:22s} RMSE {m['rmse']:.3f} MAE {m['mae']:.3f} bias {m['bias']:+.3f} | "
                + " | ".join(f"@{t}: POD {m[f'pod_{t}']} FAR {m[f'far_{t}']} CSI {m[f'csi_{t}']} ETS {m[f'ets_{t}']}"
                             .replace("None", "n/a") for t in (15.6, 64.5))
                + f" | FSS5 15.6 {m['fss5_15.6']:.3f} 64.5 {m['fss5_64.5']:.3f}")
    for n, k in (("RAW", "raw_nwp"), ("B1 quantile map", "b1_quantile_mapping"), ("B2 xgb tweedie", "b2_xgboost_tweedie")):
        print(row(n, k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
