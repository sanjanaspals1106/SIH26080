#!/usr/bin/env python
"""Drive the M3 rainfall-correction models on real, merged data (PRD 10, 12, 13, 17).

Uses only the repo's own functions (ml.orchestration, ml.training.hyperparameter_search, probability.*,
verification.*); nothing here is a new model or metric. Two modes:

  python scripts/run_m3.py smoke   # small settings grid, a coarse cell stride: proves the flow works, minutes
  python scripts/run_m3.py full    # the real 20-configuration search, PRD default settings: hours

Both modes:
  1. load the merged M3 input frame (27 + 14 features, region_code, regime_source) for every season on disk;
  2. compute development event counts and the heavy-rain model-availability decision (PRD 13.2), and write
     docs/event-counts.md;
  3. settings search (PRD 12.2) for B2 and B3, each fold's clim_mean/clim_p95 refit on that fold's training
     seasons only (PRD 10.4, ml/fold_climatology.py);
  4. leave-one-season-out OOF predictions (B0, B1, B2, B3, probability, range) with the chosen settings;
  5. fit M4 probability calibrators on the pooled OOF predictions (never on holdout);
  6. verify OOF: raw vs B1 vs B2 vs B3 (RMSE, contingency metrics, FSS), range coverage, Brier/BSS;
  7. fit the final models on ALL development seasons and save them.

Holdout (a separate, explicit step, `--run-holdout`): acquires/reads the PRD 10.5 lock, predicts the holdout
season with the final models and calibrators, and verifies it the same way. Never run before dev results are
sound: PRD 10.5 says a repeated look at holdout after a change burns it.

Run from the repository root:
  python scripts/run_m3.py smoke --out-dir /tmp/m3_smoke
  python scripts/run_m3.py full  --out-dir ml/models/m3
  python scripts/run_m3.py full  --out-dir ml/models/m3 --run-holdout   # only once dev results are accepted
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment.cells import load_valid_cells  # noqa: E402
from data_pipeline.alignment.golden import read_golden  # noqa: E402
from data_pipeline.features import get_climatology  # noqa: E402
from data_pipeline.ingestion import load_config  # noqa: E402
from data_pipeline.ingestion.imd import imd_axes  # noqa: E402
from ml.feature_contracts import B2_FEATURES, B3_FEATURES  # noqa: E402
from ml.fold_climatology import ClimProvider  # noqa: E402
from ml.orchestration import fit_final_m3_models, predict_final_m3, run_development_oof_pipeline  # noqa: E402
from ml.training.hyperparameter_search import evaluate_settings_search, generate_shared_hyperparameter_configs  # noqa: E402
from probability.calibration import ProbabilityCalibrator  # noqa: E402
from probability.coverage import check_range_coverage  # noqa: E402
from protocol.locking import HoldoutLockedError, acquire_holdout_lock, read_lock_file  # noqa: E402
from protocol.splits import assign_season_splits  # noqa: E402
from regime_engine.pipeline import load_m3_frame  # noqa: E402
from verification.events import (  # noqa: E402
    DailyObservation,
    count_events_in_grid_series,
    determine_model_availability,
    load_rain_thresholds,
)
from verification.metrics.contingency import compute_contingency_counts, compute_contingency_metrics  # noqa: E402
from verification.metrics.continuous import compute_continuous_metrics  # noqa: E402
from verification.metrics.probabilistic import compute_brier_score, compute_brier_skill_score  # noqa: E402
from verification.metrics.spatial import aggregate_fss, compute_fss_components  # noqa: E402

THRESHOLDS = (15.6, 64.5, 115.6)
# acquire_holdout_lock()/run_holdout.py both default to <repo root>/holdout.lock (the .gitignore note about
# "protocol/holdout.lock" does not match that default; this follows the code, not the stale comment).
LOCK_PATH = Path(__file__).resolve().parent.parent / "holdout.lock"


def dev_event_counts(config, golden: pd.DataFrame, dev_seasons: list[int]) -> tuple[dict[float, int], object]:
    """Development-season event totals per threshold (PRD 10.7) and the model-availability decision (PRD 13.2).

    Same logic as `scripts/build_event_counts.py`, kept in step with it deliberately: both read the Golden
    Dataset and count 8-connected, date-merged events, never invented from the feature-table row counts.
    """
    valid = load_valid_cells(config)
    lat, lon = imd_axes(config.imd)
    n_lat, n_lon = lat.size, lon.size
    mask = np.zeros((n_lat, n_lon), dtype=bool)
    ids = valid.loc[valid["is_valid"], "cell_id"].to_numpy()
    mask[ids // n_lon, ids % n_lon] = True
    thresholds = load_rain_thresholds()
    dev = golden[golden["season"].isin(dev_seasons)]
    obs = []
    for (season, lead, date), rows in dev.groupby(["season", "lead_day", "imd_date"], sort=False):
        grid = np.full((n_lat, n_lon), np.nan)
        cid = rows["cell_id"].to_numpy()
        grid[cid // n_lon, cid % n_lon] = rows["obs_mm"].to_numpy()
        obs.append(DailyObservation(season=int(season), lead_day=int(lead), date=date, observed_rain=grid, valid_mask=mask))
    records = count_events_in_grid_series(obs, thresholds=thresholds)
    totals = {t: sum(r.n_events for r in records if r.threshold_mm == t) for t in thresholds}
    return totals, determine_model_availability(totals, min_events=30)


def metrics_table(pred: pd.DataFrame, cols: dict[str, str], n_lat: int, n_lon: int) -> dict:
    """RMSE/MAE/bias, contingency metrics and 5x5 FSS at every threshold, for each named column against obs_mm."""
    out = {}
    for name, col in cols.items():
        f, o = pred[col].to_numpy(), pred["obs_mm"].to_numpy()
        row = {k: v.value for k, v in compute_continuous_metrics(f, o).items()}
        for t in THRESHOLDS:
            c = compute_contingency_counts(f, o, t)
            row[f"n_obs_events_{t}"] = c.a + c.c
            row.update({f"{k}_{t}": v.value for k, v in compute_contingency_metrics(c).items()})
        fss = {}
        for t in (THRESHOLDS[0], THRESHOLDS[1]):
            comps = []
            for _, g in pred.groupby(["run_id", "lead_day"], sort=False):
                F = np.full(n_lat * n_lon, np.nan)
                O = np.full(n_lat * n_lon, np.nan)
                F[g["cell_id"].to_numpy()], O[g["cell_id"].to_numpy()] = g[col].to_numpy(), g["obs_mm"].to_numpy()
                comps.append(compute_fss_components(F.reshape(n_lat, n_lon), O.reshape(n_lat, n_lon), t, 5))
            fss[f"fss5_{t}"] = aggregate_fss(comps).value
        row.update(fss)
        out[name] = row
    return out


def fit_calibrators(oof: pd.DataFrame, availability) -> dict[float, ProbabilityCalibrator]:
    """One calibrator per available threshold, fitted on the pooled OOF predictions only (PRD 10.4, 13.4)."""
    cals: dict[float, ProbabilityCalibrator] = {}
    for t, has, col in ((15.6, availability.p_15_6_available, "uncalibrated_p_ge_15_6"),
                        (64.5, availability.p_64_5_available, "uncalibrated_p_ge_64_5"),
                        (115.6, availability.p_115_6_available, "uncalibrated_p_ge_115_6")):
        if not has:
            continue
        obs_bin = (oof["obs_mm"].to_numpy() >= t).astype(float)
        cal = ProbabilityCalibrator(threshold_mm=t)
        cal.fit(oof[col].to_numpy(), obs_bin, evaluation_set="development", prediction_source="oof")
        cals[t] = cal
    return cals


def probability_metrics(oof: pd.DataFrame, cals: dict[float, ProbabilityCalibrator]) -> dict:
    out = {}
    for t, cal in cals.items():
        col = f"uncalibrated_p_ge_{str(t).replace('.', '_')}"
        obs_bin = (oof["obs_mm"].to_numpy() >= t).astype(float)
        p_raw = oof[col].to_numpy()
        p_cal = cal.predict(p_raw)  # already clipped to [0, 1]
        clim = float(obs_bin.mean())
        bs_raw = compute_brier_score(p_raw, obs_bin)
        bs_cal = compute_brier_score(p_cal, obs_bin)
        bs_clim = compute_brier_score(np.full_like(obs_bin, clim), obs_bin)
        out[str(t)] = {
            "brier_uncalibrated": bs_raw.value, "brier_calibrated": bs_cal.value,
            "bss_vs_climatology": compute_brier_skill_score(bs_cal, bs_clim).value, "climatology_rate": clim,
        }
    return out


def range_metrics(oof: pd.DataFrame) -> dict:
    res = check_range_coverage(oof["lead_day"].to_numpy(), oof["obs_mm"].to_numpy(), oof["q10_mm"].to_numpy(),
                               oof["q50_mm"].to_numpy(), oof["q90_mm"].to_numpy())
    return {r.lead_day: r.to_dict() for r in res} if isinstance(res, (list, tuple)) else res


def run(mode: str, out_dir: Path, run_holdout: bool, force_lock: bool, seed: int) -> None:
    t0 = time.time()
    n_configs, stride, val_seasons_n = (2, 8, 3) if mode == "smoke" else (20, 1, 3)
    config = load_config()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    golden = read_golden(config)
    all_seasons = sorted(int(s) for s in golden["season"].unique())
    split = assign_season_splits(all_seasons)
    dev, hold = split.development_seasons, split.holdout_seasons
    print(f"seasons on disk: {all_seasons} | development {dev} | holdout {hold} | status {split.status}")

    print("loading the merged M3 frame (27+14 features, region_code, regime_source) ...")
    frame = load_m3_frame(config, all_seasons)
    dev_df, hold_df = frame[frame["season"].isin(dev)].copy(), frame[frame["season"].isin(hold)].copy()
    print(f"development rows {len(dev_df):,} | holdout rows {len(hold_df):,}")

    print("counting development events (PRD 10.7, 13.2) ...")
    counts, availability = dev_event_counts(config, golden, dev)
    print("event counts:", counts, "| availability:", availability)
    print("(docs/event-counts.md is not touched here: it is a tracked, committed record meant to be regenerated "
          "deliberately from real seasons with `python scripts/build_event_counts.py`, not as a side effect of a "
          "training run, and never from a smoke run's synthetic data.)")

    clim: ClimProvider = lambda seasons: get_climatology(seasons, config)  # noqa: E731
    lat, lon = imd_axes(config.imd)
    n_lat, n_lon = lat.size, lon.size

    out_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, dict] = {}
    for model_type in ("B2", "B3"):
        print(f"settings search: {model_type} ({n_configs} configs x {val_seasons_n} validation seasons, stride {stride}) ...")
        combos = generate_shared_hyperparameter_configs(seed=seed, n_configs=n_configs)
        result = evaluate_settings_search(dev_df, dev, combinations=combos, model_type=model_type,
                                          n_validation_seasons=val_seasons_n, train_cell_stride=stride, clim_provider=clim)
        best[model_type] = result
        print(f"  best {model_type}: index {result['best_config_index']}, mean RMSE {result['best_mean_rmse']:.3f}, "
              f"config {result['best_config']}")
    (out_dir / "settings_search.json").write_text(json.dumps(best, indent=2, default=float))

    print("leave-one-season-out OOF predictions ...")
    oof = run_development_oof_pipeline(
        dev_df, dev, b2_params=best["B2"]["best_config"], b3_params=best["B3"]["best_config"],
        event_counts=counts, train_cell_stride=stride, clim_provider=clim,
    )
    oof = oof.merge(dev_df[["run_id", "lead_day", "cell_id"]], on=["run_id", "lead_day", "cell_id"], how="left")
    print(f"OOF rows {len(oof):,}")

    print("fitting probability calibrators on the pooled OOF predictions ...")
    cals = fit_calibrators(oof, availability)

    print("verifying OOF (DEVELOPMENT ONLY) ...")
    cols = {"raw_nwp": "raw_mm", "b1_quantile_mapping": "b1_corrected_mm", "b2_global_ml": "b2_corrected_mean_mm",
            "b3_regime_aware": "b3_corrected_mean_mm"}
    dev_results = {
        "correction": metrics_table(oof, cols, n_lat, n_lon),
        "probability": probability_metrics(oof, cals),
        "range_coverage": range_metrics(oof),
        "settings": {k: v["best_config"] for k, v in best.items()},
        "event_counts": counts, "availability": availability.to_dict(),
        "development_seasons": dev, "holdout_seasons": hold, "n_configs": n_configs, "train_cell_stride": stride,
    }
    verif_dir = config.data_dir / "verification" / "m3_dev"
    verif_dir.mkdir(parents=True, exist_ok=True)
    (verif_dir / "metrics.json").write_text(json.dumps(dev_results, indent=2, default=float))
    oof.to_parquet(verif_dir / "oof_predictions.parquet", index=False)
    for name, m in dev_results["correction"].items():
        print(f"  {name:20s} RMSE {m['rmse']:.3f} | ETS64.5 {m.get('ets_64.5')} | FSS64.5 {m.get('fss5_64.5')}")

    print("fitting the final models on all development seasons ...")
    final = fit_final_m3_models(
        dev_df, dev, b2_params=best["B2"]["best_config"], b3_params=best["B3"]["best_config"],
        event_counts=counts, output_dir=out_dir / "final", feature_set_version="v1", git_commit=commit,
        dev_scores=dev_results, train_cell_stride=stride,
    )
    import pickle

    with open(out_dir / "final" / "calibrators.pkl", "wb") as f:
        pickle.dump(cals, f)
    print(f"final models saved to {out_dir / 'final'} ({time.time() - t0:.0f}s so far)")

    if not run_holdout:
        print("holdout NOT run (pass --run-holdout once development results are accepted). STOP.")
        return

    print("acquiring/reading the PRD 10.5 holdout lock ...")
    try:
        lock = acquire_holdout_lock(lock_path=LOCK_PATH, git_commit=commit, force=force_lock,
                                    allow_uncommitted="+uncommitted" in commit or not commit)
        print("holdout lock acquired:", lock)
    except HoldoutLockedError:
        lock = read_lock_file(LOCK_PATH)
        print("holdout already locked, reusing:", lock)

    print("predicting the holdout season with the final models (regime_source='final') ...")
    if not cals:
        print("WARNING: no threshold has enough development events for a calibrator (PRD 13.2); "
              "serving uncalibrated probabilities explicitly (allow_uncalibrated=True).")
    # predict_final_m3/assemble_serving_grid assume one forecast run at a time (its (cell_id, lead_day) uniqueness
    # contract, PRD 20.1); the holdout season is many runs, so it is called once per run_id and the parts concatenated.
    # obs_mm is not one of predict_final_m3's own outputs (it is unknown at real serving time), so it is attached
    # afterwards from the same rows, in the same order (every predict_* function preserves row order).
    serving_dir = config.data_dir / "serving" / "grid"
    parts = []
    for run_id, sub in hold_df.groupby("run_id", sort=False):
        sub = sub.reset_index(drop=True)  # predict_final_m3 rebuilds a 0-based index internally; align to it
        s = predict_final_m3(final, sub, calibrators=cals or None, allow_uncalibrated=not cals,
                             output_serving_dir=serving_dir).reset_index(drop=True)
        s["run_id"], s["obs_mm"] = run_id, sub["obs_mm"].to_numpy()
        parts.append(s)
    serving = pd.concat(parts, ignore_index=True)
    holdout_cols = {"raw_nwp": "raw_mm", "final_corrected": "corrected_mean_mm"}
    hold_results = {"correction": metrics_table(serving, holdout_cols, n_lat, n_lon), "holdout_seasons": hold,
                    "lock": lock.__dict__ if hasattr(lock, "__dict__") else str(lock)}
    hverif = config.data_dir / "verification" / "m3_holdout"
    hverif.mkdir(parents=True, exist_ok=True)
    (hverif / "metrics.json").write_text(json.dumps(hold_results, indent=2, default=float))
    print("HOLDOUT (LOCKED) results:")
    for name, m in hold_results["correction"].items():
        print(f"  {name:20s} RMSE {m['rmse']:.3f} | ETS64.5 {m.get('ets_64.5')} | FSS64.5 {m.get('fss5_64.5')}")
    print(f"total time {time.time() - t0:.0f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["smoke", "full"])
    p.add_argument("--out-dir", type=Path, default=Path("ml/models/m3"))
    p.add_argument("--run-holdout", action="store_true", help="also predict and verify the holdout (PRD 10.5 lock)")
    p.add_argument("--force-lock", action="store_true", help="re-lock the holdout even if already locked (logged)")
    p.add_argument("--seed", type=int, default=None, help="default: config/protocol.yaml random_seed")
    args = p.parse_args()
    from ml.training.hyperparameter_search import resolve_protocol_seed

    run(args.mode, args.out_dir, args.run_holdout, args.force_lock, resolve_protocol_seed(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
