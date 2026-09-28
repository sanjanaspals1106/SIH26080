"""Read-only verification report for `/verification` (PRD 16). Reads the two `metrics.json` files written by
`scripts/run_m3.py`: `data/verification/m3_dev/` (out-of-fold development seasons) and `data/verification/m3_holdout/`
(the locked 2025 holdout). Nothing here recomputes, re-runs or writes anything, and the holdout file is only ever
opened for reading.

The files hold point estimates pooled over the three lead days and all valid cells. They carry no confidence
intervals, no per-lead breakdown, no regional breakdown and no reliability bins, so none of those are returned.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.app.errors import ApiError

# key in metrics.json -> (label, description, forecast_type as used by the frontend)
MODELS: dict[str, tuple[str, str, str]] = {
    "raw_nwp": ("B0: Raw NWP", "ECMWF TIGGE control forecast, uncorrected", "raw_nwp"),
    "b1_quantile_mapping": ("B1: Quantile Mapping", "Empirical quantile calibration", "quantile_mapping"),
    "b2_global_ml": ("B2: Global ML", "XGBoost without regime features", "global_ml"),
    "b3_regime_aware": ("B3: Regime-Aware ML", "XGBoost with regime features (41 features)", "regime_aware_ml"),
    "final_corrected": (
        "Regime-Aware ML (final model)",
        "Final regime-aware model, fitted on 2021-2024 and applied once to 2025",
        "regime_aware_ml",
    ),
}
DEV_CORRECTED = "b3_regime_aware"
HOLDOUT_CORRECTED = "final_corrected"
SCALARS = ("rmse", "mae", "bias")
PER_THRESHOLD = ("pod", "far", "csi", "ets", "frequency_bias", "fss5")


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


@lru_cache(maxsize=1)
def _files(verification_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    dev = _read(verification_dir / "m3_dev" / "metrics.json")
    if dev is None:
        raise ApiError(503, "VERIFICATION_DATA_UNAVAILABLE", "Verification metrics have not been generated.")
    return dev, _read(verification_dir / "m3_holdout" / "metrics.json")


def _thresholds(model: dict[str, Any]) -> list[str]:
    found = {k.split("_", 1)[1] for k in model if k.startswith("pod_")}
    return sorted(found, key=float)


def _num(value: Any) -> float | None:
    return None if value is None else float(value)


def _model_entry(key: str, metrics: dict[str, Any]) -> dict[str, Any]:
    label, description, forecast_type = MODELS[key]
    by_threshold = {}
    for t in _thresholds(metrics):
        row = {name: _num(metrics.get(f"{name}_{t}")) for name in PER_THRESHOLD}
        row["n_obs_events"] = int(metrics[f"n_obs_events_{t}"]) if f"n_obs_events_{t}" in metrics else None
        by_threshold[t] = row
    return {
        "key": key, "label": label, "description": description, "forecast_type": forecast_type,
        "scalars": {name: _num(metrics.get(name)) for name in SCALARS}, "thresholds": by_threshold,
    }  # fmt: skip


def _highlights(raw: dict[str, Any], corrected: dict[str, Any], threshold_keys: list[str]) -> list[dict[str, str]]:
    """Plain statements generated from the numbers, comparing the corrected forecast with raw NWP."""
    out: list[dict[str, str]] = []
    r0, r1 = raw["rmse"], corrected["rmse"]
    pct = abs(r1 - r0) / r0 * 100
    out.append({
        "kind": "improved" if r1 < r0 else "declined",
        "text": f"RMSE {r0:.2f} to {r1:.2f} mm ({pct:.1f}% {'lower' if r1 < r0 else 'higher'} than raw NWP).",
    })  # fmt: skip
    for t in threshold_keys:
        e0, e1 = raw.get(f"ets_{t}"), corrected.get(f"ets_{t}")
        if e0 is None or e1 is None:
            continue
        out.append({
            "kind": "improved" if e1 > e0 else "declined",
            "text": f"Equitable threat score at {t} mm: {e0:.3f} to {e1:.3f}.",
        })  # fmt: skip
    fb = corrected.get("frequency_bias_64.5")
    if fb is not None and fb < 0.5:
        out.append({
            "kind": "note",
            "text": (
                f"The mean-corrected forecast issues about {fb * 100:.0f}% as many 64.5 mm exceedances as were "
                "observed (frequency bias "
                f"{fb:.2f}); heavy-rain guidance in this system is carried by the calibrated probability products."
            ),
        })  # fmt: skip
    return out


def _evaluation(
    name: str, label: str, seasons: list[int], correction: dict[str, Any], keys: list[str], corrected_key: str
) -> dict[str, Any]:
    thresholds = _thresholds(correction["raw_nwp"])
    return {
        "evaluation_set": name, "label": label, "seasons": seasons,
        "models": [_model_entry(k, correction[k]) for k in keys if k in correction],
        "corrected_key": corrected_key, "thresholds_mm": [float(t) for t in thresholds],
        "highlights": _highlights(correction["raw_nwp"], correction[corrected_key], thresholds),
    }  # fmt: skip


def get_verification(verification_dir: Path) -> dict[str, Any]:
    dev, holdout = _files(verification_dir)
    development = _evaluation(
        "development",
        "Development seasons (out-of-fold predictions)",
        [int(s) for s in dev["development_seasons"]],
        dev["correction"],
        ["raw_nwp", "b1_quantile_mapping", "b2_global_ml", "b3_regime_aware"],
        DEV_CORRECTED,
    )
    holdout_eval = None
    if holdout is not None:
        holdout_eval = _evaluation(
            "holdout",
            "2025 holdout (evaluated once under the locked protocol)",
            [int(s) for s in holdout["holdout_seasons"]],
            holdout["correction"],
            ["raw_nwp", "final_corrected"],
            HOLDOUT_CORRECTED,
        )
        lock = holdout.get("lock", {})
        holdout_eval["lock"] = {
            "timestamp": lock.get("timestamp"), "git_commit": None,
            "run_count": lock.get("run_count"), "forced_rerun": bool(lock.get("forced_rerun", False)),
        }  # fmt: skip

    counts = dev.get("event_counts", {})
    probability = [
        {
            "threshold_mm": float(t), "brier_uncalibrated": v["brier_uncalibrated"],
            "brier_calibrated": v["brier_calibrated"], "brier_skill_score": v["bss_vs_climatology"],
            "climatology_rate": v["climatology_rate"], "n_events": counts.get(t),
        }
        for t, v in sorted(dev.get("probability", {}).items(), key=lambda kv: float(kv[0]))
    ]  # fmt: skip
    coverage = [
        {
            "lead_day": int(v["lead_day"]), "coverage": v["coverage"], "target": v["target"],
            "tolerance": v["tolerance"], "n_samples": v["n_samples"], "within_tolerance": not v["needs_display_warning"],
        }
        for _, v in sorted(dev.get("range_coverage", {}).items(), key=lambda kv: int(kv[0]))
    ]  # fmt: skip
    return {
        "development": development,
        "holdout": holdout_eval,
        "probability": {"evaluation_set": "development", "rows": probability},
        "range_coverage": {"evaluation_set": "development", "interval": "q10-q90", "rows": coverage},
        "pooling": "All values are point estimates pooled over lead days 1-3 and all valid IMD cells.",
    }  # fmt: skip
