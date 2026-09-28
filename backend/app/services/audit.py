"""Read-only audit trail for `/forecasts/{forecast_id}/audit` (PRD 15 F2, rule H5). One forecast is
`<run_id>_L<lead>_<district_id>` (docs/data-contracts.md). Every number comes from stored products: the
`district_forecasts` / `district_history` rows, the regime service, the cell-level regime feature file, the
Golden Dataset (raw rainfall at the wettest cell) and the measured q10-q90 coverage from `m3_dev/metrics.json`.
Nothing is recomputed by a model and nothing is written.

The audit trail states what the model did. It does not claim to prove why the weather happened (rule H5).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import Connection, select

from backend.app.db.tables import district_forecasts as f
from backend.app.db.tables import district_history as h
from backend.app.db.tables import districts as d
from backend.app.errors import ApiError
from backend.app.services import regime as regime_service
from backend.app.services import verification as verification_service
from backend.app.services.runs import get_run_row

FORECAST_ID = re.compile(r"^(tigge_ecmwf_cf_\d{10})_L([1-3])_([A-Za-z0-9_-]{1,32})$")
LPS_NONE_KM = 3000.0  # the "no LPS" sentinel (PRD 11.4)
MIN_HISTORY_CASES = 10
PHASES = ["active", "normal", "break"]


def parse_forecast_id(forecast_id: str) -> tuple[str, int, str]:
    m = FORECAST_ID.match(forecast_id)
    if m is None:
        raise ApiError(422, "INVALID_FORECAST_ID", "Forecast id must look like <run_id>_L<lead>_<district_id>.")
    return m.group(1), int(m.group(2)), m.group(3)


def _round(value: Any, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if math.isnan(v) else round(v, ndigits)


def _raw_at_cell(golden_dir: Path, run_id: str, lead_day: int, cell_id: int) -> float | None:
    month = run_id[-10:-4]
    path = golden_dir / f"season_{month[:4]}" / f"golden_{month}.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(
        path, columns=["rain_mm"],
        filters=[("run_id", "=", run_id), ("lead_day", "=", lead_day), ("cell_id", "=", cell_id)],
    )  # fmt: skip
    return _round(table["rain_mm"][0].as_py()) if table.num_rows else None


def _cell_regime(regime_dir: Path, season: int, run_id: str, lead_day: int, cell_id: int) -> dict[str, Any] | None:
    path = regime_dir / "regime_features" / f"season_{season}.parquet"
    if not path.is_file():
        return None
    cols = ["distance_to_lps_km", "bearing_sin", "bearing_cos", "lps_influence", "orographic_influence", "coastal_influence"]
    table = pq.read_table(
        path, columns=cols,
        filters=[("run_id", "=", run_id), ("lead_day", "=", lead_day), ("cell_id", "=", cell_id)],
    )  # fmt: skip
    if table.num_rows == 0:
        return None
    r = table.to_pylist()[0]
    return {
        "distance_km": r["distance_to_lps_km"],
        "bearing_deg": (math.degrees(math.atan2(r["bearing_sin"], r["bearing_cos"])) + 360) % 360,
        "influence": r["lps_influence"],
        "orographic": r["orographic_influence"],
        "coastal": r["coastal_influence"],
    }


def _history(
    conn: Connection, regime_dir: Path, verification_dir: Path, district_id: str, lead_day: int, season: int,
    phase: str, lps_present: bool,
) -> dict[str, Any] | None:  # fmt: skip
    """Raw-forecast error (observed minus raw district mean) on past dates of the development seasons, for this
    district and lead, under the same dominant phase and low-pressure-system presence. Holdout dates are never used."""
    try:
        dev, _ = verification_service._files(verification_dir)
    except ApiError:
        return None
    dev_seasons = [int(s) for s in dev["development_seasons"] if int(s) != season]
    rows = conn.execute(
        select(h.c.run_id, h.c.observed_mean_mm, h.c.raw_mean_mm).where(
            h.c.district_id == district_id, h.c.lead_day == lead_day, h.c.season.in_(dev_seasons),
            h.c.observed_mean_mm.is_not(None), h.c.raw_mean_mm.is_not(None),
        )
    ).all()  # fmt: skip
    if not rows:
        return None
    past = pd.DataFrame(rows, columns=["run_id", "observed", "raw"])
    dom = regime_service._domain_table(regime_dir)
    dom = dom[dom["lead_day"] == lead_day][["run_id", "p_active", "p_normal", "p_break", "lps_detected"]].copy()
    dom["phase"] = np.array(PHASES)[dom[["p_active", "p_normal", "p_break"]].to_numpy().argmax(axis=1)]
    merged = past.merge(dom, on="run_id")
    same = merged[(merged["phase"] == phase) & (merged["lps_detected"].astype(bool) == lps_present)]
    if same.empty:
        return None
    diff = (same["observed"] - same["raw"]).to_numpy()
    n = len(diff)
    return {
        "phase": phase, "lps_near": lps_present, "n_dates": n,
        "median_diff_mm": _round(np.median(diff)), "q25_diff_mm": _round(np.percentile(diff, 25)),
        "q75_diff_mm": _round(np.percentile(diff, 75)),
        "note": (
            f"development seasons {', '.join(str(s) for s in dev_seasons)}"
            + ("; few past cases" if n < MIN_HISTORY_CASES else "")
        ),
    }  # fmt: skip


def _model_record(models_dir: Path, alignment_method: str, fallback_used: bool) -> dict[str, Any]:
    version, features = "regime-aware XGBoost (B3)", "regime-aware feature set"
    meta_path = models_dir / "b3" / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        version = "regime-aware XGBoost (B3)"
        features = f"{len(meta.get('feature_names', []))} regime-aware features"
    return {
        "model_version": version, "feature_set_version": features,
        "alignment_method": alignment_method, "fallback_used": fallback_used,
    }  # fmt: skip


def _coverage(verification_dir: Path, lead_day: int) -> float | None:
    try:
        dev, _ = verification_service._files(verification_dir)
    except ApiError:
        return None
    cov = dev.get("range_coverage", {}).get(str(lead_day))
    return _round(cov["coverage"], 4) if cov else None


def _summary(name: str, raw: float, corrected: float, regime: dict | None, heavy: float | None) -> str:
    delta = corrected - raw
    text = f"For {name}, the model {'raised' if delta > 0 else 'lowered' if delta < 0 else 'left unchanged'} the district mean"
    text += f" from {raw:.1f} to {corrected:.1f} mm ({delta:+.1f})." if delta else f" at {raw:.1f} mm."
    if regime:
        ph = regime["phase"]
        text += f" Most likely phase: {ph['dominant_phase']} (confidence {ph['confidence_band']})."
    if heavy is not None:
        text += f" Heavy-rain probability at the wettest cell: {round(heavy * 100)}%."
    return text


def get_audit(
    conn: Connection, forecast_id: str, *, data_dir: Path, golden_dir: Path
) -> dict[str, Any]:  # fmt: skip
    run_id, lead_day, district_id = parse_forecast_id(forecast_id)
    run = get_run_row(conn, run_id)
    row = conn.execute(
        select(f, d.c.name.label("district_name")).select_from(f.join(d, d.c.district_id == f.c.district_id)).where(
            f.c.run_id == run_id, f.c.lead_day == lead_day, f.c.district_id == district_id
        )
    ).first()  # fmt: skip
    if row is None:
        raise ApiError(404, "FORECAST_NOT_FOUND", "No forecast exists for this run, lead day and district.")
    m = row._mapping
    if m["raw_mean_mm"] is None or m["corrected_mean_mm"] is None:
        raise ApiError(404, "FORECAST_NOT_FOUND", "This forecast has no raw or corrected value stored.")

    regime_dir = data_dir / "regime"
    verification_dir = data_dir / "verification"
    cell_id = m["wettest_cell_id"]

    try:
        regime = regime_service.get_regime(regime_dir, golden_dir, data_dir / "serving" / "corrected", run_id, lead_day)
    except ApiError:
        regime = None  # regime data missing for this run/lead: the regime and history steps are left out

    raw_mean = _round(m["raw_mean_mm"])
    corrected_mean = _round(m["corrected_mean_mm"])
    raw_wet = _raw_at_cell(golden_dir, run_id, lead_day, int(cell_id)) if cell_id is not None else None
    corrected_wet = _round(m["wettest_cell_mean_mm"])

    regime_step = None
    history = None
    if regime is not None:
        cell = _cell_regime(regime_dir, run.season, run_id, lead_day, int(cell_id)) if cell_id is not None else None
        lps = regime["nearest_lps"]
        run_lps_present = bool(lps.get("present"))  # run-level flag: the same definition the history archive uses
        if cell is not None:  # nearest low-pressure system as seen from the wettest cell
            present = cell["distance_km"] < LPS_NONE_KM
            lps = {
                "present": present, "settings": lps.get("settings"),
                "distance_km": _round(cell["distance_km"], 1) if present else None,
                "bearing_deg": _round(cell["bearing_deg"], 1) if present else None,
                "influence": _round(cell["influence"], 3) if present else None,
            }  # fmt: skip
        phase = regime["phase"]
        regime_step = {
            "phase": {
                "active": phase["active"], "normal": phase["normal"], "break": phase["break"],
                "confidence_band": phase["confidence_band"],
            },
            "nearest_lps": lps,
            "orographic_influence": _round(cell["orographic"], 3) if cell else None,
            "coastal_influence": _round(cell["coastal"], 3) if cell else None,
            "regime_available": regime["regime_available"], "ood_flag": regime["ood_flag"],
            "note": regime.get("note"),
        }  # fmt: skip
        history = _history(
            conn, regime_dir, verification_dir, district_id, lead_day, run.season, phase["dominant_phase"],
            run_lps_present,
        )  # fmt: skip

    fallback_used = bool(m["fallback_used"]) if m["fallback_used"] is not None else False
    record = _model_record(data_dir / "models" / "m3" / "final", run.alignment_method, fallback_used)
    record["product_type"] = m["product_type"] or "regime_aware_ml"
    record["fallback_reason"] = m["fallback_reason"]

    heavy = m["heavy_prob_max_cell"]
    steps = {
        "raw": {"district_mean_mm": raw_mean, "wettest_cell_mm": raw_wet},
        "regime": regime_step,
        "history": history,
        "correction": {
            "district_mean_mm": _round(corrected_mean - raw_mean),
            "wettest_cell_mm": _round(corrected_wet - raw_wet) if corrected_wet is not None and raw_wet is not None else None,
        },
        "corrected": {"district_mean_mm": corrected_mean, "wettest_cell_mm": corrected_wet},
        "confidence": {
            "heavy_prob_max_cell": _round(heavy, 4),
            "very_heavy_prob_max_cell": _round(m["very_heavy_prob_max_cell"], 4),
            "range_wettest_cell_mm": {
                "q10": _round(m["wettest_cell_q10_mm"]), "q50": _round(m["wettest_cell_q50_mm"]),
                "q90": _round(m["wettest_cell_q90_mm"]),
            },
            "measured_coverage_q10_q90": _coverage(verification_dir, lead_day),
        },
        "record": record,
    }  # fmt: skip
    return {
        "forecast_id": forecast_id, "run_id": run_id, "evaluation_set": run.evaluation_set, "lead_day": lead_day,
        "imd_date": m["imd_date"].isoformat() if m["imd_date"] is not None else None,
        "district_id": district_id, "district_name": m["district_name"], "steps": steps,
        "summary": _summary(m["district_name"], raw_mean, corrected_mean, regime, heavy),
    }  # fmt: skip
