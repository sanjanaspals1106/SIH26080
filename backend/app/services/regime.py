"""Read-only regime data for `/regime` and `/regime/transitions` (PRD 15 F1-F4). Reads `data/regime/`, written
once by `scripts/build_regime.py`; never recomputes, retrains or writes anything back.

`regime_domain.parquet` (one row per real forecast run and lead, ~1,800 rows total) is loaded once and cached in
memory -- it is tiny and does not change while the server runs. Per-cell lookups (the nearest low-pressure system)
read the season's `regime_features/season_<Y>.parquet` file, filtered to one (run_id, lead_day) at a time, the
same pattern `services/grid.py` already uses for the cell-level Golden Dataset files.

This module does NOT import `regime_engine` (M2's own package): `regime_engine/__init__.py` eagerly imports the
phase model, which imports scikit-learn, and the API process must never pull that in at startup (PRD 18 keeps
the API process free of training/model code; see `tests/test_backend_integration.py`). `_detect_transitions`
below is therefore a local, read-only port of `regime_engine/transitions/detector.py::TransitionDetector
.detect_transitions` (PRD 15 F4 / Appendix B) -- same rolling-mean smoothing, same confirmation rule, same event
shape -- minus the optional `district_centroid`/`distance_to_lps_km` LPS_NEAR_DISTRICT branch, which is unused
here (`scope` is always "domain"; no per-district distance series is ever passed in).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from backend.app.errors import ApiError

PHASES = ["active", "normal", "break"]
EARTH_RADIUS_KM = 6371.0
LPS_AGREEMENT_KM = 25.0  # PRD-adjacent tolerance (Appendix B's own merge_km is 500km; this is far tighter on purpose
SMOOTHING_DAYS = 3
CONFIRM_MIN_PROB = 0.50


@lru_cache(maxsize=1)
def _domain_table(regime_dir: Path) -> pd.DataFrame:
    path = regime_dir / "regime_domain.parquet"
    if not path.is_file():
        raise ApiError(503, "REGIME_DATA_UNAVAILABLE", "Regime domain data has not been generated.")
    df = pd.read_parquet(path)
    df["imd_date"] = pd.to_datetime(df["imd_date"])
    return df


@lru_cache(maxsize=1)
def _labels(regime_dir: Path) -> pd.DataFrame:
    path = regime_dir / "labels.parquet"
    if not path.is_file():
        raise ApiError(503, "REGIME_DATA_UNAVAILABLE", "Regime labels have not been generated.")
    return pd.read_parquet(path)


def _dominant(row: pd.Series) -> str:
    return PHASES[int(np.argmax([row["p_active"], row["p_normal"], row["p_break"]]))]


def _golden_mean(golden_dir: Path, run_id: str, lead_day: int, column: str) -> float | None:
    month = run_id[-10:-4]
    path = golden_dir / f"season_{month[:4]}" / f"golden_{month}.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(path, columns=[column], filters=[("run_id", "=", run_id), ("lead_day", "=", lead_day)])
    if table.num_rows == 0:
        return None
    return round(float(pc.mean(table[column]).as_py()), 3)


def _corrected_mean(corrected_dir: Path, run_id: str, lead_day: int) -> float | None:
    month = run_id[-10:-4]
    path = corrected_dir / f"season_{month[:4]}" / f"corrected_{month}.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(
        path, columns=["corrected_mean_mm"], filters=[("run_id", "=", run_id), ("lead_day", "=", lead_day)]
    )
    if table.num_rows == 0:
        return None
    return round(float(pc.mean(table["corrected_mean_mm"]).as_py()), 3)


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """The point `distance_km` away from (lat, lon) on bearing `bearing_deg` (forward geodesic, spherical earth).
    `bearing_deg` is measured cell -> target, matching `compute_bearing_and_components`'s own convention."""
    phi1, lam1, theta = math.radians(lat), math.radians(lon), math.radians(bearing_deg)
    d_r = distance_km / EARTH_RADIUS_KM
    phi2 = math.asin(math.sin(phi1) * math.cos(d_r) + math.cos(phi1) * math.sin(d_r) * math.cos(theta))
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(d_r) * math.cos(phi1), math.cos(d_r) - math.sin(phi1) * math.sin(phi2)
    )
    return math.degrees(phi2), math.degrees(lam2)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2, dphi, dlmb = map(math.radians, (lat1, lat2, lat2 - lat1, lon2 - lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_lps(regime_dir: Path, season: int, run_id: str, lead_day: int, lps_settings: str | None) -> dict:
    """Distance/bearing/influence from the closest cell, plus a best-effort triangulated centre: several cells'
    (own position + distance_to_lps_km + bearing) are projected outward; if the projections agree within
    `LPS_AGREEMENT_KM`, their average is returned as the centre, otherwise latitude/longitude stay null."""
    path = regime_dir / "regime_features" / f"season_{season}.parquet"
    if not path.is_file():
        return {"present": False, "settings": lps_settings}
    cols = ["cell_id", "distance_to_lps_km", "bearing_sin", "bearing_cos", "lps_strength", "lps_influence"]
    table = pq.read_table(path, columns=cols, filters=[("run_id", "=", run_id), ("lead_day", "=", lead_day)])
    if table.num_rows == 0:
        return {"present": False, "settings": lps_settings}
    df = table.to_pandas()
    present = df["distance_to_lps_km"].lt(3000.0).any()  # 3000 km is the "no LPS" sentinel (PRD 11.4)
    if not present:
        return {"present": False, "settings": lps_settings}

    nearest = df.loc[df["distance_to_lps_km"].idxmin()]
    bearing_deg = float((math.degrees(math.atan2(nearest["bearing_sin"], nearest["bearing_cos"])) + 360) % 360)
    out = {
        "present": True,
        "distance_km": round(float(nearest["distance_to_lps_km"]), 1),
        "bearing_deg": round(bearing_deg, 1),
        "influence": round(float(nearest["lps_influence"]), 4),
        "settings": lps_settings,
    }

    grid = pd.read_parquet(regime_dir.parent / "golden" / "grid_cells.parquet")[["cell_id", "latitude", "longitude"]]
    close = df.nsmallest(6, "distance_to_lps_km").merge(grid, on="cell_id", how="left").dropna(subset=["latitude"])
    projections = []
    for _, r in close.iterrows():
        b = (math.degrees(math.atan2(r["bearing_sin"], r["bearing_cos"])) + 360) % 360
        projections.append(_destination_point(r["latitude"], r["longitude"], b, r["distance_to_lps_km"]))
    if len(projections) >= 2:
        lats, lons = zip(*projections)
        centre = (float(np.mean(lats)), float(np.mean(lons)))
        max_spread = max(_haversine_km(*p, *centre) for p in projections)
        if max_spread <= LPS_AGREEMENT_KM:
            out["latitude"], out["longitude"] = round(centre[0], 3), round(centre[1], 3)
    return out


def get_regime(regime_dir: Path, golden_dir: Path, corrected_dir: Path, run_id: str, lead_day: int) -> dict:
    dom = _domain_table(regime_dir)
    row = dom[(dom["run_id"] == run_id) & (dom["lead_day"] == lead_day)]
    if row.empty:
        raise ApiError(404, "REGIME_NOT_FOUND", "No regime data for this run and lead day.")
    r = row.iloc[0]

    phase = {
        "active": round(float(r["p_active"]), 4), "normal": round(float(r["p_normal"]), 4),
        "break": round(float(r["p_break"]), 4), "confidence": round(float(r["regime_confidence"]), 4),
        "confidence_band": r["confidence_band"], "dominant_phase": _dominant(r),
    }  # fmt: skip
    note = None
    if not bool(r["regime_available"]):
        note = "regime_available is false for this run/lead: the values above are the engine's fallback output, not a genuine regime read."

    nearest_lps = (
        _nearest_lps(regime_dir, int(r["season"]), run_id, lead_day, r["lps_settings"])
        if r["lps_detected"]
        else {"present": False, "settings": r["lps_settings"]}
    )

    return {
        "lead_day": lead_day,
        "imd_date": r["imd_date"].strftime("%Y-%m-%d"),
        "phase": phase,
        "nearest_lps": nearest_lps,
        "indicators": [],
        "domain_mean_raw_mm": _golden_mean(golden_dir, run_id, lead_day, "rain_mm"),
        "domain_mean_corrected_mm": _corrected_mean(corrected_dir, run_id, lead_day),
        "regime_available": bool(r["regime_available"]),
        "ood_flag": bool(r["ood_flag"]),
        "note": note,
    }


def _detect_transitions(daily_series_df: pd.DataFrame) -> dict[str, Any]:
    """Local port of `TransitionDetector.detect_transitions` (see module docstring for why). `daily_series_df`
    columns: imd_date, p_active, p_normal, p_break, lps_present, sorted chronologically."""
    if daily_series_df.empty:
        return {"series": [], "events": []}

    df = daily_series_df.sort_values("imd_date").reset_index(drop=True).copy()
    df["dt"] = pd.to_datetime(df["imd_date"]).dt.date

    n = len(df)
    has_gap = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if (df.loc[i, "dt"] - df.loc[i - 1, "dt"]).days != 1:
            has_gap[i] = True

    smooth = np.column_stack([
        df[col].rolling(window=SMOOTHING_DAYS, min_periods=1, center=True).mean().to_numpy()
        for col in ("p_active", "p_normal", "p_break")
    ])
    states = [PHASES[int(np.argmax(smooth[i]))] for i in range(n)]
    df["state"] = states

    def calc_rain_change(event_idx: int) -> tuple[float | None, int, int]:
        return None, 0, 0  # domain_mean_corrected_mm is never in this series; see module docstring

    events: list[dict] = []
    event_counter = 1

    for d_idx in range(1, n):
        if has_gap[d_idx]:
            continue
        state_prev, state_curr = states[d_idx - 1], states[d_idx]
        if state_curr == state_prev:
            continue
        phase_idx = PHASES.index(state_curr)
        prob_d = float(smooth[d_idx, phase_idx])
        if prob_d < CONFIRM_MIN_PROB:
            continue
        if (d_idx + 1) < n and not has_gap[d_idx + 1]:
            prob_d1 = float(smooth[d_idx + 1, phase_idx])
            if prob_d1 < CONFIRM_MIN_PROB:
                continue
            conf_score, confirmed = round((prob_d + prob_d1) / 2.0, 2), True
        else:
            conf_score, confirmed = round(prob_d, 2), False
        change_val, n_bef, n_aft = calc_rain_change(d_idx)
        events.append({
            "event_id": event_counter, "event_type": f"PHASE:{state_prev}->{state_curr}",
            "from_state": state_prev, "to_state": state_curr, "event_date": str(df.loc[d_idx, "imd_date"]),
            "confirmed": confirmed, "confidence": conf_score, "district_id": None,
            "domain_mean_corrected_change_mm": change_val, "n_dates_before": n_bef, "n_dates_after": n_aft,
        })  # fmt: skip
        event_counter += 1

    lps_present = df["lps_present"].to_numpy(dtype=bool)
    for d_idx in range(1, n):
        if has_gap[d_idx]:
            continue
        lps_prev, lps_curr = lps_present[d_idx - 1], lps_present[d_idx]
        forms = not lps_prev and lps_curr and (d_idx + 1) < n and not has_gap[d_idx + 1] and lps_present[d_idx + 1]
        ends = lps_prev and not lps_curr and (d_idx + 1) < n and not has_gap[d_idx + 1] and not lps_present[d_idx + 1]
        if not (forms or ends):
            continue
        change_val, n_bef, n_aft = calc_rain_change(d_idx)
        events.append({
            "event_id": event_counter, "event_type": "LPS_FORMS" if forms else "LPS_ENDS",
            "from_state": "none" if forms else "detected", "to_state": "detected" if forms else "none",
            "event_date": str(df.loc[d_idx, "imd_date"]), "confirmed": True, "confidence": None,
            "district_id": None, "domain_mean_corrected_change_mm": change_val,
            "n_dates_before": n_bef, "n_dates_after": n_aft,
        })  # fmt: skip
        event_counter += 1

    series_out = [{
        "imd_date": str(df.loc[i, "imd_date"]), "p_active": round(float(df.loc[i, "p_active"]), 3),
        "p_normal": round(float(df.loc[i, "p_normal"]), 3), "p_break": round(float(df.loc[i, "p_break"]), 3),
        "lps_present": bool(df.loc[i, "lps_present"]),
    } for i in range(n)]  # fmt: skip

    return {"series": series_out, "events": events}


def _series_from_domain(dom: pd.DataFrame, season: int) -> pd.DataFrame:
    d1 = dom[(dom["season"] == season) & (dom["lead_day"] == 1)].sort_values("imd_date")
    return pd.DataFrame({
        "imd_date": d1["imd_date"], "p_active": d1["p_active"], "p_normal": d1["p_normal"],
        "p_break": d1["p_break"], "lps_present": d1["lps_detected"],
    })  # fmt: skip


def _series_from_labels(regime_dir: Path, season: int) -> pd.DataFrame:
    lab = _labels(regime_dir)
    season_rows = lab[(lab["year"] == season) & (~lab["is_base"])].copy()  # never the 1981-2010 base years
    onehot = {p: (season_rows["phase_label"] == p).astype(float) for p in PHASES}
    return pd.DataFrame({
        "imd_date": season_rows.index, "p_active": onehot["active"], "p_normal": onehot["normal"],
        "p_break": onehot["break"], "lps_present": False,  # labels.parquet carries no LPS information
    })  # fmt: skip


def get_transitions(regime_dir: Path, run_id: str, season: int, district_id: str | None) -> dict:
    dom = _domain_table(regime_dir)
    if district_id and not (dom["run_id"] == run_id).any():
        raise ApiError(404, "RUN_NOT_FOUND", "Forecast run was not found in the regime data.")

    forecast_df = _series_from_domain(dom, season)
    if forecast_df.empty:
        raise ApiError(404, "REGIME_NOT_FOUND", f"No regime domain rows for season {season}.")
    gaps = int((forecast_df["imd_date"].diff().dt.days.dropna() != 1).sum())
    if gaps:
        raise ApiError(
            503, "REGIME_SERIES_GAP",
            f"The lead-1 forecast chain for season {season} has {gaps} gap(s); refusing to synthesize a fake continuous series.",
        )  # fmt: skip

    forecast = _detect_transitions(forecast_df)
    observed = _detect_transitions(_series_from_labels(regime_dir, season))
    for result in (forecast, observed):
        for pt in result["series"]:
            pt["imd_date"] = str(pt["imd_date"])[:10]
        for ev in result["events"]:
            ev["event_date"] = str(ev["event_date"])[:10]

    return {
        "series": forecast["series"], "events": forecast["events"],
        "observed_imd": {"series": observed["series"], "events": observed["events"]},
        "scope": "domain", "district_id": district_id,
    }  # fmt: skip
