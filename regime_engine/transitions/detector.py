"""Regime Transition Detection (F4).

PRD Section 15 (F4) & Appendix B:
- Builds series from d1 - 6 to d1 + 2.
- 3-day rolling mean on phase probabilities.
- Phase transitions: PHASE:X->Y when state changes from X to Y with smoothed P(Y) >= 0.5 on d and d+1.
  If d+1 is beyond series, confirmed = False.
- LPS events: LPS_FORMS, LPS_ENDS, LPS_NEAR_DISTRICT (<= 500 km on d, > 500 km on d-1).
- No events found across missing date gaps.
- Change in domain-mean corrected rain (up to 3 dates after minus up to 3 dates before).
"""

from typing import Any, Dict, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd

from regime_engine.lps.geo_utils import haversine_distance_km

PHASES = ["active", "normal", "break"]


class TransitionDetector:
    """Detects phase shifts and LPS events across chronological forecast sequences."""

    def __init__(
        self,
        smoothing_days: int = 3,
        confirm_min_prob: float = 0.50,
        lps_near_distance_km: float = 500.0,
    ):
        self.smoothing_days = smoothing_days
        self.confirm_min_prob = confirm_min_prob
        self.lps_near_distance_km = lps_near_distance_km

    def detect_transitions(
        self,
        daily_series_df: pd.DataFrame,
        district_centroid: Optional[Tuple[float, float]] = None,
        district_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect transitions over a series of consecutive dates.

        Expected daily_series_df columns:
            ['imd_date', 'p_active', 'p_normal', 'p_break', 'lps_present']
            Optional: 'domain_mean_corrected_mm', 'nearest_lps_lat', 'nearest_lps_lon',
                      'distance_to_lps_km'.

        Args:
            daily_series_df: DataFrame sorted chronologically by imd_date.
            district_centroid: Optional (lat, lon) for LPS_NEAR_DISTRICT detection.
            district_id: Optional district_id string.

        Returns:
            Dict matching API contract: {'series': [...], 'events': [...]}
        """
        if daily_series_df.empty:
            return {"series": [], "events": []}

        df = daily_series_df.sort_values("imd_date").reset_index(drop=True).copy()
        df["dt"] = pd.to_datetime(df["imd_date"]).dt.date

        # Check for consecutive calendar day continuity (gaps)
        n = len(df)
        has_gap = np.zeros(n, dtype=bool)
        for i in range(1, n):
            gap_days = (df.loc[i, "dt"] - df.loc[i - 1, "dt"]).days
            if gap_days != 1:
                has_gap[i] = True  # There is a discontinuity before index i

        # Compute 3-day rolling mean for probabilities
        # Using centered or trailing window? PRD specifies: "Smooth p_active, p_normal, p_break with a 3-day mean"
        smooth_p_act = df["p_active"].rolling(window=self.smoothing_days, min_periods=1, center=True).mean().to_numpy()
        smooth_p_norm = df["p_normal"].rolling(window=self.smoothing_days, min_periods=1, center=True).mean().to_numpy()
        smooth_p_brk = df["p_break"].rolling(window=self.smoothing_days, min_periods=1, center=True).mean().to_numpy()

        # States: argmax among [active, normal, break]
        states = []
        smoothed_matrix = np.column_stack([smooth_p_act, smooth_p_norm, smooth_p_brk])
        for i in range(n):
            best_idx = int(np.argmax(smoothed_matrix[i]))
            states.append(PHASES[best_idx])
        df["state"] = states

        # Domain mean corrected rain series
        rain_col = "domain_mean_corrected_mm" if "domain_mean_corrected_mm" in df.columns else None

        def calc_rain_change(event_idx: int) -> Tuple[Optional[float], int, int]:
            if rain_col is None or df[rain_col].isna().all():
                return None, 0, 0
            # Up to 3 dates before (strictly before event_idx)
            before_start = max(0, event_idx - 3)
            before_vals = df.loc[before_start : event_idx - 1, rain_col].dropna().to_numpy()

            # Up to 3 dates after (including or after event_idx? "mean of up to 3 dates after minus before")
            after_end = min(n - 1, event_idx + 2)
            after_vals = df.loc[event_idx:after_end, rain_col].dropna().to_numpy()

            n_before = len(before_vals)
            n_after = len(after_vals)

            if n_before == 0 or n_after == 0:
                return None, n_before, n_after

            change = float(np.mean(after_vals) - np.mean(before_vals))
            return round(change, 1), n_before, n_after

        events = []
        event_counter = 1

        # 1. Phase transitions
        for d_idx in range(1, n):
            # PRD: If a run is missing, no event is found across the gap
            if has_gap[d_idx]:
                continue

            state_prev = states[d_idx - 1]
            state_curr = states[d_idx]

            if state_curr != state_prev:
                phase_idx = PHASES.index(state_curr)
                prob_d = float(smoothed_matrix[d_idx, phase_idx])

                # Check condition on d: smoothed probability >= confirm_min_prob (0.5)
                if prob_d >= self.confirm_min_prob:
                    # Check condition on d + 1
                    if (d_idx + 1) < n and not has_gap[d_idx + 1]:
                        prob_d1 = float(smoothed_matrix[d_idx + 1, phase_idx])
                        if prob_d1 >= self.confirm_min_prob:
                            conf_score = round(float((prob_d + prob_d1) / 2.0), 2)
                            confirmed = True
                        else:
                            # Not sustained on d+1
                            continue
                    else:
                        # Beyond series or followed by gap: confirmed = False ("pending")
                        conf_score = round(prob_d, 2)
                        confirmed = False

                    change_val, n_bef, n_aft = calc_rain_change(d_idx)
                    events.append({
                        "event_id": event_counter,
                        "event_type": f"PHASE:{state_prev}->{state_curr}",
                        "from_state": state_prev,
                        "to_state": state_curr,
                        "event_date": str(df.loc[d_idx, "imd_date"]),
                        "confirmed": confirmed,
                        "confidence": conf_score,
                        "district_id": None,
                        "domain_mean_corrected_change_mm": change_val,
                        "n_dates_before": n_bef,
                        "n_dates_after": n_aft,
                    })
                    event_counter += 1

        # 2. LPS events
        lps_present_series = df["lps_present"].to_numpy(dtype=bool)

        for d_idx in range(1, n):
            if has_gap[d_idx]:
                continue

            lps_prev = lps_present_series[d_idx - 1]
            lps_curr = lps_present_series[d_idx]

            # LPS_FORMS: No system on d-1, system on d and d+1
            if not lps_prev and lps_curr:
                if (d_idx + 1) < n and not has_gap[d_idx + 1] and lps_present_series[d_idx + 1]:
                    change_val, n_bef, n_aft = calc_rain_change(d_idx)
                    events.append({
                        "event_id": event_counter,
                        "event_type": "LPS_FORMS",
                        "from_state": "none",
                        "to_state": "detected",
                        "event_date": str(df.loc[d_idx, "imd_date"]),
                        "confirmed": True,
                        "confidence": None,  # Rule-based has no probability per PRD F4
                        "district_id": None,
                        "domain_mean_corrected_change_mm": change_val,
                        "n_dates_before": n_bef,
                        "n_dates_after": n_aft,
                    })
                    event_counter += 1

            # LPS_ENDS: System on d-1, none on d and d+1
            if lps_prev and not lps_curr:
                if (d_idx + 1) < n and not has_gap[d_idx + 1] and not lps_present_series[d_idx + 1]:
                    change_val, n_bef, n_aft = calc_rain_change(d_idx)
                    events.append({
                        "event_id": event_counter,
                        "event_type": "LPS_ENDS",
                        "from_state": "detected",
                        "to_state": "none",
                        "event_date": str(df.loc[d_idx, "imd_date"]),
                        "confirmed": True,
                        "confidence": None,
                        "district_id": None,
                        "domain_mean_corrected_change_mm": change_val,
                        "n_dates_before": n_bef,
                        "n_dates_after": n_aft,
                    })
                    event_counter += 1

        # 3. LPS_NEAR_DISTRICT (if district centroid or distance series provided)
        if district_centroid is not None or "distance_to_lps_km" in df.columns:
            dist_series = np.full(n, np.inf, dtype=float)
            if "distance_to_lps_km" in df.columns:
                dist_series = df["distance_to_lps_km"].to_numpy(dtype=float)
            elif district_centroid is not None and "nearest_lps_lat" in df.columns:
                d_lat, d_lon = district_centroid
                lats = df["nearest_lps_lat"].to_numpy(dtype=float)
                lons = df["nearest_lps_lon"].to_numpy(dtype=float)
                for i in range(n):
                    if not np.isnan(lats[i]) and not np.isnan(lons[i]):
                        dist_series[i] = haversine_distance_km(d_lat, d_lon, lats[i], lons[i])

            for d_idx in range(1, n):
                if has_gap[d_idx]:
                    continue
                d_prev = dist_series[d_idx - 1]
                d_curr = dist_series[d_idx]
                # Nearest system is <= 500 km on d, and was > 500 km on d-1
                if d_curr <= self.lps_near_distance_km and d_prev > self.lps_near_distance_km:
                    change_val, n_bef, n_aft = calc_rain_change(d_idx)
                    events.append({
                        "event_id": event_counter,
                        "event_type": "LPS_NEAR_DISTRICT",
                        "from_state": f"dist>{int(self.lps_near_distance_km)}km",
                        "to_state": f"dist<={int(self.lps_near_distance_km)}km",
                        "event_date": str(df.loc[d_idx, "imd_date"]),
                        "confirmed": True,
                        "confidence": None,
                        "district_id": district_id,
                        "domain_mean_corrected_change_mm": change_val,
                        "n_dates_before": n_bef,
                        "n_dates_after": n_aft,
                    })
                    event_counter += 1

        # Format timeline series
        series_out = []
        for i in range(n):
            series_out.append({
                "imd_date": str(df.loc[i, "imd_date"]),
                "p_active": round(float(df.loc[i, "p_active"]), 3),
                "p_normal": round(float(df.loc[i, "p_normal"]), 3),
                "p_break": round(float(df.loc[i, "p_break"]), 3),
                "lps_present": bool(df.loc[i, "lps_present"]),
            })

        return {
            "series": series_out,
            "events": events,
        }
