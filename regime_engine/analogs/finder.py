"""Historical Analog Finder (F3).

PRD Section 15 (F3) & Appendix B:
- Vector (10 numbers): p_active, p_break, A1, A2, A3, A4, A5, A6, lps_present, lps_strength.
- Library: Same lead only, development seasons only, strictly excluding the query's own season.
- Standardized Euclidean distance.
- Greedy selection: nearest analog, then next nearest separated by at least 5 days from all selected.
- K = 5 cases.
- Computes distance_percentile (rank among all library candidate distances).
- District outcome lookup from district_history.
"""

from typing import Any, Dict, List, Optional
import datetime
import numpy as np
import pandas as pd

ANALOG_VECTOR_KEYS = [
    "p_active",
    "p_break",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "lps_present",
    "lps_strength",
]


class AnalogFinder:
    """Finds top-K historical atmospheric analogues under F3 rules."""

    def __init__(
        self,
        k: int = 5,
        min_separation_days: int = 5,
    ):
        self.k = k
        self.min_separation_days = min_separation_days
        self.library_df: Optional[pd.DataFrame] = None

    def fit_library(self, library_records: pd.DataFrame) -> "AnalogFinder":
        """Load library of historical (run, lead) pairs.

        Expected columns:
            ['run_id', 'lead_day', 'imd_date', 'season'] + ANALOG_VECTOR_KEYS.
        """
        df = library_records.copy()
        df["date_dt"] = pd.to_datetime(df["imd_date"]).dt.date
        self.library_df = df
        return self

    def find_analogs(
        self,
        query_vector: Dict[str, float],
        lead_day: int,
        query_season: Optional[int],
        query_date: Optional[Any] = None,
        district_history: Optional[pd.DataFrame] = None,
        district_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find the 5 closest historical analogues for a query.

        Args:
            query_vector: Dict with the 10 keys in ANALOG_VECTOR_KEYS.
            lead_day: Target lead day (1, 2, or 3).
            query_season: Season year of query (excluded from library).
            query_date: Optional query date.
            district_history: Optional DataFrame with past district outcomes.
            district_id: Optional district_id to look up outcomes for.

        Returns:
            Dict matching F3 / API contract:
            {'analogs': [...], 'median_error_observed_minus_raw_mm': float, 'n_analogs': int}
        """
        if self.library_df is None or self.library_df.empty:
            return {"analogs": [], "median_error_observed_minus_raw_mm": 0.0, "n_analogs": 0}

        # Filter to same lead
        mask = self.library_df["lead_day"] == lead_day

        # Exclude query's own season (PRD F3 decision)
        if query_season is not None:
            mask &= self.library_df["season"] != query_season

        candidates = self.library_df.loc[mask].copy()
        if candidates.empty:
            return {"analogs": [], "median_error_observed_minus_raw_mm": 0.0, "n_analogs": 0}

        # Extract 10-feature matrix
        X_cand = candidates[ANALOG_VECTOR_KEYS].to_numpy(dtype=float)
        q_vec = np.array([query_vector.get(k, 0.0) for k in ANALOG_VECTOR_KEYS], dtype=float)

        # Standardize using library mean and spread
        mean = np.nanmean(X_cand, axis=0)
        std = np.nanstd(X_cand, axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        X_cand_z = (X_cand - mean) / std
        q_vec_z = (q_vec - mean) / std

        # Euclidean distances
        dists = np.sqrt(np.sum((X_cand_z - q_vec_z) ** 2, axis=1))
        candidates["distance"] = dists

        # Compute distance percentile (0 to 100, lower is closer)
        n_cand = len(candidates)
        ranks = np.argsort(np.argsort(dists)) + 1
        candidates["distance_percentile"] = (ranks / n_cand) * 100.0

        # Sort by distance ascending
        candidates = candidates.sort_values("distance")

        # Greedy selection with >= min_separation_days
        selected_rows = []
        selected_dates = []

        for _, row in candidates.iterrows():
            cand_date = row["date_dt"]
            too_close = False
            for d in selected_dates:
                if abs((cand_date - d).days) < self.min_separation_days:
                    too_close = True
                    break

            if not too_close:
                selected_rows.append(row)
                selected_dates.append(cand_date)
                if len(selected_rows) == self.k:
                    break

        # Build output analog list
        analogs_out = []
        errors = []

        # Prepare district history index if provided
        hist_indexed = None
        if district_history is not None and district_id is not None:
            dh = district_history.loc[district_history["district_id"] == district_id]
            hist_indexed = dh.set_index(["run_id", "lead_day"])

        for rank_idx, row in enumerate(selected_rows, start=1):
            run_id = row["run_id"]
            imd_date_str = str(row["imd_date"])
            season_val = int(row["season"])
            dist_val = float(row["distance"])
            pct_val = float(row["distance_percentile"])

            # Look up district outcomes if available
            obs_mean = 0.0
            obs_wet = 0.0
            raw_mean = 0.0
            corr_mean = 0.0
            err_val = 0.0

            if hist_indexed is not None and (run_id, lead_day) in hist_indexed.index:
                hist_row = hist_indexed.loc[(run_id, lead_day)]
                if isinstance(hist_row, pd.DataFrame):
                    hist_row = hist_row.iloc[0]
                obs_mean = float(hist_row.get("observed_mean_mm", 0.0))
                obs_wet = float(hist_row.get("observed_max_cell_mm", 0.0))
                raw_mean = float(hist_row.get("raw_mean_mm", 0.0))
                corr_mean = float(hist_row.get("corrected_mean_mm", 0.0))
                err_val = float(obs_mean - raw_mean)
            else:
                # Default / fallback outcome fields
                err_val = 0.0

            errors.append(err_val)
            analogs_out.append({
                "rank": rank_idx,
                "analog_run_id": run_id,
                "imd_date": imd_date_str,
                "season": season_val,
                "distance": round(dist_val, 3),
                "distance_percentile": round(pct_val, 1),
                "observed_mean_mm": round(obs_mean, 1),
                "observed_wettest_cell_mm": round(obs_wet, 1),
                "raw_mean_mm": round(raw_mean, 1),
                "corrected_mean_mm": round(corr_mean, 1),
                "error_observed_minus_raw_mm": round(err_val, 1),
            })

        med_err = float(np.median(errors)) if errors else 0.0

        return {
            "analogs": analogs_out,
            "median_error_observed_minus_raw_mm": round(med_err, 1),
            "n_analogs": len(analogs_out),
        }
