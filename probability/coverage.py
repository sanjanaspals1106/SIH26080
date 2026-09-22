"""Range coverage verification and checking per PRD §13.7.

Rules:
- Coverage checks use out-of-fold (OOF) development predictions only (PRD §10.4, §13.7).
- Rejects holdout data.
- Excludes rows with missing observations or quantiles.
- Defensively sorts quantiles row-wise so q10 <= q50 <= q90.
- Target coverage = 0.80, tolerance = 0.10 (from config/verification.yaml).
- Flags a lead if abs(coverage - target) > tolerance.
- Exact +/- 0.10 boundary is not flagged.
- Provides structured wording data for frontend/API without formatting UI text.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml


@dataclass(frozen=True)
class LeadCoverageResult:
    """Coverage statistics and alert status for a single lead day."""
    lead_day: int
    coverage: float
    n_samples: int
    target: float
    tolerance: float
    needs_display_warning: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_day": self.lead_day,
            "coverage": self.coverage,
            "n_samples": self.n_samples,
            "target": self.target,
            "tolerance": self.tolerance,
            "needs_display_warning": self.needs_display_warning,
        }

    def get_wording_data(self) -> Dict[str, Any]:
        """Provides structured data for API and frontend display per PRD §13.7.

        Does not generate user-interface presentation strings directly.
        """
        return {
            "lead_day": self.lead_day,
            "measured_coverage_pct": round(self.coverage * 100.0, 1),
            "target_pct": round(self.target * 100.0, 1),
            "tolerance_pct": round(self.tolerance * 100.0, 1),
            "needs_display_warning": self.needs_display_warning,
        }


def load_coverage_parameters(
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[float, float]:
    """Loads target coverage and tolerance from config/verification.yaml."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "verification.yaml"

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cov = data.get("coverage", {})
        target = float(cov.get("target", 0.80))
        tolerance = float(cov.get("tolerance", 0.10))
        return target, tolerance
    return 0.80, 0.10


def check_range_coverage(
    lead_day: Sequence[int] | np.ndarray,
    obs_mm: Sequence[float] | np.ndarray,
    q10_mm: Sequence[float] | np.ndarray,
    q50_mm: Sequence[float] | np.ndarray,
    q90_mm: Sequence[float] | np.ndarray,
    evaluation_set: Optional[Union[str, Sequence[str]]] = "development",
    prediction_source: Optional[Union[str, Sequence[str]]] = "oof",
    config_path: Optional[Union[str, Path]] = None,
) -> List[LeadCoverageResult]:
    """Calculates q10-q90 empirical coverage separately per lead day.

    Args:
        lead_day: Sequence of lead days (e.g. 1, 2, 3).
        obs_mm: Observed rainfall values in mm.
        q10_mm: 10th percentile rainfall forecasts in mm.
        q50_mm: 50th percentile (median) rainfall forecasts in mm.
        q90_mm: 90th percentile rainfall forecasts in mm.
        evaluation_set: Must be strictly 'development' (rejects holdout).
        prediction_source: Must be strictly 'oof' if provided.
        config_path: Optional path to verification.yaml.

    Returns:
        List of LeadCoverageResult objects sorted by lead_day.
    """
    # Rule check 1: Reject holdout data (PRD §10.4, §10.6, §13.7)
    if evaluation_set is not None:
        if isinstance(evaluation_set, str):
            if evaluation_set.strip().lower() == "holdout":
                raise ValueError(
                    "Holdout data cannot be used for range coverage checks per PRD §10.4, §10.6 L4."
                )
            if evaluation_set.strip().lower() != "development":
                raise ValueError(
                    f"Coverage checks must use 'development' set, got: {evaluation_set!r}."
                )
        else:
            arr_eval = np.char.lower(np.asarray(evaluation_set, dtype=str))
            if np.any(arr_eval == "holdout"):
                raise ValueError(
                    "Holdout data detected in evaluation_set; forbidden per PRD §10.4, §10.6 L4."
                )
            if np.any(arr_eval != "development"):
                raise ValueError(
                    "All rows for coverage checks must belong to the 'development' evaluation set."
                )

    # Rule check 2: Reject non-OOF predictions
    if prediction_source is not None:
        if isinstance(prediction_source, str):
            if prediction_source.strip().lower() != "oof":
                raise ValueError(
                    f"prediction_source must be 'oof', got {prediction_source!r}. "
                    "PRD §13.7 requires coverage checks on out-of-fold predictions."
                )
        else:
            arr_pred = np.char.lower(np.asarray(prediction_source, dtype=str))
            if np.any(arr_pred != "oof"):
                raise ValueError("All rows for coverage checks must have prediction_source='oof'.")

    leads = np.asarray(lead_day, dtype=int).ravel()
    obs = np.asarray(obs_mm, dtype=float).ravel()
    q10 = np.asarray(q10_mm, dtype=float).ravel()
    q50 = np.asarray(q50_mm, dtype=float).ravel()
    q90 = np.asarray(q90_mm, dtype=float).ravel()

    n = len(leads)
    if not (len(obs) == len(q10) == len(q50) == len(q90) == n):
        raise ValueError("All input arrays must have the same length.")

    # Exclude missing observations or missing quantiles
    valid_mask = ~(
        np.isnan(obs)
        | np.isnan(q10)
        | np.isnan(q50)
        | np.isnan(q90)
    )

    leads_clean = leads[valid_mask]
    obs_clean = obs[valid_mask]
    q10_clean = q10[valid_mask]
    q50_clean = q50[valid_mask]
    q90_clean = q90[valid_mask]

    target, tolerance = load_coverage_parameters(config_path)

    results: List[LeadCoverageResult] = []

    if len(leads_clean) == 0:
        return results

    # Process each lead independently
    unique_leads = np.unique(leads_clean)
    unique_leads.sort()

    for ld in unique_leads:
        lead_mask = leads_clean == ld
        ld_obs = obs_clean[lead_mask]
        ld_q10 = q10_clean[lead_mask]
        ld_q50 = q50_clean[lead_mask]
        ld_q90 = q90_clean[lead_mask]

        n_samples = len(ld_obs)
        if n_samples == 0:
            continue

        # Defensively sort each row so q10 <= q50 <= q90 (PRD §13.7)
        stacked_q = np.column_stack([ld_q10, ld_q50, ld_q90])
        sorted_q = np.sort(stacked_q, axis=1)

        q10_sorted = sorted_q[:, 0]
        # q50_sorted = sorted_q[:, 1]
        q90_sorted = sorted_q[:, 2]

        # Empirical coverage: fraction of observations inside [q10, q90] inclusive
        inside = (ld_obs >= q10_sorted) & (ld_obs <= q90_sorted)
        coverage = float(np.mean(inside))

        # Flag when abs(coverage - target) > tolerance.
        # Float tolerance margin 1e-9 avoids float inaccuracy on exact +/- 0.10 boundaries.
        diff = abs(coverage - target)
        needs_warning = bool((diff - tolerance) > 1e-9)

        results.append(
            LeadCoverageResult(
                lead_day=int(ld),
                coverage=coverage,
                n_samples=n_samples,
                target=target,
                tolerance=tolerance,
                needs_display_warning=needs_warning,
            )
        )

    return results
