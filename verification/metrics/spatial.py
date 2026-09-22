"""Spatial verification: Fractions Skill Score (FSS) per PRD §16.8, Appendix A.

Formulas:
- Pf, Po: fractions of valid cells equal to 1 in an n x n neighbourhood window.
- numerator_sum            = Σ(Pf - Po)^2
- forecast_fraction_sq_sum = ΣPf^2
- observed_fraction_sq_sum = ΣPo^2
- FSS = 1 - numerator_sum / (forecast_fraction_sq_sum + observed_fraction_sq_sum)
- f0  = observed event fraction = (count of observed events) / (count of valid cells)
- FSS_useful = 0.5 + f0 / 2

Rules (PRD §16.4, §16.8):
- Sea and missing cells must NOT count as dry.
- Summing over all valid points and dates.
- Aggregation across dates must sum components first, NEVER average daily FSS.
- Undefined denominator => MetricResult(None, reason). Never return 0.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.ndimage import uniform_filter

from verification.metrics import MetricResult


@dataclass(frozen=True)
class FSSComponents:
    """Additive components for Fractions Skill Score across grid cells.

    Can be summed across dates and runs before computing final FSS.
    """
    numerator_sum: float             # Σ(Pf - Po)^2
    forecast_fraction_sq_sum: float  # ΣPf^2
    observed_fraction_sq_sum: float  # ΣPo^2
    n_valid_cells: int               # Count of valid cells evaluated
    n_observed_events: int           # Count of observed events in valid cells

    def __add__(self, other: "FSSComponents") -> "FSSComponents":
        """Sums components across multiple dates or runs (PRD §16.4, §16.8)."""
        if not isinstance(other, FSSComponents):
            return NotImplemented
        return FSSComponents(
            numerator_sum=self.numerator_sum + other.numerator_sum,
            forecast_fraction_sq_sum=self.forecast_fraction_sq_sum + other.forecast_fraction_sq_sum,
            observed_fraction_sq_sum=self.observed_fraction_sq_sum + other.observed_fraction_sq_sum,
            n_valid_cells=self.n_valid_cells + other.n_valid_cells,
            n_observed_events=self.n_observed_events + other.n_observed_events,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numerator_sum": self.numerator_sum,
            "forecast_fraction_sq_sum": self.forecast_fraction_sq_sum,
            "observed_fraction_sq_sum": self.observed_fraction_sq_sum,
            "n_valid_cells": self.n_valid_cells,
            "n_observed_events": self.n_observed_events,
        }


def compute_fss_components(
    forecast: np.ndarray,
    observed: np.ndarray,
    threshold: float,
    neighbourhood_size: int,
    valid_mask: Optional[np.ndarray] = None,
) -> FSSComponents:
    """Computes additive FSS components for a single 2D forecast/observed grid.

    Args:
        forecast: 2D array of forecast rainfall.
        observed: 2D array of observed rainfall.
        threshold: Rainfall threshold in mm (e.g. 15.6, 64.5, 115.6).
        neighbourhood_size: Odd integer window width (1, 3, 5, 9).
        valid_mask: 2D boolean array of valid cells. If None, all cells are considered valid.

    Returns:
        FSSComponents instance.
    """
    f = np.asarray(forecast, dtype=float)
    o = np.asarray(observed, dtype=float)

    if f.ndim != 2 or o.ndim != 2:
        raise ValueError(
            f"Forecast and observed must be 2D grids, got dims: forecast {f.ndim}, observed {o.ndim}."
        )
    if f.shape != o.shape:
        raise ValueError(f"Shape mismatch: forecast {f.shape} != observed {o.shape}.")

    if neighbourhood_size < 1 or (neighbourhood_size % 2 == 0):
        raise ValueError(
            f"Neighbourhood size must be an odd positive integer (e.g. 1, 3, 5, 9), got {neighbourhood_size}."
        )

    if valid_mask is None:
        mask = np.ones(f.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != f.shape:
            raise ValueError(f"Valid mask shape {mask.shape} does not match grid shape {f.shape}.")

    # Effective valid mask: exclude sea/invalid cells AND missing observations (NaNs)
    effective_mask = mask & (~np.isnan(f)) & (~np.isnan(o))

    n_valid = int(np.count_nonzero(effective_mask))
    if n_valid == 0:
        return FSSComponents(
            numerator_sum=0.0,
            forecast_fraction_sq_sum=0.0,
            observed_fraction_sq_sum=0.0,
            n_valid_cells=0,
            n_observed_events=0,
        )

    m_float = effective_mask.astype(float)
    f_bin = np.where(effective_mask & (f >= threshold), 1.0, 0.0)
    o_bin = np.where(effective_mask & (o >= threshold), 1.0, 0.0)

    n_obs_events = int(np.count_nonzero(o_bin[effective_mask] > 0))

    if neighbourhood_size == 1:
        # For size 1, window fractions are simply the binary values on valid cells
        pf = f_bin
        po = o_bin
    else:
        # Uniform filter with mode='constant', cval=0.0 per PRD §16.8
        f_filt = uniform_filter(f_bin, size=neighbourhood_size, mode="constant", cval=0.0)
        o_filt = uniform_filter(o_bin, size=neighbourhood_size, mode="constant", cval=0.0)
        m_filt = uniform_filter(m_float, size=neighbourhood_size, mode="constant", cval=0.0)

        pf = np.zeros_like(f_filt)
        po = np.zeros_like(o_filt)

        # Non-zero valid cells inside window
        has_valid = m_filt > 0.0
        pf[has_valid] = f_filt[has_valid] / m_filt[has_valid]
        po[has_valid] = o_filt[has_valid] / m_filt[has_valid]

    # Sum only over valid cells
    pf_valid = pf[effective_mask]
    po_valid = po[effective_mask]

    num_sum = float(np.sum((pf_valid - po_valid) ** 2))
    pf_sq_sum = float(np.sum(pf_valid ** 2))
    po_sq_sum = float(np.sum(po_valid ** 2))

    return FSSComponents(
        numerator_sum=num_sum,
        forecast_fraction_sq_sum=pf_sq_sum,
        observed_fraction_sq_sum=po_sq_sum,
        n_valid_cells=n_valid,
        n_observed_events=n_obs_events,
    )


def compute_fss_from_components(components: FSSComponents) -> MetricResult:
    """Computes FSS from additive components:

    FSS = 1 - numerator_sum / (forecast_fraction_sq_sum + observed_fraction_sq_sum).
    """
    if components.n_valid_cells == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_CELLS")

    denom = components.forecast_fraction_sq_sum + components.observed_fraction_sq_sum
    if denom == 0.0:
        return MetricResult(value=None, undefined_reason="NO_EVENTS_AT_THRESHOLD")

    fss = 1.0 - (components.numerator_sum / denom)
    return MetricResult(value=float(fss), undefined_reason=None)


def compute_fss(
    forecast: np.ndarray,
    observed: np.ndarray,
    threshold: float,
    neighbourhood_size: int,
    valid_mask: Optional[np.ndarray] = None,
) -> MetricResult:
    """Computes Fractions Skill Score for a single forecast/observed field pair."""
    comps = compute_fss_components(
        forecast=forecast,
        observed=observed,
        threshold=threshold,
        neighbourhood_size=neighbourhood_size,
        valid_mask=valid_mask,
    )
    return compute_fss_from_components(comps)


def aggregate_fss(components_list: Sequence[FSSComponents]) -> MetricResult:
    """Aggregates FSS across dates or grid runs by summing components first.

    PRD Rule: FSS is always added up as (sum of numerators) / (sum of denominators),
    NEVER as an average of daily FSS.
    """
    if not components_list:
        return MetricResult(value=None, undefined_reason="EMPTY_COMPONENTS_LIST")

    total = components_list[0]
    for c in components_list[1:]:
        total = total + c
    return compute_fss_from_components(total)


def compute_observed_event_fraction(components: FSSComponents) -> MetricResult:
    """Computes the observed event fraction f0 = n_observed_events / n_valid_cells."""
    if components.n_valid_cells == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_CELLS")
    f0 = float(components.n_observed_events / components.n_valid_cells)
    return MetricResult(value=f0, undefined_reason=None)


def compute_useful_skill(
    f0: Union[float, MetricResult, None, FSSComponents],
) -> MetricResult:
    """Computes the useful skill line: FSS_useful = 0.5 + f0 / 2 (Roberts & Lean 2008)."""
    if isinstance(f0, FSSComponents):
        f0_res = compute_observed_event_fraction(f0)
        f0_val = f0_res.value
    elif isinstance(f0, MetricResult):
        f0_val = f0.value
    else:
        f0_val = f0

    if f0_val is None:
        return MetricResult(value=None, undefined_reason="NO_VALID_CELLS")

    val = 0.5 + (f0_val / 2.0)
    return MetricResult(value=float(val), undefined_reason=None)
