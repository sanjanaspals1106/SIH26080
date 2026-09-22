"""Verification component representation (PRD §16.4, §16.5, §16.10).

Per-date additive verification components representing counts and sums that can be:
- stored in Parquet (data/verification/components/)
- aggregated over dates and groups
- passed into paired block bootstrap
"""

from dataclasses import asdict, dataclass, field
from datetime import date
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from verification.metrics.contingency import ContingencyCounts
from verification.metrics.spatial import FSSComponents
from verification.stats.bootstrap import BrierComponents, RMSEComponents

VALID_FORECAST_TYPES = {
    "raw_nwp",
    "quantile_mapping",
    "global_ml",
    "regime_aware_ml",
}

VALID_EVALUATION_SETS = {
    "development",
    "dev",
    "loso",
    "holdout",
    "test",
}


@dataclass
class VerificationComponent:
    """Represents per-date additive verification sums and counts for a specific group (PRD §16.4)."""

    # Identifiers / Metadata
    imd_date: Optional[Union[str, date]] = None
    lead_day: int = 1
    forecast_type: str = "regime_aware_ml"
    evaluation_set: str = "development"
    threshold_mm: Optional[float] = None
    neighbourhood_cells: Optional[int] = None
    group_type: str = "all"
    group_value: str = "all"

    # Overall sample & event tracking
    n_samples: int = 0
    n_events: Optional[int] = None

    # Continuous components
    n: int = 0
    sum_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0

    # Categorical / Contingency components
    a: int = 0  # hits
    b: int = 0  # false alarms
    c: int = 0  # misses
    d: int = 0  # correct negatives

    # Spatial FSS components
    fss_numerator_sum: float = 0.0
    fss_forecast_fraction_sq_sum: float = 0.0
    fss_observed_fraction_sq_sum: float = 0.0

    # Probabilistic Brier components
    brier_n: int = 0
    sum_brier_terms: float = 0.0
    sum_reference_brier_terms: float = 0.0

    # Reliability Diagram Bins (PRD §16.4)
    bin_lower: Optional[float] = None
    bin_upper: Optional[float] = None
    sum_predicted_probability: float = 0.0
    n_observed_events: int = 0

    # Range / Quantile components
    pinball_q10_sum: float = 0.0
    pinball_q50_sum: float = 0.0
    pinball_q90_sum: float = 0.0
    range_n: int = 0
    coverage_count: int = 0

    def __post_init__(self) -> None:
        # Infer n_samples if not explicitly set
        if self.n_samples == 0:
            if self.n > 0:
                self.n_samples = self.n
            elif (self.a + self.b + self.c + self.d) > 0:
                self.n_samples = self.a + self.b + self.c + self.d
            elif self.brier_n > 0:
                self.n_samples = self.brier_n
            elif self.range_n > 0:
                self.n_samples = self.range_n

    def check_compatibility(self, other: "VerificationComponent") -> None:
        """Rejects addition across incompatible metadata fields."""
        if self.forecast_type != other.forecast_type:
            raise ValueError(
                f"Incompatible forecast_type: '{self.forecast_type}' vs '{other.forecast_type}'"
            )
        if self.evaluation_set != other.evaluation_set:
            raise ValueError(
                f"Incompatible evaluation_set: '{self.evaluation_set}' vs '{other.evaluation_set}'"
            )
        if self.lead_day != other.lead_day:
            raise ValueError(
                f"Incompatible lead_day: {self.lead_day} vs {other.lead_day}"
            )
        if self.threshold_mm != other.threshold_mm:
            raise ValueError(
                f"Incompatible threshold_mm: {self.threshold_mm} vs {other.threshold_mm}"
            )
        if self.neighbourhood_cells != other.neighbourhood_cells:
            raise ValueError(
                f"Incompatible neighbourhood_cells: {self.neighbourhood_cells} vs {other.neighbourhood_cells}"
            )
        if self.group_type != other.group_type or self.group_value != other.group_value:
            raise ValueError(
                f"Incompatible group: ({self.group_type}={self.group_value}) vs ({other.group_type}={other.group_value})"
            )
        if self.bin_lower != other.bin_lower or self.bin_upper != other.bin_upper:
            raise ValueError(
                f"Incompatible reliability bin: ({self.bin_lower}, {self.bin_upper}) vs ({other.bin_lower}, {other.bin_upper})"
            )

    def __add__(self, other: "VerificationComponent") -> "VerificationComponent":
        """Sums additive components while enforcing metadata compatibility."""
        if not isinstance(other, VerificationComponent):
            return NotImplemented

        self.check_compatibility(other)

        # Aggregate n_events if either is not None
        if self.n_events is not None or other.n_events is not None:
            sum_events = (self.n_events or 0) + (other.n_events or 0)
        else:
            sum_events = None

        return VerificationComponent(
            imd_date="aggregated",
            lead_day=self.lead_day,
            forecast_type=self.forecast_type,
            evaluation_set=self.evaluation_set,
            threshold_mm=self.threshold_mm,
            neighbourhood_cells=self.neighbourhood_cells,
            group_type=self.group_type,
            group_value=self.group_value,
            n_samples=self.n_samples + other.n_samples,
            n_events=sum_events,
            n=self.n + other.n,
            sum_error=self.sum_error + other.sum_error,
            sum_abs_error=self.sum_abs_error + other.sum_abs_error,
            sum_squared_error=self.sum_squared_error + other.sum_squared_error,
            a=self.a + other.a,
            b=self.b + other.b,
            c=self.c + other.c,
            d=self.d + other.d,
            fss_numerator_sum=self.fss_numerator_sum + other.fss_numerator_sum,
            fss_forecast_fraction_sq_sum=self.fss_forecast_fraction_sq_sum + other.fss_forecast_fraction_sq_sum,
            fss_observed_fraction_sq_sum=self.fss_observed_fraction_sq_sum + other.fss_observed_fraction_sq_sum,
            brier_n=self.brier_n + other.brier_n,
            sum_brier_terms=self.sum_brier_terms + other.sum_brier_terms,
            sum_reference_brier_terms=self.sum_reference_brier_terms + other.sum_reference_brier_terms,
            bin_lower=self.bin_lower,
            bin_upper=self.bin_upper,
            sum_predicted_probability=self.sum_predicted_probability + other.sum_predicted_probability,
            n_observed_events=self.n_observed_events + other.n_observed_events,
            pinball_q10_sum=self.pinball_q10_sum + other.pinball_q10_sum,
            pinball_q50_sum=self.pinball_q50_sum + other.pinball_q50_sum,
            pinball_q90_sum=self.pinball_q90_sum + other.pinball_q90_sum,
            range_n=self.range_n + other.range_n,
            coverage_count=self.coverage_count + other.coverage_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    # Conversion helpers to bridge into existing metric objects and bootstrap
    def to_contingency_counts(self) -> ContingencyCounts:
        return ContingencyCounts(
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            n=self.a + self.b + self.c + self.d,
        )

    def to_fss_components(self) -> FSSComponents:
        n_valid = self.n if self.n > 0 else self.n_samples
        if n_valid == 0 and (
            self.fss_forecast_fraction_sq_sum > 0
            or self.fss_observed_fraction_sq_sum > 0
            or self.fss_numerator_sum > 0
        ):
            n_valid = 1

        return FSSComponents(
            numerator_sum=self.fss_numerator_sum,
            forecast_fraction_sq_sum=self.fss_forecast_fraction_sq_sum,
            observed_fraction_sq_sum=self.fss_observed_fraction_sq_sum,
            n_valid_cells=n_valid,
            n_observed_events=self.n_events if self.n_events is not None else 0,
        )

    def to_rmse_components(self) -> RMSEComponents:
        return RMSEComponents(
            n_samples=self.n if self.n > 0 else self.n_samples,
            sum_squared_errors=self.sum_squared_error,
        )

    def to_brier_components(self) -> BrierComponents:
        return BrierComponents(
            n_samples=self.brier_n if self.brier_n > 0 else self.n_samples,
            sum_brier_terms=self.sum_brier_terms,
        )


def create_continuous_component(
    forecast: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
    imd_date: Union[str, date],
    lead_day: int,
    forecast_type: str = "regime_aware_ml",
    evaluation_set: str = "development",
    group_type: str = "all",
    group_value: str = "all",
) -> VerificationComponent:
    """Builds a VerificationComponent for continuous errors from clean arrays."""
    f = np.asarray(forecast, dtype=float).ravel()
    o = np.asarray(observed, dtype=float).ravel()

    valid = ~(np.isnan(f) | np.isnan(o))
    f_clean = f[valid]
    o_clean = o[valid]

    n = len(f_clean)
    if n == 0:
        return VerificationComponent(
            imd_date=imd_date,
            lead_day=lead_day,
            forecast_type=forecast_type,
            evaluation_set=evaluation_set,
            group_type=group_type,
            group_value=group_value,
            n=0,
            n_samples=0,
        )

    diff = f_clean - o_clean
    return VerificationComponent(
        imd_date=imd_date,
        lead_day=lead_day,
        forecast_type=forecast_type,
        evaluation_set=evaluation_set,
        group_type=group_type,
        group_value=group_value,
        n=n,
        n_samples=n,
        sum_error=float(np.sum(diff)),
        sum_abs_error=float(np.sum(np.abs(diff))),
        sum_squared_error=float(np.sum(diff ** 2)),
    )
