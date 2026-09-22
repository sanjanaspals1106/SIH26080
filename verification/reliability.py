"""Reliability diagram bins and probability verification components (PRD §13.5, §16.2, §16.4).

Implements:
1. Reliability bin configuration and loading from config/verification.yaml.
2. Deterministic, non-overlapping probability bin assignment (0.0 to 1.0).
3. Additive component representations for reliability diagram data.
4. Aggregation of reliability bins with mean predicted probability and observed frequency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml

from verification.components import VerificationComponent

DEFAULT_RELIABILITY_BINS: List[Tuple[float, float]] = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
]


def get_reliability_bins_from_config(
    config_path: Optional[Union[str, Path]] = None,
) -> List[Tuple[float, float]]:
    """Reads configured reliability bins from config/verification.yaml."""
    cfg_file = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parent.parent / "config" / "verification.yaml"
    )

    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            raw_bins = data.get("reliability", {}).get("bins")
            if raw_bins:
                parsed = []
                for b in raw_bins:
                    if isinstance(b, (list, tuple)) and len(b) == 2:
                        parsed.append((float(b[0]), float(b[1])))
                if parsed:
                    return parsed
        except Exception:
            pass

    return list(DEFAULT_RELIABILITY_BINS)


@dataclass(frozen=True)
class ReliabilityBinSummary:
    """Aggregated summary of a single reliability diagram bin."""
    bin_lower: float
    bin_upper: float
    n: int
    sum_predicted_probability: float
    n_observed_events: int
    mean_predicted_probability: Optional[float]
    observed_frequency: Optional[float]

    @property
    def is_empty(self) -> bool:
        return self.n == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_lower": self.bin_lower,
            "bin_upper": self.bin_upper,
            "n": self.n,
            "sum_predicted_probability": self.sum_predicted_probability,
            "n_observed_events": self.n_observed_events,
            "mean_predicted_probability": self.mean_predicted_probability,
            "observed_frequency": self.observed_frequency,
            "is_empty": self.is_empty,
        }


def assign_probability_bins(
    probabilities: Union[Sequence[float], np.ndarray],
    bins: Optional[Sequence[Tuple[float, float]]] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Assigns each valid probability to exactly one non-overlapping bin.
    
    Rules:
    - Interval for bin i < K-1 is [lower, upper)
    - Interval for the final bin i = K-1 is [lower, upper]
    - Probabilities 0.0 and 1.0 are deterministically assigned
    - Every valid probability belongs to exactly one bin
    - Returned masks have the same shape as probabilities
    
    Returns:
        List of boolean masks corresponding to each bin.
    """
    p_arr = np.asarray(probabilities, dtype=float)
    bin_list = list(bins) if bins is not None else get_reliability_bins_from_config()

    if valid_mask is not None:
        v_mask = valid_mask & (~np.isnan(p_arr))
    else:
        v_mask = ~np.isnan(p_arr)

    # Clean clip tiny float precision noise to [0.0, 1.0]
    p_clamped = np.clip(p_arr, 0.0, 1.0)

    masks: List[np.ndarray] = []
    n_bins = len(bin_list)

    for i, (lower, upper) in enumerate(bin_list):
        low_f = float(lower)
        upp_f = float(upper)

        if i == n_bins - 1:
            # Last bin includes upper boundary (e.g. 1.0)
            in_bin = v_mask & (p_clamped >= low_f) & (p_clamped <= upp_f)
        else:
            # Interior bins are half-open [lower, upper)
            in_bin = v_mask & (p_clamped >= low_f) & (p_clamped < upp_f)

        masks.append(in_bin)

    return masks


def compute_reliability_bin_components(
    predicted_probabilities: Union[Sequence[float], np.ndarray],
    observed_grid_mm: Union[Sequence[float], np.ndarray],
    threshold_mm: float,
    bins: Optional[Sequence[Tuple[float, float]]] = None,
    valid_mask: Optional[np.ndarray] = None,
    imd_date: Optional[Union[str, Any]] = None,
    lead_day: int = 1,
    forecast_type: str = "regime_aware_ml",
    evaluation_set: str = "development",
    group_type: str = "all",
    group_value: str = "all",
) -> List[VerificationComponent]:
    """Generates additive VerificationComponent records for reliability diagram bins."""
    p_arr = np.asarray(predicted_probabilities, dtype=float)
    o_arr = np.asarray(observed_grid_mm, dtype=float)

    if p_arr.shape != o_arr.shape:
        raise ValueError(
            f"Shape mismatch: probabilities {p_arr.shape} vs observed {o_arr.shape}"
        )

    t_float = float(threshold_mm)
    base_valid = (~np.isnan(p_arr)) & (~np.isnan(o_arr))
    if valid_mask is not None:
        base_valid = base_valid & np.asarray(valid_mask, dtype=bool)

    bin_list = list(bins) if bins is not None else get_reliability_bins_from_config()
    bin_masks = assign_probability_bins(p_arr, bins=bin_list, valid_mask=base_valid)

    components: List[VerificationComponent] = []

    for (low, upp), b_mask in zip(bin_list, bin_masks):
        n_in_bin = int(np.count_nonzero(b_mask))
        if n_in_bin > 0:
            p_bin = p_arr[b_mask]
            o_bin = o_arr[b_mask]
            sum_p = float(np.sum(p_bin))
            n_events = int(np.count_nonzero(o_bin >= t_float))
        else:
            sum_p = 0.0
            n_events = 0

        comp = VerificationComponent(
            imd_date=imd_date,
            lead_day=lead_day,
            forecast_type=forecast_type,
            evaluation_set=evaluation_set,
            threshold_mm=t_float,
            neighbourhood_cells=None,
            group_type=group_type,
            group_value=group_value,
            bin_lower=float(low),
            bin_upper=float(upp),
            n_samples=n_in_bin,
            n_events=n_events,
            n=n_in_bin,
            sum_predicted_probability=sum_p,
            n_observed_events=n_events,
        )
        components.append(comp)

    return components


def summarize_reliability_bins(
    bin_components: Sequence[VerificationComponent],
) -> List[ReliabilityBinSummary]:
    """Aggregates a collection of reliability bin components into ReliabilityBinSummary list."""
    # Group by (bin_lower, bin_upper)
    totals: Dict[Tuple[float, float], Dict[str, Any]] = {}

    for c in bin_components:
        if c.bin_lower is None or c.bin_upper is None:
            continue
        key = (round(float(c.bin_lower), 4), round(float(c.bin_upper), 4))
        if key not in totals:
            totals[key] = {
                "bin_lower": c.bin_lower,
                "bin_upper": c.bin_upper,
                "n": 0,
                "sum_p": 0.0,
                "n_events": 0,
            }
        totals[key]["n"] += c.n
        totals[key]["sum_p"] += c.sum_predicted_probability
        totals[key]["n_events"] += c.n_observed_events

    summaries: List[ReliabilityBinSummary] = []
    for key in sorted(totals.keys()):
        d = totals[key]
        n = d["n"]
        if n > 0:
            mean_p = float(d["sum_p"] / n)
            obs_freq = float(d["n_events"] / n)
        else:
            mean_p = None
            obs_freq = None

        summaries.append(
            ReliabilityBinSummary(
                bin_lower=d["bin_lower"],
                bin_upper=d["bin_upper"],
                n=n,
                sum_predicted_probability=d["sum_p"],
                n_observed_events=d["n_events"],
                mean_predicted_probability=mean_p,
                observed_frequency=obs_freq,
            )
        )

    return summaries
