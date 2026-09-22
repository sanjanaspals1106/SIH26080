"""Verification compute engine (PRD §16.1–§16.5, §16.10).

Converts one date/run/lead prediction + observation inputs into VerificationComponent records.

Rules:
1. All systems compared on exactly the same valid observation samples.
2. Missing observation excludes that sample.
3. Missing forecast for one system must not silently change comparison sample set
   when a common comparison mask is supplied.
4. Sea and invalid cells excluded via valid_mask.
5. Preserve evaluation_set exactly (no holdout/development mixing).
6. Undefined metrics represented by additive components; do not invent zero skill.
7. n_events comes from external event counting (§10.7, §13.2) and is NEVER inferred from a+c.
"""

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

from verification.components import (
    VALID_EVALUATION_SETS,
    VALID_FORECAST_TYPES,
    VerificationComponent,
)
from verification.metrics.spatial import compute_fss_components

DEFAULT_THRESHOLDS = (15.6, 64.5, 115.6)
DEFAULT_NEIGHBOURHOODS = (1, 3, 5, 9)


@dataclass
class VerificationComputeResult:
    """Container for all verification component records computed for one date/run/lead."""
    components: List[VerificationComponent]

    def __iter__(self) -> Iterator[VerificationComponent]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, index: int) -> VerificationComponent:
        return self.components[index]

    def get_continuous(
        self,
        group_type: str = "all",
        group_value: str = "all",
    ) -> Optional[VerificationComponent]:
        """Returns the continuous verification component for the requested group."""
        for c in self.components:
            if c.threshold_mm is None and c.neighbourhood_cells is None:
                if c.group_type == group_type and c.group_value == group_value:
                    return c
        return None

    def get_contingency(
        self,
        threshold_mm: float,
        group_type: str = "all",
        group_value: str = "all",
    ) -> Optional[VerificationComponent]:
        """Returns the categorical/contingency component for the requested threshold."""
        for c in self.components:
            if c.neighbourhood_cells is None and c.threshold_mm is not None and c.bin_lower is None:
                if abs(c.threshold_mm - threshold_mm) < 1e-4:
                    if c.group_type == group_type and c.group_value == group_value:
                        return c
        return None

    def get_reliability_bins(
        self,
        threshold_mm: float,
        group_type: str = "all",
        group_value: str = "all",
    ) -> List[VerificationComponent]:
        """Returns the list of reliability bin components for the requested threshold and group."""
        out = []
        for c in self.components:
            if c.threshold_mm is not None and c.bin_lower is not None:
                if abs(c.threshold_mm - threshold_mm) < 1e-4:
                    if c.group_type == group_type and c.group_value == group_value:
                        out.append(c)
        return sorted(out, key=lambda x: (x.bin_lower if x.bin_lower is not None else 0.0))

    def get_reliability_summary(
        self,
        threshold_mm: float,
        group_type: str = "all",
        group_value: str = "all",
    ) -> List[Any]:
        """Aggregates reliability bin components into ReliabilityBinSummary list."""
        from verification.reliability import summarize_reliability_bins
        bins = self.get_reliability_bins(threshold_mm, group_type=group_type, group_value=group_value)
        return summarize_reliability_bins(bins)

    def get_fss(
        self,
        threshold_mm: float,
        neighbourhood_cells: int,
        group_type: str = "all",
        group_value: str = "all",
    ) -> Optional[VerificationComponent]:
        """Returns the FSS component for the requested threshold and neighbourhood size."""
        for c in self.components:
            if c.neighbourhood_cells == neighbourhood_cells and c.threshold_mm is not None:
                if abs(c.threshold_mm - threshold_mm) < 1e-4:
                    if c.group_type == group_type and c.group_value == group_value:
                        return c
        return None

    def list_groups(self) -> List[Tuple[str, str]]:
        """Returns a deduplicated list of (group_type, group_value) present in results."""
        seen = set()
        out = []
        for c in self.components:
            pair = (c.group_type, c.group_value)
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
        return out

    def get_components_by_group(
        self,
        group_type: Optional[str] = None,
        group_value: Optional[str] = None,
    ) -> List[VerificationComponent]:
        """Filters components by group_type and/or group_value."""
        return [
            c for c in self.components
            if (group_type is None or c.group_type == group_type)
            and (group_value is None or c.group_value == group_value)
        ]


def _get_threshold_field(
    mapping: Optional[Dict[Any, Any]],
    threshold_mm: float,
) -> Optional[Any]:
    """Extracts field matching threshold from dict supporting float, str, int, or p_ge_ keys."""
    if mapping is None:
        return None
    t_float = float(threshold_mm)
    if t_float in mapping:
        return mapping[t_float]
    if str(t_float) in mapping:
        return mapping[str(t_float)]
    if t_float.is_integer():
        t_int = int(t_float)
        if t_int in mapping:
            return mapping[t_int]
        if str(t_int) in mapping:
            return mapping[str(t_int)]
    str_formatted = str(t_float).replace(".", "_")
    if f"p_ge_{str_formatted}" in mapping:
        return mapping[f"p_ge_{str_formatted}"]
    for k, v in mapping.items():
        try:
            if abs(float(k) - t_float) < 1e-4:
                return v
        except (ValueError, TypeError):
            pass
    return None


def compute_verification_components(
    forecast: Union[Sequence[float], np.ndarray],
    observed: Union[Sequence[float], np.ndarray],
    imd_date: Union[str, date],
    lead_day: int,
    forecast_type: str = "regime_aware_ml",
    evaluation_set: str = "development",
    valid_mask: Optional[Union[Sequence[bool], np.ndarray]] = None,
    common_mask: Optional[Union[Sequence[bool], np.ndarray]] = None,
    thresholds: Optional[Sequence[float]] = DEFAULT_THRESHOLDS,
    neighbourhood_sizes: Optional[Sequence[int]] = DEFAULT_NEIGHBOURHOODS,
    probabilities: Optional[Dict[Any, Union[Sequence[float], np.ndarray]]] = None,
    probability_reference: Optional[Dict[Any, Union[Sequence[float], np.ndarray]]] = None,
    quantiles: Optional[Dict[str, Union[Sequence[float], np.ndarray]]] = None,
    group_masks: Optional[Dict[Tuple[str, str], Optional[np.ndarray]]] = None,
    n_events: Optional[Union[int, Dict[float, int]]] = None,
    group_types: Optional[Sequence[str]] = None,
    region_code: Optional[Union[Sequence[str], np.ndarray]] = None,
    phase: Optional[Union[str, Sequence[str], np.ndarray]] = None,
    lps_near: Optional[Union[bool, str, Sequence[Union[bool, str]], np.ndarray]] = None,
    orographic_favorable: Optional[Union[bool, str, Sequence[Union[bool, str]], np.ndarray]] = None,
    coastal_favorable: Optional[Union[bool, str, Sequence[Union[bool, str]], np.ndarray]] = None,
    raw_forecast: Optional[Union[Sequence[float], np.ndarray]] = None,
    group_metadata: Optional[Dict[str, Any]] = None,
) -> VerificationComputeResult:
    """Computes additive verification component records for a single forecast day and lead.
    
    Args:
        forecast: 1D or 2D forecast rainfall in mm.
        observed: 1D or 2D observed rainfall in mm.
        imd_date: Forecast target date (YYYY-MM-DD or date object).
        lead_day: Forecast lead day (e.g. 1, 2, 3).
        forecast_type: One of 'raw_nwp', 'quantile_mapping', 'global_ml', 'regime_aware_ml'.
        evaluation_set: 'development' or 'holdout'.
        valid_mask: Optional boolean mask (True for valid land cells, False for sea/missing).
        common_mask: Optional boolean mask enforcing identical comparison sample set across models.
        thresholds: Rain thresholds for categorical/FSS/probability components (mm).
        neighbourhood_sizes: Neighbourhood window sizes for FSS (e.g. 1, 3, 5, 9).
        probabilities: Optional mapping of threshold -> model probability field.
        probability_reference: Optional mapping of threshold -> reference probability field (e.g. climatology).
        quantiles: Optional mapping of 'q10', 'q50', 'q90' -> quantile rain fields.
        group_masks: Optional mapping of (group_type, group_value) -> boolean mask.
        n_events: Optional externally computed event counts (int or dict by threshold).
        group_types: Optional list of group types to evaluate (PRD §16.5).
        region_code: Optional per-cell region codes.
        phase: Optional monsoon phase ('active', 'normal', 'break').
        lps_near: Optional LPS within 500 km indicator.
        orographic_favorable: Optional orographic flag.
        coastal_favorable: Optional coastal flag.
        raw_forecast: Optional raw NWP rain grid (used for raw_rain_size grouping).
        group_metadata: Optional dictionary with upstream group attributes.
        
    Returns:
        VerificationComputeResult containing all generated VerificationComponent objects.
    """
    if forecast_type not in VALID_FORECAST_TYPES:
        raise ValueError(
            f"Invalid forecast_type: '{forecast_type}'. Must be one of {sorted(list(VALID_FORECAST_TYPES))}"
        )

    if evaluation_set.lower() not in VALID_EVALUATION_SETS:
        raise ValueError(
            f"Invalid evaluation_set: '{evaluation_set}'. Must be one of {sorted(list(VALID_EVALUATION_SETS))}"
        )

    f_arr = np.asarray(forecast, dtype=float)
    o_arr = np.asarray(observed, dtype=float)

    if f_arr.shape != o_arr.shape:
        raise ValueError(
            f"Shape mismatch between forecast {f_arr.shape} and observed {o_arr.shape}."
        )

    # Base valid mask excludes sea cells and missing observations
    if valid_mask is not None:
        v_mask = np.asarray(valid_mask, dtype=bool)
        if v_mask.shape != o_arr.shape:
            raise ValueError(f"valid_mask shape {v_mask.shape} does not match observed {o_arr.shape}.")
    else:
        v_mask = np.ones_like(o_arr, dtype=bool)

    obs_valid = v_mask & (~np.isnan(o_arr)) & (~np.isinf(o_arr))

    # Apply common mask if supplied (PRD §16.1: all systems compared on exactly same samples)
    if common_mask is not None:
        c_mask = np.asarray(common_mask, dtype=bool)
        if c_mask.shape != o_arr.shape:
            raise ValueError(f"common_mask shape {c_mask.shape} does not match observed {o_arr.shape}.")
        effective_mask = c_mask & obs_valid
        # Check that forecast does not contain NaNs within the common mask
        if np.any(np.isnan(f_arr[effective_mask])):
            raise ValueError(
                "Forecast contains NaNs inside common_mask; missing forecast would silently alter comparison sample set."
            )
    else:
        effective_mask = obs_valid & (~np.isnan(f_arr)) & (~np.isinf(f_arr))

    # Setup groups to evaluate (default: group_type="all", group_value="all")
    if group_masks is not None:
        groups_to_evaluate = list(group_masks.items())
    elif group_types is not None:
        from verification.groups import generate_group_masks

        meta = dict(group_metadata or {})
        rc = region_code if region_code is not None else meta.get("region_code")
        ph = phase if phase is not None else meta.get("phase")
        lps = lps_near if lps_near is not None else meta.get("lps_near")
        oro = orographic_favorable if orographic_favorable is not None else meta.get("orographic_favorable")
        coa = coastal_favorable if coastal_favorable is not None else meta.get("coastal_favorable")
        raw_f = (
            raw_forecast
            if raw_forecast is not None
            else meta.get("raw_forecast", f_arr if forecast_type == "raw_nwp" else None)
        )

        generated_masks = generate_group_masks(
            shape=o_arr.shape,
            group_types=group_types,
            imd_date=imd_date,
            lead_day=lead_day,
            region_code=rc,
            phase=ph,
            lps_near=lps,
            orographic_favorable=oro,
            coastal_favorable=coa,
            raw_forecast=raw_f,
            forecast=f_arr,
            base_mask=effective_mask,
        )
        groups_to_evaluate = list(generated_masks.items())
    else:
        groups_to_evaluate = [(("all", "all"), None)]

    results: List[VerificationComponent] = []

    for (g_type, g_val), g_mask in groups_to_evaluate:
        if g_mask is not None:
            sub_mask = effective_mask & np.asarray(g_mask, dtype=bool)
        else:
            sub_mask = effective_mask

        f_sub = f_arr[sub_mask]
        o_sub = o_arr[sub_mask]
        n_samples = len(f_sub)

        # -------------------------------------------------------------------
        # 1. Continuous Component (threshold_mm=None, neighbourhood_cells=None)
        # -------------------------------------------------------------------
        if n_samples > 0:
            diff = f_sub - o_sub
            sum_err = float(np.sum(diff))
            sum_abs_err = float(np.sum(np.abs(diff)))
            sum_sq_err = float(np.sum(diff ** 2))
        else:
            sum_err = 0.0
            sum_abs_err = 0.0
            sum_sq_err = 0.0

        # Range / Quantile evaluations if quantiles dictionary is provided
        range_n = 0
        coverage_count = 0
        pinball_q10_sum = 0.0
        pinball_q50_sum = 0.0
        pinball_q90_sum = 0.0

        if quantiles is not None and "q10" in quantiles and "q50" in quantiles and "q90" in quantiles:
            q10_sub = np.asarray(quantiles["q10"], dtype=float)[sub_mask]
            q50_sub = np.asarray(quantiles["q50"], dtype=float)[sub_mask]
            q90_sub = np.asarray(quantiles["q90"], dtype=float)[sub_mask]

            range_n = len(o_sub)
            if range_n > 0:
                inside = (o_sub >= q10_sub) & (o_sub <= q90_sub)
                coverage_count = int(np.count_nonzero(inside))

                d10 = o_sub - q10_sub
                pinball_q10_sum = float(np.sum(np.maximum(0.10 * d10, (0.10 - 1.0) * d10)))

                d50 = o_sub - q50_sub
                pinball_q50_sum = float(np.sum(np.maximum(0.50 * d50, (0.50 - 1.0) * d50)))

                d90 = o_sub - q90_sub
                pinball_q90_sum = float(np.sum(np.maximum(0.90 * d90, (0.90 - 1.0) * d90)))

        cont_comp = VerificationComponent(
            imd_date=imd_date,
            lead_day=lead_day,
            forecast_type=forecast_type,
            evaluation_set=evaluation_set,
            threshold_mm=None,
            neighbourhood_cells=None,
            group_type=g_type,
            group_value=g_val,
            n_samples=n_samples,
            n_events=None,  # Continuous has no event threshold
            n=n_samples,
            sum_error=sum_err,
            sum_abs_error=sum_abs_err,
            sum_squared_error=sum_sq_err,
            pinball_q10_sum=pinball_q10_sum,
            pinball_q50_sum=pinball_q50_sum,
            pinball_q90_sum=pinball_q90_sum,
            range_n=range_n,
            coverage_count=coverage_count,
        )
        results.append(cont_comp)

        # -------------------------------------------------------------------
        # 2. Threshold-based Components (Categorical & Probability)
        # -------------------------------------------------------------------
        if thresholds is not None:
            for thresh in thresholds:
                t_float = float(thresh)

                # Determine externally supplied n_events for this threshold (if any)
                thresh_events: Optional[int] = None
                if n_events is not None:
                    if isinstance(n_events, dict):
                        thresh_events = n_events.get(t_float)
                    elif isinstance(n_events, int):
                        thresh_events = n_events

                # Contingency counts
                if n_samples > 0:
                    f_ge = f_sub >= t_float
                    o_ge = o_sub >= t_float
                    a = int(np.count_nonzero(f_ge & o_ge))
                    b = int(np.count_nonzero(f_ge & (~o_ge)))
                    c = int(np.count_nonzero((~f_ge) & o_ge))
                    d = int(np.count_nonzero((~f_ge) & (~o_ge)))
                else:
                    a, b, c, d = 0, 0, 0, 0

                # Brier score terms if probabilities provided
                brier_n = 0
                sum_brier = 0.0
                sum_ref_brier = 0.0

                p_field = _get_threshold_field(probabilities, t_float)
                p_ref_field = _get_threshold_field(probability_reference, t_float)

                p_raw = None
                p_mask = None
                if p_field is not None:
                    p_raw = np.asarray(p_field, dtype=float)
                    if p_raw.shape != o_arr.shape:
                        raise ValueError(
                            f"probabilities shape {p_raw.shape} does not match observed {o_arr.shape}."
                        )
                    p_mask = sub_mask & (~np.isnan(p_raw)) & (~np.isinf(p_raw))

                    if p_ref_field is not None:
                        p_ref_raw = np.asarray(p_ref_field, dtype=float)
                        if p_ref_raw.shape != o_arr.shape:
                            raise ValueError(
                                f"probability_reference shape {p_ref_raw.shape} does not match observed {o_arr.shape}."
                            )
                        p_mask = p_mask & (~np.isnan(p_ref_raw)) & (~np.isinf(p_ref_raw))
                    else:
                        p_ref_raw = None

                    brier_n = int(np.count_nonzero(p_mask))
                    if brier_n > 0:
                        p_sub_valid = p_raw[p_mask]
                        o_sub_valid = o_arr[p_mask]
                        y_sub_valid = np.where(o_sub_valid >= t_float, 1.0, 0.0)
                        sum_brier = float(np.sum((p_sub_valid - y_sub_valid) ** 2))

                        if p_ref_raw is not None:
                            p_ref_sub_valid = p_ref_raw[p_mask]
                            sum_ref_brier = float(np.sum((p_ref_sub_valid - y_sub_valid) ** 2))

                cat_comp = VerificationComponent(
                    imd_date=imd_date,
                    lead_day=lead_day,
                    forecast_type=forecast_type,
                    evaluation_set=evaluation_set,
                    threshold_mm=t_float,
                    neighbourhood_cells=None,
                    group_type=g_type,
                    group_value=g_val,
                    n_samples=n_samples,
                    n_events=thresh_events,  # Strictly external, NEVER inferred from a+c
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                    brier_n=brier_n,
                    sum_brier_terms=sum_brier,
                    sum_reference_brier_terms=sum_ref_brier,
                )
                results.append(cat_comp)

                # Reliability diagram bins for this probability threshold
                if p_raw is not None and p_mask is not None:
                    from verification.reliability import compute_reliability_bin_components

                    rel_comps = compute_reliability_bin_components(
                        predicted_probabilities=p_raw,
                        observed_grid_mm=o_arr,
                        threshold_mm=t_float,
                        valid_mask=p_mask,
                        imd_date=imd_date,
                        lead_day=lead_day,
                        forecast_type=forecast_type,
                        evaluation_set=evaluation_set,
                        group_type=g_type,
                        group_value=g_val,
                    )
                    results.extend(rel_comps)

                # ---------------------------------------------------------------
                # 3. Spatial FSS Components (per threshold and neighbourhood)
                # ---------------------------------------------------------------
                if neighbourhood_sizes is not None and f_arr.ndim == 2:
                    for w in neighbourhood_sizes:
                        w_int = int(w)
                        fss_c = compute_fss_components(
                            forecast=f_arr,
                            observed=o_arr,
                            threshold=t_float,
                            neighbourhood_size=w_int,
                            valid_mask=sub_mask,
                        )

                        fss_comp = VerificationComponent(
                            imd_date=imd_date,
                            lead_day=lead_day,
                            forecast_type=forecast_type,
                            evaluation_set=evaluation_set,
                            threshold_mm=t_float,
                            neighbourhood_cells=w_int,
                            group_type=g_type,
                            group_value=g_val,
                            n_samples=fss_c.n_valid_cells,
                            n_events=thresh_events,
                            fss_numerator_sum=fss_c.numerator_sum,
                            fss_forecast_fraction_sq_sum=fss_c.forecast_fraction_sq_sum,
                            fss_observed_fraction_sq_sum=fss_c.observed_fraction_sq_sum,
                        )
                        results.append(fss_comp)

    return VerificationComputeResult(components=results)


# Alias for verification.compute(...)
compute = compute_verification_components
