"""PRD §16.9 standard verification plot and display data preparation.

Provides pure data preparation functions independent of matplotlib:
1. Performance diagram data (POD vs Success Ratio, CSI, Frequency Bias).
2. FSS vs neighbourhood size (ordered neighbourhoods with approximate km).
3. Reliability diagram data (ordered bins with predicted vs observed frequencies).
4. Bias by raw-rain size (ordered raw-rain bins).
5. Average improvement map data (accumulated cell improvements).
6. Per-season improvement dots (deterministic season order with metric direction).
7. B0 -> B3 comparison table data (fixed row order, strictly unmixed evaluation sets).
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from verification.aggregation import compute_bias_from_components, evaluate_all_metrics_from_components
from verification.components import VerificationComponent
from verification.groups import RAW_RAIN_BINS
from verification.metrics import MetricResult
from verification.reliability import ReliabilityBinSummary, summarize_reliability_bins
from verification.results import VerificationResult

# PRD §16.8 neighbourhood cell-to-km mapping
NEIGHBOURHOOD_KM_MAP: Dict[int, int] = {
    1: 28,
    3: 84,
    5: 140,
    9: 250,
}

# PRD §10.9 & §16.9 the four systems in a row
B0_TO_B3_SYSTEMS: Tuple[Tuple[str, str], ...] = (
    ("B0", "raw_nwp"),
    ("B1", "quantile_mapping"),
    ("B2", "global_ml"),
    ("B3", "regime_aware_ml"),
)


# ---------------------------------------------------------------------------
# 1. Performance Diagram Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerformanceDiagramPoint:
    """Data point for a performance diagram (PRD §16.9)."""
    pod: Optional[float]
    far: Optional[float]
    success_ratio: Optional[float]
    csi: Optional[float] = None
    frequency_bias: Optional[float] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_performance_diagram_data(
    pod: Optional[float] = None,
    far: Optional[float] = None,
    csi: Optional[float] = None,
    frequency_bias: Optional[float] = None,
    label: Optional[str] = None,
    item: Optional[Any] = None,
    items: Optional[Sequence[Any]] = None,
) -> Union[PerformanceDiagramPoint, List[PerformanceDiagramPoint]]:
    """Prepares performance diagram data: POD vs. Success Ratio (1 - FAR).
    
    Rule:
    - success_ratio = 1 - FAR
    - Retains CSI and Frequency Bias without recomputation from raw grids.
    - Undefined values remain None.
    """
    def _extract_from_obj(obj: Any, lbl: Optional[str] = None) -> PerformanceDiagramPoint:
        p_val: Optional[float] = None
        f_val: Optional[float] = None
        c_val: Optional[float] = None
        fb_val: Optional[float] = None
        item_lbl: Optional[str] = lbl

        if isinstance(obj, VerificationResult):
            p_val = obj.metrics.get("pod", {}).value if hasattr(obj.metrics.get("pod"), "value") else None
            f_val = obj.metrics.get("far", {}).value if hasattr(obj.metrics.get("far"), "value") else None
            c_val = obj.metrics.get("csi", {}).value if hasattr(obj.metrics.get("csi"), "value") else None
            fb_val = obj.metrics.get("frequency_bias", {}).value if hasattr(obj.metrics.get("frequency_bias"), "value") else None
            item_lbl = lbl or obj.forecast_type
        elif isinstance(obj, VerificationComponent):
            mets = evaluate_all_metrics_from_components(obj)
            p_val = mets.get("pod").value if mets.get("pod") else None
            f_val = mets.get("far").value if mets.get("far") else None
            c_val = mets.get("csi").value if mets.get("csi") else None
            fb_val = mets.get("frequency_bias").value if mets.get("frequency_bias") else None
            item_lbl = lbl or obj.forecast_type
        elif isinstance(obj, dict):
            p_val = obj.get("pod")
            f_val = obj.get("far")
            c_val = obj.get("csi")
            fb_val = obj.get("frequency_bias")
            item_lbl = lbl or obj.get("label") or obj.get("forecast_type")

        sr_val = (1.0 - float(f_val)) if f_val is not None else None
        return PerformanceDiagramPoint(
            pod=float(p_val) if p_val is not None else None,
            far=float(f_val) if f_val is not None else None,
            success_ratio=sr_val,
            csi=float(c_val) if c_val is not None else None,
            frequency_bias=float(fb_val) if fb_val is not None else None,
            label=item_lbl,
        )

    if items is not None:
        return [_extract_from_obj(it) for it in items]

    if item is not None:
        return _extract_from_obj(item, lbl=label)

    sr = (1.0 - float(far)) if far is not None else None
    return PerformanceDiagramPoint(
        pod=float(pod) if pod is not None else None,
        far=float(far) if far is not None else None,
        success_ratio=sr,
        csi=float(csi) if csi is not None else None,
        frequency_bias=float(frequency_bias) if frequency_bias is not None else None,
        label=label,
    )


# ---------------------------------------------------------------------------
# 2. FSS vs Neighbourhood Size Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FSSNeighbourhoodPoint:
    """Data point for FSS vs. neighbourhood size (PRD §16.8, §16.9)."""
    neighbourhood_cells: int
    approximate_km: int
    fss: Optional[float]
    useful_skill_line: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_fss_neighbourhood_data(
    items: Sequence[Any],
    useful_skill: Optional[float] = None,
    f0: Optional[float] = None,
) -> List[FSSNeighbourhoodPoint]:
    """Prepares ordered FSS vs. neighbourhood size curve data.
    
    Mapping (PRD §16.8):
    - 1 cell  -> 28 km
    - 3 cells -> 84 km
    - 5 cells -> 140 km
    - 9 cells -> 250 km
    """
    points: List[FSSNeighbourhoodPoint] = []

    # Calculate default useful skill line if f0 provided: 0.5 + f0 / 2
    useful_val = useful_skill
    if useful_val is None and f0 is not None:
        useful_val = 0.5 + float(f0) / 2.0

    for it in items:
        w: Optional[int] = None
        fss_val: Optional[float] = None
        item_useful = useful_val

        if isinstance(it, VerificationResult):
            w = it.neighbourhood_cells
            fss_rec = it.metrics.get("fss")
            fss_val = fss_rec.value if fss_rec is not None else None
        elif isinstance(it, VerificationComponent):
            w = it.neighbourhood_cells
            mets = evaluate_all_metrics_from_components(it)
            fss_res = mets.get("fss")
            fss_val = fss_res.value if fss_res is not None else None
            if item_useful is None and it.n_samples > 0 and it.n_events is not None:
                item_useful = 0.5 + (it.n_events / it.n_samples) / 2.0
        elif isinstance(it, dict):
            w = it.get("neighbourhood_cells") or it.get("neighbourhood") or it.get("w")
            fss_val = it.get("fss") or it.get("value")
            if item_useful is None and "useful_skill_line" in it:
                item_useful = it["useful_skill_line"]

        if w is None:
            continue

        w_int = int(w)
        km = NEIGHBOURHOOD_KM_MAP.get(w_int, int(round(w_int * 28)))

        points.append(
            FSSNeighbourhoodPoint(
                neighbourhood_cells=w_int,
                approximate_km=km,
                fss=float(fss_val) if fss_val is not None else None,
                useful_skill_line=float(item_useful) if item_useful is not None else None,
            )
        )

    # Sort strictly by neighbourhood_cells ascending
    return sorted(points, key=lambda p: p.neighbourhood_cells)


# ---------------------------------------------------------------------------
# 3. Reliability Diagram Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReliabilityBinPoint:
    """Data point for a single reliability diagram probability bin (PRD §16.9)."""
    bin_lower: float
    bin_upper: float
    mean_predicted_probability: Optional[float]
    observed_frequency: Optional[float]
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_reliability_diagram_data(
    summaries_or_components: Sequence[Any],
) -> List[ReliabilityBinPoint]:
    """Prepares ordered reliability diagram points from summaries or components."""
    if not summaries_or_components:
        return []

    # If VerificationComponent records passed, aggregate them first
    first = summaries_or_components[0]
    if isinstance(first, VerificationComponent):
        summaries = summarize_reliability_bins(summaries_or_components)
    elif isinstance(first, ReliabilityBinSummary):
        summaries = list(summaries_or_components)
    else:
        # Sequence of dicts
        summaries = []
        for d in summaries_or_components:
            summaries.append(
                ReliabilityBinSummary(
                    bin_lower=float(d["bin_lower"]),
                    bin_upper=float(d["bin_upper"]),
                    n=int(d["n"]),
                    sum_predicted_probability=float(d.get("sum_predicted_probability", 0.0)),
                    n_observed_events=int(d.get("n_observed_events", 0)),
                    mean_predicted_probability=d.get("mean_predicted_probability"),
                    observed_frequency=d.get("observed_frequency"),
                )
            )

    points: List[ReliabilityBinPoint] = []
    for s in summaries:
        points.append(
            ReliabilityBinPoint(
                bin_lower=s.bin_lower,
                bin_upper=s.bin_upper,
                mean_predicted_probability=s.mean_predicted_probability,
                observed_frequency=s.observed_frequency,
                n=s.n,
            )
        )

    # Sort strictly by bin_lower ascending
    return sorted(points, key=lambda p: p.bin_lower)


# ---------------------------------------------------------------------------
# 4. Bias by Raw-Rain Size Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawRainBiasPoint:
    """Bias metric record for a raw-rain size category (PRD §16.9)."""
    bin_name: str
    bias: Optional[float]
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_raw_rain_bias_data(
    components_or_results: Sequence[Any],
) -> List[RawRainBiasPoint]:
    """Prepares bias by raw-rain size points ordered by PRD canonical bins:
    below_1, 1_to_15_6, 15_6_to_64_5, 64_5_and_above.
    """
    bin_map: Dict[str, Dict[str, Any]] = {}

    for it in components_or_results:
        b_name: Optional[str] = None
        bias_val: Optional[float] = None
        n_val: int = 0

        if isinstance(it, VerificationComponent):
            if it.group_type == "raw_rain_size":
                b_name = it.group_value
                bias_res = compute_bias_from_components(it)
                bias_val = bias_res.value
                n_val = it.n_samples
        elif isinstance(it, VerificationResult):
            if it.group.get("type") == "raw_rain_size":
                b_name = it.group.get("value")
                bias_rec = it.metrics.get("bias")
                bias_val = bias_rec.value if bias_rec is not None else None
                n_val = it.n_samples or 0
        elif isinstance(it, dict):
            grp = it.get("group", {})
            if it.get("group_type") == "raw_rain_size" or grp.get("type") == "raw_rain_size":
                b_name = it.get("group_value") or grp.get("value")
                bias_val = it.get("bias") or it.get("mean_bias")
                n_val = it.get("n_samples", 0)
            elif it.get("bin_name") in RAW_RAIN_BINS:
                b_name = it.get("bin_name")
                bias_val = it.get("bias")
                n_val = it.get("n_samples", 0)

        if b_name in RAW_RAIN_BINS:
            bin_map[b_name] = {
                "bias": float(bias_val) if bias_val is not None else None,
                "n_samples": int(n_val),
            }

    # Fixed output order matching PRD §16.5
    points: List[RawRainBiasPoint] = []
    for b_name in RAW_RAIN_BINS:
        info = bin_map.get(b_name, {"bias": None, "n_samples": 0})
        points.append(
            RawRainBiasPoint(
                bin_name=b_name,
                bias=info["bias"],
                n_samples=info["n_samples"],
            )
        )

    return points


# ---------------------------------------------------------------------------
# 5. Average Improvement Map Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellImprovementPoint:
    """Accumulated cell improvement data for spatial mapping (PRD §15 F1, §16.9)."""
    cell_id: Union[int, str]
    mean_improvement_mm: Optional[float]
    n_dates: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_average_improvement_map_data(
    improvements: Union[Sequence[Any], Mapping[Union[int, str], Sequence[float]]],
    lat_lon_map: Optional[Mapping[Union[int, str], Tuple[float, float]]] = None,
) -> List[CellImprovementPoint]:
    """Prepares per-cell average improvement data across multiple forecast dates.
    
    Formula:
        mean_improvement_mm = Σ(|raw - obs| - |corrected - obs|) / n_dates
    
    Rules:
    - Never invent lat/lon if not supplied.
    - Preserves deterministic ordering by cell_id.
    """
    accumulators: Dict[Union[int, str], Dict[str, Any]] = {}

    if isinstance(improvements, Mapping):
        for cell_id, vals in improvements.items():
            valid_vals = [float(v) for v in vals if v is not None]
            accumulators[cell_id] = {
                "sum_diff": sum(valid_vals),
                "count": len(valid_vals),
            }
    else:
        for it in improvements:
            c_id: Optional[Union[int, str]] = None
            diff: Optional[float] = None

            if hasattr(it, "cell_id") and hasattr(it, "improvement_mm"):
                c_id = getattr(it, "cell_id")
                diff = getattr(it, "improvement_mm")
            elif isinstance(it, dict):
                c_id = it.get("cell_id")
                diff = it.get("improvement_mm")
            elif isinstance(it, (tuple, list)) and len(it) >= 2:
                c_id = it[0]
                diff = it[1]

            if c_id is not None and diff is not None:
                if c_id not in accumulators:
                    accumulators[c_id] = {"sum_diff": 0.0, "count": 0}
                accumulators[c_id]["sum_diff"] += float(diff)
                accumulators[c_id]["count"] += 1

    points: List[CellImprovementPoint] = []
    for c_id in sorted(accumulators.keys(), key=lambda k: (str(type(k)), str(k))):
        data = accumulators[c_id]
        cnt = data["count"]
        mean_val = (data["sum_diff"] / cnt) if cnt > 0 else None

        lat = None
        lon = None
        if lat_lon_map and c_id in lat_lon_map:
            lat, lon = lat_lon_map[c_id]

        points.append(
            CellImprovementPoint(
                cell_id=c_id,
                mean_improvement_mm=float(mean_val) if mean_val is not None else None,
                n_dates=cnt,
                latitude=float(lat) if lat is not None else None,
                longitude=float(lon) if lon is not None else None,
            )
        )

    return points


# ---------------------------------------------------------------------------
# 6. Per-Season Improvement Dots Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeasonImprovementDot:
    """Per-season metric comparison dot for seasonal consistency plots (PRD §16.6, §16.9)."""
    season: Union[int, str]
    system_comparison: str
    metric_difference: Optional[float]
    metric_direction: str  # "lower" or "higher"
    metric_name: Optional[str] = None
    value_a: Optional[float] = None
    value_b: Optional[float] = None
    favored_system: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_per_season_improvement_data(
    season_metrics_a: Mapping[Union[int, str], Optional[float]],
    season_metrics_b: Mapping[Union[int, str], Optional[float]],
    system_comparison: str = "B3 vs B2",
    metric_name: str = "RMSE",
    metric_direction: str = "lower",
) -> List[SeasonImprovementDot]:
    """Prepares per-season paired difference dot data in deterministic season order."""
    direction = metric_direction.strip().lower()
    if direction not in ("lower", "higher"):
        raise ValueError(f"metric_direction must be 'lower' or 'higher', got '{metric_direction}'")

    all_seasons = sorted(
        list(set(season_metrics_a.keys()).union(set(season_metrics_b.keys()))),
        key=lambda s: (str(type(s)), str(s)),
    )

    dots: List[SeasonImprovementDot] = []
    for s in all_seasons:
        va = season_metrics_a.get(s)
        vb = season_metrics_b.get(s)

        diff: Optional[float] = None
        favored: Optional[str] = None

        if va is not None and vb is not None:
            diff = float(va) - float(vb)
            if abs(diff) < 1e-9:
                favored = "tie"
            elif direction == "lower":
                # Lower is better: diff < 0 means A has lower loss (A better)
                favored = "system_a" if diff < 0.0 else "system_b"
            else:
                # Higher is better: diff > 0 means A has higher skill (A better)
                favored = "system_a" if diff > 0.0 else "system_b"

        dots.append(
            SeasonImprovementDot(
                season=s,
                system_comparison=system_comparison,
                metric_difference=diff,
                metric_direction=direction,
                metric_name=metric_name,
                value_a=float(va) if va is not None else None,
                value_b=float(vb) if vb is not None else None,
                favored_system=favored,
            )
        )

    return dots


# ---------------------------------------------------------------------------
# 7. B0 -> B3 Comparison Table Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class B0ToB3Row:
    """Fixed-order row in the B0 -> B3 progression table (PRD §10.9, §16.9)."""
    system_id: str
    forecast_type: str
    metric_name: str
    metric_value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n_samples: Optional[int]
    n_events: Optional[int]
    undefined_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_b0_to_b3_table_data(
    records: Sequence[Any],
    metric_name: str = "rmse",
    evaluation_set: Optional[str] = None,
) -> List[B0ToB3Row]:
    """Prepares fixed-order B0 -> B3 comparison table data (PRD §10.9, §16.9).
    
    Rows:
    1. B0: raw_nwp
    2. B1: quantile_mapping
    3. B2: global_ml
    4. B3: regime_aware_ml
    
    Rule:
    - Never mix development and holdout evaluation sets; raises ValueError if mixed.
    - Undefined metrics preserve None and undefined_reason.
    """
    m_name = metric_name.strip().lower()

    # Verify evaluation set consistency
    eval_sets = set()
    for r in records:
        e_set = getattr(r, "evaluation_set", None)
        if e_set is None and isinstance(r, dict):
            e_set = r.get("evaluation_set")
        if e_set:
            eval_sets.add(str(e_set).lower())

    if len(eval_sets) > 1:
        raise ValueError(
            f"Cannot mix development and holdout evaluation sets in B0->B3 comparison: found {eval_sets}"
        )

    # Index records by forecast_type
    by_type: Dict[str, Any] = {}
    for r in records:
        f_type: Optional[str] = None
        if isinstance(r, (VerificationResult, VerificationComponent)):
            f_type = r.forecast_type
        elif isinstance(r, dict):
            f_type = r.get("forecast_type")

        if f_type:
            by_type[f_type] = r

    rows: List[B0ToB3Row] = []

    for sys_id, f_type in B0_TO_B3_SYSTEMS:
        item = by_type.get(f_type)
        val: Optional[float] = None
        ci_l: Optional[float] = None
        ci_h: Optional[float] = None
        ns: Optional[int] = None
        ne: Optional[int] = None
        reason: Optional[str] = None

        if item is not None:
            if isinstance(item, VerificationResult):
                m_rec = item.metrics.get(m_name)
                if m_rec is not None:
                    val = m_rec.value
                    ci_l = m_rec.ci_low
                    ci_h = m_rec.ci_high
                    reason = m_rec.undefined_reason
                ns = item.n_samples
                ne = item.n_events
            elif isinstance(item, VerificationComponent):
                mets = evaluate_all_metrics_from_components(item)
                m_res = mets.get(m_name)
                if m_res is not None:
                    val = m_res.value
                    reason = m_res.undefined_reason
                ns = item.n_samples
                ne = item.n_events
            elif isinstance(item, dict):
                mets = item.get("metrics", {})
                if m_name in mets and isinstance(mets[m_name], dict):
                    val = mets[m_name].get("value")
                    ci_l = mets[m_name].get("ci_low")
                    ci_h = mets[m_name].get("ci_high")
                    reason = mets[m_name].get("undefined_reason")
                else:
                    val = item.get(m_name)
                    ci_l = item.get("ci_low")
                    ci_h = item.get("ci_high")
                    reason = item.get("undefined_reason")
                ns = item.get("n_samples")
                ne = item.get("n_events")
        else:
            reason = "SYSTEM_NOT_PROVIDED"

        rows.append(
            B0ToB3Row(
                system_id=sys_id,
                forecast_type=f_type,
                metric_name=m_name,
                metric_value=float(val) if val is not None else None,
                ci_low=float(ci_l) if ci_l is not None else None,
                ci_high=float(ci_h) if ci_h is not None else None,
                n_samples=int(ns) if ns is not None else None,
                n_events=int(ne) if ne is not None else None,
                undefined_reason=reason,
            )
        )

    return rows
