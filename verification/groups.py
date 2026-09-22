"""PRD §16.5 Verification grouping and display eligibility engine.

Supports grouping verification components by:
1. all: group_value="all"
2. lead_day: one group per lead
3. month: Jun, Jul, Aug, Sep derived from imd_date
4. region_code: from per-cell region_code values
5. phase: active | normal | break supplied upstream
6. lps_near: yes / no flag supplied upstream (within 500 km)
7. orographic_favorable: true / false supplied upstream
8. coastal_favorable: true / false supplied upstream
9. raw_rain_size: below_1 | 1_to_15_6 | 15_6_to_64_5 | 64_5_and_above

Also implements display eligibility helper:
- Group is displayable only if n_events >= 10 (from config/verification.yaml)
- If n_events is None, status is unknown / not displayable yet
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml

SUPPORTED_GROUP_TYPES: Tuple[str, ...] = (
    "all",
    "lead_day",
    "month",
    "region_code",
    "phase",
    "lps_near",
    "orographic_favorable",
    "coastal_favorable",
    "raw_rain_size",
)

RAW_RAIN_BINS: Tuple[str, ...] = (
    "below_1",
    "1_to_15_6",
    "15_6_to_64_5",
    "64_5_and_above",
)

MONTH_NAMES: Dict[int, str] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

DEFAULT_MIN_EVENTS: int = 10
DEFAULT_RAIN_THRESHOLDS: Dict[str, float] = {
    "light": 1.0,
    "moderate": 15.6,
    "heavy": 64.5,
}


def get_min_events_for_grouping(config_path: Optional[Union[str, Path]] = None) -> int:
    """Reads the minimum event threshold for grouping display eligibility from config/verification.yaml."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "verification.yaml"

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            min_ev = data.get("grouping", {}).get("min_events")
            if min_ev is not None:
                return int(min_ev)
        except Exception:
            pass
    return DEFAULT_MIN_EVENTS


def get_rain_thresholds(config_path: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """Reads rain thresholds from config/thresholds.yaml."""
    thresholds = dict(DEFAULT_RAIN_THRESHOLDS)
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            r_thresh = data.get("rain_thresholds_mm", {})
            if "moderate" in r_thresh:
                thresholds["moderate"] = float(r_thresh["moderate"])
            if "heavy" in r_thresh:
                thresholds["heavy"] = float(r_thresh["heavy"])
        except Exception:
            pass
    return thresholds


@dataclass(frozen=True)
class GroupDisplayEligibility:
    """Display eligibility decision for a group per PRD §16.5."""
    is_displayable: bool
    status: str  # "displayable" | "not_displayable" | "unknown"
    message: str
    n_events: Optional[int]
    min_events: int

    def __bool__(self) -> bool:
        return self.is_displayable


def check_group_display_eligibility(
    n_events: Optional[int],
    min_events: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> GroupDisplayEligibility:
    """Evaluates whether a group satisfies the display threshold (n_events >= 10).
    
    Rules:
    - a group is displayable only if n_events >= 10
    - if n_events is None, status is unknown / not displayable yet
    - threshold 10 comes from config/verification.yaml
    """
    threshold = (
        int(min_events)
        if min_events is not None
        else get_min_events_for_grouping(config_path)
    )

    if n_events is None:
        return GroupDisplayEligibility(
            is_displayable=False,
            status="unknown",
            message="Unknown / not displayable yet (event count not determined)",
            n_events=None,
            min_events=threshold,
        )

    if int(n_events) >= threshold:
        return GroupDisplayEligibility(
            is_displayable=True,
            status="displayable",
            message="Displayable",
            n_events=int(n_events),
            min_events=threshold,
        )

    return GroupDisplayEligibility(
        is_displayable=False,
        status="not_displayable",
        message="Not enough events",
        n_events=int(n_events),
        min_events=threshold,
    )


def is_group_displayable(
    n_events: Optional[int],
    min_events: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Helper returning boolean True if n_events >= min_events, False otherwise."""
    return check_group_display_eligibility(n_events, min_events, config_path).is_displayable


def get_display_status(
    n_events: Optional[int],
    min_events: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> str:
    """Helper returning 'displayable', 'not_displayable', or 'unknown'."""
    return check_group_display_eligibility(n_events, min_events, config_path).status


def get_month_name(imd_date: Union[str, date, datetime]) -> str:
    """Extracts 3-letter month abbreviation (Jun, Jul, Aug, Sep, etc.) from imd_date."""
    if isinstance(imd_date, (date, datetime)):
        m_int = imd_date.month
    elif isinstance(imd_date, str):
        cleaned = imd_date.strip().split()[0].split("T")[0]
        try:
            d = date.fromisoformat(cleaned)
            m_int = d.month
        except ValueError:
            parts = cleaned.replace("/", "-").split("-")
            if len(parts) >= 2:
                m_int = int(parts[1])
            else:
                raise ValueError(f"Unable to parse month from date string: '{imd_date}'")
    else:
        raise TypeError(f"Expected str or date, got {type(imd_date)}")

    if m_int in MONTH_NAMES:
        return MONTH_NAMES[m_int]
    raise ValueError(f"Invalid month integer: {m_int}")


def get_raw_rain_size_masks(
    raw_rain: Union[Sequence[float], np.ndarray],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Generates masks for raw rainfall bins according to PRD §16.5.
    
    Bins:
    - below_1: rain < 1.0 mm
    - 1_to_15_6: 1.0 mm <= rain < 15.6 mm
    - 15_6_to_64_5: 15.6 mm <= rain < 64.5 mm
    - 64_5_and_above: rain >= 64.5 mm
    """
    thresh = thresholds or get_rain_thresholds()
    light = float(thresh.get("light", 1.0))
    moderate = float(thresh.get("moderate", 15.6))
    heavy = float(thresh.get("heavy", 64.5))

    r_arr = np.asarray(raw_rain, dtype=float)

    # Valid values exclude NaNs
    valid = ~np.isnan(r_arr)

    mask_below_1 = valid & (r_arr < light)
    mask_1_to_15_6 = valid & (r_arr >= light) & (r_arr < moderate)
    mask_15_6_to_64_5 = valid & (r_arr >= moderate) & (r_arr < heavy)
    mask_64_5_and_above = valid & (r_arr >= heavy)

    return {
        "below_1": mask_below_1,
        "1_to_15_6": mask_1_to_15_6,
        "15_6_to_64_5": mask_15_6_to_64_5,
        "64_5_and_above": mask_64_5_and_above,
    }


def generate_group_masks(
    shape: Tuple[int, ...],
    group_types: Optional[Sequence[str]] = None,
    imd_date: Optional[Union[str, date, datetime]] = None,
    lead_day: Optional[int] = None,
    region_code: Optional[Union[Sequence[str], np.ndarray]] = None,
    phase: Optional[Union[str, Sequence[str], np.ndarray]] = None,
    lps_near: Optional[Union[bool, str, Sequence[Union[bool, str]], np.ndarray]] = None,
    orographic_favorable: Optional[Union[bool, str, Sequence[Union[bool, str]], np.ndarray]] = None,
    coastal_favorable: Optional[Union[bool, str, Sequence[Union[bool, str]], np.ndarray]] = None,
    raw_forecast: Optional[Union[Sequence[float], np.ndarray]] = None,
    forecast: Optional[Union[Sequence[float], np.ndarray]] = None,
    base_mask: Optional[Union[Sequence[bool], np.ndarray]] = None,
    include_empty: bool = False,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Generates (group_type, group_value) -> boolean mask mapping for verification.
    
    Rules:
    - Missing group metadata causes that group to be skipped (never invent a value).
    - Preserves deterministic group_value names.
    - If base_mask is provided, all returned masks are intersected with base_mask.
    """
    if group_types is None:
        requested = ["all"]
    else:
        requested = list(group_types)

    for gt in requested:
        if gt not in SUPPORTED_GROUP_TYPES:
            raise ValueError(
                f"Unknown group_type '{gt}'. Supported types are: {list(SUPPORTED_GROUP_TYPES)}"
            )

    b_mask = np.asarray(base_mask, dtype=bool) if base_mask is not None else None
    if b_mask is not None and b_mask.shape != shape:
        raise ValueError(f"base_mask shape {b_mask.shape} does not match domain shape {shape}")

    masks: Dict[Tuple[str, str], np.ndarray] = {}

    def _add_mask(g_type: str, g_val: str, condition_mask: np.ndarray) -> None:
        if b_mask is not None:
            effective = condition_mask & b_mask
        else:
            effective = condition_mask

        # For "all", always include it even if empty so default behavior works.
        # For subgroups, include if non-empty or caller explicitly asked for include_empty.
        if g_type == "all" or include_empty or np.any(effective):
            masks[(g_type, g_val)] = effective

    for gt in requested:
        # 1. all
        if gt == "all":
            _add_mask("all", "all", np.ones(shape, dtype=bool))

        # 2. lead_day
        elif gt == "lead_day":
            if lead_day is not None:
                _add_mask("lead_day", str(lead_day), np.ones(shape, dtype=bool))

        # 3. month
        elif gt == "month":
            if imd_date is not None:
                m_name = get_month_name(imd_date)
                _add_mask("month", m_name, np.ones(shape, dtype=bool))

        # 4. region_code
        elif gt == "region_code":
            if region_code is not None:
                rc_arr = np.asarray(region_code)
                if rc_arr.shape != shape:
                    raise ValueError(
                        f"region_code shape {rc_arr.shape} does not match expected shape {shape}"
                    )
                # Extract valid unique codes
                unique_codes = sorted(
                    [
                        str(c)
                        for c in np.unique(rc_arr)
                        if c is not None and str(c).strip() not in ("", "nan", "none", "null")
                    ]
                )
                for code in unique_codes:
                    cond = rc_arr == code
                    _add_mask("region_code", code, cond)

        # 5. phase
        elif gt == "phase":
            if phase is not None:
                if isinstance(phase, str) or np.ndim(phase) == 0:
                    val = str(phase).strip().lower()
                    _add_mask("phase", val, np.ones(shape, dtype=bool))
                else:
                    p_arr = np.asarray(phase)
                    if p_arr.shape != shape:
                        raise ValueError(f"phase shape {p_arr.shape} does not match {shape}")
                    unique_phases = sorted(
                        [
                            str(p).strip().lower()
                            for p in np.unique(p_arr)
                            if p is not None and str(p).strip() not in ("", "nan", "none")
                        ]
                    )
                    for p in unique_phases:
                        cond = np.array([str(x).strip().lower() for x in p_arr.flat]).reshape(shape) == p
                        _add_mask("phase", p, cond)

        # 6. lps_near
        elif gt == "lps_near":
            if lps_near is not None:
                if isinstance(lps_near, (bool, np.bool_)) or (
                    isinstance(lps_near, str) or np.ndim(lps_near) == 0
                ):
                    if isinstance(lps_near, (bool, np.bool_)):
                        val = "yes" if lps_near else "no"
                    else:
                        s = str(lps_near).strip().lower()
                        val = "yes" if s in ("yes", "y", "true", "1") else "no"
                    _add_mask("lps_near", val, np.ones(shape, dtype=bool))
                else:
                    l_arr = np.asarray(lps_near)
                    if l_arr.shape != shape:
                        raise ValueError(f"lps_near shape {l_arr.shape} does not match {shape}")
                    # Per-cell lps
                    yes_mask = (
                        (l_arr == True)
                        | (l_arr == "yes")
                        | (l_arr == "YES")
                        | (l_arr == "true")
                        | (l_arr == "1")
                    )
                    no_mask = (
                        (l_arr == False)
                        | (l_arr == "no")
                        | (l_arr == "NO")
                        | (l_arr == "false")
                        | (l_arr == "0")
                    )
                    _add_mask("lps_near", "yes", yes_mask)
                    _add_mask("lps_near", "no", no_mask)

        # 7. orographic_favorable
        elif gt == "orographic_favorable":
            if orographic_favorable is not None:
                if isinstance(orographic_favorable, (bool, np.bool_)) or (
                    isinstance(orographic_favorable, str) or np.ndim(orographic_favorable) == 0
                ):
                    if isinstance(orographic_favorable, (bool, np.bool_)):
                        val = "true" if orographic_favorable else "false"
                    else:
                        s = str(orographic_favorable).strip().lower()
                        val = "true" if s in ("true", "yes", "1") else "false"
                    _add_mask("orographic_favorable", val, np.ones(shape, dtype=bool))
                else:
                    o_arr = np.asarray(orographic_favorable)
                    if o_arr.shape != shape:
                        raise ValueError(
                            f"orographic_favorable shape {o_arr.shape} does not match {shape}"
                        )
                    true_mask = (
                        (o_arr == True)
                        | (o_arr == "true")
                        | (o_arr == "TRUE")
                        | (o_arr == "yes")
                        | (o_arr == "1")
                    )
                    false_mask = (
                        (o_arr == False)
                        | (o_arr == "false")
                        | (o_arr == "FALSE")
                        | (o_arr == "no")
                        | (o_arr == "0")
                    )
                    _add_mask("orographic_favorable", "true", true_mask)
                    _add_mask("orographic_favorable", "false", false_mask)

        # 8. coastal_favorable
        elif gt == "coastal_favorable":
            if coastal_favorable is not None:
                if isinstance(coastal_favorable, (bool, np.bool_)) or (
                    isinstance(coastal_favorable, str) or np.ndim(coastal_favorable) == 0
                ):
                    if isinstance(coastal_favorable, (bool, np.bool_)):
                        val = "true" if coastal_favorable else "false"
                    else:
                        s = str(coastal_favorable).strip().lower()
                        val = "true" if s in ("true", "yes", "1") else "false"
                    _add_mask("coastal_favorable", val, np.ones(shape, dtype=bool))
                else:
                    c_arr = np.asarray(coastal_favorable)
                    if c_arr.shape != shape:
                        raise ValueError(
                            f"coastal_favorable shape {c_arr.shape} does not match {shape}"
                        )
                    true_mask = (
                        (c_arr == True)
                        | (c_arr == "true")
                        | (c_arr == "TRUE")
                        | (c_arr == "yes")
                        | (c_arr == "1")
                    )
                    false_mask = (
                        (c_arr == False)
                        | (c_arr == "false")
                        | (c_arr == "FALSE")
                        | (c_arr == "no")
                        | (c_arr == "0")
                    )
                    _add_mask("coastal_favorable", "true", true_mask)
                    _add_mask("coastal_favorable", "false", false_mask)

        # 9. raw_rain_size
        elif gt == "raw_rain_size":
            rain = raw_forecast if raw_forecast is not None else forecast
            if rain is not None:
                r_arr = np.asarray(rain, dtype=float)
                if r_arr.shape != shape:
                    raise ValueError(f"raw_forecast shape {r_arr.shape} does not match {shape}")
                bin_masks = get_raw_rain_size_masks(r_arr)
                for b_name in RAW_RAIN_BINS:
                    cond = bin_masks[b_name]
                    _add_mask("raw_rain_size", b_name, cond)

    return masks
