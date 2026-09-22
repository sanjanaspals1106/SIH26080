"""M4 Bias Table computation for Feature F2 (PRD §15 F2 and §10.4).

For each:
    district_id, lead_day, phase, lps_near
compute:
    bias = observed_mean_mm - raw_mean_mm

Store:
    - n_dates
    - median_diff_mm
    - q25_diff_mm
    - q75_diff_mm
    - season_excluded
    - few_past_cases (bool, True if n_dates < 20)

Rules:
1. DEVELOPMENT query:
   - Use development seasons only.
   - EXCLUDE the query's own season.
   - season_excluded = query season.
2. HOLDOUT query:
   - Use all development seasons.
   - Never use holdout rows in the history table.
   - season_excluded = null (None).
3. Phase is the most-probable phase already provided by upstream data.
4. lps_near is already provided upstream.
5. Missing observed/raw values excluded.
6. If n_dates < 20 expose few_past_cases=true; if n_dates >= 20 => false.
7. Do not invent data when no matching history exists.
8. Empty match => n_dates=0 and statistics=None.
9. Pure computation only; no DB/API/frontend.
10. No leakage from query season for development queries.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

MIN_DATES_THRESHOLD = 20

HOLDOUT_SET_NAMES = {"holdout", "test", "locked_holdout"}
DEVELOPMENT_SET_NAMES = {"development", "dev", "loso", "train", "training", "oof"}


@dataclass
class BiasRecord:
    """Represents a computed bias table record for a district and regime."""
    district_id: Optional[Union[str, int]] = None
    lead_day: Optional[int] = None
    phase: Optional[str] = None
    lps_near: Optional[bool] = None
    n_dates: int = 0
    median_diff_mm: Optional[float] = None
    q25_diff_mm: Optional[float] = None
    q75_diff_mm: Optional[float] = None
    season_excluded: Optional[int] = None
    few_past_cases: bool = True
    note: Optional[str] = "few past cases"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _is_missing(val: Any) -> bool:
    """Checks if value is None, NaN, inf, or a missing string placeholder."""
    if val is None:
        return True
    if isinstance(val, (float, int)):
        return math.isnan(val) or math.isinf(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "none", "nan", "null"):
            return True
    return False


def _to_float(val: Any) -> Optional[float]:
    """Converts value to float or returns None if missing or invalid."""
    if _is_missing(val):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _to_bool(val: Any) -> bool:
    """Converts value to boolean reliably."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "t", "y")
    return False


def _extract_field(record: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Extracts a field from a dictionary or object using candidate keys."""
    if isinstance(record, dict):
        for k in keys:
            if k in record and record[k] is not None:
                return record[k]
        return default
    for k in keys:
        if hasattr(record, k):
            val = getattr(record, k)
            if val is not None:
                return val
    return default


def _is_holdout_row(
    row: Any,
    holdout_seasons: Optional[Sequence[int]] = None,
    development_seasons: Optional[Sequence[int]] = None,
) -> bool:
    """Determines whether a historical row belongs to the holdout set and must be excluded from history."""
    eval_set = _extract_field(row, ["evaluation_set", "eval_set", "prediction_source", "source", "split"])
    if eval_set is not None:
        eval_str = str(eval_set).strip().lower()
        if eval_str in HOLDOUT_SET_NAMES:
            return True
        if eval_str in DEVELOPMENT_SET_NAMES:
            return False

    season = _extract_field(row, ["season", "year"])
    if season is not None:
        try:
            s_int = int(season)
            if holdout_seasons is not None and s_int in holdout_seasons:
                return True
            if development_seasons is not None and s_int not in development_seasons:
                return True
        except (ValueError, TypeError):
            pass

    return False


def compute_bias_statistics(
    diffs: Sequence[float],
    district_id: Optional[Union[str, int]] = None,
    lead_day: Optional[int] = None,
    phase: Optional[str] = None,
    lps_near: Optional[bool] = None,
    season_excluded: Optional[int] = None,
    min_dates_threshold: int = MIN_DATES_THRESHOLD,
) -> BiasRecord:
    """Computes median, q25, q75, and n_dates from differences (observed - raw).
    
    If diffs is empty:
        n_dates = 0, statistics = None, few_past_cases = True.
    """
    valid_diffs = [d for d in diffs if d is not None and not math.isnan(d)]
    n_dates = len(valid_diffs)

    if n_dates == 0:
        return BiasRecord(
            district_id=district_id,
            lead_day=lead_day,
            phase=phase,
            lps_near=lps_near,
            n_dates=0,
            median_diff_mm=None,
            q25_diff_mm=None,
            q75_diff_mm=None,
            season_excluded=season_excluded,
            few_past_cases=True,
            note="few past cases",
        )

    median_val = round(float(np.median(valid_diffs)), 4)
    q25_val = round(float(np.percentile(valid_diffs, 25)), 4)
    q75_val = round(float(np.percentile(valid_diffs, 75)), 4)
    few_cases = bool(n_dates < min_dates_threshold)

    return BiasRecord(
        district_id=district_id,
        lead_day=lead_day,
        phase=phase,
        lps_near=lps_near,
        n_dates=n_dates,
        median_diff_mm=median_val,
        q25_diff_mm=q25_val,
        q75_diff_mm=q75_val,
        season_excluded=season_excluded,
        few_past_cases=few_cases,
        note="few past cases" if few_cases else None,
    )


class BiasTable:
    """Pre-indexed historical bias table for querying regime differences."""

    def __init__(
        self,
        history_rows: Sequence[Any],
        development_seasons: Optional[Sequence[int]] = None,
        holdout_seasons: Optional[Sequence[int]] = None,
        min_dates_threshold: int = MIN_DATES_THRESHOLD,
    ):
        self.development_seasons = set(development_seasons) if development_seasons is not None else None
        self.holdout_seasons = set(holdout_seasons) if holdout_seasons is not None else None
        self.min_dates_threshold = min_dates_threshold

        # Store indexed entries: key -> list of (season, diff)
        # key = (norm_district_id, lead_day, norm_phase, norm_lps_near)
        self._store: Dict[Tuple[str, int, str, bool], List[Tuple[Optional[int], float]]] = defaultdict(list)

        for row in history_rows:
            # Rule: holdout rows NEVER enter the bias history table
            if _is_holdout_row(row, holdout_seasons=self.holdout_seasons, development_seasons=self.development_seasons):
                continue

            dist_raw = _extract_field(row, ["district_id", "id", "district"])
            lead_raw = _extract_field(row, ["lead_day", "lead"])
            phase_raw = _extract_field(row, ["phase", "monsoon_phase", "regime_phase"])
            lps_raw = _extract_field(row, ["lps_near", "has_lps_near", "lps_present"])

            if dist_raw is None or lead_raw is None or phase_raw is None or lps_raw is None:
                continue

            obs = _to_float(_extract_field(row, ["observed_mean_mm", "obs_mean_mm", "observed_mm", "observed", "obs"]))
            raw = _to_float(_extract_field(row, ["raw_mean_mm", "raw_mm", "raw_mean", "raw"]))

            # Missing values excluded
            if obs is None or raw is None:
                continue

            season_raw = _extract_field(row, ["season", "year"])
            season_val = int(season_raw) if season_raw is not None else None

            norm_dist = str(dist_raw).strip()
            norm_lead = int(lead_raw)
            norm_phase = str(phase_raw).strip().lower()
            norm_lps = _to_bool(lps_raw)

            diff = obs - raw
            key = (norm_dist, norm_lead, norm_phase, norm_lps)
            self._store[key].append((season_val, diff))

    def get_entry(
        self,
        district_id: Union[str, int],
        lead_day: int,
        phase: str,
        lps_near: bool,
        query_season: Optional[int] = None,
        evaluation_set: str = "development",
    ) -> BiasRecord:
        """Retrieves bias statistics for a specific query."""
        norm_dist = str(district_id).strip()
        norm_lead = int(lead_day)
        norm_phase = str(phase).strip().lower()
        norm_lps = _to_bool(lps_near)

        key = (norm_dist, norm_lead, norm_phase, norm_lps)
        entries = self._store.get(key, [])

        is_holdout = str(evaluation_set).strip().lower() in HOLDOUT_SET_NAMES

        if is_holdout:
            # HOLDOUT query:
            # - uses all development seasons
            # - season_excluded = null
            target_season_excluded = None
            diffs = [diff for (_, diff) in entries]
        else:
            # DEVELOPMENT query:
            # - uses development seasons only
            # - EXCLUDE the query's own season
            # - season_excluded = query season
            target_season_excluded = int(query_season) if query_season is not None else None
            if target_season_excluded is not None:
                diffs = [diff for (s, diff) in entries if s != target_season_excluded]
            else:
                diffs = [diff for (_, diff) in entries]

        return compute_bias_statistics(
            diffs,
            district_id=district_id,
            lead_day=lead_day,
            phase=phase,
            lps_near=lps_near,
            season_excluded=target_season_excluded,
            min_dates_threshold=self.min_dates_threshold,
        )

    def build_all(
        self,
        query_season: Optional[int] = None,
        evaluation_set: str = "development",
    ) -> Dict[Tuple[str, int, str, bool], BiasRecord]:
        """Builds lookup table of BiasRecord for all available groups."""
        out = {}
        for key in self._store:
            dist, lead, phase, lps = key
            out[key] = self.get_entry(
                district_id=dist,
                lead_day=lead,
                phase=phase,
                lps_near=lps,
                query_season=query_season,
                evaluation_set=evaluation_set,
            )
        return out


def compute_bias_table_entry(
    history_rows: Sequence[Any],
    district_id: Union[str, int],
    lead_day: int,
    phase: str,
    lps_near: bool,
    query_season: Optional[int] = None,
    evaluation_set: str = "development",
    development_seasons: Optional[Sequence[int]] = None,
    holdout_seasons: Optional[Sequence[int]] = None,
    min_dates_threshold: int = MIN_DATES_THRESHOLD,
) -> BiasRecord:
    """Computes a single bias table entry for a district and regime from history rows."""
    table = BiasTable(
        history_rows,
        development_seasons=development_seasons,
        holdout_seasons=holdout_seasons,
        min_dates_threshold=min_dates_threshold,
    )
    return table.get_entry(
        district_id=district_id,
        lead_day=lead_day,
        phase=phase,
        lps_near=lps_near,
        query_season=query_season,
        evaluation_set=evaluation_set,
    )


# Alias for convenience
compute_district_bias = compute_bias_table_entry
get_bias_table_entry = compute_bias_table_entry


def compute_bias_table(
    history_rows: Sequence[Any],
    query_season: Optional[int] = None,
    evaluation_set: str = "development",
    development_seasons: Optional[Sequence[int]] = None,
    holdout_seasons: Optional[Sequence[int]] = None,
    min_dates_threshold: int = MIN_DATES_THRESHOLD,
) -> Dict[Tuple[str, int, str, bool], BiasRecord]:
    """Computes complete bias table dictionary for all groups matching query specifications."""
    table = BiasTable(
        history_rows,
        development_seasons=development_seasons,
        holdout_seasons=holdout_seasons,
        min_dates_threshold=min_dates_threshold,
    )
    return table.build_all(query_season=query_season, evaluation_set=evaluation_set)
