"""Event counting and model availability decision logic (PRD §10.7, §13.2).

Definition of an event (PRD §10.7):
1. For one date and threshold, find 8-connected groups of observed cells >= threshold.
2. Groups on consecutive dates are merged into one event if they share at least one cell.
3. Never merge across a missing date/gap.
4. Missing and invalid cells are excluded.
5. Seasons and lead days are evaluated independently (no cross-lead or cross-season merging).

Model availability decisions (PRD §13.2):
- Threshold >= 30 events in dev seasons -> Standalone classifier.
- 115.6 mm < 30 events but 64.5 mm >= 30 events -> Chained form.
- 64.5 mm < 30 events -> No heavy rain model; only 15.6 mm available; hotspots use 15.6 mm.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from scipy.ndimage import label
import yaml


@dataclass
class DailyObservation:
    """Represents a 2D observed rainfall grid for a single date, season, and lead."""
    season: int
    lead_day: int
    date: Union[date, datetime, str]
    observed_rain: np.ndarray
    valid_mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class EventCountRecord:
    """Event and cell-day counts for a (season, lead, threshold) combination."""
    season: int
    lead_day: int
    threshold_mm: float
    n_events: int
    n_cell_days: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "season": self.season,
            "lead_day": self.lead_day,
            "threshold_mm": self.threshold_mm,
            "n_events": self.n_events,
            "n_cell_days": self.n_cell_days,
        }


@dataclass(frozen=True)
class ModelAvailabilityDecision:
    """Model availability decision derived strictly from development-season event totals."""
    p_15_6_available: bool
    p_64_5_available: bool
    p_115_6_available: bool
    p_15_6_mode: str  # "STANDALONE" | "UNAVAILABLE"
    p_64_5_mode: str  # "STANDALONE" | "UNAVAILABLE"
    p_115_6_mode: str  # "STANDALONE" | "CHAINED" | "UNAVAILABLE"
    hotspot_threshold_mm: float  # 64.5 or 15.6
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_15_6_available": self.p_15_6_available,
            "p_64_5_available": self.p_64_5_available,
            "p_115_6_available": self.p_115_6_available,
            "p_15_6_mode": self.p_15_6_mode,
            "p_64_5_mode": self.p_64_5_mode,
            "p_115_6_mode": self.p_115_6_mode,
            "hotspot_threshold_mm": self.hotspot_threshold_mm,
            "message": self.message,
        }


class _UnionFind:
    """Disjoint Set Union (Union-Find) structure with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def count_components(self) -> int:
        if not self.parent:
            return 0
        return len({self.find(i) for i in range(len(self.parent))})


def _parse_date(d: Union[date, datetime, str]) -> date:
    """Converts diverse date inputs to datetime.date."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d.strip()[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        return d.date()
    raise TypeError(f"Unsupported date type: {type(d)} ({d!r})")


def load_rain_thresholds(config_path: Optional[Union[str, Path]] = None) -> List[float]:
    """Loads rain thresholds from config/thresholds.yaml."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"

    if not path.is_file():
        raise FileNotFoundError(f"Threshold configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    thresholds_dict = data.get("rain_thresholds_mm", {})
    thresholds = [
        float(thresholds_dict.get("moderate", 15.6)),
        float(thresholds_dict.get("heavy", 64.5)),
        float(thresholds_dict.get("very_heavy", 115.6)),
    ]
    return sorted(thresholds)


def count_events_in_grid_series(
    observations: Sequence[DailyObservation],
    thresholds: Optional[Sequence[float]] = None,
) -> List[EventCountRecord]:
    """Counts spatio-temporal events and cell-days across a series of observations.

    Rules (PRD §10.7):
    - Separate lead days (never merge across leads).
    - Separate seasons (never merge across season boundaries).
    - Within a (season, lead), sort dates ascending.
    - 8-connected components on each date.
    - Consecutive dates (|d2 - d1| == 1 day) sharing >= 1 cell are merged.
    - Gaps (|d2 - d1| > 1 day) break event continuity.

    Args:
        observations: Sequence of DailyObservation objects.
        thresholds: List of rain thresholds (mm). Defaults to thresholds from config.

    Returns:
        List of EventCountRecord objects for each (season, lead, threshold).
    """
    if thresholds is None:
        target_thresholds = load_rain_thresholds()
    else:
        target_thresholds = sorted(list(thresholds))

    # Structure 8-connectivity (3x3 kernel of ones)
    structure_8 = np.ones((3, 3), dtype=int)

    # Group observations by (season, lead_day)
    grouped: Dict[Tuple[int, int], List[DailyObservation]] = defaultdict(list)
    for obs in observations:
        grouped[(obs.season, obs.lead_day)].append(obs)

    records: List[EventCountRecord] = []

    # Process each (season, lead_day) partition in isolation
    for (season, lead), obs_list in sorted(grouped.items()):
        # Sort chronologically by date
        sorted_obs = sorted(obs_list, key=lambda x: _parse_date(x.date))

        for threshold in target_thresholds:
            # First pass: find 8-connected components for each date
            daily_groups: List[Tuple[date, Set[Tuple[int, int]]]] = []
            total_cell_days = 0

            for obs in sorted_obs:
                cur_date = _parse_date(obs.date)
                grid = np.asarray(obs.observed_rain, dtype=float)

                if obs.valid_mask is None:
                    valid = np.ones(grid.shape, dtype=bool)
                else:
                    valid = np.asarray(obs.valid_mask, dtype=bool)

                # Effective mask: valid cell and non-NaN
                effective_valid = valid & (~np.isnan(grid))
                binary_event = effective_valid & (grid >= threshold)

                cell_days_today = int(np.count_nonzero(binary_event))
                total_cell_days += cell_days_today

                if cell_days_today == 0:
                    continue

                labeled_array, num_features = label(binary_event, structure=structure_8)

                for feat_id in range(1, num_features + 1):
                    coords = np.argwhere(labeled_array == feat_id)
                    coord_set = {tuple(c) for c in coords}
                    daily_groups.append((cur_date, coord_set))

            num_groups = len(daily_groups)
            if num_groups == 0:
                records.append(
                    EventCountRecord(
                        season=season,
                        lead_day=lead,
                        threshold_mm=threshold,
                        n_events=0,
                        n_cell_days=0,
                    )
                )
                continue

            # Second pass: Union groups on consecutive dates that share >= 1 cell
            uf = _UnionFind(num_groups)

            # Map date to group indices
            date_to_indices: Dict[date, List[int]] = defaultdict(list)
            for idx, (grp_date, _) in enumerate(daily_groups):
                date_to_indices[grp_date].append(idx)

            unique_dates = sorted(date_to_indices.keys())
            for i in range(len(unique_dates) - 1):
                d_prev = unique_dates[i]
                d_curr = unique_dates[i + 1]

                # Consecutive date check: must be exactly 1 day apart
                if (d_curr - d_prev).days == 1:
                    prev_indices = date_to_indices[d_prev]
                    curr_indices = date_to_indices[d_curr]

                    for p_idx in prev_indices:
                        p_cells = daily_groups[p_idx][1]
                        for c_idx in curr_indices:
                            c_cells = daily_groups[c_idx][1]
                            if not p_cells.isdisjoint(c_cells):
                                uf.union(p_idx, c_idx)
                # If days > 1, do NOT merge across the missing gap!

            n_events = uf.count_components()

            records.append(
                EventCountRecord(
                    season=season,
                    lead_day=lead,
                    threshold_mm=threshold,
                    n_events=n_events,
                    n_cell_days=total_cell_days,
                )
            )

    return records


def determine_model_availability(
    dev_event_totals: Dict[float, int],
    min_events: int = 30,
) -> ModelAvailabilityDecision:
    """Evaluates probability model availability per PRD §13.2 from development season totals.

    Args:
        dev_event_totals: Mapping from threshold in mm (15.6, 64.5, 115.6) to total event counts
                          aggregated over all development seasons.
        min_events: Minimum event threshold (default 30 from protocol.yaml / §13.2).

    Returns:
        ModelAvailabilityDecision instance.
    """
    events_15_6 = dev_event_totals.get(15.6, 0)
    events_64_5 = dev_event_totals.get(64.5, 0)
    events_115_6 = dev_event_totals.get(115.6, 0)

    p_15_6_available = bool(events_15_6 >= min_events)
    p_15_6_mode = "STANDALONE" if p_15_6_available else "UNAVAILABLE"

    # Case 1: 64.5 mm has fewer than min_events (PRD §13.2 row 3)
    if events_64_5 < min_events:
        return ModelAvailabilityDecision(
            p_15_6_available=p_15_6_available,
            p_64_5_available=False,
            p_115_6_available=False,
            p_15_6_mode=p_15_6_mode,
            p_64_5_mode="UNAVAILABLE",
            p_115_6_mode="UNAVAILABLE",
            hotspot_threshold_mm=15.6,
            message=(
                f"Not enough events for heavy-rain model ({events_64_5} < {min_events} at 64.5 mm). "
                f"Heavy and very-heavy models unavailable. Only P(>=15.6) is available. "
                f"Hotspots fallback to 15.6 mm."
            ),
        )

    # Case 2: 64.5 mm has >= min_events, but 115.6 mm has < min_events (PRD §13.2 row 2)
    if events_115_6 < min_events:
        return ModelAvailabilityDecision(
            p_15_6_available=p_15_6_available,
            p_64_5_available=True,
            p_115_6_available=True,
            p_15_6_mode=p_15_6_mode,
            p_64_5_mode="STANDALONE",
            p_115_6_mode="CHAINED",
            hotspot_threshold_mm=64.5,
            message=(
                f"Chained form for very-heavy model ({events_115_6} < {min_events} at 115.6 mm, "
                f"{events_64_5} >= {min_events} at 64.5 mm). "
                f"P(>=115.6) = P(>=64.5) * P(>=115.6 | >=64.5)."
            ),
        )

    # Case 3: Both 64.5 mm and 115.6 mm have >= min_events (PRD §13.2 row 1)
    return ModelAvailabilityDecision(
        p_15_6_available=p_15_6_available,
        p_64_5_available=True,
        p_115_6_available=True,
        p_15_6_mode=p_15_6_mode,
        p_64_5_mode="STANDALONE",
        p_115_6_mode="STANDALONE",
        hotspot_threshold_mm=64.5,
        message="Standalone classifiers allowed for all probability thresholds (>=30 events each).",
    )


def generate_event_counts_markdown(
    records: Sequence[EventCountRecord],
    dev_seasons: Sequence[int],
    decision: Optional[ModelAvailabilityDecision] = None,
    note: Optional[str] = None,
) -> str:
    """Generates formatted docs/event-counts.md report per PRD §13.2."""
    lines = [
        "# Observed Rainfall Event Counts and Model Availability",
        "",
        "**Document generated per PRD §10.7 and §13.2.**",
        "",
        "An **event** is an 8-connected group of cells at or above the threshold on one date, "
        "merged across consecutive dates sharing >= 1 cell. Gaps break continuity.",
        "",
    ]

    if note:
        lines.append(f"> [!NOTE]\n> {note}\n")

    lines.extend([
        "## 1. Development Season Totals and Decisions",
        "",
        f"**Development Seasons:** {sorted(list(dev_seasons))}",
        "",
    ])

    # Filter records for development seasons
    dev_records = [r for r in records if r.season in dev_seasons]

    # Aggregate by threshold
    totals_by_thresh: Dict[float, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in dev_records:
        ev, cd = totals_by_thresh[r.threshold_mm]
        totals_by_thresh[r.threshold_mm] = (ev + r.n_events, cd + r.n_cell_days)

    lines.extend([
        "| Threshold (mm) | Description | Development Events | Development Cell-Days | Status |",
        "|---|---|---|---|---|",
    ])

    desc_map = {15.6: "Moderate or more", 64.5: "Heavy", 115.6: "Very heavy"}
    for th in sorted(totals_by_thresh.keys()):
        ev, cd = totals_by_thresh[th]
        desc = desc_map.get(th, f">={th} mm")
        status = ">= 30 (Sufficient)" if ev >= 30 else "< 30 (Sparse)"
        lines.append(f"| {th:.1f} | {desc} | {ev} | {cd} | {status} |")

    lines.append("")

    if decision:
        lines.extend([
            "### Model Configuration",
            f"- **P(≥15.6 mm):** {decision.p_15_6_mode}",
            f"- **P(≥64.5 mm):** {decision.p_64_5_mode}",
            f"- **P(≥115.6 mm):** {decision.p_115_6_mode}",
            f"- **Hotspot Threshold:** {decision.hotspot_threshold_mm:.1f} mm",
            f"- **Rule Outcome:** {decision.message}",
            "",
        ])

    lines.extend([
        "## 2. Event Counts by Season and Lead Day",
        "",
        "| Season | Lead Day | Threshold (mm) | Events (n_events) | Cell-Days (n_cell_days) |",
        "|---|---|---|---|---|",
    ])

    for r in sorted(records, key=lambda x: (x.season, x.lead_day, x.threshold_mm)):
        lines.append(
            f"| {r.season} | {r.lead_day} | {r.threshold_mm:.1f} | {r.n_events} | {r.n_cell_days} |"
        )

    lines.append("")
    return "\n".join(lines)
