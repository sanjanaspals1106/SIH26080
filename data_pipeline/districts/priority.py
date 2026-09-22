"""District priority and attention level computation (PRD §15 F5).

Rules:
1. Fallback product => UNAVAILABLE (overrides all other criteria).
2. When 64.5 mm model is available:
   - HIGH: heavy_prob_max_cell >= 0.50 OR very_heavy_prob_max_cell >= 0.20
   - WATCH: (not HIGH) AND (heavy_prob_max_cell >= 0.20 OR corrected_mean_mm >= 15.6)
   - NORMAL: otherwise
3. When 64.5 mm model is unavailable:
   - Use highest cell P(>=15.6 mm):
     HIGH: P(>=15.6) >= 0.80
     WATCH: P(>=15.6) >= 0.50
     NORMAL: otherwise
4. Sorting order:
   - level order: HIGH, WATCH, NORMAL, UNAVAILABLE
   - heavy_prob_max_cell descending
   - heavy_area_fraction_expected descending
   - wettest_cell_mean_mm descending
   - district name ascending
5. Priority ranks start at 1.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import yaml

LEVEL_HIGH = "HIGH"
LEVEL_WATCH = "WATCH"
LEVEL_NORMAL = "NORMAL"
LEVEL_UNAVAILABLE = "UNAVAILABLE"

LEVEL_ORDER = {
    LEVEL_HIGH: 0,
    LEVEL_WATCH: 1,
    LEVEL_NORMAL: 2,
    LEVEL_UNAVAILABLE: 3,
}


@dataclass
class DistrictForecast:
    """Represents forecast summary for a single district."""
    district_id: str
    district_name: str
    state: str
    corrected_mean_mm: float
    wettest_cell_mean_mm: float
    heavy_prob_max_cell: Optional[float] = None
    very_heavy_prob_max_cell: Optional[float] = None
    moderate_prob_max_cell: Optional[float] = None
    heavy_area_fraction_expected: Optional[float] = None
    raw_mean_mm: Optional[float] = None
    is_small: bool = False
    fallback_used: bool = False
    attention_level: Optional[str] = None
    priority_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistrictForecast":
        return cls(
            district_id=str(d.get("district_id", "")),
            district_name=str(d.get("district_name", d.get("name", ""))),
            state=str(d.get("state", "")),
            corrected_mean_mm=float(d.get("corrected_mean_mm", 0.0)),
            wettest_cell_mean_mm=float(d.get("wettest_cell_mean_mm", 0.0)),
            heavy_prob_max_cell=d.get("heavy_prob_max_cell"),
            very_heavy_prob_max_cell=d.get("very_heavy_prob_max_cell"),
            moderate_prob_max_cell=d.get("moderate_prob_max_cell"),
            heavy_area_fraction_expected=d.get("heavy_area_fraction_expected"),
            raw_mean_mm=d.get("raw_mean_mm"),
            is_small=bool(d.get("is_small", False)),
            fallback_used=bool(d.get("fallback_used", False)),
            attention_level=d.get("attention_level"),
            priority_rank=d.get("priority_rank"),
        )


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_priority_thresholds(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads attention level thresholds from config/thresholds.yaml."""
    path = (
        Path(config_path)
        if config_path
        else _find_repo_root() / "config" / "thresholds.yaml"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Thresholds config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    att = data.get("attention_levels", {})
    return att


def determine_attention_level(
    district: Union[DistrictForecast, Dict[str, Any]],
    is_64_5_available: bool = True,
    thresholds_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Determines attention level (HIGH, WATCH, NORMAL, UNAVAILABLE) for a district.

    Args:
        district: DistrictForecast instance or dict.
        is_64_5_available: Whether the 64.5 mm heavy rain model was built and available.
        thresholds_config: Loaded attention_levels dict from thresholds.yaml.

    Returns:
        One of 'HIGH', 'WATCH', 'NORMAL', 'UNAVAILABLE'.
    """
    if isinstance(district, dict):
        d = DistrictForecast.from_dict(district)
    else:
        d = district

    # Rule: fallback district is ALWAYS UNAVAILABLE (PRD §15 F5)
    if d.fallback_used:
        return LEVEL_UNAVAILABLE

    cfg = thresholds_config if thresholds_config else load_priority_thresholds()
    std_cfg = cfg.get("standard", {})
    fb_cfg = cfg.get("fallback_15_6", {})

    if is_64_5_available:
        # Thresholds loaded from config/thresholds.yaml
        high_heavy = float(std_cfg.get("high", {}).get("heavy_prob_max_cell", 0.50))
        high_very_heavy = float(std_cfg.get("high", {}).get("very_heavy_prob_max_cell", 0.20))
        watch_heavy = float(std_cfg.get("watch", {}).get("heavy_prob_max_cell", 0.20))
        watch_mean = float(std_cfg.get("watch", {}).get("corrected_mean_mm", 15.6))

        p_heavy = d.heavy_prob_max_cell
        p_very_heavy = d.very_heavy_prob_max_cell
        mean_mm = d.corrected_mean_mm

        # Check HIGH
        if (p_heavy is not None and p_heavy >= high_heavy) or (
            p_very_heavy is not None and p_very_heavy >= high_very_heavy
        ):
            return LEVEL_HIGH

        # Check WATCH
        if (p_heavy is not None and p_heavy >= watch_heavy) or (
            mean_mm is not None and mean_mm >= watch_mean
        ):
            return LEVEL_WATCH

        return LEVEL_NORMAL
    else:
        # Fallback rule using P(>=15.6 mm)
        fb_high_p15 = float(fb_cfg.get("high", {}).get("moderate_prob_max_cell", 0.80))
        fb_watch_p15 = float(fb_cfg.get("watch", {}).get("moderate_prob_max_cell", 0.50))

        p15 = d.moderate_prob_max_cell

        if p15 is not None and p15 >= fb_high_p15:
            return LEVEL_HIGH
        elif p15 is not None and p15 >= fb_watch_p15:
            return LEVEL_WATCH
        else:
            return LEVEL_NORMAL


def assign_district_priorities(
    districts: Sequence[Union[DistrictForecast, Dict[str, Any]]],
    is_64_5_available: bool = True,
    thresholds_config: Optional[Dict[str, Any]] = None,
) -> List[DistrictForecast]:
    """Computes attention level, sorts according to PRD §15 F5, and assigns priority ranks.

    Sorting:
    1. Level order: HIGH, WATCH, NORMAL, UNAVAILABLE
    2. heavy_prob_max_cell descending
    3. heavy_area_fraction_expected descending
    4. wettest_cell_mean_mm descending
    5. district name ascending

    Args:
        districts: Sequence of DistrictForecast instances or dicts.
        is_64_5_available: Whether 64.5 mm heavy rain model is available.
        thresholds_config: Optional config dict.

    Returns:
        List of DistrictForecast objects sorted with priority_rank starting at 1.
    """
    cfg = thresholds_config if thresholds_config else load_priority_thresholds()

    processed: List[DistrictForecast] = []
    for item in districts:
        df = DistrictForecast.from_dict(item) if isinstance(item, dict) else item
        # Determine attention level
        level = determine_attention_level(df, is_64_5_available=is_64_5_available, thresholds_config=cfg)
        df.attention_level = level
        processed.append(df)

    def sort_key(d: DistrictForecast) -> Tuple[int, float, float, float, str]:
        level_idx = LEVEL_ORDER.get(d.attention_level, 99)
        # Descending keys use negative floats; None is treated as -inf (sorted after valid numbers)
        heavy_prob = d.heavy_prob_max_cell if d.heavy_prob_max_cell is not None else float("-inf")
        area_frac = d.heavy_area_fraction_expected if d.heavy_area_fraction_expected is not None else float("-inf")
        wettest_mean = d.wettest_cell_mean_mm if d.wettest_cell_mean_mm is not None else float("-inf")
        name = d.district_name.strip().lower()

        return (
            level_idx,
            -heavy_prob,
            -area_frac,
            -wettest_mean,
            name,
        )

    processed.sort(key=sort_key)

    # Assign ranks starting at 1
    for rank, df in enumerate(processed, start=1):
        df.priority_rank = rank

    return processed
