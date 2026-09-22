"""Season-based experiment split logic and LOSO fold generator (PRD §10, Appendix B).

Rules:
- Season is the strict unit of splitting (PRD §10.1).
- No random row splits or shuffled cross-validation anywhere (PRD §10.6 L5).
- Deterministic assignments and seeds loaded from configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import yaml


@dataclass(frozen=True)
class SeasonSplit:
    """Holds development and holdout season allocations and status."""
    development_seasons: List[int]
    holdout_seasons: List[int]
    status: str
    n_seasons: int
    random_seed: int


@dataclass(frozen=True)
class LOSOFold:
    """Represents a single Leave-One-Season-Out (LOSO) fold for development."""
    fold_index: int
    val_season: int
    train_seasons: List[int]


def _find_default_protocol_config() -> Path:
    """Locates the default config/protocol.yaml relative to this module."""
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    config_file = repo_root / "config" / "protocol.yaml"
    return config_file


def get_protocol_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads protocol configuration from YAML file."""
    path = Path(config_path) if config_path else _find_default_protocol_config()
    if not path.is_file():
        raise FileNotFoundError(f"Protocol configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML content in {path}: expected a dictionary.")
    return config


def get_protocol_seed(config_path: Optional[str | Path] = None) -> int:
    """Returns the deterministic protocol random seed from protocol.yaml."""
    cfg = get_protocol_config(config_path)
    seed = cfg.get("random_seed")
    if seed is None:
        raise KeyError("Protocol config must contain 'random_seed'.")
    return int(seed)


def assign_season_splits(
    seasons: Sequence[int],
    config: Optional[Dict[str, Any] | str | Path] = None,
) -> SeasonSplit:
    """Assigns available seasons into development and holdout splits per PRD §10.2.

    Args:
        seasons: Sequence of integer monsoon seasons (e.g. [2016, 2017, ...]).
        config: Optional pre-loaded config dict or path to protocol.yaml.

    Returns:
        SeasonSplit containing sorted development and holdout seasons and status.

    Raises:
        ValueError: If input is invalid, contains duplicates, or N <= 2 seasons.
    """
    if not seasons:
        raise ValueError("Cannot assign season splits on an empty sequence of seasons.")

    # Validate elements are integers
    for s in seasons:
        if not isinstance(s, int):
            raise TypeError(f"All seasons must be integers, got: {type(s).__name__} ({s!r})")

    # Check for duplicate seasons
    if len(set(seasons)) != len(seasons):
        duplicates = [s for s in set(seasons) if list(seasons).count(s) > 1]
        raise ValueError(f"Duplicate seasons detected in input: {duplicates}")

    # Sort seasons ascending before assigning (PRD §10.2)
    sorted_seasons = sorted(list(seasons))
    n = len(sorted_seasons)

    # Load configuration
    if isinstance(config, (str, Path)):
        cfg = get_protocol_config(config)
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = get_protocol_config()

    split_rules = cfg.get("split_rules", {})
    min_required = int(split_rules.get("min_seasons_required", 3))
    tiers = split_rules.get("tiers", {})

    if n < min_required:
        raise ValueError(
            f"Insufficient seasons: N={n} ({sorted_seasons}). PRD §10.2 requires at least "
            f"{min_required} seasons. The project cannot be evaluated; more seasons must be downloaded."
        )

    seed = int(cfg.get("random_seed", 42))

    full_tier = tiers.get("full", {})
    full_min = int(full_tier.get("min_seasons", 8))
    full_holdout = int(full_tier.get("n_holdout", 2))
    full_status = str(full_tier.get("status", "FULL_PROTOCOL"))

    reduced_tier = tiers.get("reduced", {})
    reduced_min = int(reduced_tier.get("min_seasons", 5))
    reduced_max = int(reduced_tier.get("max_seasons", 7))
    reduced_holdout = int(reduced_tier.get("n_holdout", 1))
    reduced_status = str(reduced_tier.get("status", "REDUCED_HOLDOUT"))

    dev_only_tier = tiers.get("dev_only", {})
    dev_only_min = int(dev_only_tier.get("min_seasons", 3))
    dev_only_max = int(dev_only_tier.get("max_seasons", 4))
    dev_only_holdout = int(dev_only_tier.get("n_holdout", 0))
    dev_only_status = str(dev_only_tier.get("status", "DEVELOPMENT_ONLY"))

    if n >= full_min:
        dev = sorted_seasons[:-full_holdout]
        holdout = sorted_seasons[-full_holdout:]
        status = full_status
    elif reduced_min <= n <= reduced_max:
        dev = sorted_seasons[:-reduced_holdout]
        holdout = sorted_seasons[-reduced_holdout:]
        status = reduced_status
    elif dev_only_min <= n <= dev_only_max:
        dev = sorted_seasons[:]
        holdout = []
        status = dev_only_status
    else:
        raise ValueError(f"Unable to match N={n} seasons to any protocol tier.")

    return SeasonSplit(
        development_seasons=dev,
        holdout_seasons=holdout,
        status=status,
        n_seasons=n,
        random_seed=seed,
    )


def generate_loso_folds(development_seasons: Sequence[int]) -> List[LOSOFold]:
    """Generates Leave-One-Season-Out (LOSO) folds for development seasons (PRD §10.1, §10.3).

    Args:
        development_seasons: Sequence of development seasons.

    Returns:
        List of LOSOFold objects, one per held-out season.

    Raises:
        ValueError: If fewer than 2 development seasons are provided.
    """
    if len(development_seasons) < 2:
        raise ValueError(
            f"LOSO requires at least 2 development seasons to form train/val folds, "
            f"got {len(development_seasons)}."
        )

    # Validate types and check for duplicates
    for s in development_seasons:
        if not isinstance(s, int):
            raise TypeError(f"All development seasons must be integers, got: {type(s).__name__}")
    if len(set(development_seasons)) != len(development_seasons):
        raise ValueError(f"Duplicate seasons in development seasons: {development_seasons}")

    sorted_dev = sorted(list(development_seasons))
    folds: List[LOSOFold] = []

    for idx, val_season in enumerate(sorted_dev):
        train_seasons = [s for s in sorted_dev if s != val_season]
        folds.append(
            LOSOFold(
                fold_index=idx,
                val_season=val_season,
                train_seasons=train_seasons,
            )
        )

    return folds
