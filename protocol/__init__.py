"""Protocol module for experiment splits, protocol configuration, and verification locks."""

from protocol.locking import (
    HoldoutLockedError,
    LockMetadata,
    MalformedLockError,
    acquire_holdout_lock,
    compute_config_hash,
    get_git_commit,
    read_lock_file,
)
from protocol.run_holdout import run_holdout
from protocol.splits import (
    LOSOFold,
    SeasonSplit,
    assign_season_splits,
    generate_loso_folds,
    get_protocol_config,
    get_protocol_seed,
)

__all__ = [
    "HoldoutLockedError",
    "LockMetadata",
    "MalformedLockError",
    "acquire_holdout_lock",
    "compute_config_hash",
    "get_git_commit",
    "read_lock_file",
    "run_holdout",
    "LOSOFold",
    "SeasonSplit",
    "assign_season_splits",
    "generate_loso_folds",
    "get_protocol_config",
    "get_protocol_seed",
]
