"""Holdout locking mechanism per PRD §10.5.

Ensures that the holdout experiment is run strictly once against a locked
git commit and deterministic config hash, unless --force is explicitly provided.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Union
import yaml


class HoldoutLockedError(Exception):
    """Raised when attempting to execute holdout while already locked without --force."""
    pass


class MalformedLockError(Exception):
    """Raised when holdout.lock exists but is corrupted, empty, or missing required fields."""
    pass


@dataclass(frozen=True)
class LockMetadata:
    """Metadata recorded in holdout.lock."""
    git_commit: str
    config_hash: str
    timestamp: str
    forced_rerun: bool = False
    run_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LockMetadata":
        required_keys = {"git_commit", "config_hash", "timestamp"}
        missing = required_keys - set(data.keys())
        if missing:
            raise MalformedLockError(f"Lock data missing required fields: {sorted(list(missing))}")
        
        if not isinstance(data["git_commit"], str) or not data["git_commit"].strip():
            raise MalformedLockError("Field 'git_commit' must be a non-empty string.")
        if not isinstance(data["config_hash"], str) or not data["config_hash"].strip():
            raise MalformedLockError("Field 'config_hash' must be a non-empty string.")
        if not isinstance(data["timestamp"], str) or not data["timestamp"].strip():
            raise MalformedLockError("Field 'timestamp' must be a non-empty ISO string.")

        return cls(
            git_commit=str(data["git_commit"]).strip(),
            config_hash=str(data["config_hash"]).strip(),
            timestamp=str(data["timestamp"]).strip(),
            forced_rerun=bool(data.get("forced_rerun", False)),
            run_count=int(data.get("run_count", 1)),
        )


def _find_repo_root() -> Path:
    """Finds the repo root directory relative to this file."""
    return Path(__file__).resolve().parent.parent


def get_git_commit(
    repo_path: Optional[Union[str, Path]] = None,
    fallback: Optional[str] = "UNCOMMITTED_DEV",
) -> str:
    """Retrieves current git commit hash (HEAD).

    Args:
        repo_path: Path to git repository.
        fallback: Optional fallback string if git is not installed or repo is uninitialized.
                  If None and git fails, raises RuntimeError.

    Returns:
        Git commit hash string.
    """
    path = Path(repo_path) if repo_path else _find_repo_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except Exception as e:
        if fallback is not None:
            return fallback
        raise RuntimeError(f"Failed to obtain git commit from {path}: {e}") from e

    if fallback is not None:
        return fallback
    raise RuntimeError(f"git rev-parse HEAD returned empty commit for {path}")


def compute_config_hash(config_input: Union[str, Path, Dict[str, Any]]) -> str:
    """Computes a deterministic SHA-256 hash of protocol configuration.

    Normalizes dictionaries and JSON structures to eliminate formatting
    or cross-platform newline differences.

    Args:
        config_input: Either a file path to protocol.yaml or a dictionary.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if isinstance(config_input, (str, Path)):
        path = Path(config_input)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found for hashing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif isinstance(config_input, dict):
        data = config_input
    else:
        raise TypeError(f"Expected path or dict for config_input, got {type(config_input).__name__}")

    if not isinstance(data, dict):
        raise ValueError("Configuration content must evaluate to a dictionary.")

    # Canonical JSON string serialization for deterministic cross-platform hashing
    canonical_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def read_lock_file(lock_path: Union[str, Path]) -> Optional[LockMetadata]:
    """Reads and validates holdout.lock.

    Returns None if file does not exist.
    Raises MalformedLockError if file is invalid or empty.
    """
    path = Path(lock_path)
    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise MalformedLockError(f"Unable to read lock file {path}: {e}") from e

    if not content:
        raise MalformedLockError(f"Lock file {path} is empty.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise MalformedLockError(f"Lock file {path} contains invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise MalformedLockError(f"Lock file {path} must contain a JSON object.")

    return LockMetadata.from_dict(data)


def acquire_holdout_lock(
    lock_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    git_commit: Optional[str] = None,
    allow_uncommitted: bool = False,
) -> LockMetadata:
    """Manages holdout lock acquisition per PRD §10.5 and Gate G5.

    - If holdout.lock exists and force=False: raises HoldoutLockedError.
    - If holdout.lock exists and force=True: updates lock and logs forced_rerun=True.
    - If holdout.lock does not exist: creates lock with git commit, config hash, and timestamp.
    - Rejects uncommitted development code (commit='UNCOMMITTED_DEV') unless allow_uncommitted=True.

    Args:
        lock_path: Path to holdout.lock. Defaults to repo_root / "holdout.lock".
        config_path: Path to protocol.yaml. Defaults to repo_root / "config" / "protocol.yaml".
        force: If True, permits rerun when lock already exists.
        git_commit: Explicit git commit hash (optional override for testing).
        allow_uncommitted: If True, allows locking with UNCOMMITTED_DEV (test/dev only).

    Returns:
        LockMetadata representing the current lock state.
    """
    repo_root = _find_repo_root()
    target_lock = Path(lock_path) if lock_path else repo_root / "holdout.lock"
    target_config = Path(config_path) if config_path else repo_root / "config" / "protocol.yaml"

    existing_lock = read_lock_file(target_lock)
    run_count = 1

    if existing_lock is not None:
        if not force:
            raise HoldoutLockedError(
                f"Holdout execution refused: holdout is already locked at {existing_lock.timestamp} "
                f"(git commit: {existing_lock.git_commit}, config hash: {existing_lock.config_hash}). "
                f"Per PRD §10.5, final holdout can only be evaluated once. Use --force to override."
            )
        run_count = existing_lock.run_count + 1

    # Deterministic metadata computation
    cfg_hash = compute_config_hash(target_config)
    commit = git_commit if git_commit is not None else get_git_commit(repo_root)

    if commit == "UNCOMMITTED_DEV" and not allow_uncommitted:
        raise ValueError(
            "Holdout evaluation cannot run on uncommitted development code (commit='UNCOMMITTED_DEV'). "
            "Per PRD §10.5 & Gate G5, holdout evaluation must be executed against a clean, committed, "
            "and tagged git commit. Tag and commit your changes, or pass allow_uncommitted=True "
            "only in test environments."
        )

    timestamp = datetime.now(timezone.utc).isoformat()

    lock_meta = LockMetadata(
        git_commit=commit,
        config_hash=cfg_hash,
        timestamp=timestamp,
        forced_rerun=bool(force and existing_lock is not None),
        run_count=run_count,
    )

    # Atomic write to lock file
    temp_file = target_lock.with_suffix(".tmp")
    try:
        temp_file.write_text(lock_meta.to_json(), encoding="utf-8")
        temp_file.replace(target_lock)
    finally:
        if temp_file.exists():
            temp_file.unlink()

    return lock_meta
