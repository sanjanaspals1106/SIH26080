"""Holdout execution runner with locking mechanism (PRD §10.5).

Ensures holdout evaluation is executed under locked conditions.
Pure function run_holdout is exposed for testing and caller integration without ML execution.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Union

from protocol.locking import (
    HoldoutLockedError,
    LockMetadata,
    MalformedLockError,
    acquire_holdout_lock,
)


def run_holdout(
    config_path: Optional[Union[str, Path]] = None,
    lock_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    git_commit: Optional[str] = None,
    allow_uncommitted: bool = False,
) -> LockMetadata:
    """Executes the holdout protocol lock gate (PRD §10.5, Gate G5).

    Note: Model evaluation and metrics generation are invoked after this gate passes.
    Currently manages the lock lifecycle.

    Args:
        config_path: Path to protocol.yaml.
        lock_path: Path to holdout.lock.
        force: Allow forced re-run if True.
        git_commit: Optional explicit git commit hash.
        allow_uncommitted: Allow UNCOMMITTED_DEV commit for test/dev.

    Returns:
        LockMetadata of the acquired lock.
    """
    return acquire_holdout_lock(
        lock_path=lock_path,
        config_path=config_path,
        force=force,
        git_commit=git_commit,
        allow_uncommitted=allow_uncommitted,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for protocol/run_holdout.py."""
    parser = argparse.ArgumentParser(
        description="PRD §10.5 Holdout evaluation runner with locking."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running holdout evaluation even if holdout.lock exists.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to protocol configuration YAML (default: config/protocol.yaml).",
    )
    parser.add_argument(
        "--lock-file",
        type=str,
        default=None,
        help="Path to holdout lock file (default: holdout.lock).",
    )
    parser.add_argument(
        "--git-commit",
        type=str,
        default=None,
        help="Explicit git commit hash (optional override).",
    )
    parser.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="Permit holdout lock execution on uncommitted code (testing only).",
    )

    args = parser.parse_args(argv)

    try:
        lock_meta = run_holdout(
            config_path=args.config,
            lock_path=args.lock_file,
            force=args.force,
            git_commit=args.git_commit,
            allow_uncommitted=args.allow_uncommitted,
        )
        if lock_meta.forced_rerun:
            print(f"[WARNING] Holdout re-run forced! (run count: {lock_meta.run_count})")
        print(
            f"[OK] Holdout lock successfully acquired.\n"
            f"  Commit:      {lock_meta.git_commit}\n"
            f"  Config hash: {lock_meta.config_hash}\n"
            f"  Timestamp:   {lock_meta.timestamp}\n"
            f"  Forced:      {lock_meta.forced_rerun}"
        )
        return 0
    except HoldoutLockedError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except MalformedLockError as e:
        print(f"[CRITICAL] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
