"""Unit tests for holdout locking mechanism (PRD §10.5)."""

import json
from pathlib import Path
import pytest
import yaml

from protocol.locking import (
    HoldoutLockedError,
    LockMetadata,
    MalformedLockError,
    acquire_holdout_lock,
    compute_config_hash,
    read_lock_file,
)
from protocol.run_holdout import main, run_holdout


@pytest.fixture
def temp_protocol_config(tmp_path):
    """Creates a temporary valid protocol.yaml for isolated testing."""
    cfg_path = tmp_path / "protocol.yaml"
    data = {
        "random_seed": 42,
        "split_rules": {"min_seasons_required": 3},
        "primary_metrics": {"P1": "RMSE"},
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return cfg_path


def test_first_lock_creation_succeeds(tmp_path, temp_protocol_config):
    """Test 1: First holdout invocation creates holdout.lock with required fields."""
    lock_file = tmp_path / "holdout.lock"
    assert not lock_file.exists()

    meta = run_holdout(
        config_path=temp_protocol_config,
        lock_path=lock_file,
        force=False,
        git_commit="test_commit_sha_123456",
    )

    assert lock_file.is_file()
    assert meta.git_commit == "test_commit_sha_123456"
    assert len(meta.config_hash) == 64  # SHA-256 length
    assert meta.timestamp != ""
    assert meta.forced_rerun is False
    assert meta.run_count == 1

    # Verify written file structure
    data = json.loads(lock_file.read_text(encoding="utf-8"))
    assert data["git_commit"] == "test_commit_sha_123456"
    assert data["config_hash"] == meta.config_hash
    assert data["timestamp"] == meta.timestamp
    assert data["forced_rerun"] is False
    assert data["run_count"] == 1


def test_second_run_refuses(tmp_path, temp_protocol_config):
    """Test 2: Second invocation without --force refuses and raises HoldoutLockedError."""
    lock_file = tmp_path / "holdout.lock"

    # First run succeeds
    run_holdout(
        config_path=temp_protocol_config,
        lock_path=lock_file,
        force=False,
        git_commit="commit_1",
    )
    assert lock_file.exists()

    # Second run without force must raise HoldoutLockedError
    with pytest.raises(HoldoutLockedError, match="holdout is already locked"):
        run_holdout(
            config_path=temp_protocol_config,
            lock_path=lock_file,
            force=False,
            git_commit="commit_2",
        )

    # CLI also returns error code 1
    cli_code = main(["--config", str(temp_protocol_config), "--lock-file", str(lock_file)])
    assert cli_code == 1


def test_force_permits_rerun(tmp_path, temp_protocol_config):
    """Test 3: --force permits rerun and logs forced_rerun=True."""
    lock_file = tmp_path / "holdout.lock"

    # Initial run
    meta_initial = run_holdout(
        config_path=temp_protocol_config,
        lock_path=lock_file,
        force=False,
        git_commit="commit_initial",
    )
    assert meta_initial.forced_rerun is False
    assert meta_initial.run_count == 1

    # Forced re-run
    meta_forced = run_holdout(
        config_path=temp_protocol_config,
        lock_path=lock_file,
        force=True,
        git_commit="commit_forced",
    )
    assert meta_forced.forced_rerun is True
    assert meta_forced.run_count == 2
    assert meta_forced.git_commit == "commit_forced"

    # Verify updated lock on disk
    disk_data = json.loads(lock_file.read_text(encoding="utf-8"))
    assert disk_data["forced_rerun"] is True
    assert disk_data["run_count"] == 2

    # CLI with --force succeeds
    cli_code = main([
        "--config", str(temp_protocol_config),
        "--lock-file", str(lock_file),
        "--force",
        "--git-commit", "commit_cli",
    ])
    assert cli_code == 0


def test_same_config_same_hash(tmp_path):
    """Test 4: Same configuration content produces identical SHA-256 hash."""
    cfg1 = {"random_seed": 42, "metrics": ["P1", "P2"], "a": 1}
    cfg2 = {"a": 1, "random_seed": 42, "metrics": ["P1", "P2"]}  # different key order

    hash1 = compute_config_hash(cfg1)
    hash2 = compute_config_hash(cfg2)
    assert hash1 == hash2

    # Also test file-based hashing
    file1 = tmp_path / "cfg1.yaml"
    file2 = tmp_path / "cfg2.yaml"
    with open(file1, "w") as f:
        yaml.safe_dump(cfg1, f)
    with open(file2, "w") as f:
        yaml.safe_dump(cfg2, f)

    assert compute_config_hash(file1) == compute_config_hash(file2)


def test_changed_config_changed_hash(tmp_path):
    """Test 5: Changed configuration produces different hash."""
    cfg1 = {"random_seed": 42, "threshold": 64.5}
    cfg2 = {"random_seed": 43, "threshold": 64.5}  # modified seed

    hash1 = compute_config_hash(cfg1)
    hash2 = compute_config_hash(cfg2)
    assert hash1 != hash2


def test_malformed_lock_handled_clearly(tmp_path, temp_protocol_config):
    """Test 6: Corrupted, empty, or missing-field lock files fail clearly with MalformedLockError."""
    lock_file = tmp_path / "holdout.lock"

    # Case 1: Empty file
    lock_file.write_text("", encoding="utf-8")
    with pytest.raises(MalformedLockError, match="empty"):
        run_holdout(config_path=temp_protocol_config, lock_path=lock_file, force=False)

    # Case 2: Invalid JSON
    lock_file.write_text("NOT_JSON{[[", encoding="utf-8")
    with pytest.raises(MalformedLockError, match="invalid JSON"):
        run_holdout(config_path=temp_protocol_config, lock_path=lock_file, force=False)

    # Case 3: Missing required fields
    lock_file.write_text(json.dumps({"git_commit": "abc"}), encoding="utf-8")
    with pytest.raises(MalformedLockError, match="missing required fields"):
        run_holdout(config_path=temp_protocol_config, lock_path=lock_file, force=False)

    # Case 4: Non-string empty fields
    lock_file.write_text(
        json.dumps({"git_commit": "", "config_hash": "abc", "timestamp": "2026-09-21"}),
        encoding="utf-8",
    )
    with pytest.raises(MalformedLockError, match="non-empty string"):
        run_holdout(config_path=temp_protocol_config, lock_path=lock_file, force=False)

    # CLI exit code on malformed lock is 2
    cli_code = main(["--config", str(temp_protocol_config), "--lock-file", str(lock_file)])
    assert cli_code == 2


def test_uncommitted_dev_rejected_for_real_holdout(tmp_path, temp_protocol_config):
    """Test 7: 'UNCOMMITTED_DEV' git commit is rejected for holdout locking per PRD §10.5 and Gate G5."""
    lock_file = tmp_path / "holdout.lock"

    # Reject uncommitted code by default
    with pytest.raises(ValueError, match="uncommitted development code"):
        run_holdout(
            config_path=temp_protocol_config,
            lock_path=lock_file,
            git_commit="UNCOMMITTED_DEV",
            allow_uncommitted=False,
        )

    # CLI also returns error code 3 on uncommitted code
    cli_code = main([
        "--config", str(temp_protocol_config),
        "--lock-file", str(lock_file),
        "--git-commit", "UNCOMMITTED_DEV",
    ])
    assert cli_code == 3

    # Explicit allow_uncommitted permits it (for testing only)
    meta = run_holdout(
        config_path=temp_protocol_config,
        lock_path=lock_file,
        git_commit="UNCOMMITTED_DEV",
        allow_uncommitted=True,
    )
    assert meta.git_commit == "UNCOMMITTED_DEV"

