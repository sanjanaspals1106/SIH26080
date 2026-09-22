"""Unit tests for M4 protocol splits and configuration (PRD §10, Appendix B)."""

import pytest
from pathlib import Path
import yaml

from protocol.splits import (
    SeasonSplit,
    assign_season_splits,
    generate_loso_folds,
    get_protocol_config,
    get_protocol_seed,
)


def test_protocol_config_load():
    """Verify config/protocol.yaml loads correctly and has required PRD parameters."""
    cfg = get_protocol_config()
    assert isinstance(cfg, dict)
    assert cfg["random_seed"] == 42

    # Check split rule tiers from PRD §10.2
    tiers = cfg["split_rules"]["tiers"]
    assert tiers["full"]["min_seasons"] == 8
    assert tiers["full"]["n_holdout"] == 2
    assert tiers["full"]["status"] == "FULL_PROTOCOL"

    assert tiers["reduced"]["min_seasons"] == 5
    assert tiers["reduced"]["max_seasons"] == 7
    assert tiers["reduced"]["n_holdout"] == 1
    assert tiers["reduced"]["status"] == "REDUCED_HOLDOUT"

    assert tiers["dev_only"]["min_seasons"] == 3
    assert tiers["dev_only"]["max_seasons"] == 4
    assert tiers["dev_only"]["n_holdout"] == 0
    assert tiers["dev_only"]["status"] == "DEVELOPMENT_ONLY"

    # Check memory and search parameters
    assert cfg["settings_search"]["n_combinations"] == 20
    assert cfg["settings_search"]["n_validation_seasons"] == 3
    assert cfg["event_counting"]["min_events_primary"] == 30
    assert cfg["memory"]["train_cell_stride"] == 1


def test_thresholds_and_verification_configs_exist():
    """Verify thresholds.yaml and verification.yaml exist and match Appendix B."""
    repo_root = Path(__file__).resolve().parent.parent

    thresholds_file = repo_root / "config" / "thresholds.yaml"
    assert thresholds_file.is_file()
    with open(thresholds_file, "r", encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)
    assert t_cfg["rain_thresholds_mm"]["moderate"] == 15.6
    assert t_cfg["rain_thresholds_mm"]["heavy"] == 64.5
    assert t_cfg["rain_thresholds_mm"]["very_heavy"] == 115.6
    assert t_cfg["calibration"]["isotonic_min_events"] == 200
    assert t_cfg["attention_levels"]["standard"]["high"]["heavy_prob_max_cell"] == 0.50

    verification_file = repo_root / "config" / "verification.yaml"
    assert verification_file.is_file()
    with open(verification_file, "r", encoding="utf-8") as f:
        v_cfg = yaml.safe_load(f)
    assert v_cfg["coverage"]["target"] == 0.80
    assert v_cfg["coverage"]["tolerance"] == 0.10
    assert v_cfg["spatial"]["primary_neighbourhood"] == 5
    assert v_cfg["bootstrap"]["primary_block_days"] == 7
    assert v_cfg["bootstrap"]["resamples"] == 2000


def test_protocol_seed():
    """Check get_protocol_seed returns integer seed."""
    seed = get_protocol_seed()
    assert seed == 42


def test_split_n_ge_8():
    """N >= 8: oldest N-2 development, newest 2 holdout (PRD §10.2)."""
    # 10 seasons (target 2016-2025)
    seasons_10 = list(range(2016, 2026))
    split = assign_season_splits(seasons_10)

    assert split.n_seasons == 10
    assert split.development_seasons == list(range(2016, 2024))
    assert split.holdout_seasons == [2024, 2025]
    assert split.status == "FULL_PROTOCOL"
    assert split.random_seed == 42

    # 8 seasons
    seasons_8 = list(range(2018, 2026))
    split_8 = assign_season_splits(seasons_8)
    assert split_8.development_seasons == list(range(2018, 2024))
    assert split_8.holdout_seasons == [2024, 2025]
    assert split_8.status == "FULL_PROTOCOL"


def test_split_5_to_7():
    """5 <= N <= 7: oldest N-1 development, newest 1 holdout (PRD §10.2)."""
    # N = 7
    seasons_7 = list(range(2019, 2026))
    split_7 = assign_season_splits(seasons_7)
    assert split_7.n_seasons == 7
    assert split_7.development_seasons == list(range(2019, 2025))
    assert split_7.holdout_seasons == [2025]
    assert split_7.status == "REDUCED_HOLDOUT"

    # N = 5
    seasons_5 = list(range(2021, 2026))
    split_5 = assign_season_splits(seasons_5)
    assert split_5.n_seasons == 5
    assert split_5.development_seasons == list(range(2021, 2025))
    assert split_5.holdout_seasons == [2025]
    assert split_5.status == "REDUCED_HOLDOUT"


def test_split_3_to_4():
    """3 <= N <= 4: all development, no holdout, status DEVELOPMENT_ONLY (PRD §10.2)."""
    # N = 4
    seasons_4 = [2022, 2023, 2024, 2025]
    split_4 = assign_season_splits(seasons_4)
    assert split_4.n_seasons == 4
    assert split_4.development_seasons == [2022, 2023, 2024, 2025]
    assert split_4.holdout_seasons == []
    assert split_4.status == "DEVELOPMENT_ONLY"

    # N = 3
    seasons_3 = [2023, 2024, 2025]
    split_3 = assign_season_splits(seasons_3)
    assert split_3.n_seasons == 3
    assert split_3.development_seasons == [2023, 2024, 2025]
    assert split_3.holdout_seasons == []
    assert split_3.status == "DEVELOPMENT_ONLY"


def test_split_n_le_2_raises_error():
    """N <= 2: raises ValueError stopping the project (PRD §10.2)."""
    with pytest.raises(ValueError, match="Insufficient seasons: N=2"):
        assign_season_splits([2024, 2025])

    with pytest.raises(ValueError, match="Insufficient seasons: N=1"):
        assign_season_splits([2025])

    with pytest.raises(ValueError, match="empty sequence"):
        assign_season_splits([])


def test_sorting_ascending_before_assignment():
    """Seasons must be sorted ascending before split assignment regardless of input order."""
    unordered = [2025, 2018, 2020, 2016, 2024, 2017, 2019, 2022, 2021, 2023]
    split = assign_season_splits(unordered)

    assert split.development_seasons == [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    assert split.holdout_seasons == [2024, 2025]
    assert split.status == "FULL_PROTOCOL"


def test_duplicate_seasons_rejected():
    """Duplicate seasons in input must raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate seasons"):
        assign_season_splits([2020, 2021, 2022, 2022, 2023])


def test_non_integer_seasons_rejected():
    """Non-integer seasons must raise TypeError."""
    with pytest.raises(TypeError, match="must be integers"):
        assign_season_splits(["2020", "2021", "2022"])


def test_loso_fold_generator():
    """Verify LOSO fold generation logic for development seasons (PRD §10.1, §10.3)."""
    dev_seasons = [2016, 2017, 2018, 2019]
    folds = generate_loso_folds(dev_seasons)

    assert len(folds) == 4

    expected_val = [2016, 2017, 2018, 2019]
    expected_train = [
        [2017, 2018, 2019],
        [2016, 2018, 2019],
        [2016, 2017, 2019],
        [2016, 2017, 2018],
    ]

    for idx, fold in enumerate(folds):
        assert fold.fold_index == idx
        assert fold.val_season == expected_val[idx]
        assert fold.train_seasons == expected_train[idx]
        # Ensure strict separation: held-out season is never in training seasons
        assert fold.val_season not in fold.train_seasons


def test_loso_fold_generator_validation():
    """LOSO fold generator must require at least 2 development seasons."""
    with pytest.raises(ValueError, match="at least 2 development seasons"):
        generate_loso_folds([2020])

    with pytest.raises(ValueError, match="Duplicate seasons"):
        generate_loso_folds([2020, 2020, 2021])


def test_custom_config_override():
    """Ensure assign_season_splits accepts custom config dictionary."""
    custom_cfg = {
        "random_seed": 999,
        "split_rules": {
            "min_seasons_required": 4,
            "tiers": {
                "full": {"min_seasons": 10, "n_holdout": 3, "status": "CUSTOM_FULL"},
                "reduced": {"min_seasons": 6, "max_seasons": 9, "n_holdout": 2, "status": "CUSTOM_REDUCED"},
                "dev_only": {"min_seasons": 4, "max_seasons": 5, "n_holdout": 0, "status": "CUSTOM_DEV"},
            },
        },
    }
    split = assign_season_splits(list(range(2010, 2020)), config=custom_cfg)
    assert split.status == "CUSTOM_FULL"
    assert split.random_seed == 999
    assert len(split.holdout_seasons) == 3
    assert split.holdout_seasons == [2017, 2018, 2019]
