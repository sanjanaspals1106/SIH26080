"""Smoke tests for the repository skeleton. No pipeline logic is tested here."""

import importlib
import pkgutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

TOP_LEVEL_PACKAGES = [
    "data_pipeline",
    "regime_engine",
    "ml",
    "probability",
    "protocol",
    "verification",
    "backend",
]

# PRD section 14.3, in order.
EXPECTED_REGIONS = [
    "HIMALAYA",
    "NORTHEAST",
    "WEST_COAST",
    "NORTHWEST_WEST",
    "SOUTH_EAST",
    "CENTRAL_EAST",
]


def _all_module_names() -> list[str]:
    names: list[str] = []
    for top in TOP_LEVEL_PACKAGES:
        package = importlib.import_module(top)
        names.append(top)
        names += [m.name for m in pkgutil.walk_packages(package.__path__, prefix=f"{top}.")]
    return names


@pytest.mark.parametrize("name", _all_module_names())
def test_module_imports(name):
    importlib.import_module(name)


def test_every_top_level_package_is_covered():
    assert len(_all_module_names()) > len(TOP_LEVEL_PACKAGES)


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_config_parses(path):
    assert isinstance(yaml.safe_load(path.read_text()), dict)


def test_all_expected_config_files_exist():
    expected = {"alignment", "regime", "protocol", "thresholds", "districts", "verification", "regions"}
    assert {p.stem for p in CONFIG_DIR.glob("*.yaml")} == expected


def test_regions_are_the_six_in_prd_order():
    regions = yaml.safe_load((CONFIG_DIR / "regions.yaml").read_text())["regions"]
    assert [r["region_code"] for r in regions] == EXPECTED_REGIONS
    assert [r["order"] for r in regions] == [1, 2, 3, 4, 5, 6]
