"""`region_code` of every cell (PRD 14.3), from `config/regions.yaml`. Owner: M1.

Simple boxes, not official regions. The cell centre is used and a cell gets the FIRST region (by `order`) whose
rule fits. Rule keys: `lat_gte`, `lat_lt`, `lon_gte`, `lon_lt` (all must hold) or `default: true` (everything else).

The code is derived from `latitude` / `longitude`, so it is not stored in the feature table: `read_features(...,
with_region_code=True)` adds it when the rows are read, and no existing table needs rebuilding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data_pipeline.ingestion.config import CONFIG_DIR
from data_pipeline.ingestion.errors import ValidationError

_BOUNDS = {"lat_gte", "lat_lt", "lon_gte", "lon_lt", "default"}


def load_region_rules(path: Path | str | None = None) -> list[dict[str, Any]]:
    """The rules of `regions.yaml`, sorted by `order`, checked (known keys, exactly one final default)."""
    rules = sorted(yaml.safe_load(Path(path or CONFIG_DIR / "regions.yaml").read_text())["regions"], key=lambda r: r["order"])
    for r in rules:
        unknown = set(r["rule"]) - _BOUNDS
        if unknown:
            raise ValidationError(f"regions.yaml: region {r['region_code']} has unknown rule keys {sorted(unknown)}.")
    if not rules or not rules[-1]["rule"].get("default"):
        raise ValidationError("regions.yaml: the last region (highest order) must be the default (`default: true`).")
    if any(r["rule"].get("default") for r in rules[:-1]):
        raise ValidationError("regions.yaml: only the last region may be the default.")
    return rules


def assign_region_codes(
    latitude: np.ndarray | pd.Series, longitude: np.ndarray | pd.Series, rules: list[dict[str, Any]] | None = None
) -> np.ndarray:
    """`region_code` (str) for each cell centre: the first rule that fits."""
    rules = rules or load_region_rules()
    lat, lon = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    out = np.full(lat.shape, None, dtype=object)
    for r in rules:
        rule, fits = r["rule"], np.ones(lat.shape, dtype=bool)
        if "lat_gte" in rule:
            fits &= lat >= rule["lat_gte"]
        if "lat_lt" in rule:
            fits &= lat < rule["lat_lt"]
        if "lon_gte" in rule:
            fits &= lon >= rule["lon_gte"]
        if "lon_lt" in rule:
            fits &= lon < rule["lon_lt"]
        out[fits & (out == None)] = r["region_code"]  # noqa: E711  (elementwise on an object array)
    if (out == None).any():  # noqa: E711
        raise ValidationError("Some cells fit no region (NaN latitude/longitude?).")
    return out.astype(str)
