"""F6 spatial hotspots (PRD 15 F6), computed on request from the served corrected cell store.

The rule and the 8-connected grouping live in `spatial/hotspots.py` (numpy + yaml only, so importing it keeps the
API process free of sklearn / geopandas / regime_engine). This module only feeds it real cells:
corrected mean and P(>=64.5 mm) from `data/serving/corrected/`, positions from the Golden Dataset, and the
districts that "contain at least one main cell" from the `cell_district_weights` table.

A district with no main cell (24 large, fragmented ones: every cell < 5% of the area) counts all of its cells as
main, the same convention `scripts/build_district_corrected.py` uses for the district products.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Connection, select

from backend.app.db.tables import cell_district_weights as w
from backend.app.errors import ApiError
from backend.app.services.grid import _read, corrected_file, golden_file
from spatial.hotspots import compute_hotspots, load_hotspot_thresholds

_MAX_CACHED = 512  # one season replay is 366 (run, lead) pairs
_RESULTS: dict[tuple, dict] = {}


_MAPS: dict[str, dict[int, list[str]]] = {}


def cell_district_map(conn: Connection) -> dict[int, list[str]]:
    """cell_id -> ids of the districts for which the cell counts as main (loaded once per database)."""
    key = str(conn.engine.url)
    if key not in _MAPS:
        rows = [(int(r.cell_id), r.district_id, bool(r.is_main)) for r in conn.execute(select(w.c.cell_id, w.c.district_id, w.c.is_main))]
        has_main = {d for _, d, main in rows if main}
        out: dict[int, list[str]] = defaultdict(list)
        for cell_id, district_id, main in rows:
            if main or district_id not in has_main:
                out[cell_id].append(district_id)
        _MAPS[key] = dict(out)
    return _MAPS[key]


@lru_cache(maxsize=4)
def _availability(metrics_path: str) -> bool:
    path = Path(metrics_path)
    if not path.is_file():
        return True  # nothing recorded: every threshold had enough events in this project
    return bool(json.loads(path.read_text()).get("availability", {}).get("p_64_5_available", True))


def get_hotspots(conn: Connection, data_dir: Path, golden_dir: Path, run_id: str, lead_day: int) -> dict:
    if not _availability(str(data_dir / "verification" / "m3_dev" / "metrics.json")):
        # The 15.6 mm fallback needs the P(>=15.6) layer, which the served store does not carry.
        raise ApiError(503, "HOTSPOTS_UNAVAILABLE", "The 64.5 mm heavy-rain model is not available for hotspots.")
    key = (run_id, lead_day, str(golden_dir), str(data_dir), str(conn.engine.url))
    if key in _RESULTS:
        return _RESULTS[key]

    corrected_dir = data_dir / "serving" / "corrected"
    geo = _read(golden_file(golden_dir, run_id), ["cell_id", "latitude", "longitude", "imd_date"], run_id, lead_day)
    cor = _read(corrected_file(corrected_dir, run_id), ["cell_id", "corrected_mean_mm", "p_ge_64_5"], run_id, lead_day)
    if geo is None or cor is None:
        raise ApiError(503, "GRID_DATA_UNAVAILABLE", "Cell-level data for this run is not available.")

    cdm = cell_district_map(conn)
    latlon = {
        int(c): (float(la), float(lo))
        for c, la, lo in zip(
            geo["cell_id"].to_pylist(), geo["latitude"].to_pylist(), geo["longitude"].to_pylist(), strict=True
        )
    }
    cells = [
        {
            "lat": latlon[int(c)][0], "lon": latlon[int(c)][1], "corrected_mean_mm": mean, "p_ge_64_5": p,
            "district_ids": cdm.get(int(c), []),
        }
        for c, mean, p in zip(
            cor["cell_id"].to_pylist(), cor["corrected_mean_mm"].to_pylist(), cor["p_ge_64_5"].to_pylist(), strict=True
        )
        if int(c) in latlon
    ]  # fmt: skip
    cfg = load_hotspot_thresholds()
    found = compute_hotspots(cells, is_64_5_available=True, thresholds_config=cfg)
    result = {
        "imd_date": geo["imd_date"][0].as_py(),
        "threshold_mm": float(cfg.get("primary_threshold_mm", 64.5)),
        "probability_min": float(cfg.get("probability_min", 0.5)),
        "corrected_mean_min_mm": float(cfg.get("corrected_mean_min_mm", 15.6)),
        "hotspots": [
            {
                "hotspot_id": h.hotspot_id,
                "n_cells": h.n_cells,
                "max_probability": h.max_probability,
                "max_corrected_mean_mm": h.max_corrected_mean_mm,
                "centroid": {"lat": h.centroid_lat, "lon": h.centroid_lon},
                "district_ids": h.district_ids,
                "outline": h.outline,
            }
            for h in found
        ],
    }
    if len(_RESULTS) >= _MAX_CACHED:
        _RESULTS.pop(next(iter(_RESULTS)))
    _RESULTS[key] = result
    return result
