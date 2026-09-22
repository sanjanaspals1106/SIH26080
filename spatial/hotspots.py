"""Spatial Hotspots computation for Feature F6 (PRD §15 F6).

A cell is a hotspot candidate if:
    Normal mode (64.5 mm model available):
        P(>= 64.5) >= 0.50 AND corrected_mean_mm >= 15.6
    Fallback mode (64.5 mm model unavailable):
        P(>= 15.6) >= 0.50 AND corrected_mean_mm >= 15.6

Thresholds loaded from config/thresholds.yaml:
    hotspots:
      primary_threshold_mm: 64.5
      fallback_threshold_mm: 15.6
      probability_min: 0.50
      corrected_mean_min_mm: 15.6

Rules:
1. Fallback cells are NEVER hotspot candidates.
2. Missing required probability or corrected mean => not a hotspot.
3. 8-connected spatial grouping (touching edges or corners belong to same hotspot).
4. Separate disconnected groups.
5. Geometry outline is a GeoJSON Polygon or MultiPolygon representing the exact
   union of member cell squares (NOT a convex hull).
6. Deterministic hotspot_id ordering (1-indexed).
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import yaml

DEFAULT_PRIMARY_THRESHOLD_MM = 64.5
DEFAULT_FALLBACK_THRESHOLD_MM = 15.6
DEFAULT_PROBABILITY_MIN = 0.50
DEFAULT_CORRECTED_MEAN_MIN_MM = 15.6
DEFAULT_CELL_SIZE = 0.25


@dataclass
class Hotspot:
    """Represents a spatial heavy-rain hotspot cluster (PRD §15 F6)."""
    hotspot_id: int
    n_cells: int
    max_probability: float
    max_corrected_mean_mm: float
    centroid_lat: float
    centroid_lon: float
    outline: Dict[str, Any]
    district_ids: List[str]
    threshold_mm: float
    cells: Optional[List[Any]] = None

    @property
    def centroid(self) -> Dict[str, float]:
        return {"lat": self.centroid_lat, "lon": self.centroid_lon}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hotspot_id": self.hotspot_id,
            "n_cells": self.n_cells,
            "max_probability": self.max_probability,
            "max_corrected_mean_mm": self.max_corrected_mean_mm,
            "centroid_lat": self.centroid_lat,
            "centroid_lon": self.centroid_lon,
            "centroid": self.centroid,
            "outline": self.outline,
            "district_ids": self.district_ids,
            "threshold_mm": self.threshold_mm,
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_hotspot_thresholds(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads hotspot thresholds configuration from config/thresholds.yaml."""
    path = Path(config_path) if config_path else _find_repo_root() / "config" / "thresholds.yaml"
    if not path.is_file():
        return {
            "primary_threshold_mm": DEFAULT_PRIMARY_THRESHOLD_MM,
            "fallback_threshold_mm": DEFAULT_FALLBACK_THRESHOLD_MM,
            "probability_min": DEFAULT_PROBABILITY_MIN,
            "corrected_mean_min_mm": DEFAULT_CORRECTED_MEAN_MIN_MM,
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("hotspots", {})


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (float, int)):
        return math.isnan(val) or math.isinf(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "none", "nan", "null"):
            return True
    return False


def _to_float(val: Any) -> Optional[float]:
    if _is_missing(val):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "t", "y")
    return False


def _extract_field(record: Any, keys: Sequence[str], default: Any = None) -> Any:
    if isinstance(record, dict):
        for k in keys:
            if k in record and record[k] is not None:
                return record[k]
        return default
    for k in keys:
        if hasattr(record, k):
            val = getattr(record, k)
            if val is not None:
                return val
    return default


def _extract_district_ids(cell: Any) -> List[str]:
    """Extracts and normalizes district IDs associated with a cell."""
    dists = _extract_field(cell, ["district_ids", "districts"])
    if dists is not None:
        if isinstance(dists, (list, tuple, set)):
            return [str(d).strip() for d in dists if d is not None and str(d).strip()]
        if isinstance(dists, str) and dists.strip():
            return [dists.strip()]

    single = _extract_field(cell, ["district_id", "district"])
    if single is not None and str(single).strip():
        return [str(single).strip()]
    return []


def _get_grid_coords(cell: Any, cell_size: float = DEFAULT_CELL_SIZE) -> Tuple[int, int]:
    """Returns integer grid row and column for 8-connected topology."""
    r = _extract_field(cell, ["row", "r", "grid_row", "i"])
    c = _extract_field(cell, ["col", "c", "grid_col", "j"])
    if r is not None and c is not None:
        return (int(r), int(c))

    lat = _extract_field(cell, ["latitude", "lat", "y"])
    lon = _extract_field(cell, ["longitude", "lon", "x"])
    if lat is not None and lon is not None:
        return (int(round(float(lat) / cell_size)), int(round(float(lon) / cell_size)))

    raise ValueError(f"Cell must provide either (row, col) or (lat, lon): {cell}")


def _get_lat_lon(cell: Any, cell_size: float = DEFAULT_CELL_SIZE) -> Tuple[float, float]:
    """Returns spatial latitude and longitude for a cell center."""
    lat = _extract_field(cell, ["latitude", "lat", "y"])
    lon = _extract_field(cell, ["longitude", "lon", "x"])
    if lat is not None and lon is not None:
        return (float(lat), float(lon))

    r = _extract_field(cell, ["row", "r", "grid_row", "i"])
    c = _extract_field(cell, ["col", "c", "grid_col", "j"])
    if r is not None and c is not None:
        return (float(r) * cell_size, float(c) * cell_size)

    raise ValueError(f"Cell must provide either (lat, lon) or (row, col): {cell}")


def is_hotspot_cell(
    cell: Any,
    is_64_5_available: bool = True,
    thresholds_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Evaluates whether a single cell qualifies as a hotspot candidate (PRD §15 F6)."""
    # Rule 1: Fallback cells are NEVER hotspot candidates
    fb = _extract_field(cell, ["fallback_used", "is_fallback"], default=False)
    if _to_bool(fb):
        return False

    cfg = thresholds_config if thresholds_config is not None else load_hotspot_thresholds()
    prob_min = float(cfg.get("probability_min", DEFAULT_PROBABILITY_MIN))
    mean_min = float(cfg.get("corrected_mean_min_mm", DEFAULT_CORRECTED_MEAN_MIN_MM))

    # Check corrected mean rain
    mean_val = _to_float(_extract_field(cell, ["corrected_mean_mm", "corrected_mean", "corrected_mm", "corrected"]))
    if mean_val is None or mean_val < mean_min:
        return False

    # Check heavy-rain probability
    if is_64_5_available:
        prob_val = _to_float(_extract_field(cell, ["p_ge_64_5", "p_64_5", "prob_64_5", "p64_5", "heavy_prob"]))
    else:
        prob_val = _to_float(_extract_field(cell, ["p_ge_15_6", "p_15_6", "prob_15_6", "p15_6", "moderate_prob"]))

    if prob_val is None or prob_val < prob_min:
        return False

    return True


def group_hotspot_cells_8conn(
    cells: Sequence[Any],
    cell_size: float = DEFAULT_CELL_SIZE,
) -> List[List[Any]]:
    """Groups candidate hotspot cells into 8-connected components.
    
    Touching sides or corners (diagonals) belong to the same hotspot.
    """
    if not cells:
        return []

    coord_to_cells: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
    for cell in cells:
        coord = _get_grid_coords(cell, cell_size=cell_size)
        coord_to_cells[coord].append(cell)

    offsets_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    visited: Set[Tuple[int, int]] = set()
    components: List[List[Any]] = []

    for start_coord in sorted(list(coord_to_cells.keys())):
        if start_coord in visited:
            continue

        comp_cells: List[Any] = []
        queue = [start_coord]
        visited.add(start_coord)

        while queue:
            curr = queue.pop(0)
            comp_cells.extend(coord_to_cells[curr])
            r, c = curr
            for dr, dc in offsets_8:
                nbr = (r + dr, c + dc)
                if nbr in coord_to_cells and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        components.append(comp_cells)

    return components


def _simplify_collinear_ring(ring: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Removes intermediate collinear vertices along straight polygon edges."""
    if len(ring) < 4:
        return ring
    pts = ring[:-1]
    simplified = []
    n = len(pts)
    for i in range(n):
        prev_p = pts[(i - 1) % n]
        curr_p = pts[i]
        nxt_p = pts[(i + 1) % n]
        v1 = (curr_p[0] - prev_p[0], curr_p[1] - prev_p[1])
        v2 = (nxt_p[0] - curr_p[0], nxt_p[1] - curr_p[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if abs(cross) > 1e-9 or dot < 0:
            simplified.append(curr_p)
    if not simplified:
        return ring
    simplified.append(simplified[0])
    return simplified


def _turn_angle(v_in: Tuple[float, float], v_out: Tuple[float, float]) -> float:
    """Computes counter-clockwise turn angle in (0, 2*pi]."""
    ang_in = math.atan2(v_in[1], v_in[0])
    ang_out = math.atan2(v_out[1], v_out[0])
    diff = (ang_out - ang_in) % (2 * math.pi)
    if diff <= 1e-9:
        diff = 2 * math.pi
    return diff


def _signed_ring_area(ring: List[Tuple[float, float]]) -> float:
    """Calculates signed area via the Shoelace formula."""
    return 0.5 * sum(ring[j][0] * ring[j + 1][1] - ring[j + 1][0] * ring[j][1] for j in range(len(ring) - 1))


def build_hotspot_outline(
    cells: Sequence[Any],
    cell_size: float = DEFAULT_CELL_SIZE,
) -> Dict[str, Any]:
    """Constructs GeoJSON Polygon or MultiPolygon representing the exact union of member cell squares.
    
    Does NOT use a convex hull. Uses boundary edge cancellation and counter-clockwise loop assembly.
    Coordinates are formatted as [longitude, latitude].
    """
    if not cells:
        return {"type": "Polygon", "coordinates": []}

    half = cell_size / 2.0
    edge_counts: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = defaultdict(int)

    for cell in cells:
        lat, lon = _get_lat_lon(cell, cell_size=cell_size)
        x, y = lon, lat
        p0 = (round(x - half, 6), round(y - half, 6))
        p1 = (round(x + half, 6), round(y - half, 6))
        p2 = (round(x + half, 6), round(y + half, 6))
        p3 = (round(x - half, 6), round(y + half, 6))

        # Counter-clockwise square edges
        edge_counts[(p0, p1)] += 1
        edge_counts[(p1, p2)] += 1
        edge_counts[(p2, p3)] += 1
        edge_counts[(p3, p0)] += 1

    # Retain only un-cancelled boundary edges (shared interior edges cancel out)
    adj: Dict[Tuple[float, float], List[Tuple[float, float]]] = defaultdict(list)
    for (a, b) in list(edge_counts.keys()):
        if (b, a) not in edge_counts:
            adj[a].append(b)

    visited_edges: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
    raw_loops: List[List[Tuple[float, float]]] = []

    for start_node in sorted(list(adj.keys())):
        while any((start_node, nxt) not in visited_edges for nxt in adj[start_node]):
            curr = start_node
            prev = None
            loop = [curr]
            while True:
                unvisited = [nxt for nxt in adj[curr] if (curr, nxt) not in visited_edges]
                if not unvisited:
                    break
                if prev is None or len(unvisited) == 1:
                    nxt = unvisited[0]
                else:
                    v_in = (curr[0] - prev[0], curr[1] - prev[1])
                    # Sharpest left turn (smallest counter-clockwise angle) keeps interior on left
                    nxt = min(unvisited, key=lambda n: _turn_angle(v_in, (n[0] - curr[0], n[1] - curr[1])))
                visited_edges.add((curr, nxt))
                loop.append(nxt)
                prev = curr
                curr = nxt
                if curr == start_node:
                    break
            if len(loop) >= 4 and loop[0] == loop[-1]:
                raw_loops.append(loop)

    simplified_loops = [_simplify_collinear_ring(l) for l in raw_loops]

    # Classify loops into exterior boundaries (area > 0) and interior holes (area < 0)
    exterior_rings = [l for l in simplified_loops if _signed_ring_area(l) > 0]
    hole_rings = [l for l in simplified_loops if _signed_ring_area(l) < 0]

    if len(exterior_rings) == 0:
        # Fallback if area was degenerate
        exterior_rings = simplified_loops

    if len(exterior_rings) == 1 and len(hole_rings) == 0:
        return {
            "type": "Polygon",
            "coordinates": [[[pt[0], pt[1]] for pt in exterior_rings[0]]],
        }
    elif len(exterior_rings) == 1 and len(hole_rings) > 0:
        coords = [[[pt[0], pt[1]] for pt in exterior_rings[0]]]
        for hole in hole_rings:
            coords.append([[pt[0], pt[1]] for pt in hole])
        return {
            "type": "Polygon",
            "coordinates": coords,
        }
    else:
        # Multiple exterior rings (e.g. diagonal touching or disjoint multi-polygons)
        multipoly_coords = []
        for ext in exterior_rings:
            multipoly_coords.append([[[pt[0], pt[1]] for pt in ext]])
        return {
            "type": "MultiPolygon",
            "coordinates": multipoly_coords,
        }


def compute_hotspots(
    cells: Sequence[Any],
    is_64_5_available: bool = True,
    thresholds_config: Optional[Dict[str, Any]] = None,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> List[Hotspot]:
    """Computes all spatial hotspots from cells for a given run and lead day (PRD §15 F6).
    
    Args:
        cells: Sequence of cell records (dicts or objects).
        is_64_5_available: Whether 64.5 mm heavy rain model is available.
        thresholds_config: Optional configuration dict from thresholds.yaml.
        cell_size: Grid spacing in degrees (default 0.25).
        
    Returns:
        List of Hotspot objects, sorted deterministically by max_probability descending,
        n_cells descending, max_corrected_mean_mm descending.
    """
    cfg = thresholds_config if thresholds_config is not None else load_hotspot_thresholds()
    threshold_mm = float(
        cfg.get("primary_threshold_mm", DEFAULT_PRIMARY_THRESHOLD_MM)
        if is_64_5_available
        else cfg.get("fallback_threshold_mm", DEFAULT_FALLBACK_THRESHOLD_MM)
    )

    candidate_cells = [
        c for c in cells
        if is_hotspot_cell(c, is_64_5_available=is_64_5_available, thresholds_config=cfg)
    ]

    if not candidate_cells:
        return []

    components = group_hotspot_cells_8conn(candidate_cells, cell_size=cell_size)

    raw_hotspots = []
    for comp in components:
        probs = []
        for c in comp:
            if is_64_5_available:
                p = _to_float(_extract_field(c, ["p_ge_64_5", "p_64_5", "prob_64_5", "p64_5", "heavy_prob"]))
            else:
                p = _to_float(_extract_field(c, ["p_ge_15_6", "p_15_6", "prob_15_6", "p15_6", "moderate_prob"]))
            if p is not None:
                probs.append(p)
        max_prob = round(float(max(probs)), 4) if probs else 0.0

        means = []
        for c in comp:
            m = _to_float(_extract_field(c, ["corrected_mean_mm", "corrected_mean", "corrected_mm", "corrected"]))
            if m is not None:
                means.append(m)
        max_mean = round(float(max(means)), 4) if means else 0.0

        lats = [_get_lat_lon(c, cell_size=cell_size)[0] for c in comp]
        lons = [_get_lat_lon(c, cell_size=cell_size)[1] for c in comp]
        c_lat = round(float(np.mean(lats)), 4)
        c_lon = round(float(np.mean(lons)), 4)

        outline = build_hotspot_outline(comp, cell_size=cell_size)

        all_dists = set()
        for c in comp:
            for d in _extract_district_ids(c):
                all_dists.add(d)
        sorted_dists = sorted(list(all_dists))

        raw_hotspots.append({
            "n_cells": len(comp),
            "max_probability": max_prob,
            "max_corrected_mean_mm": max_mean,
            "centroid_lat": c_lat,
            "centroid_lon": c_lon,
            "outline": outline,
            "district_ids": sorted_dists,
            "threshold_mm": threshold_mm,
            "cells": comp,
        })

    # Sort deterministically: max_probability descending, n_cells descending, max_mean descending, spatial coords
    def sort_key(h: Dict[str, Any]) -> Tuple[float, int, float, float, float]:
        return (
            -h["max_probability"],
            -h["n_cells"],
            -h["max_corrected_mean_mm"],
            -h["centroid_lat"],
            h["centroid_lon"],
        )

    raw_hotspots.sort(key=sort_key)

    hotspots: List[Hotspot] = []
    for rank, h in enumerate(raw_hotspots, start=1):
        hotspots.append(
            Hotspot(
                hotspot_id=rank,
                n_cells=h["n_cells"],
                max_probability=h["max_probability"],
                max_corrected_mean_mm=h["max_corrected_mean_mm"],
                centroid_lat=h["centroid_lat"],
                centroid_lon=h["centroid_lon"],
                outline=h["outline"],
                district_ids=h["district_ids"],
                threshold_mm=h["threshold_mm"],
                cells=h["cells"],
            )
        )

    return hotspots
