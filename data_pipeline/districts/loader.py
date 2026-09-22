"""District boundary loader (PRD 14, D7, CK9). Owner: M1.

Reads the DataMeet 2011 district boundaries and returns a GeoDataFrame that Stage 3 (area weights)
can use directly. **No weighting is done here.**

Output columns:

* `district_id`  stable text ID `D001`, `D002`, ... (PRD 9.7 / 20.2 style). Assigned after sorting by
  state, name and source order, so the same file always gives the same IDs.
* `name`, `state`  text (`state` is "" if the file has none).
* `source_id`  the file's own district code, if `districts.yaml` names such a column.
* `geometry`  valid Polygon/MultiPolygon in EPSG:4326.

The file name and the column names come from `config/districts.yaml`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import (
    IngestionError,
    MissingInputError,
    ValidationError,
)

log = logging.getLogger(__name__)

TARGET_CRS = "EPSG:4326"
# Sanity box for India in degrees. Catches swapped lat/lon or a projected CRS mislabelled as degrees.
_INDIA_BOX = (60.0, 5.0, 100.0, 40.0)  # minx, miny, maxx, maxy


def _polygonal_part(geom: BaseGeometry) -> BaseGeometry:
    """Keep only the polygons of a repaired geometry (make_valid may return a collection)."""
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    return shapely.union_all(polys) if polys else Polygon()


def load_districts(path: Path | str | None = None, config: IngestionConfig | None = None) -> gpd.GeoDataFrame:
    """Load, validate and normalise the district boundaries (see module docstring)."""
    config = config or load_config()
    cfg = config.districts
    if path is None:
        if not cfg.file:
            raise MissingInputError(
                "No district file is configured. Download the DataMeet 2011 district boundaries, "
                f"put them in {cfg.raw_dir}, and set source.file in config/districts.yaml (CK9). "
                "You can also pass the path to load_districts()."
            )
        path = cfg.raw_dir / cfg.file
    path = Path(path)
    if not path.exists():
        raise MissingInputError(f"District boundary file not found: {path}.")

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        raise IngestionError(
            f"Could not read {path.name} as a boundary file ({type(exc).__name__}: {exc}). "
            "Use a shapefile (with its .shx/.dbf/.prj siblings) or GeoJSON."
        ) from exc

    if len(gdf) == 0:
        raise ValidationError(f"{path.name} contains no districts.")

    if gdf.crs is None:
        if not cfg.assume_crs:
            raise ValidationError(
                f"{path.name} has no CRS (a shapefile without .prj?). If you know it is lat/lon "
                "degrees, set source.assume_crs: EPSG:4326 in config/districts.yaml."
            )
        log.warning("%s has no CRS; assuming %s as configured", path.name, cfg.assume_crs)
        gdf = gdf.set_crs(cfg.assume_crs)
    gdf = gdf.to_crs(TARGET_CRS)

    wanted = {"name": cfg.name_column, "state": cfg.state_column, "source_id": cfg.source_id_column}
    missing = [c for c in (cfg.name_column,) if c not in gdf.columns]
    if missing:
        raise ValidationError(
            f"{path.name} has no column {missing}. Columns found: {list(gdf.columns)}. "
            "Fix source.columns in config/districts.yaml (CK9)."
        )
    for label in ("state", "source_id"):
        col = wanted[label]
        if col and col not in gdf.columns:
            log.warning("Column '%s' (%s) is not in %s; leaving it empty", col, label, path.name)
            wanted[label] = None

    bad = gdf.geometry.isna() | gdf.geometry.is_empty
    if bad.any():
        names = gdf.loc[bad, cfg.name_column].astype(str).tolist()[:10]
        raise ValidationError(f"{int(bad.sum())} district(s) have no geometry, for example {names}.")

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        log.warning("Repairing %d invalid district geometries with make_valid", int(invalid.sum()))
        fixed = gdf.geometry.copy()
        fixed[invalid] = [_polygonal_part(shapely.make_valid(g)) for g in gdf.geometry[invalid]]
        gdf = gdf.set_geometry(fixed)
        still_bad = gdf.geometry.is_empty | ~gdf.geometry.is_valid
        if still_bad.any():
            names = gdf.loc[still_bad, cfg.name_column].astype(str).tolist()[:10]
            raise ValidationError(f"Could not repair the geometry of district(s) {names}.")

    minx, miny, maxx, maxy = gdf.total_bounds
    bx = _INDIA_BOX
    if minx < bx[0] or miny < bx[1] or maxx > bx[2] or maxy > bx[3]:
        raise ValidationError(
            f"District bounds lon {minx:.1f}..{maxx:.1f}, lat {miny:.1f}..{maxy:.1f} are outside India "
            f"(lon {bx[0]:g}..{bx[2]:g}, lat {bx[1]:g}..{bx[3]:g}). Wrong CRS or swapped axes?"
        )

    out = gpd.GeoDataFrame(
        {
            "name": gdf[cfg.name_column].astype(str).str.strip(),
            "state": gdf[wanted["state"]].astype(str).str.strip() if wanted["state"] else "",
            "source_id": gdf[wanted["source_id"]].astype(str).str.strip() if wanted["source_id"] else None,
        },
        geometry=gdf.geometry.values,
        crs=TARGET_CRS,
    )
    if (out["name"] == "").any() or out["name"].isin(["nan", "None"]).any():
        raise ValidationError("Some districts have an empty name. The district name column is not usable.")

    out = out.reset_index(drop=True)
    out["_order"] = out.index
    out = (
        out.sort_values(["state", "name", "_order"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    out.insert(0, "district_id", [f"D{i:03d}" for i in range(1, len(out) + 1)])
    if wanted["source_id"] is None:
        out = out.drop(columns="source_id")

    if cfg.n_districts is not None and len(out) != cfg.n_districts:
        log.warning("File has %d districts, districts.yaml says %d (CK9)", len(out), cfg.n_districts)
    log.info("Loaded %d districts from %s", len(out), path.name)
    return out
