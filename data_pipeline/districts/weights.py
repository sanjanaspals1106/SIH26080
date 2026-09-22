"""District-cell area weights (PRD 14.2). Owner: M1.

For district j and cell i: `w_ij = area(cell i ∩ district j) / area(district j)`, with all areas computed in
the equal-area projection **EPSG:6933** (never in degrees). Then, as the PRD requires:

1. only **valid** cells count: the overlaps of non-valid cells (sea, outside India) are dropped;
2. the remaining weights are **divided by their sum**, so they add up to 1 for every district
   (`renormalise_after_dropping_invalid_cells`; since the denominator is the same for all cells of a district,
   this is `overlap / sum of the valid overlaps`);
3. a cell is **main** if its (renormalised) weight is at least `w_min` = 0.05.

A cell is the exact 0.25 degree square around its IMD centre. No point-in-polygon shortcut is used.

Two tables come out (both cached as Parquet under `<DATA_DIR>/features/districts/`, with a hash of the inputs, so
they are recomputed only if the districts, the valid-cell mask or the settings change):

* weights: `district_id, cell_id, overlap_km2, area_weight, is_main`
* district summary: `district_id, name, state, district_area_km2, valid_area_fraction, n_cells, n_main_cells,
  n_effective_cells (= 1 / sum w^2), is_small (= n_effective_cells < 4), centroid_lat, centroid_lon`

Districts without any valid cell (for example a group of islands outside the grid) get no rows and are listed in
`summary.attrs["districts_without_cells"]` and in a warning.
"""

from __future__ import annotations

import hashlib
import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box

from data_pipeline.alignment.cells import load_valid_cells
from data_pipeline.districts.loader import load_districts
from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import ValidationError

log = logging.getLogger(__name__)

WEIGHT_COLUMNS = ["district_id", "cell_id", "overlap_km2", "area_weight", "is_main"]
SUMMARY_COLUMNS = [
    "district_id", "name", "state", "district_area_km2", "valid_area_fraction", "n_cells", "n_main_cells",
    "n_effective_cells", "is_small", "centroid_lat", "centroid_lon",
]  # fmt: skip
_M2_PER_KM2 = 1.0e6


def cell_squares(grid: pd.DataFrame, config: IngestionConfig) -> gpd.GeoDataFrame:
    """The 0.25 degree squares of the given grid rows, in EPSG:4326 (`cell_id`, `geometry`)."""
    h = config.imd.grid_spacing_deg / 2.0
    geoms = [
        box(lo - h, la - h, lo + h, la + h)
        for la, lo in zip(grid["latitude"], grid["longitude"], strict=True)
    ]
    return gpd.GeoDataFrame({"cell_id": grid["cell_id"].to_numpy()}, geometry=geoms, crs="EPSG:4326")


def compute_district_weights(
    districts: gpd.GeoDataFrame, grid: pd.DataFrame, config: IngestionConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weights, summary) for `districts` (output of `load_districts`) and the valid cells of `grid`
    (the table of `load_valid_cells`, with `is_valid`)."""
    config = config or load_config()
    f = config.features
    if not f.renormalise_after_dropping_invalid_cells:
        raise ValidationError(
            "Only renormalise_after_dropping_invalid_cells: true is implemented (PRD 14.2)."
        )
    if districts.crs is None:
        raise ValidationError("District geometries have no CRS; use load_districts().")
    epsg = f"EPSG:{f.projection_epsg}"

    dist = districts[["district_id", "name", "state", "geometry"]].to_crs(epsg)
    dist["district_area_km2"] = dist.geometry.area / _M2_PER_KM2
    valid_cells = cell_squares(grid[grid["is_valid"]], config).to_crs(epsg)

    inter = gpd.overlay(
        valid_cells, dist[["district_id", "geometry"]], how="intersection", keep_geom_type=True
    )
    inter["overlap_km2"] = inter.geometry.area / _M2_PER_KM2
    inter = inter[inter["overlap_km2"] > 0]
    w = inter.groupby(["district_id", "cell_id"], as_index=False, sort=True)[
        "overlap_km2"
    ].sum()  # a district can meet a cell in several pieces
    total = w.groupby("district_id")["overlap_km2"].transform("sum")
    w["area_weight"] = w["overlap_km2"] / total
    w["is_main"] = w["area_weight"] >= f.w_min
    w = w[WEIGHT_COLUMNS].astype({"cell_id": "int32", "overlap_km2": "float64", "area_weight": "float64"})

    by = w.groupby("district_id")
    stats = pd.DataFrame(
        {
            "valid_km2": by["overlap_km2"].sum(),
            "n_cells": by["cell_id"].size(),
            "n_main_cells": by["is_main"].sum(),
            "n_effective_cells": 1.0 / by["area_weight"].agg(lambda s: float((s**2).sum())),
        }
    ).reset_index()
    cen = dist.geometry.centroid.to_crs("EPSG:4326")
    summary = dist[["district_id", "name", "state", "district_area_km2"]].assign(
        centroid_lat=cen.y.to_numpy(), centroid_lon=cen.x.to_numpy()
    )
    summary = summary.merge(stats, on="district_id", how="inner")
    summary["valid_area_fraction"] = summary["valid_km2"] / summary["district_area_km2"]
    summary["is_small"] = summary["n_effective_cells"] < f.small_n_effective_max_exclusive
    summary = summary[SUMMARY_COLUMNS].astype({"n_cells": "int32", "n_main_cells": "int32"})

    none = sorted(set(dist["district_id"]) - set(summary["district_id"]))
    if none:
        log.warning("%d district(s) have no valid cell and get no weights: %s", len(none), none[:10])
    summary.attrs["districts_without_cells"] = none
    w = w.sort_values(["district_id", "cell_id"]).reset_index(drop=True)
    summary = summary.sort_values("district_id").reset_index(drop=True)
    check_weights(w)
    return w, summary


def check_weights(weights: pd.DataFrame, tol: float = 1e-9) -> None:
    """Weights must add up to 1 for every district and be unique per (district, cell)."""
    if weights.duplicated(["district_id", "cell_id"]).any():
        raise ValidationError("Duplicate (district_id, cell_id) weights.")
    sums = weights.groupby("district_id")["area_weight"].sum()
    if not np.allclose(sums, 1.0, atol=tol):
        raise ValidationError(
            f"District weights do not add up to 1 (worst: {float((sums - 1).abs().max()):.2e})."
        )


# ---- cache -----------------------------------------------------------------------------------------


def _paths(config: IngestionConfig):
    d = config.features.dir / "districts"
    return d / "cell_district_weights.parquet", d / "district_summary.parquet"


def _input_hash(districts: gpd.GeoDataFrame, grid: pd.DataFrame, config: IngestionConfig) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(grid.loc[grid["is_valid"], "cell_id"].to_numpy().astype("int32")).tobytes())
    h.update(json.dumps([districts["district_id"].tolist(), districts["name"].tolist()]).encode())
    h.update(b"".join(g.wkb for g in districts.geometry))
    f = config.features
    h.update(
        f"{f.projection_epsg}|{f.w_min}|{f.small_n_effective_max_exclusive}|{config.imd.grid_spacing_deg}".encode()
    )
    return h.hexdigest()[:16]


def get_district_weights(
    districts: gpd.GeoDataFrame | None = None,
    grid: pd.DataFrame | None = None,
    config: IngestionConfig | None = None,
    recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached `(weights, summary)`. Without arguments it loads the configured district file (Stage 1 loader,
    which fails clearly if none is configured) and the saved valid-cell mask (Stage 2)."""
    config = config or load_config()
    districts = districts if districts is not None else load_districts(config=config)
    grid = grid if grid is not None else load_valid_cells(config)
    key = _input_hash(districts, grid, config)
    wpath, spath = _paths(config)
    if wpath.is_file() and spath.is_file() and not recompute:
        wt, st = pq.read_table(wpath), pq.read_table(spath)
        if (wt.schema.metadata or {}).get(b"input_hash", b"").decode() == key:
            summary = st.to_pandas()
            summary.attrs["districts_without_cells"] = json.loads(
                st.schema.metadata[b"districts_without_cells"]
            )
            return wt.to_pandas(), summary
        log.info("district weight cache is stale (districts or valid cells changed); recomputing")
    weights, summary = compute_district_weights(districts, grid, config)
    wpath.parent.mkdir(parents=True, exist_ok=True)
    for path, df, meta in (
        (wpath, weights, {"input_hash": key}),
        (
            spath,
            summary,
            {
                "input_hash": key,
                "districts_without_cells": json.dumps(summary.attrs["districts_without_cells"]),
            },
        ),
    ):
        tmp = path.with_name(path.name + ".part")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(meta), tmp)
        tmp.replace(path)
    return weights, summary
