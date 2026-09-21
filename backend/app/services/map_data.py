"""Map metadata and district outlines: a handful of aggregate queries and one SELECT of the stored outlines."""

from __future__ import annotations

from sqlalchemy import Connection, case, func, select

from backend.app.config import Settings
from backend.app.db.tables import district_forecasts as f
from backend.app.db.tables import districts as d
from backend.app.db.tables import grid_cells as g
from backend.app.db.tables import nwp_runs as r
from backend.app.services.grid import AVAILABLE, VARIABLES
from backend.app.services.runs import as_utc

_LATER = "Produced by a later pipeline stage; not available yet."


def metadata(conn: Connection, s: Settings) -> dict:
    n_runs, first, last = conn.execute(
        select(func.count(), func.min(r.c.initialization_time), func.max(r.c.initialization_time))
    ).one()
    seasons = [
        x for (x,) in conn.execute(select(r.c.season).distinct().order_by(r.c.season)) if x is not None
    ]
    first_date, last_date = conn.execute(select(func.min(f.c.imd_date), func.max(f.c.imd_date))).one()
    n_cells, n_valid = conn.execute(
        select(func.count(), func.sum(case((g.c.is_valid.is_(True), 1), else_=0)))
    ).one()
    n_districts = conn.execute(select(func.count()).select_from(d)).scalar_one()
    return {
        "grid": {
            **s.grid,
            "crs": "EPSG:4326",
            "n_cells": int(n_cells),
            "n_valid_cells": int(n_valid or 0),
            "cell_id": "i_lat * n_lon + i_lon, from lat_min / lon_min",
        },
        "layers": [
            {"variable": v, "available": v in AVAILABLE, "note": None if v in AVAILABLE else _LATER}
            for v in VARIABLES
        ],
        "lead_days": s.lead_days,
        "thresholds_mm": s.thresholds_mm,
        "runs": {
            "total": n_runs,
            "first_initialization_time": as_utc(first),
            "last_initialization_time": as_utc(last),
            "seasons": [int(x) for x in seasons],
        },
        "imd_dates": {"first": first_date, "last": last_date},
        "districts": {
            "count": n_districts,
            "source": s.district_source,
            "census_basis_year": s.district_census_basis,
            "note": "Districts created after 2011 are not included.",
        },
    }


def district_outlines(conn: Connection) -> list[dict]:
    """GeoJSON features (stored, simplified outlines) with the district columns as properties."""
    features = []
    for row in conn.execute(select(d).order_by(d.c.district_id)):
        stored = row.geojson or {}
        features.append(
            {
                "type": "Feature",
                "id": row.district_id,
                "geometry": stored.get("geometry"),
                "properties": {
                    **stored.get("properties", {}),
                    "district_id": row.district_id,
                    "name": row.name,
                    "state": row.state,
                    "is_small": row.is_small,
                    "n_effective_cells": row.n_effective_cells,
                    "source_year": row.source_year,
                },
            }
        )
    return features
