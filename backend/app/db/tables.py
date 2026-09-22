"""SQLAlchemy Core definitions of the M1 tables. They mirror `database/schema.sql` (PRD 20.2) exactly.

`schema.sql` is the DDL that creates the real PostgreSQL database; these definitions are what the loader and the
API query with. A test compares the two so they cannot drift apart. Only the tables M1 owns are described here.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()
_Json = JSON().with_variant(JSONB(), "postgresql")
_Real = Float(24)  # REAL (float32) in PostgreSQL

districts = Table(
    "districts",
    metadata,
    Column("district_id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("geojson", _Json),
    Column("centroid_lat", _Real),
    Column("centroid_lon", _Real),
    Column("n_effective_cells", _Real),
    Column("is_small", Boolean),
    Column("source_year", Integer),
)

grid_cells = Table(
    "grid_cells",
    metadata,
    Column("cell_id", Integer, primary_key=True),
    Column("latitude", Float, nullable=False),  # NUMERIC(6,3) in PostgreSQL
    Column("longitude", Float, nullable=False),
    Column("is_valid", Boolean),
    Column("elevation_m", _Real),
    Column("slope", _Real),
    Column("dist_coast_km", _Real),
    Column("region_code", Text),
)

cell_district_weights = Table(
    "cell_district_weights",
    metadata,
    Column("cell_id", Integer, ForeignKey("grid_cells.cell_id"), primary_key=True),
    Column("district_id", Text, ForeignKey("districts.district_id"), primary_key=True),
    Column("area_weight", _Real, nullable=False),
    Column("is_main", Boolean),
    Index("idx_cell_district_weights_district", "district_id"),
)

nwp_runs = Table(
    "nwp_runs",
    metadata,
    Column("run_id", Text, primary_key=True),
    Column("source", Text),
    Column("initialization_time", DateTime(timezone=True), nullable=False),
    Column("season", Integer),
    Column("evaluation_set", Text),  # development | holdout: set by the protocol (M4), NULL at M1
    Column("mode", Text, server_default="replay"),
    Column("alignment_method", Text),
    Column("alignment_offset_hours", SmallInteger, server_default="0"),
    Column("imd_stamp", Text),
    Column("status", Text),
    Column("created_at", DateTime(timezone=True)),
    Index("idx_nwp_runs_initialization_time", "initialization_time"),
)

district_forecasts = Table(
    "district_forecasts",
    metadata,
    Column("run_id", Text, ForeignKey("nwp_runs.run_id"), primary_key=True),
    Column("lead_day", SmallInteger, primary_key=True),
    Column("district_id", Text, ForeignKey("districts.district_id"), primary_key=True),
    Column("imd_date", Date),
    Column("raw_mean_mm", _Real),
    Column("corrected_mean_mm", _Real),
    Column("observed_mean_mm", _Real),
    Column("wettest_cell_id", Integer),
    Column("wettest_cell_mean_mm", _Real),
    Column("wettest_cell_q10_mm", _Real),
    Column("wettest_cell_q50_mm", _Real),
    Column("wettest_cell_q90_mm", _Real),
    Column("heavy_prob_max_cell", _Real),
    Column("very_heavy_prob_max_cell", _Real),
    Column("heavy_area_fraction_expected", _Real),
    Column("very_heavy_area_fraction_expected", _Real),
    Column("attention_level", Text),
    Column("priority_rank", Integer),
    Column("is_small", Boolean),
    Column("product_type", Text),
    Column("fallback_used", Boolean),
    Column("fallback_reason", Text),
    Column("model_version_id", Integer),  # REFERENCES model_versions (a later stage's table)
    Index("idx_district_forecasts_run_district", "run_id", "district_id"),
    Index("idx_district_forecasts_imd_date", "imd_date"),
    Index("idx_district_forecasts_district_date", "district_id", "imd_date"),
)

district_history = Table(
    "district_history",
    metadata,
    Column("run_id", Text, ForeignKey("nwp_runs.run_id"), primary_key=True),
    Column("lead_day", SmallInteger, primary_key=True),
    Column("district_id", Text, primary_key=True),
    Column("imd_date", Date),
    Column("season", Integer),
    Column("observed_mean_mm", _Real),
    Column("observed_max_cell_mm", _Real),
    Column("raw_mean_mm", _Real),
    Column("corrected_mean_mm", _Real),
    Column("prediction_source", Text),  # oof | final
    Column("phase", Text),
    Column("lps_near", Boolean),
    Index("idx_district_history_district_lead_season", "district_id", "lead_day", "season"),
    Index("idx_district_history_run", "run_id"),
)

M1_TABLES = [districts, grid_cells, cell_district_weights, nwp_runs, district_forecasts, district_history]
