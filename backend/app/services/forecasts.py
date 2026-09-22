"""District forecasts: one JOIN of `district_forecasts` and `districts` per request (no N+1)."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Connection, func, select

from backend.app.db.tables import district_forecasts as f
from backend.app.db.tables import districts as d
from backend.app.errors import ApiError
from backend.app.services.runs import get_run_row

_FORECAST_COLUMNS = [
    f.c.run_id, f.c.lead_day, f.c.imd_date, f.c.district_id, d.c.name.label("district_name"), d.c.state, f.c.is_small,
    f.c.product_type, f.c.raw_mean_mm, f.c.corrected_mean_mm, f.c.observed_mean_mm, f.c.wettest_cell_id,
    f.c.wettest_cell_mean_mm, f.c.wettest_cell_q10_mm, f.c.wettest_cell_q50_mm, f.c.wettest_cell_q90_mm,
    f.c.heavy_prob_max_cell, f.c.very_heavy_prob_max_cell, f.c.heavy_area_fraction_expected,
    f.c.very_heavy_area_fraction_expected, f.c.attention_level, f.c.priority_rank, f.c.fallback_used, f.c.fallback_reason,
]  # fmt: skip


def _shape(row) -> dict:
    m = dict(row._mapping)
    m["flags"] = {"fallback_used": m.pop("fallback_used"), "fallback_reason": m.pop("fallback_reason")}
    return m


def require_district(conn: Connection, district_id: str):
    row = conn.execute(select(d).where(d.c.district_id == district_id)).first()
    if row is None:
        raise ApiError(404, "DISTRICT_NOT_FOUND", "District was not found.")
    return row


def query_forecasts(
    conn: Connection,
    *,
    run_id: str | None,
    lead_day: int | None,
    district_id: str | None,
    state: str | None,
    imd_date: date | None,
    season: int | None,
    limit: int,
    offset: int,
) -> tuple[int, list[dict]]:
    """Filtered, paginated forecasts ordered by (run_id, lead_day, district_id)."""
    conds = []
    if run_id is not None:
        conds.append(f.c.run_id == run_id)
    if lead_day is not None:
        conds.append(f.c.lead_day == lead_day)
    if district_id is not None:
        conds.append(f.c.district_id == district_id)
    if state is not None:
        conds.append(d.c.state == state)
    if imd_date is not None:
        conds.append(f.c.imd_date == imd_date)
    if season is not None:  # season lives on the run; select its runs without a join to keep the query simple
        from backend.app.db.tables import nwp_runs

        conds.append(f.c.run_id.in_(select(nwp_runs.c.run_id).where(nwp_runs.c.season == season)))
    base = select(*_FORECAST_COLUMNS).select_from(f.join(d, d.c.district_id == f.c.district_id)).where(*conds)
    total = conn.execute(
        select(func.count()).select_from(f.join(d, d.c.district_id == f.c.district_id)).where(*conds)
    ).scalar_one()
    rows = conn.execute(
        base.order_by(f.c.run_id, f.c.lead_day, f.c.district_id).limit(limit).offset(offset)
    ).all()
    return total, [_shape(r) for r in rows]


def list_district_forecasts(
    conn: Connection, *, run_id, lead_day, district_id, state, imd_date, season, limit, offset
) -> tuple[int, list[dict], str | None]:
    run = get_run_row(conn, run_id) if run_id is not None else None  # 404 for an unknown run
    if district_id is not None:
        require_district(conn, district_id)  # 404 for an unknown district
    total, rows = query_forecasts(
        conn,
        run_id=run_id,
        lead_day=lead_day,
        district_id=district_id,
        state=state,
        imd_date=imd_date,
        season=season,
        limit=limit,
        offset=offset,
    )
    return total, rows, (run.evaluation_set if run is not None else None)


def district_detail(
    conn: Connection, district_id: str, *, run_id, lead_day, imd_date, limit, offset
) -> tuple[dict, int, list[dict]]:
    district = dict(require_district(conn, district_id)._mapping)
    if run_id is not None:
        get_run_row(conn, run_id)
    total, rows = query_forecasts(
        conn,
        run_id=run_id,
        lead_day=lead_day,
        district_id=district_id,
        state=None,
        imd_date=imd_date,
        season=None,
        limit=limit,
        offset=offset,
    )
    if total == 0:
        raise ApiError(
            404, "FORECAST_NOT_FOUND", "No forecast was found for this district and these filters."
        )
    info = {
        k: district[k]
        for k in (
            "district_id",
            "name",
            "state",
            "is_small",
            "n_effective_cells",
            "centroid_lat",
            "centroid_lon",
            "source_year",
        )
    }
    return info, total, rows
