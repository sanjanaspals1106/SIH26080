"""Runs: read from `nwp_runs`. One query for the page of runs and one grouped query for their leads and dates."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Connection, func, select

from backend.app.db.tables import district_forecasts as df
from backend.app.db.tables import nwp_runs as runs
from backend.app.errors import ApiError


def as_utc(value: datetime | None) -> datetime | None:
    """Always UTC (PRD 18.3): SQLite returns naive values (they are UTC), PostgreSQL returns them in the
    session's time zone."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _run_row(row, extra: dict) -> dict:
    d = dict(row._mapping)
    d["initialization_time"] = as_utc(d["initialization_time"])
    d.pop("created_at", None)
    return {**d, **extra}


def _lead_info(conn: Connection, run_ids: list[str]) -> dict[str, dict]:
    q = (
        select(df.c.run_id, df.c.lead_day, func.min(df.c.imd_date), func.max(df.c.imd_date))
        .where(df.c.run_id.in_(run_ids))
        .group_by(df.c.run_id, df.c.lead_day)
    )
    info: dict[str, dict] = {r: {"lead_days": [], "first": None, "last": None} for r in run_ids}
    for run_id, lead, first, last in conn.execute(q):
        e = info[run_id]
        e["lead_days"].append(int(lead))
        e["first"] = first if e["first"] is None or (first and first < e["first"]) else e["first"]
        e["last"] = last if e["last"] is None or (last and last > e["last"]) else e["last"]
    return info


def list_runs(conn: Connection, season: int | None, limit: int, offset: int) -> tuple[int, list[dict]]:
    where = [runs.c.season == season] if season is not None else []
    total = conn.execute(select(func.count()).select_from(runs).where(*where)).scalar_one()
    rows = conn.execute(
        select(runs)
        .where(*where)
        .order_by(runs.c.initialization_time, runs.c.run_id)
        .limit(limit)
        .offset(offset)
    ).all()
    info = _lead_info(conn, [r.run_id for r in rows])
    return total, [
        _run_row(
            r,
            {
                "lead_days": sorted(info[r.run_id]["lead_days"]),
                "first_imd_date": info[r.run_id]["first"],
                "last_imd_date": info[r.run_id]["last"],
            },
        )
        for r in rows
    ]


def get_run_row(conn: Connection, run_id: str):
    row = conn.execute(select(runs).where(runs.c.run_id == run_id)).first()
    if row is None:
        raise ApiError(404, "RUN_NOT_FOUND", "Forecast run was not found.")
    return row


def get_run(conn: Connection, run_id: str) -> dict:
    row = get_run_row(conn, run_id)
    info = _lead_info(conn, [run_id])[run_id]
    n_districts = conn.execute(
        select(func.count(func.distinct(df.c.district_id))).where(df.c.run_id == run_id)
    ).scalar_one()
    return _run_row(
        row,
        {
            "lead_days": sorted(info["lead_days"]),
            "first_imd_date": info["first"],
            "last_imd_date": info["last"],
            "n_districts": n_districts,
        },
    )
