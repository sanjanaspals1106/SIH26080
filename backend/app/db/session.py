"""Database engine and the per-request connection. Engines are created lazily and cached per URL, so importing
the app never opens a connection. The API only reads: requests use a plain connection and never commit."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.pool import StaticPool

from backend.app.config import Settings, get_settings
from backend.app.errors import ApiError


def make_engine(url: str) -> Engine:
    """Engine for `url`. SQLite (tests, demos) gets foreign keys switched on, as PostgreSQL has by default."""
    if not url.startswith("sqlite"):
        return create_engine(url, pool_pre_ping=True)
    in_memory = url in ("sqlite://", "sqlite:///:memory:")
    engine = create_engine(
        url, connect_args={"check_same_thread": False}, poolclass=StaticPool if in_memory else None
    )

    @event.listens_for(engine, "connect")
    def _foreign_keys_on(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


@lru_cache(maxsize=8)
def engine_for(url: str) -> Engine:
    return make_engine(url)


def get_engine(settings: Settings = Depends(get_settings)) -> Engine:
    if not settings.database_url:
        raise ApiError(503, "DATABASE_NOT_CONFIGURED", "DATABASE_URL is not set. See backend/README.md.")
    return engine_for(settings.database_url)


def get_connection(engine: Engine = Depends(get_engine)) -> Iterator[Connection]:
    """FastAPI dependency: one connection per request."""
    with engine.connect() as connection:
        yield connection
