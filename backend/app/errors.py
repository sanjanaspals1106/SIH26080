"""One error format for the whole API (PRD 18.3): `{"error": {"code": ..., "message": ...}}`.

400 wrong request, 404 not found, 422 wrong parameter format, 500 server error, 503 needed data not available.
Database exceptions are logged, never sent to the client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'query')}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse(
            _body("INVALID_PARAMETER", f"Invalid request parameter: {problems}."), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        return JSONResponse(_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(OperationalError)
    async def _db_down(_: Request, exc: OperationalError) -> JSONResponse:
        log.error("database unavailable: %s", exc)
        return JSONResponse(_body("DATABASE_UNAVAILABLE", "The database is not available."), status_code=503)

    @app.exception_handler(SQLAlchemyError)
    async def _db_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        log.error("database error: %s", exc)
        return JSONResponse(_body("INTERNAL_ERROR", "The request could not be completed."), status_code=500)
