"""FastAPI application (PRD section 18). Owner: M1.

Reads stored M1 products from PostgreSQL (district level) and Parquet (cell level). It never trains a model,
never builds features or weights, and has no endpoint that changes data: data is loaded by
`scripts/load_m1.py`. Run from the repository root: uvicorn backend.app.main:app --port 8000
"""

from fastapi import FastAPI

from backend.app.api.routes import VERSION, router
from backend.app.errors import register_error_handlers

API_PREFIX = "/api/v1"  # PRD 18.3


def create_app() -> FastAPI:
    app = FastAPI(title="SIH26080 API", version=VERSION)
    register_error_handlers(app)
    app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
