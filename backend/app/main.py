"""FastAPI application (PRD section 18). Owner: M1.

Reads stored M1 products from PostgreSQL (district level) and Parquet (cell level). It never trains a model,
never builds features or weights, and has no endpoint that changes data: data is loaded by
`scripts/load_m1.py`. Run from the repository root: uvicorn backend.app.main:app --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import VERSION, router
from backend.app.errors import register_error_handlers

API_PREFIX = "/api/v1"  # PRD 18.3


def create_app() -> FastAPI:
    app = FastAPI(title="SIH26080 API", version=VERSION)
    # The frontend (Vite dev server / static build) runs on a different origin than the API
    # (PRD 6: two separate parts on the same laptop). Without this, every browser request is
    # blocked by CORS even though curl/pytest never see the problem.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
