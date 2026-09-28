"""FastAPI application (PRD section 18). Owner: M1.

Reads stored M1 products from PostgreSQL (district level) and Parquet (cell level). It never trains a model,
never builds features or weights, and has no endpoint that changes data: data is loaded by
`scripts/load_m1.py`. Run from the repository root: uvicorn backend.app.main:app --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.analog_routes import router as analog_router
from backend.app.api.hotspot_routes import router as hotspot_router
from backend.app.api.regime_routes import router as regime_router
from backend.app.api.routes import VERSION, router
from backend.app.api.summary_routes import router as summary_router
from backend.app.api.verification_routes import router as verification_router
from backend.app.errors import register_error_handlers

API_PREFIX = "/api/v1"  # PRD 18.3


def create_app() -> FastAPI:
    app = FastAPI(title="Bharat VarshAI API", version=VERSION)
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
    app.include_router(regime_router, prefix=API_PREFIX)
    app.include_router(hotspot_router, prefix=API_PREFIX)
    app.include_router(summary_router, prefix=API_PREFIX)
    app.include_router(verification_router, prefix=API_PREFIX)
    app.include_router(analog_router, prefix=API_PREFIX)
    return app


app = create_app()
