"""FastAPI skeleton (PRD section 18). Owner: M1.

Only the health endpoint exists. Every other endpoint of PRD 18.4 is still to be built.
Run from the repository root: uvicorn backend.app.main:app --port 8000
"""

from fastapi import FastAPI

VERSION = "0.1.0"

app = FastAPI(title="SIH26080 API", version=VERSION)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "replay", "version": VERSION}
