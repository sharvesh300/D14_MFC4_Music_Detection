"""
app/main.py — FastAPI application entrypoint
=============================================

Run locally:
    uvicorn app.main:app --reload

Inside Docker (via docker-compose):
    The Dockerfile CMD is set to run uvicorn.
"""

from fastapi import FastAPI

from app.api.routes import router
from app.api.websocket import ws_router

app = FastAPI(
    title="Music Detection API",
    description="Shazam-style audio fingerprinting over Redis",
    version="1.0.0",
)

app.include_router(router)
app.include_router(ws_router)


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
