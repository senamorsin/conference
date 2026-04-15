from __future__ import annotations

from contextlib import asynccontextmanager
import os

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.app.routes import router
from src.app.state import create_app_state
from src.tts import create_default_speech_synthesizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = create_app_state()
    app.state.speech = create_default_speech_synthesizer()
    try:
        yield
    finally:
        app.state.pipeline.close()


def create_app() -> FastAPI:
    app = FastAPI(title="ASL Demo MVP", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
    return app


def run() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.app.main:create_app", host=host, port=port, factory=True, reload=False)
