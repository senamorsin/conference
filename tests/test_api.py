import asyncio

import httpx
import cv2
import numpy as np

from src.app.main import create_app


def test_health_endpoint_returns_pipeline_state() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/health")

    response = asyncio.run(run_check())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "letters"


def test_predict_endpoint_handles_empty_frame_without_server_error() -> None:
    blank_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", blank_frame)
    assert ok
    blank_png = encoded.tobytes()

    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                files = {"file": ("blank.png", blank_png, "image/png")}
                return await client.post("/api/letters/predict", files=files)

    response = asyncio.run(run_check())

    assert response.status_code == 200
    assert response.json()["status"] == "no_hand_detected"


def test_tts_endpoint_returns_audio_wav_from_stub_synthesizer() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.speech = type(
                "StubSpeech",
                (),
                {
                    "is_available": True,
                    "unavailable_reason": None,
                    "synthesize_wav_bytes": staticmethod(lambda text: b"RIFFstubWAVE"),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post("/api/tts/speak", json={"text": "HELLO"})

    response = asyncio.run(run_check())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"RIFFstubWAVE"


def test_tts_get_endpoint_returns_audio_wav_from_stub_synthesizer() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.speech = type(
                "StubSpeech",
                (),
                {
                    "is_available": True,
                    "unavailable_reason": None,
                    "synthesize_wav_bytes": staticmethod(lambda text: b"RIFFstubWAVE"),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/tts/speak", params={"text": "HELLO"})

    response = asyncio.run(run_check())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"RIFFstubWAVE"


def test_tts_endpoint_falls_back_to_accepted_letters_when_final_word_is_missing() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.pipeline.decoder._accepted = ["H", "I"]
            app.state.speech = type(
                "StubSpeech",
                (),
                {
                    "is_available": True,
                    "unavailable_reason": None,
                    "synthesize_wav_bytes": staticmethod(lambda text: b"RIFFHIWAVE" if text == "HI" else b"BAD"),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/tts/speak")

    response = asyncio.run(run_check())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"RIFFHIWAVE"


def test_tts_preload_endpoint_returns_ready_for_stub_synthesizer() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.speech = type(
                "StubSpeech",
                (),
                {
                    "is_available": True,
                    "unavailable_reason": None,
                    "preload": staticmethod(lambda: None),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post("/api/tts/preload")

    response = asyncio.run(run_check())

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
