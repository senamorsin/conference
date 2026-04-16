import asyncio

import httpx
import cv2
import numpy as np

from src.app.main import create_app
from src.words.labels import WORD_LABELS


def test_page_routes_render_distinct_templates() -> None:
    """The new IA exposes home/letters/words/sequence pages on dedicated paths."""
    async def run_check() -> dict[str, httpx.Response]:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return {
                    "home": await client.get("/"),
                    "letters": await client.get("/letters"),
                    "words": await client.get("/words"),
                    "sequence": await client.get("/sequence"),
                }

    responses = asyncio.run(run_check())

    for name, response in responses.items():
        assert response.status_code == 200, f"page {name} returned {response.status_code}"
        assert response.headers["content-type"].startswith("text/html"), name

    home_html = responses["home"].text
    assert "Sign Sense" in home_html
    assert "mode-grid" in home_html
    assert "camera-preview" not in home_html, "home page should not embed the camera"

    letters_html = responses["letters"].text
    assert 'data-page="letters"' in letters_html
    assert 'id="camera-preview"' in letters_html
    assert 'id="accepted-letters-value"' in letters_html

    words_html = responses["words"].text
    assert 'data-page="words"' in words_html
    assert 'id="word-recording-preview"' in words_html
    assert 'id="word-result-card"' in words_html

    sequence_html = responses["sequence"].text
    assert 'data-page="sequence"' in sequence_html
    assert 'id="sequence-recording-preview"' in sequence_html
    assert 'id="sequence-chip-list"' in sequence_html


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
    assert payload["word_feature_dim"] > 0
    assert payload["word_vocab_size"] == len(WORD_LABELS)
    assert len(payload["word_vocab"]) == len(WORD_LABELS)
    assert payload["sequence_frame_extract"] in ("legacy", "seek", "fast", "linear")
    assert payload["sequence_sample_fps"] > 0
    assert "sequence_max_input_side" in payload


def test_word_sequence_capabilities_endpoint_returns_json() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/words/sequence/capabilities")

    response = asyncio.run(run_check())
    assert response.status_code == 200
    payload = response.json()
    assert "extract_modes" in payload
    assert "fast" in payload["extract_modes"]
    assert payload["streaming"]["incremental"] is False
    assert "active_settings" in payload


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


def test_tts_endpoint_speaks_full_phrase_from_history_and_buffer() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.pipeline.word_builder._history = ["HELLO", "WORLD"]
            app.state.pipeline.decoder._accepted = ["T", "O", "D", "A", "Y"]
            app.state.speech = type(
                "StubSpeech",
                (),
                {
                    "is_available": True,
                    "unavailable_reason": None,
                    "synthesize_wav_bytes": staticmethod(lambda text: text.encode("utf-8")),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/tts/speak")

    response = asyncio.run(run_check())

    assert response.status_code == 200
    assert response.content == b"HELLO WORLD TODAY"


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


def test_delete_last_endpoint_updates_buffer() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.pipeline.decoder._accepted = ["H", "I"]
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post("/api/delete-last")

    response = asyncio.run(run_check())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "deleted"
    assert payload["accepted_letters"] == "H"


def test_word_predict_endpoint_returns_stubbed_prediction() -> None:
    fake_video = b"not-a-real-video-but-stubbed"

    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.word_service = type(
                "StubWordService",
                (),
                {
                    "predict_from_video_bytes": staticmethod(
                        lambda video_bytes, filename: {
                            "status": "word_predicted",
                            "accepted_prediction": True,
                            "predicted_word": "HELLO",
                            "predicted_word_display": "HELLO",
                            "confidence": 0.91,
                            "confidence_level": "high",
                            "rejection_reason": None,
                            "top_prediction_margin": 0.87,
                            "top_predictions": [
                                {"label": "HELLO", "confidence": 0.91},
                                {"label": "HELP", "confidence": 0.04},
                            ],
                            "detected_steps": 9,
                            "sampled_steps": 12,
                            "frame_source": filename,
                        }
                    ),
                    "close": staticmethod(lambda: None),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                files = {"file": ("clip.mp4", fake_video, "video/mp4")}
                return await client.post("/api/words/predict", files=files)

    response = asyncio.run(run_check())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "word_predicted"
    assert payload["predicted_word"] == "HELLO"
    assert payload["accepted_prediction"] is True
    assert payload["confidence_level"] == "high"
    assert len(payload["top_predictions"]) == 2
    assert payload["detected_steps"] == 9


def test_word_predict_endpoint_returns_rejected_prediction_state() -> None:
    fake_video = b"not-a-real-video-but-stubbed"

    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.word_service = type(
                "StubWordService",
                (),
                {
                    "predict_from_video_bytes": staticmethod(
                        lambda video_bytes, filename: {
                            "status": "rejected_ambiguous",
                            "accepted_prediction": False,
                            "predicted_word": "HELP",
                            "predicted_word_display": "HELP",
                            "confidence": 0.48,
                            "confidence_level": "medium",
                            "rejection_reason": "top_predictions_too_close",
                            "top_prediction_margin": 0.05,
                            "top_predictions": [
                                {"label": "HELP", "confidence": 0.48},
                                {"label": "HELLO", "confidence": 0.43},
                            ],
                            "detected_steps": 8,
                            "sampled_steps": 12,
                            "frame_source": filename,
                        }
                    ),
                    "close": staticmethod(lambda: None),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                files = {"file": ("clip.mp4", fake_video, "video/mp4")}
                return await client.post("/api/words/predict", files=files)

    response = asyncio.run(run_check())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "rejected_ambiguous"
    assert payload["accepted_prediction"] is False
    assert payload["rejection_reason"] == "top_predictions_too_close"
    assert payload["predicted_word"] == "HELP"


def test_word_sequence_endpoint_returns_stubbed_transcript() -> None:
    fake_video = b"not-a-real-video-but-stubbed"

    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.word_service = type(
                "StubWordService",
                (),
                {
                    "predict_sequence_from_video_bytes": staticmethod(
                        lambda video_bytes, filename: {
                            "status": "sequence_predicted",
                            "transcript": "HELLO PLEASE HELP",
                            "total_segments": 3,
                            "accepted_segments": 3,
                            "segments": [
                                {
                                    "status": "word_predicted",
                                    "accepted_prediction": True,
                                    "predicted_word": "HELLO",
                                    "predicted_word_display": "HELLO",
                                    "confidence": 0.82,
                                    "confidence_level": "high",
                                    "rejection_reason": None,
                                    "top_prediction_margin": 0.3,
                                    "top_predictions": [
                                        {"label": "HELLO", "confidence": 0.82},
                                        {"label": "HELP", "confidence": 0.05},
                                    ],
                                    "detected_steps": 10,
                                    "sampled_steps": 12,
                                    "start_time": 0.2,
                                    "end_time": 1.4,
                                },
                                {
                                    "status": "word_predicted",
                                    "accepted_prediction": True,
                                    "predicted_word": "PLEASE",
                                    "predicted_word_display": "PLEASE",
                                    "confidence": 0.71,
                                    "confidence_level": "high",
                                    "rejection_reason": None,
                                    "top_prediction_margin": 0.22,
                                    "top_predictions": [
                                        {"label": "PLEASE", "confidence": 0.71},
                                        {"label": "SORRY", "confidence": 0.10},
                                    ],
                                    "detected_steps": 9,
                                    "sampled_steps": 12,
                                    "start_time": 1.8,
                                    "end_time": 3.1,
                                },
                                {
                                    "status": "word_predicted",
                                    "accepted_prediction": True,
                                    "predicted_word": "HELP",
                                    "predicted_word_display": "HELP",
                                    "confidence": 0.66,
                                    "confidence_level": "high",
                                    "rejection_reason": None,
                                    "top_prediction_margin": 0.25,
                                    "top_predictions": [
                                        {"label": "HELP", "confidence": 0.66},
                                        {"label": "HELLO", "confidence": 0.09},
                                    ],
                                    "detected_steps": 11,
                                    "sampled_steps": 12,
                                    "start_time": 3.6,
                                    "end_time": 4.9,
                                },
                            ],
                            "duration_seconds": 5.2,
                            "sample_fps": 10.0,
                            "frame_source": filename,
                        }
                    ),
                    "close": staticmethod(lambda: None),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                files = {"file": ("clip.webm", fake_video, "video/webm")}
                return await client.post("/api/words/sequence", files=files)

    response = asyncio.run(run_check())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "sequence_predicted"
    assert payload["transcript"] == "HELLO PLEASE HELP"
    assert payload["total_segments"] == 3
    assert payload["accepted_segments"] == 3
    assert len(payload["segments"]) == 3
    first = payload["segments"][0]
    assert first["predicted_word"] == "HELLO"
    assert first["start_time"] == 0.2
    assert first["end_time"] == 1.4


def test_word_sequence_endpoint_reports_empty_transcript_when_no_hand() -> None:
    fake_video = b"not-a-real-video-but-stubbed"

    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.word_service = type(
                "StubWordService",
                (),
                {
                    "predict_sequence_from_video_bytes": staticmethod(
                        lambda video_bytes, filename: {
                            "status": "no_hand_detected",
                            "transcript": "",
                            "total_segments": 0,
                            "accepted_segments": 0,
                            "segments": [],
                            "duration_seconds": 2.1,
                            "sample_fps": 10.0,
                            "frame_source": filename,
                        }
                    ),
                    "close": staticmethod(lambda: None),
                },
            )()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                files = {"file": ("clip.webm", fake_video, "video/webm")}
                return await client.post("/api/words/sequence", files=files)

    response = asyncio.run(run_check())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "no_hand_detected"
    assert payload["segments"] == []
    assert payload["transcript"] == ""
