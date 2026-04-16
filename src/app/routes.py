from __future__ import annotations

import re

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from src.app.schemas import (
    HealthResponse,
    PredictResponse,
    ResetResponse,
    TTSRequest,
    WordPredictResponse,
    WordSequenceResponse,
)
from src.app.state import AppPipelineState
from src.tts import PiperSpeechSynthesizer
from src.words.labels import WORD_FEATURE_COLUMNS, WORD_LABELS, display_word_label
from src.words.service import WordRecognitionService, create_default_word_service

router = APIRouter()
templates = Jinja2Templates(directory="src/app/templates")


def _get_pipeline(request: Request) -> AppPipelineState:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline state is not initialized")
    return pipeline


def _get_speech_synthesizer(request: Request) -> PiperSpeechSynthesizer:
    synthesizer = getattr(request.app.state, "speech", None)
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Speech synthesizer is not initialized")
    return synthesizer


def _get_word_service(request: Request) -> WordRecognitionService:
    service = getattr(request.app.state, "word_service", None)
    if service is None:
        service = create_default_word_service()
        request.app.state.word_service = service
    return service


def _build_audio_filename(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_") or "speech"
    return f"{slug[:48]}.wav"


def _resolve_speech_text(pipeline: AppPipelineState, text: str | None) -> str:
    if text is not None:
        normalized = " ".join(text.split()).strip()
        if normalized:
            return normalized
        return ""

    phrase_parts = list(pipeline.word_builder.history)
    current_buffer = pipeline.decoder.accepted_letters.strip()
    if current_buffer:
        phrase_parts.append(current_buffer)
    return " ".join(part for part in phrase_parts if part.strip())


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "home.html",
        {"page": "home"},
    )


@router.get("/letters", response_class=HTMLResponse)
async def letters_page(request: Request) -> HTMLResponse:
    pipeline = _get_pipeline(request)
    return templates.TemplateResponse(
        request,
        "letters.html",
        {
            "page": "letters",
            "mode": pipeline.mode,
            "accepted_letters": pipeline.decoder.accepted_letters,
            "current_letter": pipeline.decoder.current_letter,
            "confidence": pipeline.decoder.current_confidence,
            "status": pipeline.last_status,
            "feature_dim": pipeline.extractor.feature_dim,
            "final_word": pipeline.word_builder.latest.final_word if pipeline.word_builder.latest else None,
            "final_word_status": pipeline.word_builder.latest.status if pipeline.word_builder.latest else None,
            "word_history": list(pipeline.word_builder.history),
        },
    )


@router.get("/words", response_class=HTMLResponse)
async def words_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "words.html",
        {"page": "words"},
    )


@router.get("/sequence", response_class=HTMLResponse)
async def sequence_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "sequence.html",
        {"page": "sequence"},
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    pipeline = _get_pipeline(request)
    word_service = _get_word_service(request)
    return HealthResponse(
        status="ok",
        mode=pipeline.mode,
        accepted_letters=pipeline.decoder.accepted_letters,
        feature_dim=pipeline.extractor.feature_dim,
        word_feature_dim=len(WORD_FEATURE_COLUMNS),
        word_vocab_size=len(WORD_LABELS),
        word_vocab=[display_word_label(w) or w for w in WORD_LABELS],
        final_word=pipeline.word_builder.latest.final_word if pipeline.word_builder.latest else None,
        final_word_status=pipeline.word_builder.latest.status if pipeline.word_builder.latest else None,
        word_history=list(pipeline.word_builder.history),
        sequence_frame_extract=word_service.sequence_frame_extract,
        sequence_sample_fps=float(word_service.sequence_sample_fps),
        sequence_max_input_side=word_service.sequence_max_input_side,
    )


@router.get("/api/words/sequence/capabilities")
async def word_sequence_capabilities(request: Request) -> dict[str, object]:
    """Discoverable settings for clients; incremental streaming is not implemented yet."""

    service = _get_word_service(request)
    return {
        "extract_modes": ["legacy", "seek", "fast", "linear"],
        "streaming": {
            "incremental": False,
            "hint": (
                "Send shorter overlapping clips with separate POSTs for lower latency until "
                "WebSocket/SSE lands; each request still runs full clip inference."
            ),
        },
        "active_settings": {
            "SEQUENCE_FRAME_EXTRACT": service.sequence_frame_extract,
            "SEQUENCE_SAMPLE_FPS": service.sequence_sample_fps,
            "SEQUENCE_MAX_INPUT_SIDE": service.sequence_max_input_side,
        },
    }


@router.post("/api/letters/predict", response_model=PredictResponse)
async def predict_letter(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    pipeline = _get_pipeline(request)
    image_bytes = await file.read()
    result = pipeline.process_image_bytes(image_bytes=image_bytes, filename=file.filename or "upload")
    return PredictResponse(**result)


@router.post("/api/words/predict", response_model=WordPredictResponse)
async def predict_word(request: Request, file: UploadFile = File(...)) -> WordPredictResponse:
    service = _get_word_service(request)
    video_bytes = await file.read()
    try:
        result = service.predict_from_video_bytes(video_bytes=video_bytes, filename=file.filename or "upload.mp4")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WordPredictResponse(**result)


@router.post("/api/words/sequence", response_model=WordSequenceResponse)
async def predict_word_sequence(request: Request, file: UploadFile = File(...)) -> WordSequenceResponse:
    service = _get_word_service(request)
    video_bytes = await file.read()
    try:
        result = service.predict_sequence_from_video_bytes(
            video_bytes=video_bytes,
            filename=file.filename or "upload.mp4",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WordSequenceResponse(**result)


@router.post("/api/reset", response_model=ResetResponse)
async def reset(request: Request) -> ResetResponse:
    pipeline = _get_pipeline(request)
    pipeline.reset()
    return ResetResponse(status="reset", accepted_letters="", word_history=[])


@router.post("/api/delete-last", response_model=ResetResponse)
async def delete_last(request: Request) -> ResetResponse:
    pipeline = _get_pipeline(request)
    pipeline.delete_last_letter()
    return ResetResponse(
        status="deleted",
        accepted_letters=pipeline.decoder.accepted_letters,
        word_history=list(pipeline.word_builder.history),
    )


@router.post("/api/tts/speak")
async def synthesize_speech(request: Request, payload: TTSRequest) -> Response:
    return _synthesize_speech_response(request=request, text=payload.text)


@router.get("/api/tts/speak")
async def synthesize_speech_get(request: Request, text: str | None = Query(default=None)) -> Response:
    return _synthesize_speech_response(request=request, text=text)


@router.post("/api/tts/preload")
async def preload_speech(request: Request) -> dict[str, str]:
    synthesizer = _get_speech_synthesizer(request)
    if not synthesizer.is_available:
        raise HTTPException(status_code=503, detail=synthesizer.unavailable_reason or "Speech synthesis is unavailable")

    try:
        synthesizer.preload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech preload failed: {exc}") from exc

    return {"status": "ready"}


def _synthesize_speech_response(request: Request, text: str | None) -> Response:
    pipeline = _get_pipeline(request)
    synthesizer = _get_speech_synthesizer(request)

    normalized = _resolve_speech_text(pipeline, text)
    if not normalized:
        raise HTTPException(status_code=400, detail="No finalized word or accepted letters available for speech synthesis")
    if not synthesizer.is_available:
        raise HTTPException(status_code=503, detail=synthesizer.unavailable_reason or "Speech synthesis is unavailable")

    try:
        audio_bytes = synthesizer.synthesize_wav_bytes(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}") from exc

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{_build_audio_filename(normalized)}"'},
    )
