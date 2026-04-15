from __future__ import annotations

import re

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from src.app.schemas import HealthResponse, PredictResponse, ResetResponse, TTSRequest
from src.app.state import AppPipelineState
from src.tts import PiperSpeechSynthesizer

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


def _build_audio_filename(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_") or "speech"
    return f"{slug[:48]}.wav"


def _resolve_speech_text(pipeline: AppPipelineState, text: str | None) -> str:
    if text is not None:
        normalized = " ".join(text.split()).strip()
        if normalized:
            return normalized
        return ""

    latest = pipeline.word_builder.latest
    if latest is not None and latest.final_word.strip():
        return latest.final_word.strip()

    return pipeline.decoder.accepted_letters.strip()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    pipeline = _get_pipeline(request)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
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


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    pipeline = _get_pipeline(request)
    return HealthResponse(
        status="ok",
        mode=pipeline.mode,
        accepted_letters=pipeline.decoder.accepted_letters,
        feature_dim=pipeline.extractor.feature_dim,
        final_word=pipeline.word_builder.latest.final_word if pipeline.word_builder.latest else None,
        final_word_status=pipeline.word_builder.latest.status if pipeline.word_builder.latest else None,
        word_history=list(pipeline.word_builder.history),
    )


@router.post("/api/letters/predict", response_model=PredictResponse)
async def predict_letter(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    pipeline = _get_pipeline(request)
    image_bytes = await file.read()
    result = pipeline.process_image_bytes(image_bytes=image_bytes, filename=file.filename or "upload")
    return PredictResponse(**result)


@router.post("/api/reset", response_model=ResetResponse)
async def reset(request: Request) -> ResetResponse:
    pipeline = _get_pipeline(request)
    pipeline.reset()
    return ResetResponse(status="reset", accepted_letters="", word_history=[])


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
