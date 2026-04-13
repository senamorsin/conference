from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.app.schemas import HealthResponse, PredictResponse, ResetResponse
from src.app.state import AppPipelineState

router = APIRouter()
templates = Jinja2Templates(directory="src/app/templates")


def _get_pipeline(request: Request) -> AppPipelineState:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline state is not initialized")
    return pipeline


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
