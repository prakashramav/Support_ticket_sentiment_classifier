import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root and backend dir are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logger = logging.getLogger("sentiment_classifier.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

try:
    from backend.app.services.model_loader import ModelService, get_model_service, preprocess_input_text
except ImportError:
    from app.services.model_loader import ModelService, get_model_service, preprocess_input_text

# Global model service instance
model_service: Optional[ModelService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    logger.info("Initializing Sentiment Classifier Backend...")
    try:
        model_service = get_model_service(lazy=False, strict=False)
        diag = model_service.get_diagnostics()
        logger.info(
            "Startup model readiness -> bert_loaded=%s, tfidf_loaded=%s, bert_temp=%.4f",
            diag["bert_loaded"],
            diag["tfidf_loaded"],
            diag["bert_temperature"],
        )
        if diag["bert_error"]:
            logger.error("BERT unavailable: %s", diag["bert_error"])
        if diag["tfidf_error"]:
            logger.error("TF-IDF unavailable: %s", diag["tfidf_error"])
    except Exception as exc:
        logger.exception("Critical failure initializing ModelService: %s", exc)
        model_service = None
    yield
    logger.info("Shutting down Sentiment Classifier Backend...")


app = FastAPI(
    title="Support-Ticket Sentiment Classifier API",
    description="Production inference API for fine-tuned BERT and TF-IDF baseline sentiment classifiers (SST-2 GLUE benchmark)",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS cleanly for local Next.js dev, Vercel deployments, and optional FRONTEND_URL
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
extra_origin = os.getenv("FRONTEND_URL", "").strip()
if extra_origin:
    allowed_origins.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Root endpoint welcoming visitors and providing API navigation."""
    return {
        "status": "online",
        "message": "Support-Ticket Sentiment Classifier Backend is running.",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    device: str = "cpu"
    models_loaded: bool = False
    bert_loaded: bool = False
    tfidf_loaded: bool = False
    available_models: List[str] = []
    bert_error: Optional[str] = None
    tfidf_error: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input customer support text to analyze")
    model: Optional[str] = Field("bert", description="'bert', 'tfidf', or 'both'")


class SingleModelResult(BaseModel):
    text: Optional[str] = None
    sentiment: Optional[str] = None
    rawClassId: Optional[int] = None
    rawLabel: Optional[str] = None
    rawLogits: Optional[List[float]] = None
    modelProbability: Optional[float] = None
    calibratedConfidence: Optional[float] = None
    confidence: Optional[float] = None
    confidencePercentage: Optional[int] = None
    confidenceBand: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    calibratedProbabilities: Optional[Dict[str, float]] = None
    calibration: Optional[Dict[str, Any]] = None
    isTruncated: Optional[bool] = None
    vocabularyHits: Optional[int] = None
    hasOutOfVocabularyWarning: Optional[bool] = None
    source: Optional[str] = None
    model: Optional[str] = None
    modelKey: str
    device: Optional[str] = "CPU"
    latencyMs: Optional[Union[float, int]] = None
    tokens: Optional[List[str]] = None
    unavailable: Optional[bool] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    text: Optional[str] = None
    sentiment: Optional[str] = None
    rawClassId: Optional[int] = None
    rawLabel: Optional[str] = None
    rawLogits: Optional[List[float]] = None
    modelProbability: Optional[float] = None
    calibratedConfidence: Optional[float] = None
    confidence: Optional[float] = None
    confidencePercentage: Optional[int] = None
    confidenceBand: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    calibratedProbabilities: Optional[Dict[str, float]] = None
    calibration: Optional[Dict[str, Any]] = None
    isTruncated: Optional[bool] = None
    vocabularyHits: Optional[int] = None
    hasOutOfVocabularyWarning: Optional[bool] = None
    source: Optional[str] = None
    model: Optional[str] = None
    modelKey: Optional[str] = None
    device: Optional[str] = None
    latencyMs: Optional[Union[float, int]] = None
    tokens: Optional[List[str]] = None
    bert: Optional[SingleModelResult] = None
    tfidf: Optional[SingleModelResult] = None


@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    global model_service
    device_str = "cpu"
    bert_ok = False
    tfidf_ok = False
    available_models: List[str] = []
    bert_error = "ModelService not initialized"
    tfidf_error = "ModelService not initialized"
    diagnostics = None

    if model_service is not None:
        bert_ok = model_service.bert_model is not None
        tfidf_ok = model_service.tfidf_pipeline is not None
        bert_error = model_service.bert_error
        tfidf_error = model_service.tfidf_error
        diagnostics = model_service.get_diagnostics()
        if model_service.active_device:
            device_str = str(model_service.active_device).lower()
        if bert_ok:
            available_models.append("bert")
        if tfidf_ok:
            available_models.append("tfidf")

    models_ok = bert_ok and tfidf_ok
    if models_ok:
        status = "ok"
    elif bert_ok or tfidf_ok:
        status = "degraded"
    else:
        status = "unavailable"

    return {
        "status": status,
        "version": "1.0.0",
        "device": device_str,
        "models_loaded": models_ok,
        "bert_loaded": bert_ok,
        "tfidf_loaded": tfidf_ok,
        "available_models": available_models,
        "bert_error": bert_error,
        "tfidf_error": tfidf_error,
        "diagnostics": diagnostics,
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict_sentiment(payload: PredictRequest):
    global model_service

    try:
        clean_text = preprocess_input_text(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    requested_model = (payload.model or "bert").strip().lower()
    if requested_model not in {"bert", "tfidf", "both"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{payload.model}'. Choose 'bert', 'tfidf', or 'both'.",
        )

    if model_service is None or not model_service.has_any_model():
        raise HTTPException(
            status_code=503,
            detail="No trained ML models are currently loaded on the server. Check /api/v1/health diagnostics.",
        )

    if requested_model == "bert":
        if model_service.bert_model is None:
            raise HTTPException(
                status_code=503,
                detail=f"BERT model is unavailable: {model_service.bert_error or 'not loaded'}",
            )
        return model_service.predict_bert(clean_text)

    if requested_model == "tfidf":
        if model_service.tfidf_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail=f"TF-IDF model is unavailable: {model_service.tfidf_error or 'not loaded'}",
            )
        return model_service.predict_tfidf(clean_text)

    # requested_model == "both": return live model results, or transparent error block if one model is absent
    response_payload: Dict[str, Any] = {}
    if model_service.bert_model is not None:
        response_payload["bert"] = model_service.predict_bert(clean_text)
    else:
        response_payload["bert"] = {
            "text": clean_text,
            "modelKey": "bert",
            "model": "BERT Fine-Tuned (bert-base-uncased)",
            "unavailable": True,
            "error": model_service.bert_error or "BERT model weights are not loaded on this server.",
        }

    if model_service.tfidf_pipeline is not None:
        response_payload["tfidf"] = model_service.predict_tfidf(clean_text)
    else:
        response_payload["tfidf"] = {
            "text": clean_text,
            "modelKey": "tfidf",
            "model": "TF-IDF + LogisticRegression",
            "unavailable": True,
            "error": model_service.tfidf_error or "TF-IDF model is not loaded on this server.",
        }

    return response_payload
