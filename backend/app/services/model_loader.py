"""
Production Model Loading & Inference Utility
Support-Ticket Sentiment Classifier

Provides production-ready loaders and inference services for:
1. Fine-tuned BERT model and tokenizer (from artifacts/bert/).
2. Fitted TF-IDF + Logistic Regression baseline pipeline (from artifacts/tfidf/).

Security Notice:
----------------
Scikit-learn / joblib serialization relies on Python's pickle protocol.
Loading joblib files from untrusted or unauthenticated sources can result
in arbitrary code execution. ONLY load joblib artifacts from trusted,
cryptographically verified, or local storage.

Architecture:
-------------
- Loads models ONCE during application startup (singleton pattern).
- Uses torch.inference_mode() for optimized, low-overhead tensor execution.
- Automatically leverages CUDA when available; gracefully falls back to CPU.
- Fails fast with clear FileNotFoundError / ValueError if artifacts are missing.
- NEVER silently falls back to an untrained model.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger("sentiment_classifier.model_loader")

# Robust resolution of project root directory (3 levels above this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BERT_DIR = Path(
    os.getenv("BERT_MODEL_DIR", str(PROJECT_ROOT / "artifacts" / "bert"))
)
DEFAULT_TFIDF_PATH = Path(
    os.getenv("TFIDF_MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_logistic_regression.joblib"))
)


def load_bert_model(
    model_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """
    Loads fine-tuned BERT model and its matching tokenizer from disk.

    Args:
        model_dir: Path to directory containing config.json, model.safetensors, and tokenizer files.
        device: Target torch device (cuda / cpu). If None, auto-detected.

    Returns:
        Tuple of (tokenizer, model, active_device)

    Raises:
        FileNotFoundError: If model_dir or any required Hugging Face files are missing.
    """
    path = Path(model_dir) if model_dir else DEFAULT_BERT_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"[ModelLoader] BERT artifact directory does not exist: {path.resolve()}. "
            "Please ensure artifacts/bert is populated with the fine-tuned model before starting the service."
        )

    config_file = path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"[ModelLoader] Missing config.json in BERT directory: {path.resolve()}"
        )

    # Check for weights file (.safetensors or .bin)
    weights_safetensors = path / "model.safetensors"
    weights_bin = path / "pytorch_model.bin"
    if not weights_safetensors.exists() and not weights_bin.exists():
        raise FileNotFoundError(
            f"[ModelLoader] No model weights (model.safetensors or pytorch_model.bin) found in {path.resolve()}"
        )

    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading BERT tokenizer and model from {path} onto {device}...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))

    # Move to target device and enable evaluation mode
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()

    load_time_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f"BERT loaded successfully onto {device} in {load_time_ms:.2f} ms.")

    return tokenizer, model, device


def load_tfidf_pipeline(
    pipeline_path: Optional[Union[str, Path]] = None
) -> Pipeline:
    """
    Loads fitted TF-IDF + Logistic Regression scikit-learn Pipeline from disk.

    Security Warning:
        Loading joblib files executes deserialization code. Only load files from trusted sources.

    Args:
        pipeline_path: Path to serialized .joblib pipeline file.

    Returns:
        Fitted scikit-learn Pipeline object.

    Raises:
        FileNotFoundError: If pipeline file does not exist.
        ValueError: If loaded object is not a valid scikit-learn Pipeline or estimator.
    """
    path = Path(pipeline_path) if pipeline_path else DEFAULT_TFIDF_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"[ModelLoader] TF-IDF baseline artifact not found: {path.resolve()}. "
            "Please ensure artifacts/tfidf/tfidf_logistic_regression.joblib is present."
        )

    logger.info(f"Loading TF-IDF pipeline from {path}...")
    t0 = time.perf_counter()

    try:
        pipeline = joblib.load(str(path))
    except Exception as exc:
        raise ValueError(
            f"[ModelLoader] Failed to deserialize joblib artifact at {path}: {exc}"
        ) from exc

    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            f"[ModelLoader] Loaded artifact from {path} is not a valid classifier pipeline."
        )

    load_time_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f"TF-IDF pipeline loaded successfully in {load_time_ms:.2f} ms.")
    return pipeline


class ModelService:
    """
    Singleton inference manager that maintains preloaded models in memory
    for fast, low-latency API serving.
    """

    def __init__(
        self,
        bert_dir: Optional[Union[str, Path]] = None,
        tfidf_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        lazy: bool = False,
    ):
        self.bert_dir = bert_dir
        self.tfidf_path = tfidf_path
        self.device = device

        self.tokenizer: Optional[AutoTokenizer] = None
        self.bert_model: Optional[AutoModelForSequenceClassification] = None
        self.active_device: Optional[torch.device] = None
        self.tfidf_pipeline: Optional[Pipeline] = None

        if not lazy:
            self.load_all()

    def load_all(self) -> None:
        """Loads both models during service startup."""
        self.tokenizer, self.bert_model, self.active_device = load_bert_model(
            self.bert_dir, self.device
        )
        self.tfidf_pipeline = load_tfidf_pipeline(self.tfidf_path)

    def predict_bert(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        """
        Runs sentiment classification using the fine-tuned BERT model.

        Returns:
            Dictionary containing sentiment, confidence, confidencePercentage,
            probabilities, latencyMs, and extracted tokens.
        """
        if self.bert_model is None or self.tokenizer is None:
            raise RuntimeError("[ModelService] BERT model has not been loaded.")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text cannot be empty.")

        t0 = time.perf_counter()
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        if self.active_device and self.active_device.type == "cuda":
            inputs = {k: v.to(self.active_device) for k, v in inputs.items()}
            torch.cuda.synchronize()

        with torch.inference_mode():
            logits = self.bert_model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        if self.active_device and self.active_device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(probs.argmax())
        # Use config id2label mapping if present, else standard binary convention
        id2label = getattr(self.bert_model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})
        sentiment = id2label.get(str(pred_idx), id2label.get(pred_idx, "POSITIVE" if pred_idx == 1 else "NEGATIVE"))

        confidence = float(probs[pred_idx])
        tokens = [t for t in clean_text.split() if t][:16]

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "confidencePercentage": int(round(confidence * 100)),
            "probabilities": {
                "NEGATIVE": round(float(probs[0]), 4),
                "POSITIVE": round(float(probs[1]), 4),
            },
            "model": "BERT Fine-Tuned (bert-base-uncased)",
            "modelKey": "bert",
            "device": str(self.active_device).upper() if self.active_device else "CPU",
            "latencyMs": round(latency_ms, 2),
            "tokens": tokens,
        }

    def predict_tfidf(self, text: str) -> Dict[str, Any]:
        """
        Runs sentiment classification using the classical TF-IDF baseline.

        Returns:
            Dictionary containing sentiment, confidence, confidencePercentage,
            probabilities, latencyMs, and extracted tokens.
        """
        if self.tfidf_pipeline is None:
            raise RuntimeError("[ModelService] TF-IDF pipeline has not been loaded.")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text cannot be empty.")

        t0 = time.perf_counter()
        probs = self.tfidf_pipeline.predict_proba([clean_text])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(probs.argmax())
        sentiment = "POSITIVE" if pred_idx == 1 else "NEGATIVE"
        confidence = float(probs[pred_idx])
        tokens = [t for t in clean_text.split() if t][:16]

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "confidencePercentage": int(round(confidence * 100)),
            "probabilities": {
                "NEGATIVE": round(float(probs[0]), 4),
                "POSITIVE": round(float(probs[1]), 4),
            },
            "model": "TF-IDF + Logistic Regression",
            "modelKey": "tfidf",
            "device": "CPU",
            "latencyMs": round(latency_ms, 2),
            "tokens": tokens,
        }

    def predict(self, text: str, model_type: str = "bert") -> Dict[str, Any]:
        """
        Unified prediction router.

        Args:
            text: Customer support text input.
            model_type: 'bert', 'tfidf', or 'both'.
        """
        mode = model_type.lower()
        if mode == "both":
            return {
                "bert": self.predict_bert(text),
                "tfidf": self.predict_tfidf(text),
            }
        elif mode == "tfidf":
            return self.predict_tfidf(text)
        else:
            return self.predict_bert(text)


# Global singleton instance for FastAPI application lifecycle
_GLOBAL_MODEL_SERVICE: Optional[ModelService] = None


def get_model_service(lazy: bool = False) -> ModelService:
    """Returns or initializes the singleton ModelService instance."""
    global _GLOBAL_MODEL_SERVICE
    if _GLOBAL_MODEL_SERVICE is None:
        _GLOBAL_MODEL_SERVICE = ModelService(lazy=lazy)
    return _GLOBAL_MODEL_SERVICE
