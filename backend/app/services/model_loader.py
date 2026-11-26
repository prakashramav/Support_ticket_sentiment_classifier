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
- Independently tracks BERT and TF-IDF readiness (never allows one failure to mask the other).
- Uses torch.inference_mode() for optimized, low-overhead tensor execution.
- Validates explicit class/label mappings (id2label and classes_) to guarantee probability alignment.
- Distinguishes raw model probability from post-hoc temperature-scaled calibrated confidence.
- Rejects empty, whitespace-only, or non-alphanumeric inputs rather than returning biased priors.
- NEVER falls back to an untrained model or toy keyword heuristics.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("sentiment_classifier.model_loader")

# Robust resolution of project root directory (3 levels above this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BERT_DIR = Path(
    os.getenv("BERT_MODEL_DIR", str(PROJECT_ROOT / "artifacts" / "bert"))
)
DEFAULT_TFIDF_PATH = Path(
    os.getenv(
        "TFIDF_MODEL_PATH",
        str(PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_logistic_regression.joblib"),
    )
)
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"

# Temperature scaling factor fitted via NLL minimization on held-out SST-2 validation logits
# Reduces NLL (0.2428 -> 0.2000) and ECE (0.0489 -> 0.0263) without changing predicted class argmax.
DEFAULT_BERT_TEMPERATURE = 1.6650
EXPECTED_BINARY_LABELS = {"NEGATIVE", "POSITIVE"}
MAX_TEXT_LENGTH = 5000


def _load_default_temperature() -> float:
    """Reads validated BERT calibration temperature from env or model_metadata.json if present."""
    env_temp = os.getenv("BERT_CALIBRATION_TEMPERATURE")
    if env_temp:
        try:
            val = float(env_temp)
            if val > 0:
                return val
        except ValueError:
            pass

    if DEFAULT_METADATA_PATH.exists():
        try:
            with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            val = (
                meta.get("models", {})
                .get("bert", {})
                .get("calibration", {})
                .get("temperature")
            )
            if val is not None and float(val) > 0:
                return float(val)
        except Exception:
            pass

    return DEFAULT_BERT_TEMPERATURE


def preprocess_input_text(text: str, max_chars: int = MAX_TEXT_LENGTH) -> str:
    """
    Validates and normalizes input text consistently with training expectations.

    - Preserves original casing (bert-base-uncased & TfidfVectorizer lowercase consistently).
    - Preserves negations, punctuation, and sentence syntax needed by contextual models.
    - Rejects empty, whitespace-only, non-string, oversized, or zero-word inputs.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a valid string.")

    clean_text = " ".join(text.split())
    if not clean_text:
        raise ValueError("Input text cannot be empty.")

    if len(clean_text) > max_chars:
        raise ValueError(f"Input text exceeds maximum allowed length of {max_chars} characters.")

    if not any(ch.isalnum() for ch in clean_text):
        raise ValueError("Input text must contain readable words or alphanumeric characters.")

    return clean_text


def _normalize_bert_id2label(config: Any) -> Dict[int, str]:
    """
    Extracts and validates explicit integer-keyed id2label mapping from BERT config.
    Ensures exact alignment between logit indices, softmax probabilities, and class labels.
    """
    raw_id2label = getattr(config, "id2label", None)
    if not raw_id2label:
        raise ValueError("[ModelLoader] BERT config is missing required 'id2label' mapping.")

    normalized: Dict[int, str] = {}
    for k, v in raw_id2label.items():
        try:
            idx = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[ModelLoader] Invalid non-integer label key '{k}' in id2label.") from exc
        normalized[idx] = str(v).strip().upper()

    if set(normalized.values()) != EXPECTED_BINARY_LABELS or len(normalized) != 2:
        raise ValueError(
            f"[ModelLoader] Unexpected BERT id2label mapping: {normalized}. "
            f"Expected exact binary labels {sorted(EXPECTED_BINARY_LABELS)}."
        )

    return normalized


def _normalize_tfidf_class_mapping(pipeline: Pipeline) -> Dict[int, str]:
    """
    Extracts and validates class column indices from the scikit-learn classifier's classes_.
    Guarantees probabilities['NEGATIVE'] and probabilities['POSITIVE'] match fitted class order.
    """
    classes = getattr(pipeline, "classes_", None)
    if classes is None or len(classes) != 2:
        raise ValueError(f"[ModelLoader] Invalid or missing classes_ on TF-IDF pipeline: {classes}")

    col_to_label: Dict[int, str] = {}
    for col_idx, cls_val in enumerate(classes):
        if cls_val in (0, "0") or str(cls_val).strip().upper() == "NEGATIVE":
            col_to_label[col_idx] = "NEGATIVE"
        elif cls_val in (1, "1") or str(cls_val).strip().upper() == "POSITIVE":
            col_to_label[col_idx] = "POSITIVE"
        else:
            raise ValueError(
                f"[ModelLoader] Unrecognized class value '{cls_val}' in TF-IDF pipeline classes_: {classes}"
            )

    if set(col_to_label.values()) != EXPECTED_BINARY_LABELS:
        raise ValueError(
            f"[ModelLoader] TF-IDF pipeline classes_ {classes} did not map cleanly to NEGATIVE/POSITIVE."
        )

    return col_to_label


def _compute_confidence_band(calibrated_confidence: float) -> str:
    """Categorizes calibrated confidence into interpretable reliability bands."""
    if calibrated_confidence >= 0.85:
        return "HIGH"
    if calibrated_confidence >= 0.65:
        return "MODERATE"
    return "LOW"


def load_bert_model(
    model_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """
    Loads fine-tuned BERT model and its matching tokenizer from local disk or optional BERT_HF_REPO.

    Args:
        model_dir: Path to directory containing config.json, model.safetensors, and tokenizer files.
        device: Target torch device (cuda / cpu). If None, auto-detected.

    Returns:
        Tuple of (tokenizer, model, active_device)

    Raises:
        FileNotFoundError: If neither valid local weights nor BERT_HF_REPO are available.
        ValueError: If loaded model config has invalid id2label mapping.
    """
    path = Path(model_dir) if model_dir else DEFAULT_BERT_DIR
    hf_repo = os.getenv("BERT_HF_REPO", "").strip()

    has_local_weights = (
        path.exists()
        and (path / "config.json").exists()
        and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
    )

    if not has_local_weights:
        if hf_repo and model_dir is None:
            source_id = hf_repo
            logger.info(f"Local BERT weights absent at {path}; loading from BERT_HF_REPO={hf_repo}...")
        else:
            if not path.exists():
                raise FileNotFoundError(
                    f"[ModelLoader] BERT artifact directory does not exist: {path.resolve()}. "
                    "Please ensure artifacts/bert is populated with the fine-tuned model before starting the service."
                )
            if not (path / "config.json").exists():
                raise FileNotFoundError(
                    f"[ModelLoader] Missing config.json in BERT directory: {path.resolve()}"
                )
            raise FileNotFoundError(
                f"[ModelLoader] No model weights (model.safetensors or pytorch_model.bin) found in {path.resolve()}"
            )
    else:
        source_id = str(path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading BERT tokenizer and model from {source_id} onto {device}...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(source_id)
    model = AutoModelForSequenceClassification.from_pretrained(source_id)

    # Validate explicit binary label mapping on startup
    _normalize_bert_id2label(model.config)

    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()

    load_time_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f"BERT loaded successfully onto {device} in {load_time_ms:.2f} ms.")

    return tokenizer, model, device


def load_tfidf_pipeline(
    pipeline_path: Optional[Union[str, Path]] = None,
) -> Pipeline:
    """
    Loads fitted TF-IDF + Logistic Regression scikit-learn Pipeline from disk.

    Security Warning:
        Loading joblib files executes deserialization code. Only load files from trusted sources.
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

    _normalize_tfidf_class_mapping(pipeline)

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
        strict: bool = False,
        bert_temperature: Optional[float] = None,
    ):
        self.bert_dir = bert_dir
        self.tfidf_path = tfidf_path
        self.device = device

        self.tokenizer: Optional[AutoTokenizer] = None
        self.bert_model: Optional[AutoModelForSequenceClassification] = None
        self.bert_id2label: Optional[Dict[int, str]] = None
        self.active_device: Optional[torch.device] = None
        self.bert_error: Optional[str] = None

        self.tfidf_pipeline: Optional[Pipeline] = None
        self.tfidf_col_to_label: Optional[Dict[int, str]] = None
        self.tfidf_error: Optional[str] = None

        self.bert_temperature: float = (
            float(bert_temperature)
            if bert_temperature is not None and bert_temperature > 0
            else _load_default_temperature()
        )

        if not lazy:
            self.load_all(strict=strict)

    def load_all(self, strict: bool = False) -> None:
        """
        Independently loads BERT and TF-IDF models so failure of one artifact
        does not prevent serving the other unless strict=True.
        """
        try:
            self.tokenizer, self.bert_model, self.active_device = load_bert_model(
                self.bert_dir, self.device
            )
            self.bert_id2label = _normalize_bert_id2label(self.bert_model.config)
            self.bert_error = None
        except Exception as exc:
            self.tokenizer = None
            self.bert_model = None
            self.bert_id2label = None
            self.bert_error = str(exc)
            logger.error(f"Failed to load BERT model: {exc}")
            if strict:
                raise

        try:
            self.tfidf_pipeline = load_tfidf_pipeline(self.tfidf_path)
            self.tfidf_col_to_label = _normalize_tfidf_class_mapping(self.tfidf_pipeline)
            self.tfidf_error = None
        except Exception as exc:
            self.tfidf_pipeline = None
            self.tfidf_col_to_label = None
            self.tfidf_error = str(exc)
            logger.error(f"Failed to load TF-IDF pipeline: {exc}")
            if strict:
                raise

    def has_any_model(self) -> bool:
        """Returns True if at least one ML model is loaded and ready for inference."""
        return self.bert_model is not None or self.tfidf_pipeline is not None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns inspectable model configuration and label mapping metadata."""
        clf_type = None
        vocab_size = None
        classes_list = None
        if self.tfidf_pipeline is not None:
            vec = self.tfidf_pipeline[0]
            clf = self.tfidf_pipeline[-1]
            clf_type = type(clf).__name__
            vocab_size = len(getattr(vec, "vocabulary_", {}))
            classes_list = [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in self.tfidf_pipeline.classes_]

        return {
            "bert_loaded": self.bert_model is not None,
            "bert_error": self.bert_error,
            "bert_id2label": {str(k): v for k, v in self.bert_id2label.items()} if self.bert_id2label else None,
            "bert_temperature": round(self.bert_temperature, 4),
            "tfidf_loaded": self.tfidf_pipeline is not None,
            "tfidf_error": self.tfidf_error,
            "tfidf_classifier_type": clf_type,
            "tfidf_classes": classes_list,
            "tfidf_vocab_size": vocab_size,
        }

    def predict_bert(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        """
        Runs sentiment classification using the fine-tuned BERT model.
        Exposes both raw softmax probability and temperature-scaled calibrated confidence,
        along with raw predicted class ID and logits for full transparency.
        """
        if self.bert_model is None or self.tokenizer is None:
            reason = f" ({self.bert_error})" if self.bert_error else ""
            raise RuntimeError(f"[ModelService] BERT model has not been loaded{reason}.")

        clean_text = preprocess_input_text(text)
        id2label = self.bert_id2label or _normalize_bert_id2label(self.bert_model.config)

        t0 = time.perf_counter()
        full_tokens = self.tokenizer.tokenize(clean_text)
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Effective content tokens = len(input_ids[0]) minus [CLS] and [SEP]
        encoded_len = int(inputs["input_ids"].shape[-1])
        is_truncated = len(full_tokens) > max(encoded_len - 2, 0)

        if self.active_device and self.active_device.type == "cuda":
            inputs = {k: v.to(self.active_device) for k, v in inputs.items()}
            torch.cuda.synchronize()

        with torch.inference_mode():
            logits = self.bert_model(**inputs).logits
            raw_probs_t = F.softmax(logits, dim=-1).squeeze(0)
            cal_probs_t = F.softmax(logits / self.bert_temperature, dim=-1).squeeze(0)

        if self.active_device and self.active_device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits_np = logits.squeeze(0).detach().cpu().numpy()
        raw_probs = raw_probs_t.detach().cpu().numpy()
        cal_probs = cal_probs_t.detach().cpu().numpy()

        raw_class_id = int(np.argmax(raw_probs))
        raw_label = id2label[raw_class_id]

        raw_prob_map = {
            label: round(float(raw_probs[idx]), 4)
            for idx, label in id2label.items()
        }
        cal_prob_map = {
            label: round(float(cal_probs[idx]), 4)
            for idx, label in id2label.items()
        }

        model_prob = round(float(raw_probs[raw_class_id]), 4)
        calibrated_conf = round(float(cal_probs[raw_class_id]), 4)
        display_tokens = full_tokens[:24] if full_tokens else clean_text.split()[:16]

        return {
            "text": clean_text,
            "sentiment": raw_label,
            "rawClassId": raw_class_id,
            "rawLabel": raw_label,
            "rawLogits": [round(float(x), 4) for x in logits_np.tolist()],
            "modelProbability": model_prob,
            "calibratedConfidence": calibrated_conf,
            "confidence": calibrated_conf,
            "confidencePercentage": int(round(calibrated_conf * 100)),
            "confidenceBand": _compute_confidence_band(calibrated_conf),
            "probabilities": raw_prob_map,
            "calibratedProbabilities": cal_prob_map,
            "calibration": {
                "method": "temperature_scaling",
                "temperature": round(self.bert_temperature, 4),
            },
            "isTruncated": is_truncated,
            "source": "live-model",
            "model": "BERT Fine-Tuned (bert-base-uncased)",
            "modelKey": "bert",
            "device": str(self.active_device).upper() if self.active_device else "CPU",
            "latencyMs": round(latency_ms, 2),
            "tokens": display_tokens,
        }

    def predict_tfidf(self, text: str) -> Dict[str, Any]:
        """
        Runs sentiment classification using the classical TF-IDF + Logistic Regression pipeline.
        Aligns probabilities explicitly using pipeline.classes_ and reports OOV warnings.
        """
        if self.tfidf_pipeline is None:
            reason = f" ({self.tfidf_error})" if self.tfidf_error else ""
            raise RuntimeError(f"[ModelService] TF-IDF pipeline has not been loaded{reason}.")

        clean_text = preprocess_input_text(text)
        col_to_label = self.tfidf_col_to_label or _normalize_tfidf_class_mapping(self.tfidf_pipeline)

        vectorizer = self.tfidf_pipeline[0]
        classifier = self.tfidf_pipeline[-1]

        t0 = time.perf_counter()
        X_vec = vectorizer.transform([clean_text])
        probs = classifier.predict_proba(X_vec)[0]
        decision_score = (
            float(classifier.decision_function(X_vec)[0])
            if hasattr(classifier, "decision_function")
            else None
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_col = int(np.argmax(probs))
        raw_class_val = self.tfidf_pipeline.classes_[pred_col]
        raw_class_id = int(raw_class_val) if isinstance(raw_class_val, (np.integer, int)) else pred_col
        sentiment = col_to_label[pred_col]

        prob_map = {
            label: round(float(probs[col_idx]), 4)
            for col_idx, label in col_to_label.items()
        }

        model_prob = round(float(probs[pred_col]), 4)
        vocab_hits = int(X_vec.nnz)
        has_oov_warning = vocab_hits == 0

        analyzer = vectorizer.build_analyzer()
        matched_terms = [tok for tok in analyzer(clean_text) if tok in vectorizer.vocabulary_]
        display_tokens = matched_terms[:16] if matched_terms else clean_text.split()[:16]

        return {
            "text": clean_text,
            "sentiment": sentiment,
            "rawClassId": raw_class_id,
            "rawLabel": sentiment,
            "rawLogits": [round(decision_score, 4)] if decision_score is not None else None,
            "modelProbability": model_prob,
            "calibratedConfidence": model_prob,
            "confidence": model_prob,
            "confidencePercentage": int(round(model_prob * 100)),
            "confidenceBand": "LOW" if has_oov_warning else _compute_confidence_band(model_prob),
            "probabilities": prob_map,
            "calibratedProbabilities": prob_map,
            "calibration": {
                "method": "logistic_sigmoid",
                "temperature": 1.0,
            },
            "vocabularyHits": vocab_hits,
            "hasOutOfVocabularyWarning": has_oov_warning,
            "source": "live-model",
            "model": f"TF-IDF + {type(classifier).__name__}",
            "modelKey": "tfidf",
            "device": "CPU",
            "latencyMs": round(latency_ms, 2),
            "tokens": display_tokens,
        }

    def predict(self, text: str, model_type: str = "bert") -> Dict[str, Any]:
        """
        Unified prediction router.

        Args:
            text: Customer support text input.
            model_type: 'bert', 'tfidf', or 'both'.
        """
        if not isinstance(model_type, str):
            raise ValueError("model_type must be one of 'bert', 'tfidf', or 'both'.")

        mode = model_type.strip().lower()
        if mode not in {"bert", "tfidf", "both"}:
            raise ValueError(f"Unsupported model '{model_type}'. Expected 'bert', 'tfidf', or 'both'.")

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


def get_model_service(lazy: bool = False, strict: bool = False) -> ModelService:
    """Returns or initializes the singleton ModelService instance."""
    global _GLOBAL_MODEL_SERVICE
    if _GLOBAL_MODEL_SERVICE is None:
        _GLOBAL_MODEL_SERVICE = ModelService(lazy=lazy, strict=strict)
    return _GLOBAL_MODEL_SERVICE
