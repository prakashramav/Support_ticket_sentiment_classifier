"""
Production Model Loading & Inference Utility
Support-Ticket Sentiment Classifier

Provides production-ready loaders and inference services for:
1. Fine-tuned BERT model and tokenizer (from artifacts/bert/):
   - Uses PyTorch + Hugging Face Transformers (CUDA/CPU) when torch + model.safetensors are present.
   - Automatically uses ONNX Runtime INT8 (model_int8.onnx / model_int8.onnx.xz) when running in
     PyTorch-free or 512MB RAM cloud environments (such as Render free tier).
2. Fitted TF-IDF + Logistic Regression baseline pipeline (from artifacts/tfidf/).

Security Notice:
----------------
Scikit-learn / joblib serialization relies on Python's pickle protocol.
Loading joblib files from untrusted or unauthenticated sources can result
in arbitrary code execution. ONLY load joblib artifacts from trusted,
cryptographically verified, or local storage.
"""

from __future__ import annotations

import json
import logging
import lzma
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    PYTORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer as FastTokenizer

    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    FastTokenizer = None
    ONNX_AVAILABLE = False

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


def _softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable numpy softmax with optional temperature scaling."""
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


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


class _OnnxCpuDevice:
    """Minimal device descriptor compatible with `.type` and `str(device)` checks."""

    type: str = "cpu"

    def __str__(self) -> str:
        return "cpu"

    def __repr__(self) -> str:
        return "device(type='cpu')"


class OnnxTokenizerWrapper:
    """Lightweight tokenizer wrapper around Hugging Face fast `tokenizers.Tokenizer` JSON."""

    def __init__(self, tokenizer_json_path: Path):
        if not ONNX_AVAILABLE:
            raise ImportError("tokenizers package is required for ONNX BERT tokenization.")
        self._raw_tok = FastTokenizer.from_file(str(tokenizer_json_path))
        self._sub_tok = FastTokenizer.from_file(str(tokenizer_json_path))
        self._sub_tok.no_truncation()

    def tokenize(self, text: str) -> List[str]:
        enc = self._sub_tok.encode(text, add_special_tokens=False)
        return list(enc.tokens)

    def encode_numpy(self, text: str, max_length: int = 128) -> Dict[str, np.ndarray]:
        self._raw_tok.enable_truncation(max_length=max_length)
        enc = self._raw_tok.encode(text, add_special_tokens=True)
        return {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }


class OnnxBertModel:
    """Lightweight ONNX Runtime wrapper exposing config, training=False, and logits inference."""

    def __init__(self, onnx_path: Path, config_data: Dict[str, Any]):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is required for ONNX BERT inference.")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.config = SimpleNamespace(**config_data)
        self.training = False
        self.is_onnx = True

    def predict_logits(self, encoded_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        feed = {
            name: encoded_inputs[name]
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in encoded_inputs
        }
        return self.session.run(["logits"], feed)[0]


def _ensure_onnx_extracted(onnx_path: Path, xz_path: Path) -> Path:
    """Ensures uncompressed ONNX weights exist on disk, extracting cleanly from .xz if needed."""
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return onnx_path

    if not xz_path.exists():
        raise FileNotFoundError(
            f"[ModelLoader] Neither {onnx_path.name} nor {xz_path.name} exists in {onnx_path.parent}"
        )

    logger.info(f"Extracting compressed ONNX BERT weights from {xz_path.name}...")
    t0 = time.perf_counter()
    with lzma.open(xz_path, "rb") as src:
        raw_bytes = src.read()

    # Atomic write into same directory (or temp dir if filesystem is read-only)
    target_dir = onnx_path.parent
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(target_dir), prefix=".tmp_bert_", suffix=".onnx")
        try:
            with os.fdopen(fd, "wb") as dst:
                dst.write(raw_bytes)
            os.replace(tmp_name, onnx_path)
            dest_path = onnx_path
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise
    except OSError:
        fallback_path = Path(tempfile.gettempdir()) / "model_int8.onnx"
        with open(fallback_path, "wb") as dst:
            dst.write(raw_bytes)
        dest_path = fallback_path

    logger.info(f"Extracted ONNX BERT weights ({len(raw_bytes) / (1024**2):.1f} MB) in {(time.perf_counter() - t0):.2f}s.")
    return dest_path


def load_bert_model(
    model_dir: Optional[Union[str, Path]] = None,
    device: Optional[Any] = None,
    backend: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """
    Loads fine-tuned BERT model and matching tokenizer from local artifacts (or BERT_HF_REPO).

    Supports:
    - PyTorch + Transformers backend (uses artifacts/bert/model.safetensors on CUDA or CPU)
    - Lightweight ONNX Runtime INT8 backend (uses artifacts/bert/model_int8.onnx[.xz] on CPU)

    Args:
        model_dir: Directory containing BERT artifacts.
        device: Optional target device for PyTorch ('cuda'/'cpu' or torch.device).
        backend: 'auto' (default), 'pytorch', or 'onnx'.
    """
    path = Path(model_dir) if model_dir else DEFAULT_BERT_DIR
    requested_backend = (backend or os.getenv("BERT_BACKEND", "auto")).strip().lower()
    hf_repo = os.getenv("BERT_HF_REPO", "").strip()

    config_file = path / "config.json"
    has_pytorch_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    onnx_file = path / "model_int8.onnx"
    onnx_xz_file = path / "model_int8.onnx.xz"
    has_onnx_weights = onnx_file.exists() or onnx_xz_file.exists()

    if not path.exists() and not (hf_repo and model_dir is None):
        raise FileNotFoundError(
            f"[ModelLoader] BERT artifact directory does not exist: {path.resolve()}. "
            "Please ensure artifacts/bert is populated before starting the service."
        )

    if path.exists() and not config_file.exists() and not (hf_repo and model_dir is None):
        raise FileNotFoundError(
            f"[ModelLoader] Missing config.json in BERT directory: {path.resolve()}"
        )

    # Decide runtime engine
    use_pytorch = False
    if requested_backend == "pytorch":
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("[ModelLoader] BERT_BACKEND='pytorch' requested, but torch/transformers is not installed.")
        use_pytorch = True
    elif requested_backend == "onnx":
        if not ONNX_AVAILABLE:
            raise RuntimeError("[ModelLoader] BERT_BACKEND='onnx' requested, but onnxruntime/tokenizers is not installed.")
        use_pytorch = False
    else:
        # 'auto': prefer PyTorch locally when torch + weights exist; otherwise use ONNX INT8
        if PYTORCH_AVAILABLE and (has_pytorch_weights or (hf_repo and model_dir is None)):
            use_pytorch = True
        elif ONNX_AVAILABLE and has_onnx_weights:
            use_pytorch = False
        elif PYTORCH_AVAILABLE:
            use_pytorch = True  # will raise clean FileNotFoundError below if weights missing

    if use_pytorch:
        source_id = str(path) if (has_pytorch_weights or not (hf_repo and model_dir is None)) else hf_repo
        if not has_pytorch_weights and not (hf_repo and model_dir is None):
            raise FileNotFoundError(
                f"[ModelLoader] No model weights (model.safetensors, pytorch_model.bin, or model_int8.onnx) found in {path.resolve()}"
            )

        if device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            resolved_device = torch.device(device)
        else:
            resolved_device = device

        logger.info(f"Loading BERT tokenizer and PyTorch model from {source_id} onto {resolved_device}...")
        t0 = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(source_id)
        model = AutoModelForSequenceClassification.from_pretrained(source_id)
        _normalize_bert_id2label(model.config)

        model.to(resolved_device)
        model.eval()
        if resolved_device.type == "cuda":
            torch.cuda.synchronize()

        logger.info(f"BERT (PyTorch) loaded onto {resolved_device} in {(time.perf_counter() - t0) * 1000.0:.2f} ms.")
        return tokenizer, model, resolved_device

    # Lightweight ONNX Runtime INT8 path
    if not ONNX_AVAILABLE:
        raise FileNotFoundError(
            f"[ModelLoader] Neither PyTorch weights nor ONNX Runtime are available for {path.resolve()}"
        )
    if not has_onnx_weights:
        raise FileNotFoundError(
            f"[ModelLoader] No model weights (model.safetensors, pytorch_model.bin, or model_int8.onnx[.xz]) found in {path.resolve()}"
        )
    tokenizer_json = path / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"[ModelLoader] Missing tokenizer.json in {path.resolve()}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    t0 = time.perf_counter()
    resolved_onnx = _ensure_onnx_extracted(onnx_file, onnx_xz_file)
    tokenizer = OnnxTokenizerWrapper(tokenizer_json)
    model = OnnxBertModel(resolved_onnx, config_data)
    _normalize_bert_id2label(model.config)
    onnx_device = _OnnxCpuDevice()

    logger.info(f"BERT (ONNX INT8) loaded onto CPU in {(time.perf_counter() - t0) * 1000.0:.2f} ms.")
    return tokenizer, model, onnx_device


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
        device: Optional[Any] = None,
        lazy: bool = False,
        strict: bool = False,
        bert_temperature: Optional[float] = None,
        bert_backend: Optional[str] = None,
    ):
        self.bert_dir = bert_dir
        self.tfidf_path = tfidf_path
        self.device = device
        self.bert_backend = bert_backend

        self.tokenizer: Optional[Any] = None
        self.bert_model: Optional[Any] = None
        self.bert_id2label: Optional[Dict[int, str]] = None
        self.active_device: Optional[Any] = None
        self.bert_runtime: Optional[str] = None
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
                self.bert_dir, self.device, backend=self.bert_backend
            )
            self.bert_id2label = _normalize_bert_id2label(self.bert_model.config)
            self.bert_runtime = "onnx-int8" if getattr(self.bert_model, "is_onnx", False) else "pytorch"
            self.bert_error = None
        except Exception as exc:
            self.tokenizer = None
            self.bert_model = None
            self.bert_id2label = None
            self.bert_runtime = None
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
            "bert_runtime": self.bert_runtime,
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
        Runs sentiment classification using the fine-tuned BERT model (PyTorch or ONNX INT8).
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

        if getattr(self.bert_model, "is_onnx", False):
            encoded = self.tokenizer.encode_numpy(clean_text, max_length=max_length)
            encoded_len = int(encoded["input_ids"].shape[-1])
            is_truncated = len(full_tokens) > max(encoded_len - 2, 0)
            logits_np = self.bert_model.predict_logits(encoded)[0]
            model_display = "BERT Fine-Tuned (bert-base-uncased, INT8 ONNX)"
            device_label = "CPU"
        else:
            inputs = self.tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            encoded_len = int(inputs["input_ids"].shape[-1])
            is_truncated = len(full_tokens) > max(encoded_len - 2, 0)

            if self.active_device and self.active_device.type == "cuda":
                inputs = {k: v.to(self.active_device) for k, v in inputs.items()}
                torch.cuda.synchronize()

            with torch.inference_mode():
                logits = self.bert_model(**inputs).logits

            if self.active_device and self.active_device.type == "cuda":
                torch.cuda.synchronize()

            logits_np = logits.squeeze(0).detach().cpu().numpy()
            model_display = "BERT Fine-Tuned (bert-base-uncased)"
            device_label = str(self.active_device).upper() if self.active_device else "CPU"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_probs = _softmax_np(logits_np, temperature=1.0)
        cal_probs = _softmax_np(logits_np, temperature=self.bert_temperature)

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
            "model": model_display,
            "modelKey": "bert",
            "device": device_label,
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
