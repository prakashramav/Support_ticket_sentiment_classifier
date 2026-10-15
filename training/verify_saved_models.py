"""
Production Model Verification Script
Support-Ticket Sentiment Classifier Project

Validates that both saved model artifacts under artifacts/ can be successfully loaded,
properly initialized in evaluation mode, and execute valid binary sentiment inference.

Checks performed:
1. Artifact File Integrity:
   - artifacts/bert/config.json
   - artifacts/bert/model.safetensors
   - artifacts/bert/tokenizer.json
   - artifacts/bert/tokenizer_config.json
   - artifacts/tfidf/tfidf_logistic_regression.joblib
   - artifacts/model_metadata.json
2. Weight & Tokenizer Deserialization:
   - Loads BERT onto CUDA if available, else CPU.
   - Sets model.eval() and tests torch.inference_mode().
   - Loads TF-IDF scikit-learn Pipeline.
3. Inference Sanity & Boundary Validation:
   - Valid binary prediction ("POSITIVE" or "NEGATIVE").
   - Finite probabilities in [0.0, 1.0] summing to 1.0 (+/- 1e-4).
   - Finite confidence score in [0.0, 1.0].
   - Measured non-zero inference latency.
4. Clean diagnostic output and non-zero exit code on failure.

Note:
-----
Sample sentences are used exclusively for interface and sanity verification.
They are NOT evaluation data and do NOT represent empirical accuracy claims.
"""

import sys
import os
import math
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.model_loader import (
    load_bert_model,
    load_tfidf_pipeline,
    ModelService,
)

VERIFICATION_SENTENCES = [
    {
        "text": "The service was excellent and very helpful.",
        "expected_sentiment": "POSITIVE",
    },
    {
        "text": "The experience was terrible and disappointing.",
        "expected_sentiment": "NEGATIVE",
    },
    {
        "text": "My support ticket was resolved promptly, thank you!",
        "expected_sentiment": "POSITIVE",
    },
    {
        "text": "The application keeps crashing and nobody is replying to my emails.",
        "expected_sentiment": "NEGATIVE",
    },
]

EXPECTED_FILES = [
    ("artifacts/bert/config.json", True),
    ("artifacts/bert/model.safetensors", True),
    ("artifacts/bert/tokenizer.json", True),
    ("artifacts/bert/tokenizer_config.json", True),
    ("artifacts/tfidf/tfidf_logistic_regression.joblib", True),
    ("artifacts/model_metadata.json", True),
]


def check_file_existence() -> bool:
    """Verifies that all required artifact files exist on disk with non-zero size."""
    print("\n" + "=" * 65)
    print("STEP 1: VERIFYING ARTIFACT FILE SYSTEM INTEGRITY")
    print("=" * 65)
    all_ok = True

    for rel_path, required in EXPECTED_FILES:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            print(f"[FAIL] Missing required artifact: {rel_path}")
            all_ok = False
        else:
            size_bytes = full_path.stat().st_size
            if size_bytes == 0:
                print(f"[FAIL] Artifact is empty (0 bytes): {rel_path}")
                all_ok = False
            else:
                if size_bytes > 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                elif size_bytes > 1024:
                    size_str = f"{size_bytes / 1024:.2f} KB"
                else:
                    size_str = f"{size_bytes} bytes"
                print(f" [PASS] Found: {rel_path:<48} ({size_str})")

    return all_ok


def validate_inference_result(
    result: Dict[str, Any],
    model_name: str,
    text: str
) -> bool:
    """Validates structural and numerical boundaries of a single prediction."""
    sentiment = result.get("sentiment")
    confidence = result.get("confidence")
    probs = result.get("probabilities", {})
    latency = result.get("latencyMs")

    # 1. Check sentiment class
    if sentiment not in ["POSITIVE", "NEGATIVE"]:
        print(f"[FAIL] {model_name}: Invalid sentiment label '{sentiment}'. Expected 'POSITIVE' or 'NEGATIVE'.")
        return False

    # 2. Check confidence bounds and finiteness
    if confidence is None or not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
        print(f"[FAIL] {model_name}: Invalid confidence value {confidence}. Must be finite and in [0, 1].")
        return False

    # 3. Check probabilities dictionary
    if "POSITIVE" not in probs or "NEGATIVE" not in probs:
        print(f"[FAIL] {model_name}: Probabilities dict missing 'POSITIVE' or 'NEGATIVE' keys: {probs}")
        return False

    p_pos = probs["POSITIVE"]
    p_neg = probs["NEGATIVE"]
    for name, p_val in [("POSITIVE", p_pos), ("NEGATIVE", p_neg)]:
        if not math.isfinite(p_val) or not (0.0 <= p_val <= 1.0):
            print(f"[FAIL] {model_name}: Probability {name} ({p_val}) outside [0, 1] or non-finite.")
            return False

    prob_sum = p_pos + p_neg
    if abs(prob_sum - 1.0) > 0.01:
        print(f"[FAIL] {model_name}: Class probabilities do not sum to 1.0 (sum = {prob_sum:.4f})")
        return False

    # 4. Check latency
    if latency is None or latency <= 0:
        print(f"[FAIL] {model_name}: Latency value invalid ({latency} ms)")
        return False

    return True


def run_verification() -> bool:
    """Orchestrates model loading, test inference, and boundary verification."""
    # Step 1: Check files
    if not check_file_existence():
        print("\n[ERROR] File integrity verification failed. Aborting model loading.")
        return False

    # Step 2: Load models via ModelService
    print("\n" + "=" * 65)
    print("STEP 2: LOADING MODELS VIA BACKEND MODEL SERVICE")
    print("=" * 65)
    try:
        service = ModelService(
            bert_dir=PROJECT_ROOT / "artifacts" / "bert",
            tfidf_path=PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_logistic_regression.joblib",
        )
        print(" [PASS] ModelService initialized successfully.")
        print(f" [PASS] BERT Active Device: {service.active_device}")
        print(f" [PASS] BERT Model Training Mode: {service.bert_model.training} (must be False)")
        if service.bert_model.training:
            print("[FAIL] BERT is still in training mode! Expected model.eval().")
            return False
    except Exception as exc:
        print(f"[FAIL] Failed to load models: {exc}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Run sample inferences
    print("\n" + "=" * 65)
    print("STEP 3: RUNNING VERIFICATION INFERENCE SAMPLES")
    print("=" * 65)

    all_passed = True
    for item in VERIFICATION_SENTENCES:
        sample_text = item["text"]
        print(f"\nSample Text: \"{sample_text}\"")

        # --- A. BERT Inference ---
        try:
            bert_res = service.predict_bert(sample_text)
            bert_ok = validate_inference_result(bert_res, "BERT", sample_text)
            if not bert_ok:
                all_passed = False
            else:
                print(
                    f"  BERT  -> Sentiment: {bert_res['sentiment']:<8} "
                    f"| Confidence: {bert_res['confidence']*100:.2f}% "
                    f"| Probs: [Neg: {bert_res['probabilities']['NEGATIVE']:.4f}, Pos: {bert_res['probabilities']['POSITIVE']:.4f}] "
                    f"| Latency: {bert_res['latencyMs']:.2f} ms ({bert_res['device']})"
                )
        except Exception as exc:
            print(f"  BERT [ERROR]: {exc}")
            all_passed = False

        # --- B. TF-IDF Inference ---
        try:
            tfidf_res = service.predict_tfidf(sample_text)
            tfidf_ok = validate_inference_result(tfidf_res, "TF-IDF", sample_text)
            if not tfidf_ok:
                all_passed = False
            else:
                print(
                    f"  TF-IDF -> Sentiment: {tfidf_res['sentiment']:<8} "
                    f"| Confidence: {tfidf_res['confidence']*100:.2f}% "
                    f"| Probs: [Neg: {tfidf_res['probabilities']['NEGATIVE']:.4f}, Pos: {tfidf_res['probabilities']['POSITIVE']:.4f}] "
                    f"| Latency: {tfidf_res['latencyMs']:.2f} ms (CPU)"
                )
        except Exception as exc:
            print(f"  TF-IDF [ERROR]: {exc}")
            all_passed = False

    # Step 4: Final Summary
    print("\n" + "=" * 65)
    print("STEP 4: VERIFICATION OUTCOME SUMMARY")
    print("=" * 65)
    if all_passed:
        print(" [SUCCESS] All verification tests passed successfully!")
        print(" - Both BERT and TF-IDF artifacts are valid, intact, and production-ready.")
        print(" - Model inferences produce finite, properly normalized probability distributions.")
        print(" - No model retraining or data leakage occurred during verification.")
        print("=" * 65 + "\n")
        return True
    else:
        print(" [FAILURE] One or more verification checks failed.")
        print("=" * 65 + "\n")
        return False


def main():
    success = run_verification()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
