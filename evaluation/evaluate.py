"""
SST-2 Model Evaluation Runner
Support-Ticket Sentiment Classifier Project

Evaluates saved sentiment models (BERT and TF-IDF baseline) against
the specified split of the official Stanford Sentiment Treebank (SST-2) dataset.

Usage:
    python evaluation/evaluate.py --split validation --models bert,tfidf
    python evaluation/evaluate.py --dataset data/sst2 --models bert,tfidf
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.model_loader import load_bert_model, load_tfidf_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on SST-2 dataset for BERT and TF-IDF baseline."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on (default: validation)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sst2",
        help="Directory path containing SST-2 CSV files (default: data/sst2)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="bert,tfidf",
        help="Comma-separated list of models to evaluate: 'bert', 'tfidf', or 'bert,tfidf' (default: 'bert,tfidf')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for BERT evaluation (default: 32)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to export evaluation metrics JSON",
    )
    return parser.parse_args()


def load_dataset(dataset_dir: Path, split: str) -> pd.DataFrame:
    csv_file = dataset_dir / f"{split}.csv"
    if not csv_file.exists():
        # Check if dataset dir is already the file
        if Path(str(dataset_dir)).is_file():
            csv_file = Path(str(dataset_dir))
        else:
            raise FileNotFoundError(f"Dataset split file not found: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected 'sentence' and 'label' columns in {csv_file}")

    # Remove any unlabelled or NaN entries
    df = df.dropna(subset=["sentence", "label"])
    df["label"] = df["label"].astype(int)
    return df


def evaluate_tfidf(pipeline, texts: List[str], y_true: np.ndarray) -> Dict[str, Any]:
    t0 = time.perf_counter()
    y_pred = pipeline.predict(texts)
    y_probs = pipeline.predict_proba(texts)
    total_time = (time.perf_counter() - t0) * 1000.0
    avg_latency = total_time / len(texts)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "model": "TF-IDF + Logistic Regression",
        "sample_count": len(texts),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "confusion_matrix": cm,
        "avg_latency_ms": round(avg_latency, 3),
        "total_time_ms": round(total_time, 2),
    }


def evaluate_bert(tokenizer, model, device, texts: List[str], y_true: np.ndarray, batch_size: int = 32) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    all_preds = []
    all_probs = []
    total_samples = len(texts)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, total_samples, batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            if device.type == "cuda":
                encoded = {k: v.to(device) for k, v in encoded.items()}
                torch.cuda.synchronize()

            logits = model(**encoded).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = (time.perf_counter() - t0) * 1000.0
    avg_latency = total_time / total_samples

    y_pred = np.array(all_preds)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "model": "BERT Fine-Tuned (bert-base-uncased)",
        "sample_count": total_samples,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "confusion_matrix": cm,
        "avg_latency_ms": round(avg_latency, 3),
        "total_time_ms": round(total_time, 2),
        "device": str(device).upper(),
    }


def main():
    args = parse_args()
    print("=" * 70)
    print(f"SST-2 EVALUATION RUNNER (Split: {args.split.upper()})")
    print("=" * 70)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    print(f"Loading dataset from: {dataset_path} ...")
    df = load_dataset(dataset_path, args.split)
    texts = df["sentence"].tolist()
    y_true = df["label"].to_numpy()
    print(f"Loaded {len(texts)} samples (Positive: {(y_true == 1).sum()}, Negative: {(y_true == 0).sum()})\n")

    selected_models = [m.strip().lower() for m in args.models.split(",")]
    results: Dict[str, Any] = {}

    # 1. TF-IDF Evaluation
    if "tfidf" in selected_models:
        print("[1/2] Loading TF-IDF baseline model...")
        tfidf_path = PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_logistic_regression.joblib"
        if not tfidf_path.exists():
            tfidf_path = PROJECT_ROOT / "models" / "tfidf_baseline.joblib"

        tfidf_pipeline = load_tfidf_pipeline(tfidf_path)
        print("Evaluating TF-IDF on dataset...")
        results["tfidf"] = evaluate_tfidf(tfidf_pipeline, texts, y_true)
        print(f"  TF-IDF Accuracy: {results['tfidf']['accuracy'] * 100:.2f}% | F1: {results['tfidf']['f1_score'] * 100:.2f}% | Latency: {results['tfidf']['avg_latency_ms']:.2f} ms/sample\n")

    # 2. BERT Evaluation
    if "bert" in selected_models:
        print("[2/2] Loading BERT fine-tuned model...")
        bert_dir = PROJECT_ROOT / "artifacts" / "bert"
        if not bert_dir.exists():
            bert_dir = PROJECT_ROOT / "models" / "bert-sst2"

        tokenizer, bert_model, device = load_bert_model(bert_dir)
        print(f"Evaluating BERT on dataset using device: {device} (batch size: {args.batch_size})...")
        results["bert"] = evaluate_bert(tokenizer, bert_model, device, texts, y_true, batch_size=args.batch_size)
        print(f"  BERT Accuracy:   {results['bert']['accuracy'] * 100:.2f}% | F1: {results['bert']['f1_score'] * 100:.2f}% | Latency: {results['bert']['avg_latency_ms']:.2f} ms/sample\n")

    # Summary table
    print("=" * 70)
    print("FINAL EVALUATION BENCHMARK RESULTS")
    print("=" * 70)
    header = f"{'Metric':<25} "
    if "bert" in results:
        header += f"{'BERT Fine-Tuned':<20} "
    if "tfidf" in results:
        header += f"{'TF-IDF Baseline':<20} "
    if "bert" in results and "tfidf" in results:
        header += f"{'Delta (BERT - TFIDF)':<20}"
    print(header)
    print("-" * 70)

    metric_keys = [
        ("Accuracy", "accuracy", True),
        ("Precision (Pos)", "precision", True),
        ("Recall (Pos)", "recall", True),
        ("F1-Score (Pos)", "f1_score", True),
        ("Specificity (Neg)", "specificity", True),
        ("Neg Pred Value", "negative_predictive_value", True),
        ("Avg Latency (ms)", "avg_latency_ms", False),
    ]

    for label, key, is_pct in metric_keys:
        row = f"{label:<25} "
        b_val = results.get("bert", {}).get(key)
        t_val = results.get("tfidf", {}).get(key)

        if b_val is not None:
            b_str = f"{b_val * 100:.2f}%" if is_pct else f"{b_val:.2f} ms"
            row += f"{b_str:<20} "
        if t_val is not None:
            t_str = f"{t_val * 100:.2f}%" if is_pct else f"{t_val:.2f} ms"
            row += f"{t_str:<20} "
        if b_val is not None and t_val is not None:
            if is_pct:
                delta = (b_val - t_val) * 100
                delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
            else:
                delta = b_val - t_val
                delta_str = f"{delta:+.2f} ms"
            row += f"{delta_str:<20}"
        print(row)

    print("=" * 70)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
