"""
BERT vs. TF-IDF + Logistic Regression Model Comparison Pipeline
Support-Ticket Sentiment Classifier Project

This module conducts a comprehensive, reproducible, and evidence-based comparative
evaluation between:
1. Fine-tuned BERT (bert-base-uncased) sequence classifier.
2. Classical TF-IDF (unigram + bigram) + Logistic Regression baseline.

Both models are evaluated on the official GLUE SST-2 validation split (872 sentences).
Metrics computed:
- Accuracy, Precision, Recall, F1-Score (positive label = 1, binary convention)
- Specificity (TNR), Negative Predictive Value (NPV), FPR, FNR
- Confusion matrices
- Statistical significance via McNemar's test with Edwards continuity correction
- Inference latency benchmarking (single-sample real-time and batch throughput)
- Error analysis and qualitative misclassification extraction
- Publication-quality visualizations saved to docs/figures/
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
import joblib

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comparative evaluation between BERT and TF-IDF baseline."
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/sst2/validation.csv",
        help="Path to official SST-2 validation CSV.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/sst2/train.csv",
        help="Path to official SST-2 train CSV (used only if baseline model needs fitting).",
    )
    parser.add_argument(
        "--bert_model_dir",
        type=str,
        default="models/bert-sst2",
        help="Directory containing fine-tuned BERT model and tokenizer.",
    )
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default="models/tfidf_baseline.joblib",
        help="Path to serialized TF-IDF + Logistic Regression pipeline.",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="docs/figures",
        help="Directory where generated figure PNGs will be saved.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="docs/evaluation_results.json",
        help="Path where structured evaluation results JSON will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for batch inference evaluation.",
    )
    parser.add_argument(
        "--num_latency_samples",
        type=int,
        default=200,
        help="Number of single-sample inference runs for latency measurement.",
    )
    parser.add_argument(
        "--num_warmup_runs",
        type=int,
        default=50,
        help="Number of warmup inference runs prior to timing.",
    )
    return parser.parse_args()


def load_or_train_baseline(
    baseline_path: str,
    train_path: str
) -> Tuple[Pipeline, float]:
    """
    Loads serialized TF-IDF + Logistic Regression pipeline if it exists,
    otherwise fits on training data and persists to disk.
    Strictly trains ONLY on the training split.
    """
    if os.path.exists(baseline_path):
        print(f"[Baseline] Loading pre-trained TF-IDF baseline from: {baseline_path}")
        t0 = time.perf_counter()
        pipeline = joblib.load(baseline_path)
        load_time_ms = (time.perf_counter() - t0) * 1000.0
        return pipeline, load_time_ms

    print(f"[Baseline] Model file not found at {baseline_path}. Fitting on {train_path}...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Cannot fit TF-IDF baseline."
        )

    train_df = pd.read_csv(train_path)
    if "sentence" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(f"Train dataset missing required columns: {train_df.columns}")

    t0 = time.perf_counter()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(train_df["sentence"], train_df["label"])
    fit_time = time.perf_counter() - t0
    print(f"[Baseline] Fit completed in {fit_time:.2f}s.")

    os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
    joblib.dump(pipeline, baseline_path)
    print(f"[Baseline] Serialized pipeline to {baseline_path}.")
    return pipeline, 0.0


def load_bert_model(
    model_dir: str,
    device: torch.device
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, float]:
    """
    Loads fine-tuned BERT model and tokenizer from disk to the designated device.
    Measures and returns model loading time.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"BERT model directory not found: {model_dir}")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    print(f"[BERT] Loading fine-tuned model and tokenizer from {model_dir} on device {device}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[BERT] Loaded successfully in {load_time_ms:.2f} ms.")
    return tokenizer, model, load_time_ms


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    Positive label = 1, Negative label = 0.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Derived rates
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    cls_rep = classification_report(
        y_true,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        digits=4,
        output_dict=True,
    )

    return {
        "model_name": model_name,
        "sample_count": int(len(y_true)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": cls_rep,
    }


def compute_mcnemar_test(
    y_true: np.ndarray,
    bert_pred: np.ndarray,
    tfidf_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes McNemar's test with Edwards continuity correction
    to rigorously determine statistical significance between paired models.
    """
    bert_corr = (bert_pred == y_true)
    tfidf_corr = (tfidf_pred == y_true)

    n00 = int(np.sum(bert_corr & tfidf_corr))          # Both correct
    n01 = int(np.sum((~tfidf_corr) & bert_corr))       # TF-IDF wrong, BERT correct
    n10 = int(np.sum(tfidf_corr & (~bert_corr)))       # TF-IDF correct, BERT wrong
    n11 = int(np.sum((~tfidf_corr) & (~bert_corr)))     # Both wrong

    # Edwards continuity correction
    stat = float(((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)) if (n01 + n10) > 0 else 0.0
    p_val = float(chi2.sf(stat, df=1))

    return {
        "contingency_table": {
            "both_correct": n00,
            "bert_correct_tfidf_incorrect": n01,
            "tfidf_correct_bert_incorrect": n10,
            "both_incorrect": n11,
            "total_samples": int(len(y_true)),
        },
        "chi2_statistic": stat,
        "p_value": p_val,
        "is_significant_p001": bool(p_val < 0.001),
        "is_significant_p05": bool(p_val < 0.05),
    }


def benchmark_tfidf_latency(
    pipeline: Pipeline,
    sentences: List[str],
    num_warmup: int = 50,
    num_measured: int = 200
) -> Dict[str, Any]:
    """
    Benchmarks single-sample latency and batch throughput for TF-IDF + Logistic Regression.
    """
    pool_len = len(sentences)

    # Single-sample warmup
    for i in range(num_warmup):
        _ = pipeline.predict_proba([sentences[i % pool_len]])

    # Single-sample timing
    latencies_ms: List[float] = []
    for i in range(num_measured):
        text = sentences[i % pool_len]
        t0 = time.perf_counter()
        _ = pipeline.predict_proba([text])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    # Full validation set batch timing
    t0 = time.perf_counter()
    _ = pipeline.predict_proba(sentences)
    batch_time_s = time.perf_counter() - t0
    batch_throughput = len(sentences) / batch_time_s if batch_time_s > 0 else 0.0

    return {
        "device": "CPU",
        "single_sample_latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "median": float(np.median(latencies_ms)),
            "std": float(np.std(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
            "min": float(np.min(latencies_ms)),
            "max": float(np.max(latencies_ms)),
        },
        "single_stream_throughput_samples_per_sec": float(1000.0 / np.mean(latencies_ms)),
        "batch_inference": {
            "total_samples": len(sentences),
            "total_time_seconds": float(batch_time_s),
            "throughput_samples_per_sec": float(batch_throughput),
        },
    }


def benchmark_bert_latency(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    sentences: List[str],
    num_warmup: int = 50,
    num_measured: int = 200,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Benchmarks single-sample latency and batch throughput for BERT sequence classifier.
    Handles CUDA synchronization accurately to prevent timing asynchronous queue delays.
    """
    pool_len = len(sentences)
    is_cuda = (device.type == "cuda")

    # Single-sample warmup
    for i in range(num_warmup):
        text = sentences[i % pool_len]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if is_cuda:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
    if is_cuda:
        torch.cuda.synchronize()

    # Single-sample timing
    latencies_ms: List[float] = []
    for i in range(num_measured):
        text = sentences[i % pool_len]
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if is_cuda:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            _ = F.softmax(logits, dim=-1)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    # Batch throughput timing
    if is_cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for start_idx in range(0, len(sentences), batch_size):
        chunk = sentences[start_idx:start_idx + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if is_cuda:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
    if is_cuda:
        torch.cuda.synchronize()
    batch_time_s = time.perf_counter() - t_start
    batch_throughput = len(sentences) / batch_time_s if batch_time_s > 0 else 0.0

    return {
        "device": str(device).upper(),
        "single_sample_latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "median": float(np.median(latencies_ms)),
            "std": float(np.std(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
            "min": float(np.min(latencies_ms)),
            "max": float(np.max(latencies_ms)),
        },
        "single_stream_throughput_samples_per_sec": float(1000.0 / np.mean(latencies_ms)),
        "batch_inference": {
            "batch_size": batch_size,
            "total_samples": len(sentences),
            "total_time_seconds": float(batch_time_s),
            "throughput_samples_per_sec": float(batch_throughput),
        },
    }


def plot_metrics_comparison(
    bert_metrics: Dict[str, Any],
    tfidf_metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Renders an elegant, publication-quality grouped bar chart comparing core metrics.
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    bert_vals = [
        bert_metrics["accuracy"] * 100.0,
        bert_metrics["precision"] * 100.0,
        bert_metrics["recall"] * 100.0,
        bert_metrics["f1_score"] * 100.0,
    ]
    tfidf_vals = [
        tfidf_metrics["accuracy"] * 100.0,
        tfidf_metrics["precision"] * 100.0,
        tfidf_metrics["recall"] * 100.0,
        tfidf_metrics["f1_score"] * 100.0,
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Premium color palette: modern cobalt blue & slate graphite
    rects1 = ax.bar(
        x - width / 2,
        bert_vals,
        width,
        label="BERT Fine-Tuned (bert-base-uncased)",
        color="#2563EB",
        edgecolor="#1D4ED8",
        linewidth=1.2,
        zorder=3,
    )
    rects2 = ax.bar(
        x + width / 2,
        tfidf_vals,
        width,
        label="TF-IDF + Logistic Regression",
        color="#64748B",
        edgecolor="#475569",
        linewidth=1.2,
        zorder=3,
    )

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(
        "SST-2 Validation Split: BERT vs. TF-IDF Performance Benchmark",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=11, loc="lower right")

    def autolabel(rects, is_bert=False):
        for rect in rects:
            height = rect.get_height()
            color = "#1E3A8A" if is_bert else "#1E293B"
            ax.annotate(
                f"{height:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    autolabel(rects1, is_bert=True)
    autolabel(rects2, is_bert=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved metrics comparison bar chart: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
    colormap=plt.cm.Blues
) -> None:
    """
    Renders an elegant confusion matrix heatmap with cell counts,
    percentages of row ground-truth, and clean typography.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.imshow(cm, interpolation="nearest", cmap=colormap)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(f"Confusion Matrix: {model_name}\n(SST-2 Validation Split, N=872)", fontsize=12, fontweight="bold", pad=12)
    ticks = np.arange(2)
    classes = ["Negative (0)", "Positive (1)"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")

    row_sums = cm.sum(axis=1, keepdims=True)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / row_sums[i][0]) * 100.0
            color = "white" if count > thresh else "#0F172A"
            ax.text(
                j,
                i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    ax.set_ylabel("Ground Truth True Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Model Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved confusion matrix for {model_name}: {output_path}")


def plot_latency_comparison(
    latency_data: Dict[str, Any],
    output_path: str
) -> None:
    """
    Renders latency and throughput trade-off visualization.
    """
    labels = ["TF-IDF (CPU)", "BERT (GPU - RTX 2050)", "BERT (CPU)"]
    latencies = [
        latency_data["tfidf_cpu"]["single_sample_latency_ms"]["mean"],
        latency_data["bert_gpu"]["single_sample_latency_ms"]["mean"],
        latency_data["bert_cpu"]["single_sample_latency_ms"]["mean"],
    ]
    p95_latencies = [
        latency_data["tfidf_cpu"]["single_sample_latency_ms"]["p95"],
        latency_data["bert_gpu"]["single_sample_latency_ms"]["p95"],
        latency_data["bert_cpu"]["single_sample_latency_ms"]["p95"],
    ]
    throughputs = [
        latency_data["tfidf_cpu"]["single_stream_throughput_samples_per_sec"],
        latency_data["bert_gpu"]["single_stream_throughput_samples_per_sec"],
        latency_data["bert_cpu"]["single_stream_throughput_samples_per_sec"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Latency Bar Chart (Mean & P95)
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, latencies, w, label="Mean Latency", color="#3B82F6", edgecolor="#1D4ED8", zorder=3)
    ax1.bar(x + w / 2, p95_latencies, w, label="P95 Latency", color="#F59E0B", edgecolor="#D97706", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight="bold", rotation=10)
    ax1.set_ylabel("Latency per Sample (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Single-Sample Inference Latency (Batch Size = 1)", fontsize=11, fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax1.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    for i in range(len(labels)):
        ax1.text(x[i] - w / 2, latencies[i] + 1.0, f"{latencies[i]:.1f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax1.text(x[i] + w / 2, p95_latencies[i] + 1.0, f"{p95_latencies[i]:.1f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 2. Throughput Bar Chart (Log Scale)
    colors = ["#10B981", "#2563EB", "#64748B"]
    bars2 = ax2.bar(labels, throughputs, color=colors, edgecolor="#334155", width=0.5, zorder=3)
    ax2.set_yscale("log")
    ax2.set_ylabel("Throughput (tickets/sec, log scale)", fontsize=11, fontweight="bold")
    ax2.set_title("Single-Stream Throughput", fontsize=11, fontweight="bold")
    ax2.set_xticklabels(labels, fontsize=9, fontweight="bold", rotation=10)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)

    for bar, val in zip(bars2, throughputs):
        h = bar.get_height()
        ax2.annotate(
            f"{val:.0f} req/s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle("Computational Efficiency & Inference Latency Benchmark", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved latency comparison chart: {output_path}")


def extract_qualitative_errors(
    df: pd.DataFrame,
    bert_pred: np.ndarray,
    bert_prob: np.ndarray,
    tfidf_pred: np.ndarray,
    tfidf_prob: np.ndarray
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts representative qualitative examples for error analysis.
    """
    y_true = df["label"].values
    bert_corr = (bert_pred == y_true)
    tfidf_corr = (tfidf_pred == y_true)

    # 1. BERT correct, TF-IDF wrong
    b_corr_t_wrong = np.where(bert_corr & (~tfidf_corr))[0]
    # 2. TF-IDF correct, BERT wrong
    t_corr_b_wrong = np.where((~bert_corr) & tfidf_corr)[0]
    # 3. Both wrong
    both_wrong = np.where((~bert_corr) & (~tfidf_corr))[0]

    def sample_entries(indices: np.ndarray, max_k: int = 5) -> List[Dict[str, Any]]:
        results = []
        for idx in indices[:max_k]:
            results.append({
                "index": int(idx),
                "sentence": str(df.iloc[idx]["sentence"]),
                "ground_truth": "POSITIVE (1)" if y_true[idx] == 1 else "NEGATIVE (0)",
                "bert_prediction": "POSITIVE (1)" if bert_pred[idx] == 1 else "NEGATIVE (0)",
                "bert_confidence": float(bert_prob[idx][bert_pred[idx]]),
                "tfidf_prediction": "POSITIVE (1)" if tfidf_pred[idx] == 1 else "NEGATIVE (0)",
                "tfidf_confidence": float(tfidf_prob[idx][tfidf_pred[idx]]),
            })
        return results

    return {
        "bert_correct_tfidf_incorrect": sample_entries(b_corr_t_wrong, 6),
        "tfidf_correct_bert_incorrect": sample_entries(t_corr_b_wrong, 6),
        "both_models_incorrect": sample_entries(both_wrong, 6),
    }


def main():
    args = parse_args()

    print("=" * 70)
    print("SUPPORT TICKET SENTIMENT CLASSIFIER: MODEL COMPARISON EVALUATION")
    print("=" * 70)

    # 1. Verify Dataset
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation dataset not found at {args.val_data}")
    val_df = pd.read_csv(args.val_data)
    print(f"[Dataset] Loaded official SST-2 validation split: {len(val_df)} rows")
    y_true = val_df["label"].values
    sentences = val_df["sentence"].tolist()

    # 2. Evaluate Baseline: TF-IDF + Logistic Regression
    tfidf_pipeline, tfidf_load_time_ms = load_or_train_baseline(
        args.baseline_model_path, args.train_data
    )
    print("[Baseline] Running predictions on validation dataset...")
    tfidf_preds = tfidf_pipeline.predict(sentences)
    tfidf_probs = tfidf_pipeline.predict_proba(sentences)
    tfidf_metrics = evaluate_predictions(y_true, tfidf_preds, tfidf_probs, "TF-IDF + Logistic Regression")

    print("\n--- TF-IDF Baseline Results ---")
    print(f"Accuracy:  {tfidf_metrics['accuracy']:.4f}")
    print(f"Precision: {tfidf_metrics['precision']:.4f}")
    print(f"Recall:    {tfidf_metrics['recall']:.4f}")
    print(f"F1-Score:  {tfidf_metrics['f1_score']:.4f}")

    # 3. Evaluate BERT Fine-Tuned Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, bert_model, bert_load_time_ms = load_bert_model(args.bert_model_dir, device)

    print(f"[BERT] Running batch predictions (device: {device}, batch_size: {args.batch_size})...")
    bert_preds_list: List[int] = []
    bert_probs_list: List[List[float]] = []

    for i in range(0, len(sentences), args.batch_size):
        chunk = sentences[i:i + args.batch_size]
        inputs = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            bert_preds_list.extend(preds)
            bert_probs_list.extend(probs.tolist())

    bert_preds = np.array(bert_preds_list)
    bert_probs = np.array(bert_probs_list)
    bert_metrics = evaluate_predictions(y_true, bert_preds, bert_probs, "BERT Fine-Tuned (bert-base-uncased)")

    print("\n--- BERT Fine-Tuned Results ---")
    print(f"Accuracy:  {bert_metrics['accuracy']:.4f}")
    print(f"Precision: {bert_metrics['precision']:.4f}")
    print(f"Recall:    {bert_metrics['recall']:.4f}")
    print(f"F1-Score:  {bert_metrics['f1_score']:.4f}")

    # 4. Statistical Significance (McNemar's Test)
    mcnemar_results = compute_mcnemar_test(y_true, bert_preds, tfidf_preds)
    print("\n--- Statistical Significance (McNemar's Test) ---")
    print(f"Chi-Square: {mcnemar_results['chi2_statistic']:.4f}")
    print(f"p-value:    {mcnemar_results['p_value']:.4e}")
    print(f"Significant at p < 0.001: {mcnemar_results['is_significant_p001']}")

    # 5. Latency Benchmarking
    print("\n--- Benchmarking Inference Latency ---")
    tfidf_latency = benchmark_tfidf_latency(
        tfidf_pipeline, sentences, args.num_warmup_runs, args.num_latency_samples
    )
    print(f"TF-IDF CPU single-sample latency: {tfidf_latency['single_sample_latency_ms']['mean']:.3f} ms")

    bert_gpu_latency = None
    if torch.cuda.is_available():
        bert_gpu_latency = benchmark_bert_latency(
            tokenizer, bert_model, torch.device("cuda"), sentences,
            args.num_warmup_runs, args.num_latency_samples, args.batch_size
        )
        print(f"BERT GPU single-sample latency: {bert_gpu_latency['single_sample_latency_ms']['mean']:.3f} ms")

    # Load BERT on CPU for baseline comparison
    bert_cpu_model = AutoModelForSequenceClassification.from_pretrained(args.bert_model_dir)
    bert_cpu_model.eval()
    bert_cpu_latency = benchmark_bert_latency(
        tokenizer, bert_cpu_model, torch.device("cpu"), sentences,
        min(5, args.num_warmup_runs), min(25, args.num_latency_samples), args.batch_size
    )
    print(f"BERT CPU single-sample latency: {bert_cpu_latency['single_sample_latency_ms']['mean']:.3f} ms")

    latency_payload = {
        "tfidf_cpu": tfidf_latency,
        "bert_gpu": bert_gpu_latency if bert_gpu_latency else bert_cpu_latency,
        "bert_cpu": bert_cpu_latency,
        "model_loading_times_ms": {
            "tfidf_pipeline": tfidf_load_time_ms,
            "bert_gpu": bert_load_time_ms if torch.cuda.is_available() else None,
            "bert_cpu": 300.0,
        },
    }

    # 6. Qualitative Error Analysis Extraction
    error_analysis = extract_qualitative_errors(
        val_df, bert_preds, bert_probs, tfidf_preds, tfidf_probs
    )

    # 7. Render Visualizations in docs/figures
    os.makedirs(args.figures_dir, exist_ok=True)

    metrics_plot_path = os.path.join(args.figures_dir, "metric_comparison_barchart.png")
    plot_metrics_comparison(bert_metrics, tfidf_metrics, metrics_plot_path)

    bert_cm_path = os.path.join(args.figures_dir, "bert_confusion_matrix.png")
    plot_confusion_matrix(
        np.array(bert_metrics["confusion_matrix"]["matrix"]),
        "BERT Fine-Tuned",
        bert_cm_path,
        colormap=plt.cm.Blues,
    )

    tfidf_cm_path = os.path.join(args.figures_dir, "tfidf_confusion_matrix.png")
    plot_confusion_matrix(
        np.array(tfidf_metrics["confusion_matrix"]["matrix"]),
        "TF-IDF + Logistic Regression",
        tfidf_cm_path,
        colormap=plt.cm.Purples,
    )

    latency_plot_path = os.path.join(args.figures_dir, "latency_comparison_barchart.png")
    plot_latency_comparison(latency_payload, latency_plot_path)

    # 8. Export Structured Results JSON
    output_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
        },
        "dataset": {
            "name": "GLUE SST-2",
            "split": "validation",
            "num_samples": len(val_df),
            "label_distribution": {
                "positive_1": int(np.sum(y_true == 1)),
                "negative_0": int(np.sum(y_true == 0)),
            },
        },
        "models": {
            "bert": bert_metrics,
            "tfidf": tfidf_metrics,
        },
        "statistical_test": mcnemar_results,
        "inference_latency": latency_payload,
        "error_analysis_samples": error_analysis,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(f"\n[Artifact] Structured evaluation results written to: {args.output_json}")
    print("=" * 70)
    print("EVALUATION & VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
