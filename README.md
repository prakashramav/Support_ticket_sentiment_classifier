# Support-Ticket Sentiment Classifier

A sentiment classification system designed to analyze and classify customer support tickets into positive and negative sentiments.

---

## Project Overview: BERT Fine-Tuning vs. TF-IDF Baseline

This repository compares two distinct sentiment classification paradigms on the official **GLUE SST-2 (Stanford Sentiment Treebank)** benchmark:
1. **TF-IDF + Logistic Regression:** Classical n-gram bag-of-words baseline.
2. **BERT Fine-Tuning (`bert-base-uncased`):** Pretrained bidirectional transformer fine-tuned for sequence classification.

---

## Tech Stack

- **Frontend:** Next.js (App Router), React, JavaScript, Tailwind CSS, Axios, Lucide Icons
- **Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic
- **Machine Learning & NLP:**
  - Hugging Face `transformers` (`bert-base-uncased`, `Trainer`, `TrainingArguments`)
  - Hugging Face `datasets` (Official GLUE SST-2)
  - Hugging Face `accelerate` (Hardware & GPU execution)
  - `torch` (PyTorch with CUDA 12.4 acceleration)
  - `scikit-learn` (TF-IDF Vectorizer, Logistic Regression, Evaluation Metrics)
  - `matplotlib` (Confusion matrix visualization)

---

## Folder Structure

```
support-ticket-sentiment-classifier/
│
├── artifacts/                    # Production-ready model artifacts & metadata
│   ├── bert/                     # Fine-tuned BERT model (config.json, model.safetensors, tokenizer)
│   ├── tfidf/                    # Fitted TF-IDF baseline (tfidf_logistic_regression.joblib)
│   └── model_metadata.json       # Provenance, hyperparams, architecture, and metrics
├── frontend/                     # Next.js frontend project (React, JS, Tailwind CSS, Axios)
├── backend/                      # FastAPI backend project (main.py, requirements.txt)
│   └── app/                      # Backend application package
│       └── services/             # Core service utilities
│           └── model_loader.py   # Production singleton model loader & inference service
├── data/                         # Dataset storage and scripts
│   ├── sst2/                     # Official SST-2 GLUE benchmark dataset splits (train, validation)
│   └── download_sst2.py          # Script to download official SST-2 dataset
├── models/                       # Checkpoint and intermediate model storage
│   ├── bert/                     # Pretrained BERT base weights & vocab
│   ├── bert-sst2/                # Fine-tuned BERT checkpoint
│   └── tfidf_baseline.joblib     # Pre-serialized baseline pipeline
├── training/                     # Model training & verification scripts
│   ├── bert_finetuning.ipynb     # Interactive fine-tuning notebook
│   ├── compare_models.py         # Empirical comparison & latency benchmark pipeline
│   └── verify_saved_models.py    # Production model sanity & verification runner
├── docs/                         # Project documentation & benchmark artifacts
│   ├── model_comparison_report.md# Comprehensive 14-section evaluation report
│   ├── evaluation_results.json   # Full empirical results export
│   └── figures/                  # Publication-quality benchmark visualizations
├── requirements.txt              # Core project dependencies (Transformers, Torch, Scikit-Learn)
├── .gitignore                    # Git ignore rules for venv, node_modules, datasets, weights
└── README.md                     # Project overview, setup, and run instructions
```

---

## Environment Setup

### 1. Python Virtual Environment

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Project Dependencies

```bash
pip install -r requirements.txt
```

*For GPU acceleration (NVIDIA GPU with CUDA 12.4):*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## BERT Fine-Tuning Experiment

The fine-tuning pipeline is implemented in [`training/bert_finetuning.ipynb`](training/bert_finetuning.ipynb) and consists of 13 structured sections:

1. **Imports:** Loads PyTorch, Hugging Face Transformers, Datasets, Scikit-Learn, and Matplotlib.
2. **GPU Check:** Detects GPU device, CUDA availability, and enables TF32 / Mixed Precision (`fp16`).
3. **Load SST-2:** Loads official GLUE SST-2 splits (`train`: 67,349, `validation`: 872) without custom splits.
4. **Dataset Inspection:** Analyzes label distributions and inspects sample texts.
5. **Load Pretrained BERT:** Loads `bert-base-uncased` with `AutoModelForSequenceClassification` (`num_labels=2`).
6. **Tokenization:** WordPiece tokenization with `max_length=128`, `truncation=True`, and dynamic batch padding via `DataCollatorWithPadding`.
7. **Training Configuration:** Hugging Face `TrainingArguments` (`learning_rate=2e-5`, `weight_decay=0.01`, `load_best_model_at_end=True`, `eval_strategy="epoch"`).
8. **Fine-Tuning:** Updates pretrained encoder weights and classification head using SST-2 training data.
9. **Evaluation:** Evaluates strictly on the official SST-2 validation split (Accuracy, Precision, Recall, F1, Classification Report).
10. **Confusion Matrix:** Renders and saves [`evaluation/figures/bert_confusion_matrix.png`](evaluation/figures/bert_confusion_matrix.png).
11. **Save Model:** Serializes model, tokenizer, and config to [`models/bert-sst2/`](models/bert-sst2/).
12. **Inference Test:** Runs predictions and confidence scores on support-ticket text examples.
13. **Results:** Prints final metrics and comparison matrix.

---

## Empirical Benchmark Results

Evaluated on the official **SST-2 validation split (872 sentences)**:

### 1. Comparison Table (Actual Empirical Metrics)

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 0.8131 | 0.7933 | 0.8559 | 0.8234 |
| **BERT Fine-Tuned (`bert-base-uncased`)** | **0.9289** | **0.9244** | **0.9369** | **0.9306** |

### 2. BERT Classification Report

```text
              precision    recall  f1-score   support

Negative (0)     0.9336    0.9206    0.9271       428
Positive (1)     0.9244    0.9369    0.9306       444

    accuracy                         0.9289       872
   macro avg     0.9290    0.9287    0.9289       872
weighted avg     0.9290    0.9289    0.9289       872
```

### 3. Confusion Matrix

![BERT Confusion Matrix](evaluation/figures/bert_confusion_matrix.png)

- **True Negatives:** 394
- **False Positives:** 34
- **False Negatives:** 28
- **True Positives:** 416

### 4. Support-Ticket Inference Samples

| Customer Support Sample | Predicted Sentiment | Confidence | Probabilities (Neg / Pos) |
| :--- | :---: | :---: | :---: |
| *"The support team solved my problem quickly."* | **POSITIVE** | **77.69%** | `[0.2231, 0.7769]` |
| *"The issue was never fixed and the service was terrible."* | **NEGATIVE** | **99.69%** | `[0.9969, 0.0031]` |
| *"I appreciate the prompt resolution, exceptional customer care!"* | **POSITIVE** | **99.92%** | `[0.0008, 0.9992]` |
| *"My ticket was closed without any explanation or refund."* | **NEGATIVE** | **97.65%** | `[0.9765, 0.0235]` |

---

## Dataset & Domain Limitation Disclosure

> **Scientific Transparency & Domain Mismatch:**
> - The SST-2 dataset consists of single-sentence movie reviews from Rotten Tomatoes.
> - While SST-2 is the recognized academic benchmark for evaluating NLP sequence classification pipelines, customer support tickets exhibit distinct domain characteristics (technical logs, billing disputes, frustration indicators, mixed bug descriptions and polite sign-offs).
> - Performance on SST-2 demonstrates the sentiment classification pipeline and transfer learning capability, but does not establish production performance on real-world support-ticket data without domain-specific data fine-tuning or evaluation.

---

## Production Model Artifacts & Serving

### 1. Artifact Storage Locations

Both trained models and their complete configuration metadata are persisted under `artifacts/` for decoupled, production-grade serving:

| Component | Storage Path | Key Files & Artifacts | Purpose |
| :--- | :--- | :--- | :--- |
| **Fine-Tuned BERT** | `artifacts/bert/` | `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` | 110M parameter sequence classifier with subword tokenization and explicit binary label mappings (`0: NEGATIVE`, `1: POSITIVE`). |
| **TF-IDF Baseline** | `artifacts/tfidf/` | `tfidf_logistic_regression.joblib` | Scikit-learn `Pipeline` encapsulating fitted `TfidfVectorizer(ngram_range=(1,2), max_features=10000)` and `LogisticRegression(C=1.0)`. |
| **Model Metadata** | `artifacts/` | `model_metadata.json` | Full experiment provenance, architecture dimensions, hyperparams, library versions, git commit, and validation metrics. |

> [!SECURITY]
> **Joblib Security Advisory:**
> Loading `.joblib` files relies on Python's pickle protocol which can execute arbitrary code upon deserialization. Only load joblib artifacts originating from trusted, authentic repositories.

### 2. Saving or Refreshing Model Artifacts

To refresh the production model artifacts after fine-tuning or retraining:

```python
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# 1. Save Fine-Tuned BERT & Tokenizer
bert_dir = Path("artifacts/bert")
bert_dir.mkdir(parents=True, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained("models/bert-sst2")
tokenizer = AutoTokenizer.from_pretrained("models/bert-sst2")

model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model.save_pretrained(bert_dir)
tokenizer.save_pretrained(bert_dir)

# 2. Save TF-IDF Baseline Pipeline
tfidf_dir = Path("artifacts/tfidf")
tfidf_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(tfidf_pipeline, tfidf_dir / "tfidf_logistic_regression.joblib")
```

### 3. Model Verification

A standalone verification suite ensures all artifact files exist, loads both models into memory, validates evaluation modes, and verifies finite, normalized probability distributions:

```bash
# Windows PowerShell
.\.venv\Scripts\python training/verify_saved_models.py

# Linux / macOS
python training/verify_saved_models.py
```

The script evaluates boundary conditions on sample sentences, confirms `model.eval()` mode, measures inference latency, and exits with code `0` on success or code `1` on failure.

### 4. How FastAPI Loads Artifacts

The FastAPI backend uses the singleton loader in [`backend/app/services/model_loader.py`](backend/app/services/model_loader.py):
- **Startup Singleton:** Models are loaded into memory once during application startup, preventing disk I/O bottlenecks during live requests.
- **Hardware Acceleration:** Auto-detects and uses CUDA when available, falling back to CPU gracefully.
- **Low-Overhead Execution:** Evaluates predictions inside `torch.inference_mode()`.
- **Fail-Fast Protection:** Raises explicit `FileNotFoundError` if artifacts are missing, preventing silent fallbacks to untrained models.

```python
from backend.app.services.model_loader import get_model_service

# Retrieve singleton instance (loads models on startup)
model_service = get_model_service()

# Fast inference with probabilities and confidence scores
result = model_service.predict("The issue was resolved immediately, thank you!", model_type="bert")
# Returns: {'sentiment': 'POSITIVE', 'confidence': 0.9992, 'probabilities': {...}, 'latencyMs': 12.5}
```

### 5. Dependency Compatibility & Requirements

| Package | Minimum Version | Tested Version | Notes |
| :--- | :--- | :--- | :--- |
| `python` | `>=3.10.0` | `3.11.9` | 64-bit runtime |
| `torch` | `>=2.2.0` | `2.6.0+cu124` | Compatible with CUDA 12.4+ / CPU |
| `transformers` | `>=4.38.0` | `5.17.0` | Requires `safetensors` support |
| `scikit-learn` | `>=1.4.0` | `1.9.1` | Required for TF-IDF pipeline deserialization |
| `joblib` | `>=1.3.0` | `1.4.2` | Pipeline serialization |
| `fastapi` | `>=0.110.0` | `0.115.11` | Inference API |

---

## Running the Application

### 1. Run the Training Notebook

Open and run all cells in [`training/bert_finetuning.ipynb`](training/bert_finetuning.ipynb) via VS Code, JupyterLab, or Google Colab.

### 2. Run Backend Server

```bash
# Windows PowerShell
.\.venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

- Health Endpoint: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- Interactive Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Run Frontend Server

```bash
cd frontend
npm install
npm run dev
```

- Web UI: [http://localhost:3000](http://localhost:3000)
