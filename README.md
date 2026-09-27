# Support-Ticket Sentiment Classifier

A sentiment classification system designed to analyze and classify customer support tickets into positive and negative sentiments.

---

## Tech Stack

- **Frontend:** Next.js (App Router), React, JavaScript, Tailwind CSS, Axios
- **Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic
- **Data & Models:** Hugging Face `datasets` (GLUE SST-2), Hugging Face `transformers` (Pretrained BERT - `bert-base-uncased`), PyTorch

---

## Folder Structure

```
support-ticket-sentiment-classifier/
│
├── frontend/             # Next.js frontend project (React, JS, Tailwind CSS, Axios)
├── backend/              # FastAPI backend project (main.py, requirements.txt)
├── data/                 # Dataset storage and scripts
│   ├── sst2/             # Official SST-2 GLUE benchmark dataset splits (train, validation)
│   └── download_sst2.py  # Script to download official SST-2 dataset
├── models/               # Model weights storage and scripts
│   ├── bert/             # Pretrained BERT weights and tokenizer config/vocab
│   └── download_bert.py  # Script to download pretrained BERT & tokenizer
├── training/             # Future model training scripts
├── evaluation/           # Future evaluation and metrics scripts
├── docs/                 # Project documentation
├── tests/                # Automated tests
├── .gitignore            # Git ignore rules for venv, node_modules, datasets, weights
└── README.md             # Project overview, setup, and run instructions
```

---

## Frontend Setup

1. Navigate to the `frontend/` folder:
   ```bash
   cd frontend
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Axios is included in dependencies for API communication with the backend.

---

## Backend Setup

1. Create and activate a Python virtual environment (Python 3.11 recommended):
   ```bash
   # From root directory
   python -m venv .venv
   
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   
   # Linux / macOS
   source .venv/bin/activate
   ```

2. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## Dataset Download Instructions

The project uses the official SST-2 dataset from the GLUE benchmark via Hugging Face `datasets` (`load_dataset("glue", "sst2")`).

To download both official splits (`train` and `validation`) into `data/sst2/`:

```bash
python data/download_sst2.py
```

*Note: No custom splits, preprocessing, or modifications are applied.*

---

## Pretrained BERT Download Instructions

The project downloads the official pretrained `bert-base-uncased` model weights and tokenizer into `models/bert/`:

```bash
python models/download_bert.py
```

*Note: The model is saved purely in its pretrained state without fine-tuning or training.*

---

## Run Commands

### 1. Run Backend Server

```bash
# Windows PowerShell
.\.venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

Verify backend health status at:
- Endpoint: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- Interactive Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Run Frontend Server

```bash
cd frontend
npm run dev
```

Access the frontend application at:
- [http://localhost:3000](http://localhost:3000)
