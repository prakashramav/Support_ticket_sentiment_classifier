# Lightweight production Dockerfile for FastAPI sentiment backend (~330MB RSS, fits Render 512MB RAM)
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    BERT_BACKEND=onnx \
    ORT_INTRA_OP_THREADS=2

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install slim production inference dependencies (ONNX Runtime + Tokenizers + Scikit-Learn + FastAPI)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code and production model artifacts
COPY backend /app/backend
COPY artifacts /app/artifacts

# Decompress bundled INT8 ONNX BERT weights once at image build time for instant startup
RUN python -c "from pathlib import Path; import lzma; xz = Path('/app/artifacts/bert/model_int8.onnx.xz'); onnx = Path('/app/artifacts/bert/model_int8.onnx'); onnx.write_bytes(lzma.open(xz, 'rb').read()) if xz.exists() and not onnx.exists() else None; xz.unlink() if xz.exists() and onnx.exists() else None"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Bind to Render's dynamic $PORT environment variable (defaults to 8000)
CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"
