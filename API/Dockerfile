# ---------- Base ----------
FROM python:3.11-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Deps (common for prod & dev, but WITHOUT the model) ----------
FROM base AS deps

# 1) Torch CPU (no CUDA)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

# 2) Runtime deps
COPY ./API/requirements_api.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 3) App code
COPY ./API /app/API

# ---------- PROD (adds the model only here) ----------
FROM deps AS prod

# Copy trained model ONLY in prod
COPY ./API/artifacts/linear_svc_pointwise.joblib /app/artifacts/linear_svc_pointwise.joblib
ENV RANK_MODEL_PATH=/app/artifacts/linear_svc_pointwise.joblib \
    SKIP_RANK_MODEL=0

EXPOSE 8000
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- DEV (no model here) ----------
FROM deps AS dev

COPY ./API/requirements_dev.txt /tmp/requirements_dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements_dev.txt

# Make sure app doesn't try to load the rank model in dev
ENV SKIP_RANK_MODEL=1

EXPOSE 8000
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]

# (Optional) Make prod the default if you build without --target
FROM prod AS release
