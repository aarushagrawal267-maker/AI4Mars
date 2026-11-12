# Backend image: FastAPI + PyTorch (CPU build). Build context is the repository root:
#   docker compose build backend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch first, so requirements.txt does not pull a different (larger) build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code only; data, checkpoints and experiments are mounted as volumes at run time.
COPY backend ./backend
COPY ml ./ml
COPY scripts ./scripts

RUN mkdir -p data/raw/imports data/processed data/analyses checkpoints experiments

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# 0.0.0.0 inside the container; docker-compose.yml publishes it on the host's 127.0.0.1 only.
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
