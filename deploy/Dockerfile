# Single-container image for public hosting (Hugging Face Spaces, Render, Railway, Fly.io...).
# FastAPI serves both the REST API and the pre-built website on one port.
# Built from the staging folder produced by `python scripts/deploy_space.py --stage`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
RUN mkdir -p data/raw/imports data/processed data/analyses checkpoints experiments && chown -R user /home/user/app

USER user
ENV MTI_PUBLIC_MODE=true \
    MTI_FRONTEND_DIST=/home/user/app/frontend_dist \
    MTI_CORS_ORIGINS="" \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
