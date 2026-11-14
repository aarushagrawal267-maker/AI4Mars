# Render, Hugging Face Spaces or any Docker host, building from the repository source.
# Stage 1 builds the website; stage 2 runs FastAPI, which serves the API and the website on $PORT.
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend ./backend
COPY ml ./ml
COPY scripts ./scripts
COPY --from=web /web/dist ./frontend_dist
RUN mkdir -p data/raw/imports data/processed data/analyses checkpoints experiments
# Run as an unprivileged user (uid 1000, which Hugging Face Spaces also require); it owns the writable folders.
RUN useradd -m -u 1000 user && chown -R user /app
USER user
ENV MTI_FRONTEND_DIST=/app/frontend_dist PORT=10000
EXPOSE 10000
# Fetch the trained models + results bundle (if MTI_DEMO_BUNDLE_URL is set), then start the server.
CMD ["sh", "-c", "python scripts/fetch_demo_bundle.py && exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
