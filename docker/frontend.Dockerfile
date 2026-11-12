# Frontend image: Vite development server. Build context is the repository root:
#   docker compose build frontend
# /api requests are proxied by Vite to VITE_API_PROXY_TARGET (set in docker-compose.yml).
FROM node:20-slim

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY frontend/ ./

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173", "--strictPort"]
