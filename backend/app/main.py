"""FastAPI application factory.

Run locally with:  uvicorn backend.app.main:app --port 8000
Interactive API documentation is served at /docs (Swagger UI) and /redoc.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import hmac
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import dataset, evaluate, experiments, navigation, predict, system, training
from .api.deps import build_context
from .core.config import APP_NAME, APP_VERSION, Settings, get_settings

log = logging.getLogger("mti")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.ctx.training.shutdown()
        app.state.ctx.db.dispose()

    app = FastAPI(
        title=APP_NAME, version=APP_VERSION, lifespan=lifespan,
        description="Local research API for Mars terrain segmentation with AI4Mars: dataset management, "
                    "U-Net / DeepLabV3 training, evaluation, inference and an experimental route-planning "
                    "simulator. Independent student research project; not a NASA system.",
    )
    app.state.ctx = build_context(settings)
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Not found: {exc.args[0] if exc.args else ''}"})

    @app.exception_handler(LookupError)
    async def _lookup(_: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def _conflict(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error. See the backend log."})

    # Public demo mode: anyone may browse, analyse images and plan routes; everything that changes
    # shared state or starts heavy work needs the admin token.
    public_ok = {("POST", "/api/predict"), ("POST", "/api/navigation/plan")}

    @app.middleware("http")
    async def _public_guard(request: Request, call_next):
        if settings.public_mode and request.method in ("POST", "PUT", "PATCH", "DELETE") \
                and request.url.path.startswith("/api/") and (request.method, request.url.path) not in public_ok:
            supplied = request.headers.get("x-admin-token", "")
            if not settings.admin_token or not hmac.compare_digest(supplied, settings.admin_token):
                return JSONResponse(status_code=403, content={"detail": (
                    "This is a public demo: training, dataset changes, evaluation, deletion and settings are "
                    "disabled. Enter the admin token in Settings to unlock them.")})
        return await call_next(request)

    for module in (system, dataset, training, experiments, predict, evaluate, navigation):
        app.include_router(module.router, prefix="/api")

    dist = Path(settings.frontend_dist) if settings.frontend_dist else None
    if dist and (dist / "index.html").is_file():
        root = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        async def _frontend(path: str):
            if path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            target = (root / path).resolve()
            if path and target.is_file() and root in target.parents:
                return FileResponse(target)
            return FileResponse(root / "index.html")  # client-side routes such as /analysis
    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    """Create the ASGI app lazily on first access of ``backend.app.main.app`` (what uvicorn does).

    Importing this module (e.g. from tests that call ``create_app`` with their own settings) therefore never
    opens the real data directory or touches the experiment database.
    """
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
