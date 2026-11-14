"""Shared application context and FastAPI dependency."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from ..core.config import Settings
from ..database.db import Database
from ..services.dataset_service import DatasetService
from ..services.evaluation_service import EvaluationService
from ..services.inference_service import InferenceService
from ..services.jobs import JobRunner
from ..services.model_registry import ModelRegistry
from ..services.settings_service import SettingsService
from ..services.training_manager import TrainingManager


@dataclass
class Context:
    settings: Settings
    db: Database
    jobs: JobRunner
    dataset: DatasetService
    training: TrainingManager
    registry: ModelRegistry
    inference: InferenceService
    evaluation: EvaluationService
    prefs: SettingsService


def build_context(settings: Settings) -> Context:
    settings.ensure_dirs()
    db = Database(settings.db_url)
    jobs = JobRunner()
    dataset = DatasetService(settings, jobs)
    registry = ModelRegistry(settings, db, cache_size=max(1, settings.model_cache_size))
    ctx = Context(settings=settings, db=db, jobs=jobs, dataset=dataset,
                  training=TrainingManager(settings, db, dataset), registry=registry,
                  inference=InferenceService(settings, db, registry),
                  evaluation=EvaluationService(db, jobs, registry, dataset), prefs=SettingsService(db))
    if settings.enable_test_fixture:
        registry.ensure_fixture()
    return ctx


def get_ctx(request: Request) -> Context:
    return request.app.state.ctx
