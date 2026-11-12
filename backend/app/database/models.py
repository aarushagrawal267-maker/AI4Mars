"""SQLAlchemy models: experiments, evaluations, image analyses and key/value settings."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    arch: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(24), default="queued")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    history: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)
    best_val_miou: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parameters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_train: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_val: Mapped[int | None] = mapped_column(Integer, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    pretrained_loaded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    is_fixture: Mapped[bool] = mapped_column(Boolean, default=False)

    evaluations: Mapped[list["Evaluation"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    split: Mapped[str] = mapped_column(String(16))
    label_variant: Mapped[str] = mapped_column(String(32))
    images: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON)

    experiment: Mapped[Experiment] = relationship(back_populates="evaluations")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    experiment_id: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(200))
    arch: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16))          # upload | dataset
    filename: Mapped[str] = mapped_column(String(300))
    sample_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_fixture: Mapped[bool] = mapped_column(Boolean, default=False)
    stats: Mapped[dict] = mapped_column(JSON)
    meta: Mapped[dict] = mapped_column(JSON)


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
