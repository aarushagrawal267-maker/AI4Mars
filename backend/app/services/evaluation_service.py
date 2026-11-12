"""Runs held-out evaluations in the background, stores results and builds CSV/JSON reports."""
from __future__ import annotations

import csv
import io
import json
import logging

from ml.classes import CLASS_KEYS, CLASS_LABELS
from ml.evaluation.evaluate import evaluate_model, records_for

from ..core.config import APP_VERSION
from ..database.db import Database
from ..database.models import Evaluation, Experiment
from .dataset_service import DatasetService
from .jobs import Job, JobRunner
from .model_registry import ModelRegistry

log = logging.getLogger(__name__)
JOB_KIND = "evaluation"
SUMMARY_KEYS = ("pixel_accuracy", "mean_iou", "mean_dice", "mean_precision", "mean_recall")


class EvaluationService:
    def __init__(self, db: Database, jobs: JobRunner, registry: ModelRegistry, dataset: DatasetService):
        self.db, self.jobs, self.registry, self.dataset = db, jobs, registry, dataset

    def start(self, experiment_id: str, split: str, variant: str | None, max_images: int | None) -> Job:
        index = self.dataset.require_index()
        recs, label_key = records_for(index, split, variant)  # validates split/variant early
        if not recs:
            raise LookupError(f"The loaded dataset has no labelled images in the {split} split.")
        model, meta, device, info = self.registry.load(experiment_id)

        def run(job: Job) -> dict:
            with self.registry.inference_lock:
                metrics = evaluate_model(model, int(meta["resolution"]), index, self.dataset.settings.raw_dir,
                                         split=split, variant=variant, device=device, max_images=max_images,
                                         progress=lambda f, m: job.update(f, m, "evaluate"),
                                         should_stop=job.cancel_event.is_set)
            metrics["model"] = {
                "experiment_id": info["id"], "name": info["name"], "arch": info["arch"],
                "is_fixture": info["is_fixture"], "parameters": meta.get("parameters"),
                "checkpoint_bytes": meta.get("file_size_bytes"),
                "checkpoint_mb": round(meta.get("file_size_bytes", 0) / 1e6, 3),
                "model_config": meta.get("model_config"), "checkpoint_epoch": meta.get("extra", {}).get("epoch"),
            }
            metrics["dataset_fingerprint"] = index.get("fingerprint")
            metrics["subset_of_split"] = max_images is not None and max_images < len(recs)
            with self.db.session() as s:
                row = Evaluation(experiment_id=info["id"], split=split, label_variant=metrics["label_variant"],
                                 images=metrics["images"], metrics=metrics)
                s.add(row)
                s.flush()
                return {"evaluation_id": row.id}

        return self.jobs.submit(JOB_KIND, run, {"experiment_id": experiment_id, "split": split,
                                                "variant": label_key if split == "test" else None})

    # ---------------------------------------------------------------- queries
    @staticmethod
    def _public(e: Evaluation, exp: Experiment | None) -> dict:
        return {"id": e.id, "experiment_id": e.experiment_id, "experiment_name": exp.name if exp else None,
                "arch": exp.arch if exp else e.metrics.get("model", {}).get("arch"),
                "is_fixture": bool(exp.is_fixture) if exp else False,
                "created_at": e.created_at.isoformat(), "split": e.split, "label_variant": e.label_variant,
                "images": e.images, "metrics": e.metrics}

    def list(self, experiment_id: str | None = None) -> list[dict]:
        with self.db.session() as s:
            q = s.query(Evaluation)
            if experiment_id:
                q = q.filter(Evaluation.experiment_id == experiment_id)
            return [self._public(e, e.experiment) for e in q.order_by(Evaluation.created_at.desc()).all()]

    def get(self, evaluation_id: int) -> dict:
        with self.db.session() as s:
            e = s.get(Evaluation, evaluation_id)
            if e is None:
                raise KeyError(evaluation_id)
            return self._public(e, e.experiment)

    def delete(self, evaluation_id: int) -> None:
        with self.db.session() as s:
            e = s.get(Evaluation, evaluation_id)
            if e is None:
                raise KeyError(evaluation_id)
            s.delete(e)

    # ---------------------------------------------------------------- reports
    def report_json(self, evaluation_ids: list[int]) -> str:
        items = [self.get(i) for i in evaluation_ids]
        return json.dumps({"application": "Mars Terrain Intelligence", "version": APP_VERSION,
                           "note": "All values were computed by this application from held-out labelled data. "
                                   "Unlabelled pixels (255) are excluded.",
                           "evaluations": items}, indent=2)

    def report_csv(self, evaluation_ids: list[int]) -> str:
        """Long-format CSV: one row per (evaluation, section, class, metric)."""
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["evaluation_id", "experiment_id", "experiment_name", "arch", "split", "label_variant", "images",
                    "section", "class", "metric", "value"])
        label = dict(zip(CLASS_KEYS, CLASS_LABELS))
        for ev in (self.get(i) for i in evaluation_ids):
            m = ev["metrics"]
            base = [ev["id"], ev["experiment_id"], ev["experiment_name"], ev["arch"], ev["split"],
                    ev["label_variant"], ev["images"]]
            for k in SUMMARY_KEYS:
                w.writerow(base + ["summary", "", k, m.get(k)])
            w.writerow(base + ["summary", "", "evaluated_pixels", m.get("evaluated_pixels")])
            for k in CLASS_KEYS:
                for metric in ("iou", "dice", "precision", "recall", "support_pixels", "predicted_pixels"):
                    w.writerow(base + ["per_class", label[k], metric, m["per_class"][k][metric]])
            for i, tk in enumerate(CLASS_KEYS):
                for j, pk in enumerate(CLASS_KEYS):
                    w.writerow(base + ["confusion_matrix", f"true={label[tk]}|pred={label[pk]}", "pixels",
                                       m["confusion_matrix"][i][j]])
            inf = m.get("inference", {})
            for k in ("device", "forward_ms_mean", "forward_ms_median", "forward_ms_p95", "end_to_end_ms_mean",
                      "images_per_second"):
                w.writerow(base + ["inference", "", k, inf.get(k)])
            mod = m.get("model", {})
            for k in ("parameters", "checkpoint_mb", "checkpoint_epoch"):
                w.writerow(base + ["model", "", k, mod.get(k)])
        return buf.getvalue()
