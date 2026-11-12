"""Application settings. Values come from environment variables (prefix ``MTI_``) or a ``.env`` file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "Mars Terrain Intelligence"
APP_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MTI_", env_file=str(REPO_ROOT / ".env"), extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    checkpoint_dir: Path = REPO_ROOT / "checkpoints"
    experiments_dir: Path = REPO_ROOT / "experiments"
    database_url: str = ""
    max_upload_mb: int = 20
    max_output_side: int = 1536
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    val_fraction: float = 0.1
    split_seed: int = 42
    # Registers a clearly labelled, untrained fixture model for automated UI tests only.
    enable_test_fixture: bool = False
    log_level: str = "INFO"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def imports_dir(self) -> Path:
        return self.data_dir / "raw" / "imports"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def analyses_dir(self) -> Path:
        return self.data_dir / "analyses"

    @property
    def index_path(self) -> Path:
        return self.processed_dir / "index.json"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'mti.sqlite3').as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.imports_dir, self.processed_dir, self.analyses_dir,
                  self.checkpoint_dir, self.experiments_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
