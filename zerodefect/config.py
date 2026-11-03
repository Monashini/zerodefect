"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    api_key: str | None = field(default_factory=lambda: os.environ.get("MODEL_API_KEY") or None)
    base_url: str = field(default_factory=lambda: os.environ.get("MODEL_API_BASE", "https://api.meta.ai/v1"))
    spark_model: str = field(default_factory=lambda: os.environ.get("ZD_SPARK_MODEL", "muse-spark-1.3"))
    sam_model: str = field(default_factory=lambda: os.environ.get("ZD_SAM_MODEL", "sam-3.1"))
    backbone_path: Path = field(
        default_factory=lambda: Path(os.environ.get("ZD_BACKBONE", ROOT / "models" / "resnet18-v1-7.onnx"))
    )
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ZD_DATA", ROOT / "data")))
    # Force local-only mode (no Meta Model API calls) even if a key is present.
    offline: bool = field(default_factory=lambda: os.environ.get("ZD_OFFLINE", "0") == "1")
    request_timeout: float = field(default_factory=lambda: float(os.environ.get("ZD_TIMEOUT", "60")))

    @property
    def meta_enabled(self) -> bool:
        return bool(self.api_key) and not self.offline


settings = Settings()
