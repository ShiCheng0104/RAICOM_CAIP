from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/fraudsim.yaml")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("FRAUDSIM_CONFIG", DEFAULT_CONFIG_PATH))
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if os.getenv("FRAUDSIM_DATASET"):
        config.setdefault("dataset", {})["name"] = os.environ["FRAUDSIM_DATASET"]
    if os.getenv("FRAUDSIM_DATA_ROOT"):
        config.setdefault("dataset", {})["root"] = os.environ["FRAUDSIM_DATA_ROOT"]
    if os.getenv("FRAUDSIM_MODEL_PATH"):
        config.setdefault("model", {})["path"] = os.environ["FRAUDSIM_MODEL_PATH"]
    if os.getenv("FRAUDSIM_KAFKA_BOOTSTRAP"):
        config.setdefault("kafka", {})["bootstrap_servers"] = os.environ["FRAUDSIM_KAFKA_BOOTSTRAP"]
    if os.getenv("FRAUDSIM_REDIS_URL"):
        config.setdefault("redis", {})["url"] = os.environ["FRAUDSIM_REDIS_URL"]
    if os.getenv("FRAUDSIM_API_URL"):
        config.setdefault("api", {})["url"] = os.environ["FRAUDSIM_API_URL"]

    return config


def dataset_dir(config: dict[str, Any], dataset: str | None = None) -> Path:
    root = Path(config.get("dataset", {}).get("root", "data/processed"))
    name = dataset or config.get("dataset", {}).get("name", "fp_fraudsim_injected")
    return root / name


def model_dir(config: dict[str, Any], model: str | None = None) -> Path:
    if model:
        return Path("models") / model / "latest"
    return Path(config.get("model", {}).get("path", "models/lightgbm/latest"))
