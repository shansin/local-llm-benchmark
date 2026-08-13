"""Model discovery against the Ollama API."""

from __future__ import annotations

import re
from typing import Any

import requests

_SIZE_UNITS = {"B": 1e9, "M": 1e6, "K": 1e3, "G": 1e9, "T": 1e12, "": 1.0}

Model = dict[str, Any]


def param_sort_key(model: Model) -> float:
    """Sort key turning a parameter_size string like "27.8B" into a number."""
    size = model.get("details", {}).get("parameter_size", "0")
    match = re.match(r"([\d.]+)\s*([BKMGT]?)", str(size).upper())
    if not match:
        return 0.0
    value, unit = float(match.group(1)), match.group(2)
    return value * _SIZE_UNITS.get(unit, 1.0)


def is_embedding_model(model: Model) -> bool:
    """Embedding models can't answer prompts, so they're excluded from runs."""
    name = model.get("name", "").lower()
    families = [f.lower() for f in (model.get("details", {}).get("families") or [])]
    return "embed" in name or any("embed" in f for f in families)


def get_models(base_url: str) -> list[Model]:
    """Fetch available models from Ollama, excluding embedding models."""
    resp = requests.get(f"{base_url}/api/tags")
    resp.raise_for_status()
    models: list[Model] = resp.json().get("models", [])
    return sorted((m for m in models if not is_embedding_model(m)), key=param_sort_key)


def get_model_details(base_url: str, model_name: str) -> Model:
    """Fetch detailed model info."""
    resp = requests.post(f"{base_url}/api/show", json={"name": model_name})
    resp.raise_for_status()
    result: Model = resp.json()
    return result
