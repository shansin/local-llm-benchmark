"""Small shared helpers with no dependencies on the rest of the package."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def safe_dirname(model_name: str) -> str:
    """Turn a model name into a filesystem-safe directory name."""
    return model_name.replace(":", "_").replace("/", "_")
