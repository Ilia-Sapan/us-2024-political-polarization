"""Utility helpers used by multiple modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import logging


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure consistent app-wide logging."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def dump_json(payload: dict[str, Any], output_path: Path) -> None:
    """Serialize a dictionary to JSON with UTF-8 encoding."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(input_path: Path) -> dict[str, Any]:
    """Load a JSON file from disk."""

    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown(output_path: Path, content: str) -> None:
    """Write markdown content to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

