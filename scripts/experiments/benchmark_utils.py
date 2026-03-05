from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

MIN_FREE_SPACE_BYTES = 1 * 1024 ** 3


def ensure_free_space(path: Path, *, min_free_bytes: int = MIN_FREE_SPACE_BYTES) -> None:
    """Ensure the target path is on a filesystem with at least `min_free_bytes` free."""
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    target = candidate if candidate.exists() else Path.cwd()
    usage = shutil.disk_usage(str(target))
    if usage.free < min_free_bytes:
        raise RuntimeError(
            f"Insufficient disk space at {target}. Need {min_free_bytes} bytes free; only {usage.free} available."
        )


def safe_write_text(path: Path, content: str, description: str) -> tuple[bool, str | None]:
    """Write the provided text to disk while surfacing any errors."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True, None
    except OSError as exc:
        err = str(exc)
        print(f"[ERROR] Failed to write {description} ({path}): {err}", file=sys.stderr)
        return False, err


def write_text_or_raise(path: Path, content: str, description: str) -> None:
    ok, err = safe_write_text(path, content, description)
    if not ok:
        raise OSError(f"Unable to persist {description}: {err}")


def write_json_payload(path: Path, payload: Any, description: str, *, indent: int = 2) -> None:
    """Serialize a payload as JSON and persist it."""
    content = json.dumps(payload, indent=indent) + "\n"
    write_text_or_raise(path, content, description)


def write_lines(path: Path, lines: Iterable[str], description: str) -> None:
    """Persist a sequence of lines with newline separators."""
    content = "\n".join(lines) + "\n"
    write_text_or_raise(path, content, description)
