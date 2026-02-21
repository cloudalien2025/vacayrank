from __future__ import annotations

from datetime import datetime, timezone
import re


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    compact = re.sub(r"\s+", "_", value.strip().lower())
    return re.sub(r"[^a-z0-9_\-]", "", compact)
