from dataclasses import dataclass
from typing import Any


@dataclass
class Source:
    source_id: str
    source_type: str
    title: str
    channel: str
    url: str
    published_at: str | None = None
    duration_seconds: int | None = None


@dataclass
class RunMetadata:
    run_id: str
    keyword: str
    started_at: str
    ended_at: str
    config: dict[str, Any]
    errors: list[str]
    env: dict[str, Any]
