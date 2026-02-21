from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    path.write_text(content, encoding="utf-8")


def write_sources_csv(path: Path, sources: list[dict]) -> None:
    headers = [
        "source_id",
        "source_type",
        "title",
        "channel",
        "url",
        "published_at",
        "duration_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in sources:
            writer.writerow(row)
