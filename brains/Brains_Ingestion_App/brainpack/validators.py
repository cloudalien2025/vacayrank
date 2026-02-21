from __future__ import annotations

import json
from pathlib import Path
from jsonschema import Draft202012Validator, RefResolver


class SchemaValidationError(Exception):
    pass


def _load_schema(schema_dir: Path, filename: str) -> dict:
    return json.loads((schema_dir / filename).read_text(encoding="utf-8"))


def _schema_store(schema_dir: Path) -> dict:
    store = {}
    for path in schema_dir.glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        schema_id = schema.get("$id")
        if schema_id:
            store[schema_id] = schema
        store[f"file://{path.resolve()}"] = schema
        store[path.name] = schema
    return store


def build_validator(schema_dir: Path, filename: str) -> Draft202012Validator:
    root_schema = _load_schema(schema_dir, filename)
    base_uri = f"file://{schema_dir.resolve()}/"
    resolver = RefResolver(base_uri=base_uri, referrer=root_schema, store=_schema_store(schema_dir))
    return Draft202012Validator(root_schema, resolver=resolver)


def validate_or_raise(payload: dict, validator: Draft202012Validator, label: str) -> None:
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if not errors:
        return
    lines = [f"{label}: schema validation failed"]
    for err in errors:
        ptr = "/".join(str(part) for part in err.path) or "<root>"
        lines.append(f"- {ptr}: {err.message}")
    raise SchemaValidationError("\n".join(lines))
