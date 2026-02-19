from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditLog:
    def __init__(self, path: str = "logs/m3_audit.jsonl") -> None:
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log_event(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        self.entries.append(normalized)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        return normalized

    @staticmethod
    def _mask_headers(headers: Dict[str, str]) -> Dict[str, str]:
        out = dict(headers)
        if "X-Api-Key" in out:
            out["X-Api-Key"] = "***REDACTED***"
        return out

    def export_json(self) -> str:
        return json.dumps(self.entries, indent=2, ensure_ascii=False)

    def export_csv(self) -> str:
        if not self.entries:
            return ""
        fields = sorted({k for e in self.entries for k in e.keys()})
        buff = io.StringIO()
        writer = csv.DictWriter(buff, fieldnames=fields)
        writer.writeheader()
        for entry in self.entries:
            row = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in entry.items()}
            writer.writerow(row)
        return buff.getvalue()
