from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Any, Dict, List


def _stable_row_id(parts: List[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _read_table(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("rows", [])
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_m2_artifacts(inventory_path: str, missing_path: str, possible_path: str, duplicates_path: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "inventory": _read_table(inventory_path),
        "missing": _read_table(missing_path),
        "possible": _read_table(possible_path),
        "duplicates": _read_table(duplicates_path),
    }


def build_queue_tables(artifacts: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match_type, key in (("missing", "missing"), ("possible_match", "possible"), ("duplicate", "duplicates")):
        for r in artifacts.get(key, []):
            rows.append(
                {
                    "row_id": _stable_row_id([match_type, str(r.get("candidate_name", "")), str(r.get("candidate_website", ""))]),
                    "category_key": r.get("category_key", r.get("category", "")),
                    "city_key": r.get("city_key", r.get("city", "")),
                    "candidate_name": r.get("candidate_name", r.get("name", "")),
                    "candidate_address": r.get("candidate_address", r.get("address", "")),
                    "candidate_website": r.get("candidate_website", r.get("website", "")),
                    "candidate_phone": r.get("candidate_phone", r.get("phone", "")),
                    "source_url": r.get("source_url", ""),
                    "match_type": match_type,
                    "best_match_user_id": r.get("best_match_user_id"),
                    "match_score": float(r.get("match_score", 0) or 0),
                    "bd_user_ids": r.get("bd_user_ids", []),
                    "proposed_patch": r.get("proposed_patch", {}),
                    "status": "pending",
                    "notes": "",
                }
            )
    return rows


def update_row_status(rows: List[Dict[str, Any]], row_id: str, status: str, notes: str = "") -> None:
    for row in rows:
        if row["row_id"] == row_id:
            row["status"] = status
            row["notes"] = notes
            return


def enforce_max_writes_per_session(writes_done: int, max_writes: int) -> bool:
    return writes_done < max_writes
