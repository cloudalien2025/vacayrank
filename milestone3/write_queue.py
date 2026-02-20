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


def build_write_queue_from_m2(
    inventory_df,
    serp_gap_df,
    possible_matches_df,
    duplicates_obj,
    config: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    del inventory_df, config

    def _stable_row_id(match_type: str, row: Dict[str, Any]) -> str:
        seed = [
            match_type,
            str(row.get("category_key") or row.get("category") or ""),
            str(row.get("city_key") or row.get("city") or ""),
            str(row.get("candidate_name") or row.get("title") or ""),
            str(row.get("candidate_website") or row.get("website") or ""),
            str(row.get("best_match_user_id") or ""),
        ]
        return hashlib.sha256("|".join(seed).encode("utf-8")).hexdigest()[:16]

    def _normalize_row(match_type: str, row: Dict[str, Any]) -> Dict[str, Any]:
        proposed_patch = row.get("proposed_patch") if isinstance(row.get("proposed_patch"), dict) else {}
        if not proposed_patch:
            proposed_patch = {
                "company": row.get("candidate_name") or row.get("title") or "",
                "address1": row.get("candidate_address") or row.get("address") or "",
                "website": row.get("candidate_website") or row.get("website") or "",
                "phone_number": row.get("candidate_phone") or row.get("phone") or "",
            }
            proposed_patch = {k: v for k, v in proposed_patch.items() if str(v).strip()}

        raw_ids = row.get("bd_user_ids") or row.get("cluster_user_ids") or []
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except Exception:
                raw_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
        bd_user_ids = [int(x) for x in raw_ids if str(x).strip().isdigit()]

        return {
            "row_id": _stable_row_id(match_type, row),
            "match_type": match_type,
            "category_key": row.get("category_key") or row.get("category") or "",
            "city_key": row.get("city_key") or row.get("city") or "",
            "candidate_name": row.get("candidate_name") or row.get("title") or "",
            "candidate_address": row.get("candidate_address") or row.get("address") or "",
            "candidate_website": row.get("candidate_website") or row.get("website") or "",
            "candidate_phone": row.get("candidate_phone") or row.get("phone") or "",
            "source_url": row.get("source_url") or row.get("source") or "",
            "match_score": float(row.get("match_score") or row.get("best_match_score") or 0),
            "best_match_user_id": row.get("best_match_user_id"),
            "bd_user_ids": bd_user_ids,
            "proposed_patch": proposed_patch,
            "status": "pending",
            "notes": "",
        }

    missing_rows = [_normalize_row("missing", row) for row in serp_gap_df.to_dict(orient="records")]
    possible_rows = [_normalize_row("possible_match", row) for row in possible_matches_df.to_dict(orient="records")]

    duplicate_rows: List[Dict[str, Any]] = []
    if isinstance(duplicates_obj, dict):
        source_rows = duplicates_obj.get("clusters") or duplicates_obj.get("rows") or []
    else:
        source_rows = duplicates_obj or []
    for row in source_rows:
        duplicate_rows.append(_normalize_row("duplicate", row))

    return {
        "missing": missing_rows,
        "possible_matches": possible_rows,
        "duplicates": duplicate_rows,
    }


def update_row_status(rows: List[Dict[str, Any]], row_id: str, status: str, notes: str = "") -> None:
    for row in rows:
        if row["row_id"] == row_id:
            row["status"] = status
            row["notes"] = notes
            return


def enforce_max_writes_per_session(writes_done: int, max_writes: int) -> bool:
    return writes_done < max_writes
