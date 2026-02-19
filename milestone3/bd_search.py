from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None


def build_search_payload(candidate: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    payload = {}
    if config.get("use_search_keyword", True):
        payload["search_keyword"] = candidate.get("candidate_name") or candidate.get("name") or ""
    if config.get("use_email") and candidate.get("email"):
        payload["email"] = candidate["email"]
    if config.get("use_website", True) and candidate.get("candidate_website"):
        payload["website"] = candidate["candidate_website"]
    if config.get("strict_city_geo_filter", True) and candidate.get("city_key"):
        payload["city"] = candidate["city_key"]
    return {k: v for k, v in payload.items() if str(v).strip()}


def pick_best_match(results: List[Dict[str, Any]], candidate: Dict[str, Any]) -> Tuple[int | None, float]:
    if not results:
        return None, 0.0
    target = (candidate.get("candidate_name") or "").lower()
    best_id, best_score = None, 0.0
    for row in results:
        name = str(row.get("name") or row.get("company") or "").lower()
        if fuzz:
            score = float(fuzz.token_sort_ratio(target, name))
        else:
            score = 100.0 if target == name else 0.0
        if score > best_score:
            best_score = score
            best_id = int(row.get("user_id") or row.get("id") or 0) or None
    return best_id, best_score


def cache_search_pattern(base_url: str, config: Dict[str, Any], path: str = "cache/search_payload_patterns.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    current = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            current = json.load(f)
    current[base_url] = config
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
