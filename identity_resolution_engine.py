from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rapidfuzz import fuzz


@dataclass
class IdentityDecision:
    action: str
    best_match_user_id: Optional[int]
    score: float
    evidence: Dict[str, Any]


def _domain(url: str) -> str:
    if not url:
        return ""
    return (urlparse(url).hostname or "").replace("www.", "").lower()


def resolve_identity(candidate: Dict[str, Any], inventory_rows: List[Dict[str, Any]], threshold: int = 86) -> IdentityDecision:
    best_score = 0.0
    best_user_id: Optional[int] = None
    best_evidence: Dict[str, Any] = {}

    for listing in inventory_rows:
        name_exact = (candidate.get("business_name") or "").strip().lower() == (listing.get("business_name") or "").strip().lower()
        website_match = _domain(candidate.get("website", "")) and _domain(candidate.get("website", "")) == _domain(listing.get("website", ""))
        fuzzy_name = float(fuzz.token_set_ratio(candidate.get("business_name", ""), listing.get("business_name", "")))
        address_similarity = float(fuzz.partial_ratio(candidate.get("address", ""), listing.get("address", "")))
        phone_match = bool(candidate.get("phone") and listing.get("phone") and candidate.get("phone") == listing.get("phone"))

        score = max(fuzzy_name, address_similarity)
        if name_exact:
            score = max(score, 100.0)
        if website_match:
            score = max(score, 98.0)
        if phone_match:
            score = max(score, 96.0)

        if score > best_score:
            best_score = score
            best_user_id = int(listing.get("user_id")) if listing.get("user_id") else None
            best_evidence = {
                "name_exact": name_exact,
                "website_match": website_match,
                "fuzzy_name": round(fuzzy_name, 2),
                "address_similarity": round(address_similarity, 2),
                "phone_match": phone_match,
            }

    action = "CREATE" if best_score < threshold else "UPDATE"
    return IdentityDecision(action=action, best_match_user_id=best_user_id, score=round(best_score, 2), evidence=best_evidence)
