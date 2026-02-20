from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
from rapidfuzz import fuzz

from inventory_engine import InventoryBundle

SERP_API_URL = "https://serpapi.com/search.json"


@dataclass
class SerpGapResult:
    rows: List[Dict[str, Any]]


def _domain(url: str) -> str:
    if not url:
        return ""
    return (urlparse(url).hostname or "").replace("www.", "").lower()


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _extract_organic(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "business_name": item.get("title") or item.get("displayed_link") or "",
        "address": item.get("address") or item.get("snippet") or "",
        "website": item.get("link") or "",
        "category_classification": item.get("type") or "unknown",
        "geo_location": item.get("address") or "",
    }


def _classify_match(candidate: Dict[str, Any], listing: Dict[str, Any], fuzzy_threshold: int = 86) -> Tuple[str, float]:
    c_name = _norm(candidate.get("business_name", ""))
    l_name = _norm(listing.get("business_name", ""))
    name_exact = c_name and c_name == l_name
    fuzzy_score = float(fuzz.token_sort_ratio(c_name, l_name)) if c_name and l_name else 0.0
    website_match = _domain(candidate.get("website", "")) and _domain(candidate.get("website", "")) == _domain(listing.get("website", ""))
    address_score = float(fuzz.partial_ratio(_norm(candidate.get("address", "")), _norm(listing.get("address", "")))) if candidate.get("address") and listing.get("address") else 0.0
    phone_match = bool(candidate.get("phone") and listing.get("phone") and candidate.get("phone") == listing.get("phone"))

    if name_exact or website_match:
        return "Already Exists", 100.0
    if fuzzy_score >= fuzzy_threshold or address_score >= 88 or phone_match:
        return "Possible Match", max(fuzzy_score, address_score)
    return "Missing", max(fuzzy_score, address_score)


def run_serp_gap_analysis(
    serp_api_key: str,
    inventory: InventoryBundle,
    category_geo_pairs: List[Tuple[str, str]],
    search_depth: int = 10,
    allow_secondary_categories: bool = False,
) -> SerpGapResult:
    rows: List[Dict[str, Any]] = []
    for category, geo in category_geo_pairs:
        query = f"{category} in {geo}"
        response = requests.get(
            SERP_API_URL,
            params={"engine": "google", "q": query, "api_key": serp_api_key, "num": search_depth},
            timeout=20,
        )
        data = response.json() if response.ok else {}
        organic = data.get("organic_results", [])
        inventory_rows = inventory.inventory_index.get((category.strip().lower(), geo.strip().lower()), [])

        for item in organic:
            candidate = _extract_organic(item)
            best_status = "Missing"
            best_score = 0.0
            best_user_id = None
            out_of_scope = geo.lower() not in _norm(candidate.get("geo_location", "") + " " + candidate.get("address", "")) and candidate.get("geo_location")
            category_misalignment = False

            for listing in inventory_rows:
                if not allow_secondary_categories and listing.get("primary_category", "").strip().lower() != category.strip().lower():
                    continue
                status, score = _classify_match(candidate, listing)
                if score > best_score:
                    best_score = score
                    best_status = status
                    best_user_id = listing.get("user_id")

            if out_of_scope:
                best_status = "Out of Scope (wrong geo)"
            if best_status == "Missing" and inventory_rows:
                category_misalignment = True
                best_status = "Category Misalignment"

            rows.append(
                {
                    "category": category,
                    "geo": geo,
                    "query": query,
                    "business_name": candidate.get("business_name"),
                    "address": candidate.get("address"),
                    "website": candidate.get("website"),
                    "classification": candidate.get("category_classification"),
                    "status": best_status,
                    "best_score": round(best_score, 2),
                    "best_match_user_id": best_user_id,
                    "category_misalignment": category_misalignment,
                }
            )
    return SerpGapResult(rows=rows)
