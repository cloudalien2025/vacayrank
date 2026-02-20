from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from milestone3.bd_client import BDClient


@dataclass
class StructuralAuditResult:
    report: Dict[str, Any]
    risk_level: str


def _extract_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("data", "results", "items", "users"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def _read_config_or_api(client: BDClient, endpoint: str, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    response = client.request_get(endpoint)
    if not response.ok:
        return fallback
    rows = _extract_list(response.json_data)
    return rows or fallback


def run_structural_audit(
    client: BDClient,
    inventory_records: List[Dict[str, Any]],
    settings_override: Dict[str, Any],
) -> StructuralAuditResult:
    categories = settings_override.get("categories") or _read_config_or_api(client, "/api/v2/category/search", [])
    subcategories = settings_override.get("subcategories") or _read_config_or_api(client, "/api/v2/subcategory/search", [])
    membership_plans = settings_override.get("membership_plans") or _read_config_or_api(client, "/api/v2/subscription/search", [])

    settings = {
        "pretty_url": settings_override.get("pretty_url", True),
        "google_api_enabled": settings_override.get("google_api_enabled", False),
        "service_area_logic": settings_override.get("service_area_logic", True),
        "multi_location_logic": settings_override.get("multi_location_logic", False),
        "hidden_profile_addons": settings_override.get("hidden_profile_addons", []),
        "business_toolkit_visibility_per_plan": settings_override.get("business_toolkit_visibility_per_plan", {}),
    }

    category_names = {str(c.get("name") or c.get("category") or "").strip().lower() for c in categories}
    subcat_names = [str(s.get("name") or s.get("subcategory") or "").strip().lower() for s in subcategories if s]

    empty_categories = []
    overlap_candidates = []
    category_counts: Dict[str, int] = {}
    for listing in inventory_records:
        key = str(listing.get("primary_category") or "uncategorized").strip().lower()
        category_counts[key] = category_counts.get(key, 0) + 1

    for name in sorted(category_names):
        if not name:
            continue
        if category_counts.get(name, 0) == 0:
            empty_categories.append(name)

    for name in subcat_names:
        if name and name in category_names:
            overlap_candidates.append(name)

    wrong_geo_count = 0
    service_area_distortion_count = 0
    for listing in inventory_records:
        if not listing.get("location"):
            wrong_geo_count += 1
        if settings["service_area_logic"] and listing.get("service_areas") and listing.get("city"):
            service_areas = {x.lower() for x in listing.get("service_areas", [])}
            if listing["city"].lower() not in service_areas:
                service_area_distortion_count += 1

    subcategory_jungle = len(subcat_names) > max(50, len(category_names) * 4)
    category_inflation = sum(1 for _, count in category_counts.items() if count <= 1)

    findings = {
        "top_level_categories": categories,
        "subcategories": subcategories,
        "membership_plans": membership_plans,
        "hidden_profile_addons": settings["hidden_profile_addons"],
        "pretty_url": settings["pretty_url"],
        "google_api_enabled": settings["google_api_enabled"],
        "service_area_logic": settings["service_area_logic"],
        "multi_location_logic": settings["multi_location_logic"],
        "business_toolkit_visibility_per_plan": settings["business_toolkit_visibility_per_plan"],
        "risks": {
            "subcategory_jungle_risk": subcategory_jungle,
            "empty_categories": empty_categories,
            "overlapping_categories": sorted(set(overlap_candidates)),
            "service_area_distortions": service_area_distortion_count,
            "listings_in_wrong_geo": wrong_geo_count,
            "category_inflation_count": category_inflation,
        },
    }

    score = 0
    score += 2 if subcategory_jungle else 0
    score += 2 if empty_categories else 0
    score += 2 if overlap_candidates else 0
    score += 2 if service_area_distortion_count > 10 else (1 if service_area_distortion_count else 0)
    score += 2 if wrong_geo_count > 10 else (1 if wrong_geo_count else 0)
    score += 2 if category_inflation > 20 else (1 if category_inflation > 0 else 0)

    risk_level = "Low" if score <= 2 else "Medium" if score <= 6 else "High"
    return StructuralAuditResult(report=findings, risk_level=risk_level)
