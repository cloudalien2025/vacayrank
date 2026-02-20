from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from milestone3.bd_client import BDClient


@dataclass
class InventoryBundle:
    records: List[Dict[str, Any]]
    inventory_index: Dict[Tuple[str, str], List[Dict[str, Any]]]
    summary: Dict[str, Any]


def _first_non_empty(payload: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def normalize_user(user: Dict[str, Any], base_url: str = "") -> Dict[str, Any]:
    city = _first_non_empty(user, ("city", "location_city"))
    state = _first_non_empty(user, ("state", "location_state"))
    country = _first_non_empty(user, ("country", "location_country"))
    primary_category = _first_non_empty(user, ("primary_category", "category", "category_name", "category_title"))
    secondary_categories = user.get("secondary_categories") or user.get("subcategories") or []
    if isinstance(secondary_categories, str):
        secondary_categories = [x.strip() for x in secondary_categories.split(",") if x.strip()]
    service_areas = user.get("service_areas") or user.get("service_area") or []
    if isinstance(service_areas, str):
        service_areas = [x.strip() for x in service_areas.split(",") if x.strip()]

    slug = _first_non_empty(user, ("slug", "profile_slug", "url_slug"))
    profile_url = _first_non_empty(user, ("profile_url", "url"))
    if not profile_url and slug and base_url:
        profile_url = f"{base_url.rstrip('/')}/{slug.lstrip('/')}"

    active_raw = str(user.get("active", user.get("status", ""))).strip().lower()
    active_status = "active" if active_raw in {"1", "true", "active", "yes"} else "inactive"

    return {
        "user_id": user.get("user_id") or user.get("id"),
        "business_name": _first_non_empty(user, ("business_name", "company", "name", "listing_name")),
        "primary_category": primary_category,
        "secondary_categories": secondary_categories,
        "city": city,
        "state": state,
        "country": country,
        "location": ", ".join([x for x in [city, state, country] if x]),
        "service_areas": service_areas,
        "membership_plan": _first_non_empty(user, ("membership_plan", "plan", "subscription_name", "subscription_id")),
        "status": active_status,
        "slug": slug,
        "profile_url": profile_url,
        "website": _first_non_empty(user, ("website", "website_url")),
        "phone": _first_non_empty(user, ("phone", "phone_number", "phone1")),
        "address": _first_non_empty(user, ("address", "address1", "street")),
        "raw": user,
    }


def fetch_inventory_index(client: BDClient, page_size: int = 200, max_pages: int = 200) -> InventoryBundle:
    records: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = {"page": page, "per_page": page_size}
        response = client.search_users(payload)
        if not response.ok:
            break
        data = response.json_data
        rows = data.get("data") or data.get("users") or data.get("results") or []
        if not rows:
            break
        normalized = [normalize_user(row, client.base_url) for row in rows]
        records.extend(normalized)
        if len(rows) < page_size:
            break

    inventory_index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    cat_counter: Counter[str] = Counter()
    geo_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    plan_counter: Counter[str] = Counter()

    for row in records:
        category = (row["primary_category"] or "Uncategorized").strip().lower()
        geo = (row["location"] or "Unknown").strip().lower()
        inventory_index[(category, geo)].append(row)
        cat_counter[row["primary_category"] or "Uncategorized"] += 1
        geo_counter[row["location"] or "Unknown"] += 1
        status_counter[row["status"]] += 1
        plan_counter[row["membership_plan"] or "Unknown"] += 1

    summary = {
        "total_listings": len(records),
        "by_category": dict(cat_counter),
        "by_geo": dict(geo_counter),
        "status_distribution": dict(status_counter),
        "membership_plan_distribution": dict(plan_counter),
    }
    return InventoryBundle(records=records, inventory_index=dict(inventory_index), summary=summary)


def cache_inventory_to_disk(bundle: InventoryBundle, path: str) -> None:
    payload = {
        "records": bundle.records,
        "summary": bundle.summary,
        "inventory_index": {f"{k[0]}|||{k[1]}": v for k, v in bundle.inventory_index.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_inventory_from_cache(path: str) -> InventoryBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    inventory_index = {}
    for key, value in payload.get("inventory_index", {}).items():
        category, geo = key.split("|||", 1)
        inventory_index[(category, geo)] = value
    return InventoryBundle(
        records=payload.get("records", []),
        inventory_index=inventory_index,
        summary=payload.get("summary", {}),
    )


def inventory_to_csv(records: List[Dict[str, Any]]) -> str:
    if not records:
        return ""
    frame = pd.DataFrame(records).drop(columns=["raw"], errors="ignore")
    return frame.to_csv(index=False)
