from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from milestone3.bd_client import BDClient, BDClientError


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
        "email": _first_non_empty(user, ("email",)),
        "name": _first_non_empty(user, ("name", "full_name", "business_name", "company", "listing_name")),
        "business_name": _first_non_empty(user, ("business_name", "company", "name", "listing_name")),
        "primary_category": primary_category,
        "secondary_categories": secondary_categories,
        "city": city,
        "state": state,
        "country": country,
        "location": ", ".join([x for x in [city, state, country] if x]),
        "service_areas": service_areas,
        "plan": _first_non_empty(user, ("membership_plan", "plan", "subscription_name", "subscription_id")),
        "status": active_status,
        "slug": slug,
        "profile_url": profile_url,
        "website": _first_non_empty(user, ("website", "website_url")),
        "phone": _first_non_empty(user, ("phone", "phone_number", "phone1")),
        "address": _first_non_empty(user, ("address", "address1", "street")),
        "raw": user,
    }


def normalize_records(payload: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    parse_errors: List[str] = []

    if isinstance(payload, list):
        return payload, parse_errors

    if isinstance(payload, dict):
        status = payload.get("status")
        if status is not None and str(status).lower() != "success":
            parse_errors.append(f"BD status not success: {status}")

        candidate_rows = None
        for key in ("message", "data", "results", "users"):
            value = payload.get(key)
            if isinstance(value, list):
                candidate_rows = value
                break

        if candidate_rows is not None:
            if len(candidate_rows) == 0 and str(status).lower() == "success":
                parse_errors.append("BD returned success but no records in expected keys (message/data).")
            return candidate_rows, parse_errors

        if str(status).lower() == "success":
            parse_errors.append("BD returned success but no records in expected keys (message/data).")

        message = payload.get("error") or payload.get("message")
        if isinstance(message, str) and message.strip():
            parse_errors.append(f"BD payload message: {message.strip()}")
        return [], parse_errors

    raise BDClientError("Unexpected payload shape from /api/v2/user/search")


def fetch_inventory_index(
    client: BDClient,
    page_size: int = 100,
    max_pages: int = 200,
    delay_seconds: float = 0.15,
) -> InventoryBundle:
    records: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = {"output_type": "array", "page": page, "limit": page_size}
        response = client.search_users(payload)
        content_type = (response.content_type or "").lower()
        is_html = "text/html" in content_type or response.text.lstrip().startswith("<")
        if is_html:
            client.annotate_last_evidence(parse_error="HTML response detected; output_type=array missing or endpoint returned HTML")
            raise BDClientError("Brilliant Directories returned HTML instead of JSON. Ensure output_type=array is present.")
        if not response.ok:
            client.annotate_last_evidence(parse_error=f"HTTP {response.status_code}")
            raise BDClientError(f"Inventory fetch failed with HTTP {response.status_code}")

        try:
            rows, parse_errors = normalize_records(response.json_data)
        except Exception as exc:
            client.annotate_last_evidence(parse_error=f"JSON parse error: {exc}")
            raise

        normalized = [normalize_user(row, client.base_url) for row in rows]
        records.extend(normalized)
        client.annotate_last_evidence(records_parsed=len(rows))
        for parse_error in parse_errors:
            client.annotate_last_evidence(records_parsed=len(rows), parse_error=parse_error)

        if len(rows) == 0:
            break
        if delay_seconds > 0:
            time.sleep(delay_seconds)

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
        plan_counter[row["plan"] or "Unknown"] += 1

    summary = {
        "total_members": len(records),
        "by_category": dict(cat_counter),
        "by_geo": dict(geo_counter),
        "status_distribution": dict(status_counter),
        "membership_plan_distribution": dict(plan_counter),
    }
    return InventoryBundle(records=records, inventory_index=dict(inventory_index), summary=summary)


def cache_inventory_to_disk(bundle: InventoryBundle, path: str, *, base_url_used: str = "", payload_defaults: Dict[str, Any] | None = None) -> None:
    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_url_used": base_url_used,
            "endpoint": "/api/v2/user/search",
            "payload_defaults": payload_defaults or {"output_type": "array", "limit": 100},
            "total_records": len(bundle.records),
        },
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
    preferred = ["user_id", "email", "name", "status", "plan", "profile_url"]
    ordered = [column for column in preferred if column in frame.columns] + [column for column in frame.columns if column not in preferred]
    return frame[ordered].to_csv(index=False)
