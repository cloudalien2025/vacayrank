from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from milestone3.bd_client import BDClient, BDClientError, BDResponse

__all__ = [
    "InventoryFetchRateLimited",
    "RateLimiter",
    "InventoryBundle",
    "fetch_inventory_index",
    "load_inventory_from_cache",
    "cache_inventory_to_disk",
    "inventory_to_csv",
]


class InventoryFetchRateLimited(RuntimeError):
    def __init__(self, message: str, *, last_page: int, attempts: int, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.error_type = "RATE_LIMIT"
        self.last_page = last_page
        self.attempts = attempts
        self.retry_after = retry_after


@dataclass
class InventoryBundle:
    records: List[Dict[str, Any]]
    inventory_index: Dict[Tuple[str, str], List[Dict[str, Any]]]
    summary: Dict[str, Any]
    status: str = "complete"
    meta: Dict[str, Any] | None = None


class RateLimiter:
    def __init__(self, rpm: int = 30) -> None:
        self.rpm = max(1, int(rpm))
        self.min_interval_sec = 60.0 / float(self.rpm)
        self._last_request_ts: float | None = None

    def wait(self) -> float:
        now = time.monotonic()
        jitter = self.min_interval_sec * random.uniform(-0.1, 0.1)
        target_interval = max(0.0, self.min_interval_sec + jitter)
        wait_s = 0.0
        if self._last_request_ts is not None:
            elapsed = now - self._last_request_ts
            wait_s = max(0.0, target_interval - elapsed)
        if wait_s > 0:
            time.sleep(wait_s)
        self._last_request_ts = time.monotonic()
        return wait_s


def _build_inventory_bundle(records: List[Dict[str, Any]], *, status: str = "complete", meta: Dict[str, Any] | None = None) -> InventoryBundle:
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
    return InventoryBundle(records=records, inventory_index=dict(inventory_index), summary=summary, status=status, meta=meta or {})


def _parse_retry_after(response: BDResponse) -> float | None:
    retry_after = (response.headers or {}).get("Retry-After")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return None


def _request_with_backoff(
    client: BDClient,
    payload: Dict[str, Any],
    limiter: RateLimiter,
    *,
    max_attempts: int = 6,
    max_5xx_attempts: int = 3,
) -> BDResponse:
    retry_after_hint: float | None = None
    for attempt in range(1, max_attempts + 1):
        pre_wait = limiter.wait()
        response = client.search_users(payload)
        evidence = client.evidence_log[-1] if client.evidence_log else None
        if evidence is not None:
            evidence["attempt_no"] = attempt
            evidence["limiter_rpm"] = limiter.rpm
            evidence["wait_s"] = round(pre_wait, 3)

        if 200 <= response.status_code < 300:
            if evidence is not None and "error_type" not in evidence:
                evidence["error_type"] = "NONE"
            return response

        is_429 = response.status_code == 429
        is_5xx = 500 <= response.status_code <= 599
        if not (is_429 or is_5xx):
            return response

        if is_429:
            retry_after = _parse_retry_after(response)
            retry_after_hint = retry_after if retry_after is not None else retry_after_hint
            backoff = retry_after if retry_after is not None else min(60.0, 2.0 * (2 ** (attempt - 1)))
            backoff += random.uniform(0, 0.5)
            if evidence is not None:
                evidence["error_type"] = "RATE_LIMIT"
                evidence["retry_after_s"] = retry_after
                evidence["backoff_s"] = round(backoff, 3)
            if attempt == max_attempts:
                break
            time.sleep(backoff)
            continue

        if is_5xx:
            backoff = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            if evidence is not None:
                evidence["error_type"] = "UPSTREAM_5XX"
                evidence["backoff_s"] = round(backoff, 3)
            if attempt >= max_5xx_attempts:
                return response
            time.sleep(backoff)

    raise InventoryFetchRateLimited(
        "Too many API requests per minute",
        last_page=int(payload.get("page", 1)) - 1,
        attempts=max_attempts,
        retry_after=retry_after_hint,
    )


def _write_inventory_progress(progress_path: Path, *, last_page: int, records: int) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "last_page": int(last_page),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "records": int(records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_inventory_progress(path: str) -> Dict[str, Any]:
    progress_path = Path(path)
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _decode_presentation_fields(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Placeholder for future UI decode pass (URL encoded values, etc).
    return records


def _normalize_records(payload: Any) -> tuple[List[Dict[str, Any]], List[str], str]:
    parse_errors: List[str] = []
    bd_status = ""

    if isinstance(payload, list):
        return payload, parse_errors, bd_status

    if isinstance(payload, dict):
        bd_status = str(payload.get("status", "") or "")
        if bd_status and bd_status.lower() != "success":
            detail_parts = [bd_status]
            for key in ("message", "error"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    detail_parts.append(value.strip())
            parse_errors.append("BD status not success: " + " | ".join(detail_parts))

        if isinstance(payload.get("message"), list):
            return payload["message"], parse_errors, bd_status
        if isinstance(payload.get("data"), list):
            return payload["data"], parse_errors, bd_status
        return [], parse_errors, bd_status

    return [], ["Unexpected payload shape from /api/v2/user/search"], bd_status


def _apply_response_classification(response: BDResponse, client: BDClient) -> None:
    if not client.evidence_log:
        return

    content_type = (response.content_type or "").lower()
    snippet = (response.text or "")[:500]
    lower_snippet = snippet.lower()
    evidence = client.evidence_log[-1]

    if response.status_code == 403 and ("cloudflare" in lower_snippet or "attention required" in lower_snippet):
        evidence["error_type"] = "CLOUDFLARE_BLOCK"
        evidence["action_hint"] = "Cloudflare custom rule blocking non-human browsers; exclude /api/"
        client.annotate_last_evidence(parse_error="CLOUDFLARE_BLOCK")
        return

    if response.status_code in {401, 403}:
        evidence["error_type"] = "AUTH_FORBIDDEN"
        client.annotate_last_evidence(parse_error="AUTH_FORBIDDEN")
        return

    if response.status_code == 200 and ("text/html" in content_type or response.text.lstrip().startswith("<")):
        evidence["error_type"] = "BD_HTML_RESPONSE"
        evidence["action_hint"] = "Ensure output_type=array; verify endpoint/method"
        client.annotate_last_evidence(parse_error="BD_HTML_RESPONSE")


def _update_last_evidence_parse(client: BDClient, *, records_parsed: int, bd_status: str, parse_errors: List[str], payload: Any) -> None:
    client.annotate_last_evidence(records_parsed=records_parsed)
    if not client.evidence_log:
        return
    evidence = client.evidence_log[-1]
    if bd_status:
        evidence["bd_status"] = bd_status
    evidence["response_json_type"] = type(payload).__name__
    for parse_error in parse_errors:
        client.annotate_last_evidence(records_parsed=records_parsed, parse_error=parse_error)


def fetch_inventory_index(
    client: BDClient,
    page_size: int = 100,
    max_pages: int = 200,
    delay_seconds: float = 0.0,
    requests_per_minute: int = 30,
    start_page: int = 1,
    cache_path: str | None = None,
    progress_path: str = "cache/inventory_progress.json",
) -> InventoryBundle:
    records: List[Dict[str, Any]] = []
    progress_file = Path(progress_path)
    if cache_path and Path(cache_path).exists():
        existing = load_inventory_from_cache(cache_path)
        records = list(existing.records)

    limiter = RateLimiter(rpm=requests_per_minute)
    end_page = start_page + max_pages - 1
    for page in range(start_page, end_page + 1):
        payload = {"output_type": "array", "page": page, "limit": page_size}
        try:
            response = _request_with_backoff(client, payload, limiter)
        except InventoryFetchRateLimited as exc:
            _write_inventory_progress(progress_file, last_page=max(start_page - 1, page - 1), records=len(records))
            partial_bundle = _build_inventory_bundle(
                records,
                status="partial",
                meta={
                    "error_type": exc.error_type,
                    "last_page": max(start_page - 1, page - 1),
                    "resume_from_page": page,
                    "attempts": exc.attempts,
                    "retry_after_s": exc.retry_after,
                    "message": f"Rate limited; resume from page {page} after cooldown",
                },
            )
            if cache_path:
                cache_inventory_to_disk(partial_bundle, cache_path)
            return partial_bundle
        _apply_response_classification(response, client)

        if not response.ok:
            raise BDClientError(f"Inventory fetch failed with HTTP {response.status_code}")

        if response.status_code == 200 and (
            "text/html" in (response.content_type or "").lower() or response.text.lstrip().startswith("<")
        ):
            raise BDClientError("Brilliant Directories returned HTML instead of JSON. Ensure endpoint and output_type=array.")

        rows, parse_errors, bd_status = _normalize_records(response.json_data)
        normalized = _decode_presentation_fields([normalize_user(row, client.base_url) for row in rows])
        records.extend(normalized)
        _update_last_evidence_parse(client, records_parsed=len(rows), bd_status=bd_status, parse_errors=parse_errors, payload=response.json_data)

        if len(rows) == 0:
            break
        _write_inventory_progress(progress_file, last_page=page, records=len(records))
        if cache_path:
            cache_inventory_to_disk(_build_inventory_bundle(records), cache_path)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return _build_inventory_bundle(records)


def cache_inventory_to_disk(bundle: InventoryBundle, path: str, *, base_url_used: str = "", payload_defaults: Dict[str, Any] | None = None) -> None:
    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_url_used": base_url_used,
            "endpoint": "/api/v2/user/search",
            "payload_defaults": payload_defaults or {"output_type": "array", "limit": 100},
            "total_records": len(bundle.records),
            "status": bundle.status,
            "meta": bundle.meta or {},
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
        status=(payload.get("metadata", {}) or {}).get("status", "complete"),
        meta=(payload.get("metadata", {}) or {}).get("meta", {}),
    )


def inventory_to_csv(records: List[Dict[str, Any]]) -> str:
    if not records:
        return ""
    frame = pd.DataFrame(records).drop(columns=["raw"], errors="ignore")
    preferred = ["user_id", "email", "name", "status", "plan", "profile_url"]
    ordered = [column for column in preferred if column in frame.columns] + [column for column in frame.columns if column not in preferred]
    return frame[ordered].to_csv(index=False)
