from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import requests

from inventory_engine import RateLimiter
from milestone3.bd_client import BDClient

LISTINGS_INVENTORY_CACHE = Path("cache/listings_inventory.json")
LISTINGS_PROGRESS_CACHE = Path("cache/listings_progress.json")
LISTINGS_AUDIT_CACHE = Path("cache/audit_log.jsonl")
LISTINGS_DATA_ID = 75
LISTINGS_SEARCH_ENDPOINT = "/api/v2/users_portfolio_groups/search"
LISTINGS_GET_ENDPOINT = "/api/v2/users_portfolio_groups/get/{group_id}"

VALID_HINT_KEYS = {"group_id", "group_filename", "group_name", "post_location"}


@dataclass
class EndpointDecision:
    selected_endpoint: Optional[str]
    accepted_reason: str
    rejected_reasons: List[str]
    probe_details: Dict[str, Any]


@dataclass
class ListingsInventoryResult:
    records: List[Dict[str, Any]]
    source: str
    by_category: Dict[str, int]
    by_geo: Dict[str, int]
    endpoint_decision: EndpointDecision
    last_completed_page: int = 0
    total_pages: int = 0
    total_posts: int = 0


def _request_json(
    client: BDClient,
    session: requests.Session,
    *,
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    label: str = "Listings Probe",
    dry_run: bool = False,
    fixtures: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[int, Dict[str, Any], str]:
    key = f"{method.upper()}:{path}"
    if dry_run:
        payload = (fixtures or {}).get(key, {"status": "error", "message": "dry_run fixture missing"})
        return 200, payload, json.dumps(payload)

    url = f"{client.base_url}{path}"
    headers = {"X-Api-Key": client.api_key}
    if method.upper() == "POST":
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        response = session.post(url, headers=headers, data=data or {}, timeout=client.timeout)
    else:
        response = session.get(url, headers=headers, timeout=client.timeout)
    try:
        payload = response.json() if response.text else {}
    except Exception:
        payload = {"raw_text": response.text}
    client.evidence_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "method": method.upper(),
            "url": url,
            "status_code": response.status_code,
            "request_body_keys": sorted((data or {}).keys()),
            "response_text_snippet": (response.text or "")[:500],
            "parse_result_summary": {"records_parsed": 0, "parse_errors": []},
        }
    )
    return response.status_code, payload, response.text


def _extract_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    msg = payload.get("message")
    if isinstance(msg, list):
        return [x for x in msg if isinstance(x, dict)]
    for key in ("data", "records", "results", "rows", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = value.get("data") or value.get("rows")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
    return []


def build_listings_search_payload(*, limit: int, page: int) -> Dict[str, Any]:
    return {
        "action": "search",
        "output_type": "array",
        "data_id": LISTINGS_DATA_ID,
        "limit": int(limit),
        "page": int(page),
    }


def parse_users_portfolio_groups_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    status_value = str(payload.get("status", "")).lower()
    records = _extract_records(payload)
    total_posts = int(payload.get("total_posts", 0) or 0)
    total_pages = int(payload.get("total_pages", 0) or 0)
    return {
        "ok": status_value == "success" and isinstance(records, list),
        "records": records,
        "total_posts": total_posts,
        "total_pages": total_pages,
        "has_totals": "total_posts" in payload and "total_pages" in payload,
        "has_group_id": bool(records and records[0].get("group_id")),
    }


def _payload_has_listing_signals(records: List[Dict[str, Any]]) -> bool:
    for record in records:
        if any(k in record for k in VALID_HINT_KEYS):
            return True
        if str(record.get("data_id", "")).strip() == "75":
            return True
    return False


def canonical_listing_key(record: Dict[str, Any]) -> str:
    group_id = str(record.get("group_id") or "").strip()
    if group_id:
        return group_id
    user_id = str(record.get("user_id") or "").strip()
    filename = str(record.get("group_filename") or "").strip()
    if user_id and filename:
        return f"{user_id}:{filename.lower()}"
    digest = hashlib.sha1(json.dumps(record, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest


def detect_repetition(first_keys: Iterable[str], repeat_threshold: int = 3) -> bool:
    streak = 0
    prev = ""
    for key in first_keys:
        if not key:
            streak = 0
            prev = ""
            continue
        if key == prev:
            streak += 1
        else:
            streak = 1
            prev = key
        if streak >= repeat_threshold:
            return True
    return False


def _audit_append(row: Dict[str, Any], max_rows: int = 200) -> None:
    LISTINGS_AUDIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if LISTINGS_AUDIT_CACHE.exists():
        rows = [json.loads(line) for line in LISTINGS_AUDIT_CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.append(row)
    rows = rows[-max_rows:]
    LISTINGS_AUDIT_CACHE.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def load_audit_rows() -> List[Dict[str, Any]]:
    if not LISTINGS_AUDIT_CACHE.exists():
        return []
    return [json.loads(line) for line in LISTINGS_AUDIT_CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]


def probe_listings_endpoints(client: BDClient, dry_run: bool = False, fixtures: Optional[Dict[str, Dict[str, Any]]] = None) -> EndpointDecision:
    rejected: List[str] = []
    session = requests.Session()
    details: Dict[str, Any] = {}

    status, payload, text = _request_json(
        client,
        session,
        method="GET",
        path=f"/api/v2/data_categories/get/{LISTINGS_DATA_ID}",
        label="Listings Probe",
        dry_run=dry_run,
        fixtures=fixtures,
    )
    records = _extract_records(payload)
    has_expected_type = any(str(r.get("data_id", "")).strip() == str(LISTINGS_DATA_ID) and str(r.get("data_type", "")).strip() == "4" for r in records)
    details["data_categories_get"] = {"http_status": status, "status": payload.get("status"), "records": len(records), "matched": has_expected_type}
    _audit_append({"timestamp": datetime.now(timezone.utc).isoformat(), "action": "probe", "endpoint": f"/api/v2/data_categories/get/{LISTINGS_DATA_ID}", "method": "GET", "params": {}, "outcome": "success" if has_expected_type else "fail", "http_status": status, "response_text": text[:300], "counts_returned": {"records": len(records)}})
    if status != 200 or not has_expected_type:
        rejected.append(f"GET /api/v2/data_categories/get/{LISTINGS_DATA_ID} failed (http={status}, status={payload.get('status')})")
        return EndpointDecision(selected_endpoint=None, accepted_reason="", rejected_reasons=rejected, probe_details=details)

    form = build_listings_search_payload(limit=1, page=1)
    status, payload, text = _request_json(
        client,
        session,
        method="POST",
        path=LISTINGS_SEARCH_ENDPOINT,
        data=form,
        label="Listings Probe",
        dry_run=dry_run,
        fixtures=fixtures,
    )
    parsed = parse_users_portfolio_groups_response(payload)
    probe_ok = status == 200 and parsed["ok"] and parsed["has_group_id"] and parsed["has_totals"]
    details["users_portfolio_groups_search"] = {
        "http_status": status,
        "status": payload.get("status"),
        "records": len(parsed["records"]),
        "total_pages": parsed["total_pages"],
        "total_posts": parsed["total_posts"],
        "has_group_id": parsed["has_group_id"],
        "has_totals": parsed["has_totals"],
    }
    _audit_append({"timestamp": datetime.now(timezone.utc).isoformat(), "action": "probe", "endpoint": LISTINGS_SEARCH_ENDPOINT, "method": "POST", "params": form, "outcome": "success" if probe_ok else "fail", "http_status": status, "response_text": text[:300], "counts_returned": {"records": len(parsed["records"]), "total_pages": parsed["total_pages"], "total_posts": parsed["total_posts"]}})
    if probe_ok and parsed["records"]:
        return EndpointDecision(selected_endpoint=LISTINGS_SEARCH_ENDPOINT, accepted_reason="Listings API usable via users_portfolio_groups/search", rejected_reasons=rejected, probe_details=details)

    rejected.append(f"POST {LISTINGS_SEARCH_ENDPOINT} failed (http={status}, status={payload.get('status')}, records={len(parsed['records'])}, total_posts={parsed['total_posts']})")
    secondary_form = build_listings_search_payload(limit=1, page=1)
    status2, payload2, text2 = _request_json(
        client,
        session,
        method="POST",
        path="/api/v2/data_posts/search",
        data=secondary_form,
        label="Listings Probe",
        dry_run=dry_run,
        fixtures=fixtures,
    )
    secondary_records = _extract_records(payload2)
    details["data_posts_search"] = {"http_status": status2, "status": payload2.get("status"), "records": len(secondary_records)}
    _audit_append({"timestamp": datetime.now(timezone.utc).isoformat(), "action": "probe", "endpoint": "/api/v2/data_posts/search", "method": "POST", "params": secondary_form, "outcome": "fail" if status2 != 200 else "success", "http_status": status2, "response_text": text2[:300], "counts_returned": {"records": len(secondary_records)}})
    rejected.append(f"POST /api/v2/data_posts/search secondary check http={status2}, records={len(secondary_records)}")
    return EndpointDecision(selected_endpoint=None, accepted_reason="", rejected_reasons=rejected, probe_details=details)


def fetch_listings_via_api(
    client: BDClient,
    endpoint: str,
    *,
    page_limit: int = 100,
    max_pages: int = 30,
    rpm: int = 30,
    start_page: int = 1,
    existing_records: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
    fixtures: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ListingsInventoryResult:
    method = "POST"
    path = endpoint
    session = requests.Session()
    limiter = RateLimiter(rpm)
    seen: Dict[str, Dict[str, Any]] = {canonical_listing_key(r): r for r in (existing_records or [])}
    first_keys: List[str] = []
    last_completed_page = start_page - 1
    total_pages = 0
    total_posts = 0

    for offset in range(max_pages):
        page = start_page + offset
        limiter.wait()
        payload = build_listings_search_payload(limit=page_limit, page=page)
        status, body, text = _request_json(
            client,
            session,
            method=method,
            path=path,
            data=payload,
            label="Listings API Fetch",
            dry_run=dry_run,
            fixtures=fixtures,
        )
        parsed = parse_users_portfolio_groups_response(body)
        records = parsed["records"]
        total_pages = parsed["total_pages"]
        total_posts = parsed["total_posts"]
        first_key = canonical_listing_key(records[0]) if records else ""
        first_keys.append(first_key)
        _audit_append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "fetch",
                "method": method,
                "endpoint": path,
                "status_code": status,
                "params": payload,
                "response_text": text[:300],
                "outcome": "success" if status == 200 and parsed["ok"] else "fail",
                "counts_returned": {"records": len(records), "page": page, "total_pages": total_pages, "total_posts": total_posts},
            }
        )
        if status != 200 or not parsed["ok"]:
            raise RuntimeError(f"Listings fetch failed endpoint={path} method=POST http={status} response={text[:300]}")
        if not records:
            break
        for record in records:
            norm = normalize_listing_record(record, source="api")
            seen[canonical_listing_key(norm)] = norm
        last_completed_page = page
        LISTINGS_PROGRESS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LISTINGS_PROGRESS_CACHE.write_text(json.dumps({"last_completed_page": last_completed_page, "total_pages": total_pages, "total_posts": total_posts}, indent=2), encoding="utf-8")
        if detect_repetition(first_keys[-3:]):
            raise RuntimeError("Endpoint rejected: first listing key repeats across 3 consecutive pages")
        if total_pages and page >= total_pages:
            break

    result = list(seen.values())
    persist_listings_inventory(result, selected_endpoint=endpoint)
    return ListingsInventoryResult(
        records=result,
        source="api",
        by_category=_count_by(result, "category"),
        by_geo=_count_by(result, "geo"),
        endpoint_decision=EndpointDecision(selected_endpoint=endpoint, accepted_reason="locked endpoint fetch", rejected_reasons=[], probe_details={}),
        last_completed_page=last_completed_page,
        total_pages=total_pages,
        total_posts=total_posts,
    )


def normalize_listing_record(record: Dict[str, Any], *, source: str, url: str = "") -> Dict[str, Any]:
    city = str(record.get("city") or record.get("post_city") or "").strip()
    state = str(record.get("state") or record.get("post_state") or "").strip()
    country = str(record.get("country") or "").strip()
    geo = ", ".join([x for x in [city, state, country] if x])
    return {
        "listing_key": canonical_listing_key(record),
        "url": url or str(record.get("url") or record.get("listing_url") or "").strip(),
        "group_id": record.get("group_id") or record.get("group_id_guess"),
        "group_id_guess": record.get("group_id") or record.get("group_id_guess"),
        "user_id": record.get("user_id") or record.get("data-userid"),
        "group_name": record.get("group_name") or record.get("title") or record.get("name"),
        "group_filename": record.get("group_filename") or record.get("slug"),
        "data_id": record.get("data_id") or LISTINGS_DATA_ID,
        "category": str(record.get("group_category") or record.get("category") or record.get("category_guess") or "").strip(),
        "city": city,
        "state": state,
        "country": country,
        "geo": geo,
        "lat": record.get("lat") or record.get("latitude"),
        "lon": record.get("lon") or record.get("longitude"),
        "post_tags": record.get("post_tags"),
        "post_location": record.get("post_location"),
        "revision_timestamp": record.get("revision_timestamp"),
        "date_updated": record.get("date_updated"),
        "group_status": record.get("group_status"),
        "users_portfolio": record.get("users_portfolio") if isinstance(record.get("users_portfolio"), list) else [],
        "source": source,
        "raw": record,
    }


def _extract_listing_urls_from_sitemap(base_url: str, session: requests.Session) -> List[str]:
    sitemap_url = f"{base_url.rstrip('/')}/sitemap.xml"
    response = session.get(sitemap_url, timeout=20)
    if response.status_code != 200:
        return []

    def parse(xml_text: str) -> List[str]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locs = [elem.text.strip() for elem in root.findall(".//s:loc", ns) if elem.text]
        return locs

    locs = parse(response.text)
    child_sitemaps = [u for u in locs if u.endswith(".xml")]
    urls = [u for u in locs if "/listings" in u]
    for child in child_sitemaps:
        child_resp = session.get(child, timeout=20)
        if child_resp.status_code != 200:
            continue
        urls.extend([u for u in parse(child_resp.text) if "/listings" in u])
    return sorted(set(urls))


def _parse_listing_page(url: str, html: str) -> Dict[str, Any]:
    userid_match = re.search(r"data-userid=['\"](\d+)['\"]", html, flags=re.IGNORECASE)
    postid_match = re.search(r"data-postid=['\"](\d+)['\"]", html, flags=re.IGNORECASE)
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
    title_match = re.search(r"property=['\"]og:title['\"][^>]*content=['\"](.*?)['\"]", html, flags=re.IGNORECASE)
    breadcrumb = re.findall(r"<li[^>]*class=['\"][^\"]*breadcrumb[^\"]*['\"][^>]*>(.*?)</li>", html, flags=re.IGNORECASE | re.DOTALL)

    group_name = ""
    if h1_match:
        group_name = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
    if not group_name and title_match:
        group_name = title_match.group(1).strip()

    category_guess = re.sub(r"<[^>]+>", "", breadcrumb[-1]).strip() if breadcrumb else ""
    group_id_guess = postid_match.group(1) if postid_match else ""
    listing_key = group_id_guess or hashlib.sha1(url.encode("utf-8")).hexdigest()
    return {
        "listing_key": listing_key,
        "url": url,
        "group_id_guess": group_id_guess,
        "user_id": userid_match.group(1) if userid_match else "",
        "group_name": group_name,
        "category_guess": category_guess,
        "source": "scrape",
    }


def build_listings_via_scrape(
    client: BDClient,
    *,
    seed_url: str,
    max_pages: int = 150,
    rpm: int = 30,
    enrich_users: bool = True,
) -> ListingsInventoryResult:
    session = requests.Session()
    limiter = RateLimiter(rpm)
    urls = _extract_listing_urls_from_sitemap(client.base_url, session)
    if not urls:
        urls = [seed_url]

    rows: Dict[str, Dict[str, Any]] = {}
    for url in urls[:max_pages]:
        limiter.wait()
        try:
            response = session.get(url, timeout=20)
        except requests.RequestException as exc:
            _audit_append({"method": "GET", "url": url, "status_code": 0, "request_body_keys": [], "response_snippet": str(exc)[:220], "parse_summary": {}, "error_type": "REQUEST_EXCEPTION"})
            continue
        _audit_append({"method": "GET", "url": url, "status_code": response.status_code, "request_body_keys": [], "response_snippet": response.text[:220], "parse_summary": {}, "error_type": "NONE" if response.status_code == 200 else "HTTP_ERROR"})
        if response.status_code != 200:
            continue
        parsed = _parse_listing_page(url, response.text)
        rows[parsed["listing_key"]] = normalize_listing_record(parsed, source="scrape", url=url)

    if enrich_users:
        user_cache: Dict[str, Dict[str, Any]] = {}
        for row in rows.values():
            user_id = str(row.get("user_id") or "").strip()
            if not user_id or user_id in user_cache:
                continue
            response = client.get_user(int(user_id))
            user_cache[user_id] = response.json_data if isinstance(response.json_data, dict) else {}
            _audit_append(
                {
                    "method": "GET",
                    "url": f"{client.base_url}/api/v2/user/get/{user_id}",
                    "status_code": response.status_code,
                    "request_body_keys": [],
                    "response_snippet": json.dumps(response.json_data)[:220],
                    "parse_summary": {"user_id": user_id},
                    "error_type": "NONE" if response.ok else "HTTP_ERROR",
                }
            )

    inventory = list(rows.values())
    persist_listings_inventory(inventory, selected_endpoint=None)
    return ListingsInventoryResult(
        records=inventory,
        source="scrape",
        by_category=_count_by(inventory, "category"),
        by_geo=_count_by(inventory, "geo"),
        endpoint_decision=EndpointDecision(selected_endpoint=None, accepted_reason="scrape fallback", rejected_reasons=[], probe_details={}),
    )


def _count_by(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    c = Counter()
    for row in records:
        value = str(row.get(key) or "Unknown").strip() or "Unknown"
        c[value] += 1
    return dict(c)


def persist_listings_inventory(records: List[Dict[str, Any]], selected_endpoint: Optional[str]) -> None:
    LISTINGS_INVENTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LISTINGS_INVENTORY_CACHE.write_text(json.dumps({"records": records, "selected_endpoint": selected_endpoint}, indent=2), encoding="utf-8")
    if not LISTINGS_PROGRESS_CACHE.exists():
        LISTINGS_PROGRESS_CACHE.write_text(json.dumps({"count": len(records), "timestamp": time.time(), "last_completed_page": 0, "total_pages": 0, "total_posts": 0}, indent=2), encoding="utf-8")


def load_listings_inventory() -> Dict[str, Any]:
    if not LISTINGS_INVENTORY_CACHE.exists():
        return {"records": [], "selected_endpoint": None}
    try:
        return json.loads(LISTINGS_INVENTORY_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {"records": [], "selected_endpoint": None}


def clear_listings_inventory_cache() -> None:
    for p in (LISTINGS_INVENTORY_CACHE, LISTINGS_PROGRESS_CACHE, LISTINGS_AUDIT_CACHE):
        if p.exists():
            p.unlink()


def load_listings_progress() -> Dict[str, Any]:
    if not LISTINGS_PROGRESS_CACHE.exists():
        return {"last_completed_page": 0, "total_pages": 0, "total_posts": 0}
    try:
        return json.loads(LISTINGS_PROGRESS_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {"last_completed_page": 0, "total_pages": 0, "total_posts": 0}

def _pick_cover_image(portfolio: List[Dict[str, Any]]) -> str:
    if not portfolio:
        return ""
    cover = next((p for p in portfolio if str(p.get("group_cover", "")).strip() == "1"), None)
    if cover:
        return str(cover.get("file_main_full_url") or "")
    order_one = next((p for p in portfolio if str(p.get("group_order", "")).strip() == "1"), None)
    if order_one:
        return str(order_one.get("file_main_full_url") or "")
    return str(portfolio[0].get("file_main_full_url") or "")


def listings_to_csv(records: List[Dict[str, Any]]) -> str:
    fields = [
        "group_id", "user_id", "group_name", "group_filename", "group_category", "post_tags", "post_location",
        "lat", "lon", "revision_timestamp", "date_updated", "group_status", "cover_image_url", "image_urls",
        "raw_users_portfolio_count", "source", "data_id",
    ]
    lines = [",".join(fields)]
    for row in records:
        portfolio = row.get("users_portfolio") if isinstance(row.get("users_portfolio"), list) else []
        export = {
            "group_id": row.get("group_id", ""),
            "user_id": row.get("user_id", ""),
            "group_name": row.get("group_name", ""),
            "group_filename": row.get("group_filename", ""),
            "group_category": row.get("category", ""),
            "post_tags": row.get("post_tags", ""),
            "post_location": row.get("post_location", ""),
            "lat": row.get("lat", ""),
            "lon": row.get("lon", ""),
            "revision_timestamp": row.get("revision_timestamp", ""),
            "date_updated": row.get("date_updated", ""),
            "group_status": row.get("group_status", ""),
            "cover_image_url": _pick_cover_image(portfolio),
            "image_urls": "|".join([str(p.get("file_main_full_url") or "") for p in portfolio if p.get("file_main_full_url")]),
            "raw_users_portfolio_count": len(portfolio),
            "source": row.get("source", "api"),
            "data_id": row.get("data_id", LISTINGS_DATA_ID),
        }
        line = []
        for field in fields:
            value = str(export.get(field, ""))
            value = value.replace('"', '""')
            if "," in value or '"' in value:
                value = f'"{value}"'
            line.append(value)
        lines.append(",".join(line))
    return "\n".join(lines) + "\n"


def discover_listing_urls_from_seed(seed_url: str, html: str) -> List[str]:
    links = re.findall(r"href=['\"](.*?)['\"]", html, flags=re.IGNORECASE)
    urls = [urljoin(seed_url, link) for link in links if "/listings" in link]
    return sorted(set(urls))
