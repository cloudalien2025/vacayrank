from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import requests

from inventory_engine import RateLimiter
from milestone3.bd_client import BDClient

LISTINGS_INVENTORY_CACHE = Path("cache/listings_inventory.json")
LISTINGS_PROGRESS_CACHE = Path("cache/listings_progress.json")
LISTINGS_AUDIT_CACHE = Path("cache/audit_log.jsonl")

CANDIDATE_ENDPOINTS: List[Tuple[str, str]] = [
    ("GET", "/api/v2/classified/search"),
    ("POST", "/api/v2/classified/search"),
    ("GET", "/api/v2/listings/search"),
    ("POST", "/api/v2/listings/search"),
    ("GET", "/api/v2/portfolio/search"),
    ("POST", "/api/v2/portfolio/search"),
    ("GET", "/api/v2/user_portfolio/search"),
    ("POST", "/api/v2/user_portfolio/search"),
    ("GET", "/api/v2/users_portfolio/search"),
    ("POST", "/api/v2/users_portfolio/search"),
    ("GET", "/api/v2/users_portfolio_groups/search"),
    ("POST", "/api/v2/users_portfolio_groups/search"),
    ("GET", "/api/v2/listing/search"),
    ("POST", "/api/v2/listing/search"),
]

CANDIDATE_GET_SINGLE = [
    "/api/v2/classified/get/1",
    "/api/v2/portfolio/get/1",
    "/api/v2/listings/get/1",
    "/api/v2/users_portfolio_groups/get/1",
]

VALID_HINT_KEYS = {"group_id", "listing_id", "portfolio_id", "group_filename", "group_name", "post_location"}


@dataclass
class EndpointDecision:
    selected_endpoint: Optional[Dict[str, str]]
    accepted_reason: str
    rejected_reasons: List[str]


@dataclass
class ListingsInventoryResult:
    records: List[Dict[str, Any]]
    source: str
    by_category: Dict[str, int]
    by_geo: Dict[str, int]
    endpoint_decision: EndpointDecision


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

    for method, path in CANDIDATE_ENDPOINTS:
        found = None
        for page in (1, 2, 3):
            body = {"limit": 10 if page > 1 else 1, "page": page, "output_type": "json"}
            status, payload, text = _request_json(
                client,
                session,
                method=method,
                path=path,
                data=body if method == "POST" else None,
                label="Listings Probe",
                dry_run=dry_run,
                fixtures=fixtures,
            )
            records = _extract_records(payload)
            parse_summary = {"records": len(records), "has_signals": _payload_has_listing_signals(records)}
            row = {
                "method": method,
                "url": f"{client.base_url}{path}",
                "status_code": status,
                "request_body_keys": sorted((body if method == "POST" else {}).keys()),
                "response_snippet": text[:220],
                "parse_summary": parse_summary,
                "error_type": "NONE" if status == 200 else "HTTP_ERROR",
            }
            _audit_append(row)
            if status != 200 or not isinstance(payload, dict):
                continue
            status_value = str(payload.get("status", "")).lower()
            if status_value not in {"success", "error"}:
                continue
            if records and _payload_has_listing_signals(records):
                found = {"method": method, "path": path}
                break
        if found:
            return EndpointDecision(selected_endpoint=found, accepted_reason="200 + records with listing hint fields", rejected_reasons=rejected)
        rejected.append(f"{method} {path}: no qualifying records")

    for path in CANDIDATE_GET_SINGLE:
        status, payload, text = _request_json(client, session, method="GET", path=path, label="Listings Probe Single", dry_run=dry_run, fixtures=fixtures)
        records = _extract_records(payload)
        row = {
            "method": "GET",
            "url": f"{client.base_url}{path}",
            "status_code": status,
            "request_body_keys": [],
            "response_snippet": text[:220],
            "parse_summary": {"records": len(records), "has_signals": _payload_has_listing_signals(records)},
            "error_type": "NONE" if status == 200 else "HTTP_ERROR",
        }
        _audit_append(row)
        rejected.append(f"GET {path}: no qualifying records")

    return EndpointDecision(selected_endpoint=None, accepted_reason="", rejected_reasons=rejected)


def fetch_listings_via_api(
    client: BDClient,
    endpoint: Dict[str, str],
    *,
    page_limit: int = 100,
    max_pages: int = 30,
    rpm: int = 30,
    dry_run: bool = False,
    fixtures: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ListingsInventoryResult:
    method = endpoint["method"].upper()
    path = endpoint["path"]
    session = requests.Session()
    limiter = RateLimiter(rpm)
    seen: Dict[str, Dict[str, Any]] = {}
    first_keys: List[str] = []

    for page in range(1, max_pages + 1):
        limiter.wait()
        payload = {"limit": page_limit, "page": page, "output_type": "json"}
        status, body, text = _request_json(
            client,
            session,
            method=method,
            path=path,
            data=payload if method == "POST" else None,
            label="Listings API Fetch",
            dry_run=dry_run,
            fixtures=fixtures,
        )
        records = _extract_records(body)
        first_key = canonical_listing_key(records[0]) if records else ""
        first_keys.append(first_key)
        _audit_append(
            {
                "method": method,
                "url": f"{client.base_url}{path}",
                "status_code": status,
                "request_body_keys": sorted((payload if method == "POST" else {}).keys()),
                "response_snippet": text[:220],
                "parse_summary": {"records": len(records), "page": page},
                "error_type": "NONE" if status == 200 else "HTTP_ERROR",
            }
        )
        if status != 200 or not records:
            break
        for record in records:
            norm = normalize_listing_record(record, source="api")
            seen[canonical_listing_key(norm)] = norm
        if detect_repetition(first_keys[-3:]):
            raise RuntimeError("Endpoint rejected: first listing key repeats across 3 consecutive pages")

    result = list(seen.values())
    persist_listings_inventory(result, selected_endpoint=endpoint)
    return ListingsInventoryResult(
        records=result,
        source="api",
        by_category=_count_by(result, "category"),
        by_geo=_count_by(result, "geo"),
        endpoint_decision=EndpointDecision(selected_endpoint=endpoint, accepted_reason="locked endpoint fetch", rejected_reasons=[]),
    )


def normalize_listing_record(record: Dict[str, Any], *, source: str, url: str = "") -> Dict[str, Any]:
    city = str(record.get("city") or record.get("post_city") or "").strip()
    state = str(record.get("state") or record.get("post_state") or "").strip()
    country = str(record.get("country") or "").strip()
    geo = ", ".join([x for x in [city, state, country] if x])
    return {
        "listing_key": canonical_listing_key(record),
        "url": url or str(record.get("url") or record.get("listing_url") or "").strip(),
        "group_id": record.get("group_id") or record.get("data-postid") or record.get("group_id_guess"),
        "group_id_guess": record.get("group_id") or record.get("group_id_guess") or record.get("data-postid"),
        "user_id": record.get("user_id") or record.get("data-userid"),
        "group_name": record.get("group_name") or record.get("title") or record.get("name"),
        "group_filename": record.get("group_filename") or record.get("slug"),
        "data_id": record.get("data_id"),
        "category": str(record.get("category") or record.get("category_guess") or "").strip(),
        "city": city,
        "state": state,
        "country": country,
        "geo": geo,
        "lat": record.get("lat") or record.get("latitude"),
        "lon": record.get("lon") or record.get("longitude"),
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
        endpoint_decision=EndpointDecision(selected_endpoint=None, accepted_reason="scrape fallback", rejected_reasons=[]),
    )


def _count_by(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    c = Counter()
    for row in records:
        value = str(row.get(key) or "Unknown").strip() or "Unknown"
        c[value] += 1
    return dict(c)


def persist_listings_inventory(records: List[Dict[str, Any]], selected_endpoint: Optional[Dict[str, str]]) -> None:
    LISTINGS_INVENTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LISTINGS_INVENTORY_CACHE.write_text(json.dumps({"records": records, "selected_endpoint": selected_endpoint}, indent=2), encoding="utf-8")
    LISTINGS_PROGRESS_CACHE.write_text(json.dumps({"count": len(records), "timestamp": time.time()}, indent=2), encoding="utf-8")


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


def listings_to_csv(records: List[Dict[str, Any]]) -> str:
    fields = ["listing_key", "url", "group_id_guess", "user_id", "group_name", "category", "city", "state", "lat", "lon", "source"]
    lines = [",".join(fields)]
    for row in records:
        line = []
        for field in fields:
            value = str(row.get(field, ""))
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
