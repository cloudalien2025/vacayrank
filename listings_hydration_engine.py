from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from milestone3.bd_client import BDClient

LISTINGS_DATA_ID = 75
LISTINGS_SEARCH_ENDPOINT = "/api/v2/users_portfolio_groups/search"
LISTINGS_GET_ENDPOINT_TEMPLATE = "/api/v2/users_portfolio_groups/get/{group_id}"

HYDRATED_JSONL_PATH = Path("data/listings_hydrated.jsonl")
HYDRATION_AUDIT_PATH = Path("data/hydration_audit.jsonl")
HYDRATION_CHECKPOINT_PATH = Path("data/hydration_checkpoint.json")
SCHEMA_VERSION = "1.0"


@dataclass
class HydrationRunResult:
    processed_count: int
    success_count: int
    failures_count: int
    completed_count: int
    last_listing_id: Optional[str]
    last_listing_name: str
    last_errors: List[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_html(html: Any) -> str:
    if html is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(html))
    return re.sub(r"\s+", " ", text).strip()


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _sort_ids(ids: Iterable[str]) -> List[str]:
    unique = sorted(set(_as_str(x) for x in ids if _as_str(x)))

    def _key(v: str) -> Tuple[int, Any]:
        return (0, int(v)) if v.isdigit() else (1, v)

    return sorted(unique, key=_key)


def build_worklist(inventory_records: Sequence[Dict[str, Any]]) -> List[str]:
    ids = [record.get("group_id") for record in inventory_records if isinstance(record, dict)]
    return _sort_ids([_as_str(x) for x in ids])


def build_get_endpoint_path(group_id: str) -> str:
    return LISTINGS_GET_ENDPOINT_TEMPLATE.format(group_id=_as_str(group_id))


def build_search_payload(*, limit: int, page: int) -> Dict[str, Any]:
    return {
        "action": "search",
        "output_type": "array",
        "data_id": LISTINGS_DATA_ID,
        "limit": int(limit),
        "page": int(page),
    }


def build_request_headers(api_key: str, is_write: bool = False) -> Dict[str, str]:
    headers = {"X-Api-Key": api_key}
    if is_write:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    return headers


def parse_success_listing_payload(payload: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    if not isinstance(payload, dict):
        return False, None, "payload_not_dict"
    if _as_str(payload.get("status")).lower() != "success":
        return False, None, "status_not_success"
    message = payload.get("message")
    if isinstance(message, dict):
        record = message
    elif isinstance(message, list) and message:
        record = message[0] if isinstance(message[0], dict) else None
    else:
        record = None
    if not isinstance(record, dict):
        return False, None, "missing_listing_record"
    if not _as_str(record.get("group_id")):
        return False, None, "missing_group_id"
    return True, record, ""


def _pick_cover_image(portfolio: List[Dict[str, Any]]) -> str:
    if not portfolio:
        return ""
    cover = next((p for p in portfolio if _as_str(p.get("group_cover")) == "1"), None)
    if cover:
        return _as_str(cover.get("file_main_full_url"))

    ordered = sorted(
        portfolio,
        key=lambda p: int(_as_str(p.get("group_order")) or "999999") if _as_str(p.get("group_order")).isdigit() else 999999,
    )
    first_with_url = next((p for p in ordered if _as_str(p.get("file_main_full_url"))), None)
    if first_with_url:
        return _as_str(first_with_url.get("file_main_full_url"))
    return _as_str(portfolio[0].get("file_main_full_url"))


def normalize_listing_record(raw: Dict[str, Any], site_base: str) -> Dict[str, Any]:
    group_filename = _as_str(raw.get("group_filename"))
    desc_html = raw.get("group_desc") or raw.get("group_desc_html")
    desc_text = strip_html(desc_html)
    tags_raw = _as_str(raw.get("post_tags"))
    tags_list = [part.strip() for part in tags_raw.split(",") if part.strip()] if tags_raw else []
    portfolio = raw.get("users_portfolio") if isinstance(raw.get("users_portfolio"), list) else []
    image_urls = [_as_str(p.get("file_main_full_url")) for p in portfolio if _as_str(p.get("file_main_full_url"))]

    canonical_url = ""
    if group_filename:
        canonical_url = f"{site_base.rstrip('/')}/{group_filename.lstrip('/')}"

    return {
        "listing_id": _as_str(raw.get("group_id")),
        "group_id": _as_str(raw.get("group_id")),
        "user_id": _as_str(raw.get("user_id")),
        "group_token": _as_str(raw.get("group_token")),
        "url_path": group_filename,
        "canonical_url": canonical_url,
        "group_name": _as_str(raw.get("group_name")),
        "group_desc_html": desc_html,
        "group_desc_text": desc_text,
        "desc_char_count": len(desc_text),
        "desc_word_count": len([w for w in desc_text.split(" ") if w]),
        "group_category": _as_str(raw.get("group_category")),
        "post_tags_raw": tags_raw,
        "tags_list": tags_list,
        "post_location": _as_str(raw.get("post_location")),
        "lat": raw.get("lat"),
        "lon": raw.get("lon"),
        "state_sn": _as_str(raw.get("state_sn")),
        "country_sn": _as_str(raw.get("country_sn")),
        "group_status": _as_str(raw.get("group_status")),
        "sticky_post": _as_str(raw.get("sticky_post")),
        "sticky": _as_str(raw.get("sticky")),
        "revision_timestamp": _as_str(raw.get("revision_timestamp")),
        "date_updated": _as_str(raw.get("date_updated")),
        "revision_count": _as_str(raw.get("revision_count")),
        "image_count": len(image_urls),
        "cover_image_url": _pick_cover_image(portfolio),
        "image_urls": "|".join(image_urls),
        "has_webp": any(url.lower().endswith(".webp") for url in image_urls),
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_checkpoint(path: Path = HYDRATION_CHECKPOINT_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "data_id": LISTINGS_DATA_ID,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "site_base": "",
            "completed_ids_count": 0,
            "completed_ids_hash": "",
            "last_completed_listing_id": None,
            "last_completed_index": -1,
            "failures_count": 0,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("data_id", LISTINGS_DATA_ID)
    payload.setdefault("created_at", _now_iso())
    payload.setdefault("updated_at", _now_iso())
    payload.setdefault("site_base", "")
    payload.setdefault("completed_ids_count", 0)
    payload.setdefault("completed_ids_hash", "")
    payload.setdefault("last_completed_listing_id", None)
    payload.setdefault("last_completed_index", -1)
    payload.setdefault("failures_count", 0)
    return payload


def _hash_ids(ids: Iterable[str]) -> str:
    packed = "|".join(_sort_ids(ids)).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def save_checkpoint(checkpoint: Dict[str, Any], path: Path = HYDRATION_CHECKPOINT_PATH) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def _completed_ids_from_jsonl(path: Path = HYDRATED_JSONL_PATH) -> set[str]:
    rows = read_jsonl(path)
    return {_as_str(r.get("listing_id")) for r in rows if _as_str(r.get("listing_id"))}


def clear_hydration_cache() -> None:
    for path in (HYDRATED_JSONL_PATH, HYDRATION_AUDIT_PATH, HYDRATION_CHECKPOINT_PATH):
        if path.exists():
            path.unlink()


def probe_get_by_id_capability(client: BDClient, listing_id: str) -> Dict[str, Any]:
    endpoint = build_get_endpoint_path(listing_id)
    response = client.request_get(endpoint, label="Hydration Capability Probe")
    ok, record, reason = parse_success_listing_payload(response.json_data)
    available = bool(response.status_code == 200 and ok and _as_str((record or {}).get("group_id")) == _as_str(listing_id))
    return {
        "checked_at": _now_iso(),
        "listing_id": _as_str(listing_id),
        "endpoint": endpoint,
        "http_status": response.status_code,
        "available": available,
        "reason": reason if not available else "ok",
    }


def _audit_attempt(action: str, endpoint: str, listing_id: str, response_status: int, ok: bool, error: Optional[str], retry_count: int, payload: Dict[str, Any]) -> None:
    append_jsonl(
        HYDRATION_AUDIT_PATH,
        {
            "ts": _now_iso(),
            "action": action,
            "endpoint": endpoint,
            "listing_id": _as_str(listing_id),
            "http_status": int(response_status or 0),
            "ok": bool(ok),
            "bytes": len(json.dumps(payload, default=str)) if payload is not None else 0,
            "error": error,
            "retry_count": retry_count,
            "response_snippet": json.dumps(payload, default=str)[:220],
        },
    )


def _fetch_listing_get_with_retries(client: BDClient, listing_id: str, retries: int = 3) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    endpoint = build_get_endpoint_path(listing_id)
    for retry in range(retries + 1):
        response = client.request_get(endpoint, label="Hydrate Listing GET")
        ok, record, reason = parse_success_listing_payload(response.json_data)
        matches = ok and _as_str((record or {}).get("group_id")) == _as_str(listing_id)
        error = None if matches else reason or f"http_{response.status_code}"
        _audit_attempt("hydrate_get", endpoint, listing_id, response.status_code, matches, error, retry, response.json_data)
        if matches:
            return True, record, ""
        if response.status_code == 429 or 500 <= int(response.status_code or 0) <= 599:
            if retry < retries:
                time.sleep((2**retry) + (0.05 * retry))
                continue
        return False, None, error or "failed"
    return False, None, "failed"


def _fetch_listing_from_search(client: BDClient, listing_id: str, limit: int = 10, max_pages: int = 10) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    endpoint = LISTINGS_SEARCH_ENDPOINT
    for page in range(1, max_pages + 1):
        payload = build_search_payload(limit=limit, page=page)
        response = client.search_users_portfolio_groups(payload, label="Hydrate Listing Search Fallback")
        status = _as_str(response.json_data.get("status")).lower()
        records = response.json_data.get("message") if isinstance(response.json_data.get("message"), list) else []
        found = next((r for r in records if _as_str((r or {}).get("group_id")) == _as_str(listing_id)), None)
        ok = response.status_code == 200 and status == "success" and isinstance(found, dict)
        err = None if ok else f"page_{page}_not_found"
        _audit_attempt("hydrate_from_search", endpoint, listing_id, response.status_code, ok, err, page - 1, response.json_data)
        if ok:
            return True, found, ""
        if response.status_code != 200 or status != "success":
            return False, None, err
    return False, None, "not_found_in_search_window"


def _to_hydrated_record(raw: Dict[str, Any], site_base: str, source_endpoint: str) -> Dict[str, Any]:
    norm = normalize_listing_record(raw, site_base)
    return {
        "listing_id": _as_str(raw.get("group_id")),
        "group_id": _as_str(raw.get("group_id")),
        "data_id": LISTINGS_DATA_ID,
        "fetched_at": _now_iso(),
        "source_endpoint": source_endpoint,
        "raw": raw,
        "norm": norm,
        "images": {
            "image_count": norm.get("image_count", 0),
            "cover_image_url": norm.get("cover_image_url", ""),
            "image_urls": norm.get("image_urls", ""),
            "has_webp": norm.get("has_webp", False),
        },
    }


def hydrate_listings(
    client: BDClient,
    inventory_records: Sequence[Dict[str, Any]],
    *,
    site_base: str,
    rpm: int,
    max_listings_per_run: int,
    resume: bool,
    get_by_id_available: bool,
) -> HydrationRunResult:
    worklist = build_worklist(inventory_records)
    checkpoint = load_checkpoint()
    completed_ids = set(_completed_ids_from_jsonl())
    processed = 0
    successes = 0
    failures = 0
    last_id: Optional[str] = None
    last_name = ""
    errors: List[str] = []

    start_index = 0
    if resume:
        for idx, item in enumerate(worklist):
            if item not in completed_ids:
                start_index = idx
                break

    for idx in range(start_index, len(worklist)):
        listing_id = worklist[idx]
        if listing_id in completed_ids:
            continue
        if processed >= max_listings_per_run:
            break

        if get_by_id_available:
            ok, raw, error = _fetch_listing_get_with_retries(client, listing_id)
            source_endpoint = build_get_endpoint_path(listing_id)
        else:
            ok, raw, error = _fetch_listing_from_search(client, listing_id)
            source_endpoint = LISTINGS_SEARCH_ENDPOINT

        processed += 1
        if ok and isinstance(raw, dict):
            hydrated = _to_hydrated_record(raw, site_base, source_endpoint)
            append_jsonl(HYDRATED_JSONL_PATH, hydrated)
            completed_ids.add(listing_id)
            successes += 1
            last_id = listing_id
            last_name = _as_str((hydrated.get("norm") or {}).get("group_name"))
            checkpoint["last_completed_listing_id"] = listing_id
            checkpoint["last_completed_index"] = idx
        else:
            failures += 1
            checkpoint["failures_count"] = int(checkpoint.get("failures_count", 0) or 0) + 1
            err_msg = f"{listing_id}: {error}"
            errors.append(err_msg)

        checkpoint["schema_version"] = SCHEMA_VERSION
        checkpoint["data_id"] = LISTINGS_DATA_ID
        checkpoint["site_base"] = site_base.rstrip("/")
        checkpoint["updated_at"] = _now_iso()
        checkpoint.setdefault("created_at", _now_iso())
        checkpoint["completed_ids_count"] = len(completed_ids)
        checkpoint["completed_ids_hash"] = _hash_ids(completed_ids)
        save_checkpoint(checkpoint)

        sleep_seconds = 60 / max(1, int(rpm))
        time.sleep(sleep_seconds)

    return HydrationRunResult(
        processed_count=processed,
        success_count=successes,
        failures_count=failures,
        completed_count=len(completed_ids),
        last_listing_id=last_id,
        last_listing_name=last_name,
        last_errors=errors[-3:],
    )


def load_hydrated_records(path: Path = HYDRATED_JSONL_PATH) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def export_hydrated_jsonl_text(path: Path = HYDRATED_JSONL_PATH) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _days_ago(ts: str) -> Optional[int]:
    text = _as_str(ts)
    if not text:
        return None
    value = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0, int((now - dt).total_seconds() // 86400))


def extract_features(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        norm = record.get("norm") if isinstance(record.get("norm"), dict) else {}
        tags = norm.get("tags_list") if isinstance(norm.get("tags_list"), list) else []
        lat = norm.get("lat")
        lon = norm.get("lon")
        updated_days = _days_ago(_as_str(norm.get("date_updated") or norm.get("revision_timestamp")))
        rows.append(
            {
                "listing_id": _as_str(norm.get("listing_id") or record.get("listing_id")),
                "canonical_url": _as_str(norm.get("canonical_url")),
                "group_name": _as_str(norm.get("group_name")),
                "category": _as_str(norm.get("group_category")),
                "tag_count": len(tags),
                "has_tags": 1 if tags else 0,
                "has_description": 1 if _as_str(norm.get("group_desc_text")) else 0,
                "desc_word_count": int(norm.get("desc_word_count") or 0),
                "desc_char_count": int(norm.get("desc_char_count") or 0),
                "has_location": 1 if _as_str(norm.get("post_location")) else 0,
                "has_geo": 1 if _as_str(lat) and _as_str(lon) else 0,
                "lat": lat,
                "lon": lon,
                "image_count": int(norm.get("image_count") or 0),
                "cover_image_present": 1 if _as_str(norm.get("cover_image_url")) else 0,
                "updated_days_ago": "" if updated_days is None else updated_days,
                "revision_count": _as_str(norm.get("revision_count")),
                "group_status": _as_str(norm.get("group_status")),
                "source": "api",
                "data_id": LISTINGS_DATA_ID,
            }
        )
    return rows


def features_to_csv(rows: Sequence[Dict[str, Any]]) -> str:
    fields = [
        "listing_id",
        "canonical_url",
        "group_name",
        "category",
        "tag_count",
        "has_tags",
        "has_description",
        "desc_word_count",
        "desc_char_count",
        "has_location",
        "has_geo",
        "lat",
        "lon",
        "image_count",
        "cover_image_present",
        "updated_days_ago",
        "revision_count",
        "group_status",
        "source",
        "data_id",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return out.getvalue()
