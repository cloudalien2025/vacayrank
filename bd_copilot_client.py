from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests


class BDCopilotClientError(RuntimeError):
    pass


def normalize_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        raise BDCopilotClientError("Base URL is required")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    if not parsed.netloc:
        raise BDCopilotClientError(f"Invalid base URL: {base_url!r}")
    if "none" in parsed.netloc.lower() and parsed.netloc.lower().endswith("none"):
        raise BDCopilotClientError("Base URL appears malformed (possible concatenation bug)")
    clean_path = parsed.path.rstrip("/")
    normalized = parsed._replace(path=clean_path, params="", query="", fragment="")
    return urlunparse(normalized).rstrip("/")


def build_api_url(base_url: str, path: str) -> str:
    safe_base = normalize_base_url(base_url)
    safe_path = "/" + str(path or "").lstrip("/")
    return urljoin(f"{safe_base}/", safe_path.lstrip("/"))


def bd_get_user(base_url: str, api_key: str, user_id: int | str, timeout: int = 20) -> Dict[str, Any]:
    url = build_api_url(base_url, f"/api/v2/user/get/{int(user_id)}")
    response = requests.get(url, headers={"X-Api-Key": api_key}, timeout=timeout)
    payload = response.json() if response.text else {}
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "json": payload,
        "text": response.text,
    }


def bd_update_user(base_url: str, api_key: str, user_id: int | str, changes_dict: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    if not changes_dict:
        raise BDCopilotClientError("changes_dict must not be empty")
    payload = {"user_id": str(user_id)}
    payload.update({str(k): str(v) for k, v in changes_dict.items()})
    url = build_api_url(base_url, "/api/v2/user/update")
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    attempts = 0
    response = None
    while attempts < 2:
        attempts += 1
        response = requests.put(url, headers=headers, data=payload, timeout=timeout)
        if response.status_code >= 500 and attempts == 1:
            time.sleep(0.6)
            continue
        break

    assert response is not None
    try:
        json_payload = response.json() if response.text else {}
    except Exception:
        json_payload = {"raw_text": response.text}
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "json": json_payload,
        "text": response.text,
        "request_payload": payload,
    }


def build_audit_entry(listing_id: str, user_id: str, request_fields: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "listing_id": listing_id,
        "user_id": user_id,
        "request_fields": {k: v for k, v in request_fields.items() if k != "X-Api-Key"},
        "api_key": "<redacted>",
        "status_code": response.get("status_code"),
        "ok": bool(response.get("ok")),
        "response": response.get("json") or response.get("text"),
    }


def validate_base_url_self_check() -> Tuple[bool, str]:
    cases = {
        "www.vailvacay.com/": "https://www.vailvacay.com",
        "https://www.vailvacay.com/listings/": "https://www.vailvacay.com/listings",
    }
    for raw, expected in cases.items():
        got = normalize_base_url(raw)
        if got != expected:
            return False, f"normalize_base_url failed for {raw}: {got} != {expected}"
    return True, "ok"


def validate_form_payload_self_check() -> Tuple[bool, str]:
    payload = {"group_name": "Demo", "group_status": "1"}
    result = {"user_id": "123", **{k: str(v) for k, v in payload.items()}}
    if "user_id" not in result:
        return False, "user_id missing"
    if set(result.keys()) != {"user_id", "group_name", "group_status"}:
        return False, f"unexpected keys {sorted(result.keys())}"
    return True, "ok"
