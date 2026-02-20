from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


class BDClientError(RuntimeError):
    pass


@dataclass
class BDResponse:
    ok: bool
    status_code: int
    json_data: Dict[str, Any]
    text: str
    content_type: str
    headers: Dict[str, str]


class BDClient:
    """Brilliant Directories API v2 client with strict write encoding rules."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.evidence_log: List[Dict[str, Any]] = []

    def _headers(self, include_form: bool = False) -> Dict[str, str]:
        headers = {"X-Api-Key": self.api_key}
        if include_form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        return headers

    @staticmethod
    def _redact_payload(payload_dict: Optional[Dict[str, Any]]) -> Dict[str, str]:
        if not payload_dict:
            return {}
        return {str(key): "<redacted>" for key in payload_dict.keys()}

    def _append_evidence(
        self,
        *,
        label: str,
        method: str,
        url: str,
        headers: Dict[str, str],
        payload_dict: Optional[Dict[str, Any]],
        response: requests.Response,
        parse_error: str = "",
    ) -> None:
        snippet = (response.text or "")[:500]
        content_type = getattr(response, "headers", {}).get("Content-Type", "")
        self.evidence_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "label": label,
                "method": method,
                "url": url,
                "request_header_keys": sorted(headers.keys()),
                "has_api_key": bool(headers.get("X-Api-Key")),
                "request_body_keys": sorted((payload_dict or {}).keys()),
                "request_body_redacted": self._redact_payload(payload_dict),
                "status_code": response.status_code,
                "response_content_type": content_type,
                "response_text_snippet": snippet,
                "parse_result_summary": {
                    "records_parsed": 0,
                    "parse_errors": [parse_error] if parse_error else [],
                },
            }
        )

    def annotate_last_evidence(self, records_parsed: int = 0, parse_error: Optional[str] = None) -> None:
        if not self.evidence_log:
            return
        summary = self.evidence_log[-1].setdefault("parse_result_summary", {})
        summary["records_parsed"] = records_parsed
        errors = list(summary.get("parse_errors", []))
        if parse_error:
            errors.append(parse_error)
        summary["parse_errors"] = errors

    def _normalize(self, response: requests.Response) -> BDResponse:
        try:
            payload = response.json() if response.text else {}
        except Exception:
            payload = {"raw_text": response.text}
        return BDResponse(
            ok=response.ok,
            status_code=response.status_code,
            json_data=payload,
            text=response.text,
            content_type=getattr(response, "headers", {}).get("Content-Type", ""),
            headers=dict(getattr(response, "headers", {}) or {}),
        )

    def _request(
        self,
        *,
        method: str,
        path: str,
        label: str,
        payload_dict: Optional[Dict[str, Any]] = None,
        include_form: bool = False,
    ) -> BDResponse:
        url = f"{self.base_url}{path}"
        headers = self._headers(include_form=include_form)
        method_upper = method.upper()
        if method_upper == "GET":
            response = requests.get(url=url, headers=headers, timeout=self.timeout)
        elif method_upper == "POST":
            response = requests.post(url=url, headers=headers, data=payload_dict, timeout=self.timeout)
        elif method_upper == "PUT":
            response = requests.put(url=url, headers=headers, data=payload_dict, timeout=self.timeout)
        elif method_upper == "DELETE":
            response = requests.delete(url=url, headers=headers, data=payload_dict, timeout=self.timeout)
        else:
            response = requests.request(
                method=method_upper,
                url=url,
                headers=headers,
                data=payload_dict,
                timeout=self.timeout,
            )
        parse_error = ""
        try:
            if response.text:
                response.json()
        except Exception as exc:
            parse_error = str(exc)
        self._append_evidence(
            label=label,
            method=method,
            url=url,
            headers=headers,
            payload_dict=payload_dict,
            response=response,
            parse_error=parse_error,
        )
        return self._normalize(response)

    def resolve_base_url(self) -> str:
        probe_url = self.base_url
        try:
            response = requests.get(probe_url, timeout=self.timeout, allow_redirects=True)
            final_url = response.url.rstrip("/")
            self._append_evidence(
                label="Base URL Resolve",
                method="GET",
                url=probe_url,
                headers=self._headers(),
                payload_dict=None,
                response=response,
            )
            self.base_url = final_url
        except requests.RequestException as exc:
            self.evidence_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "label": "Base URL Resolve",
                    "method": "GET",
                    "url": probe_url,
                    "request_header_keys": ["X-Api-Key"],
                    "has_api_key": bool(self.api_key),
                    "request_body_keys": [],
                    "request_body_redacted": {},
                    "status_code": 0,
                    "response_content_type": "",
                    "response_text_snippet": str(exc)[:500],
                    "parse_result_summary": {"records_parsed": 0, "parse_errors": [str(exc)]},
                }
            )
        return self.base_url

    def request_get(self, path: str, label: str = "GET request") -> BDResponse:
        return self._request(method="GET", path=path, label=label)

    def request_post_form(self, path: str, payload_dict: Dict[str, Any], label: str = "POST form request") -> BDResponse:
        return self._request(method="POST", path=path, label=label, payload_dict=payload_dict, include_form=True)

    def request_put_form(self, path: str, payload_dict: Dict[str, Any], label: str = "PUT form request") -> BDResponse:
        return self._request(method="PUT", path=path, label=label, payload_dict=payload_dict, include_form=True)

    def request_delete_form(self, path: str, payload_dict: Dict[str, Any], label: str = "DELETE form request") -> BDResponse:
        return self._request(method="DELETE", path=path, label=label, payload_dict=payload_dict, include_form=True)

    def search_users(self, payload: Dict[str, Any], label: str = "M1 Inventory Fetch") -> BDResponse:
        return self.request_post_form("/api/v2/user/search", payload, label=label)

    def get_user(self, user_id: int) -> BDResponse:
        return self.request_get(f"/api/v2/user/get/{int(user_id)}", label="Get User")

    def create_user(self, payload: Dict[str, Any]) -> BDResponse:
        required = {"email", "password", "subscription_id"}
        missing = required - set(payload)
        if missing:
            raise BDClientError(f"Missing required create fields: {sorted(missing)}")
        return self.request_post_form("/api/v2/user/create", payload, label="Create User")

    def update_user(self, payload: Dict[str, Any]) -> BDResponse:
        if "user_id" not in payload:
            raise BDClientError("update_user requires user_id")
        return self.request_put_form("/api/v2/user/update", payload, label="Update User")

    def delete_user(self, payload: Dict[str, Any]) -> BDResponse:
        return self.request_delete_form("/api/v2/user/delete", payload, label="Delete User")
