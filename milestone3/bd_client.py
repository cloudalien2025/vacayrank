from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class BDClientError(RuntimeError):
    pass


@dataclass
class BDResponse:
    ok: bool
    status_code: int
    json_data: Dict[str, Any]
    text: str


class BDClient:
    """Brilliant Directories API v2 client with strict write encoding rules."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self, include_form: bool = False) -> Dict[str, str]:
        headers = {"X-Api-Key": self.api_key}
        if include_form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        return headers

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
        )

    def request_get(self, path: str) -> BDResponse:
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._normalize(response)

    def request_post_form(self, path: str, payload_dict: Dict[str, Any]) -> BDResponse:
        response = requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(include_form=True),
            data=payload_dict,
            timeout=self.timeout,
        )
        return self._normalize(response)

    def request_put_form(self, path: str, payload_dict: Dict[str, Any]) -> BDResponse:
        response = requests.put(
            f"{self.base_url}{path}",
            headers=self._headers(include_form=True),
            data=payload_dict,
            timeout=self.timeout,
        )
        return self._normalize(response)

    def request_delete_form(self, path: str, payload_dict: Dict[str, Any]) -> BDResponse:
        response = requests.delete(
            f"{self.base_url}{path}",
            headers=self._headers(include_form=True),
            data=payload_dict,
            timeout=self.timeout,
        )
        return self._normalize(response)

    def search_users(self, payload: Dict[str, Any]) -> BDResponse:
        return self.request_post_form("/api/v2/user/search", payload)

    def get_user(self, user_id: int) -> BDResponse:
        return self.request_get(f"/api/v2/user/get/{int(user_id)}")

    def create_user(self, payload: Dict[str, Any]) -> BDResponse:
        required = {"email", "password", "subscription_id"}
        missing = required - set(payload)
        if missing:
            raise BDClientError(f"Missing required create fields: {sorted(missing)}")
        return self.request_post_form("/api/v2/user/create", payload)

    def update_user(self, payload: Dict[str, Any]) -> BDResponse:
        if "user_id" not in payload:
            raise BDClientError("update_user requires user_id")
        return self.request_put_form("/api/v2/user/update", payload)

    def delete_user(self, payload: Dict[str, Any]) -> BDResponse:
        return self.request_delete_form("/api/v2/user/delete", payload)
