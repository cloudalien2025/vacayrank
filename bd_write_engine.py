from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from milestone3.bd_client import BDClient
from milestone3.patch_engine import compute_patch, encode_form


@dataclass
class WriteResult:
    request: Dict[str, Any]
    response_status: int
    response_json: Dict[str, Any]
    verified: bool
    verification_payload: Dict[str, Any]


class SafeWriteEngine:
    def __init__(self, client: BDClient) -> None:
        self.client = client
        self.audit_log: List[Dict[str, Any]] = []

    def _log(self, payload: Dict[str, Any]) -> None:
        self.audit_log.append({"ts": datetime.now(timezone.utc).isoformat(), **payload})

    def preview(self, endpoint: str, form_payload: Dict[str, Any], action: str) -> None:
        self._log(
            {
                "phase": "preview",
                "action": action,
                "endpoint": endpoint,
                "encoded_payload": encode_form(form_payload),
            }
        )

    def commit_update(
        self,
        *,
        user_id: int,
        current: Dict[str, Any],
        proposed: Dict[str, Any],
        typed_confirm: str,
        dry_run: bool,
    ) -> WriteResult:
        patch = compute_patch(current, proposed, allow_clearing=False)
        payload = {"user_id": user_id, **patch}
        self.preview("/api/v2/user/update", payload, "update")

        if dry_run:
            return WriteResult(payload, 0, {"dry_run": True}, False, {})
        if typed_confirm.strip() != "CONFIRM":
            return WriteResult(payload, 0, {"error": "Typed CONFIRM required"}, False, {})

        response = self.client.update_user(payload)
        verify = self.client.get_user(user_id)
        verify_payload = verify.json_data.get("data", verify.json_data)
        verified = all(str(verify_payload.get(k, "")) == str(v) for k, v in patch.items())

        self._log(
            {
                "phase": "commit",
                "action": "update",
                "endpoint": "/api/v2/user/update",
                "encoded_payload": encode_form(payload),
                "response_status": response.status_code,
                "response_json": response.json_data,
                "verification_ok": verified,
                "verification_payload": verify_payload,
            }
        )
        return WriteResult(payload, response.status_code, response.json_data, verified, verify_payload)

    def commit_create(self, payload: Dict[str, Any], typed_confirm: str, dry_run: bool) -> WriteResult:
        self.preview("/api/v2/user/create", payload, "create")
        if dry_run:
            return WriteResult(payload, 0, {"dry_run": True}, False, {})
        if typed_confirm.strip() != "CONFIRM":
            return WriteResult(payload, 0, {"error": "Typed CONFIRM required"}, False, {})

        response = self.client.create_user(payload)
        new_id = response.json_data.get("user_id") or response.json_data.get("data", {}).get("user_id")
        verify_payload = {}
        verified = False
        if new_id:
            verify = self.client.get_user(int(new_id))
            verify_payload = verify.json_data.get("data", verify.json_data)
            verified = bool(verify.ok)

        self._log(
            {
                "phase": "commit",
                "action": "create",
                "endpoint": "/api/v2/user/create",
                "encoded_payload": encode_form(payload),
                "response_status": response.status_code,
                "response_json": response.json_data,
                "verification_ok": verified,
                "verification_payload": verify_payload,
            }
        )
        return WriteResult(payload, response.status_code, response.json_data, verified, verify_payload)
