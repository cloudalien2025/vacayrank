from __future__ import annotations

from typing import Any, Dict, Tuple

from milestone3.patch_engine import human_diff_table


def _extract_user_payload(response_json: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(response_json.get("data"), dict):
        return response_json["data"]
    if isinstance(response_json.get("user"), dict):
        return response_json["user"]
    return response_json


def verify_user_fields(client, user_id: int, expected_patch: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    response = client.get_user(user_id)
    actual = _extract_user_payload(response.json_data)
    mismatches = {}
    for key, expected in expected_patch.items():
        if str(actual.get(key, "")) != str(expected):
            mismatches[key] = {"expected": expected, "actual": actual.get(key)}
    details = {
        "checked_fields": list(expected_patch.keys()),
        "mismatches": mismatches,
        "diff_rows": human_diff_table(actual, expected_patch),
        "status_code": response.status_code,
    }
    return (len(mismatches) == 0 and response.ok, details, actual)
