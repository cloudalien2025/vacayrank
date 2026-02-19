from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlencode


def compute_patch(current_record: Dict[str, Any], proposed_values: Dict[str, Any], allow_clearing: bool = False) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    for key, new_value in proposed_values.items():
        old_value = current_record.get(key)
        if new_value == old_value:
            continue
        if not allow_clearing and (new_value is None or str(new_value).strip() == ""):
            continue
        patch[key] = new_value
    return patch


def human_diff_table(current_record: Dict[str, Any], patch_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, new_value in patch_dict.items():
        rows.append({"field": key, "current": current_record.get(key), "proposed": new_value})
    return rows


def encode_form(payload_dict: Dict[str, Any]) -> str:
    return urlencode({k: "" if v is None else str(v) for k, v in payload_dict.items()}, doseq=True)
