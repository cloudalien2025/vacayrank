from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROFILE_DIR = Path("data/profiles")
ACTIVE_PROFILE_PATH = PROFILE_DIR / "active_profile.json"
SCHEMA_VERSION = 1

REQUIRED_WEIGHTS = ("description", "media", "tags", "geo", "freshness", "status")
REQUIRED_PARAMS = (
    "desc_good_words",
    "desc_ok_words",
    "image_good",
    "image_ok",
    "tag_good",
    "tag_ok",
    "fresh_good_days",
    "fresh_ok_days",
    "inactive_penalty",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_profile() -> Dict[str, Any]:
    now = _now_iso()
    return {
        "profile_id": "v1_completeness",
        "profile_name": "V1 Completeness",
        "created_at": now,
        "updated_at": now,
        "schema_version": SCHEMA_VERSION,
        "weights": {
            "description": 0.30,
            "media": 0.20,
            "tags": 0.15,
            "geo": 0.15,
            "freshness": 0.15,
            "status": 0.05,
        },
        "params": {
            "desc_good_words": 250,
            "desc_ok_words": 120,
            "image_good": 6,
            "image_ok": 3,
            "tag_good": 6,
            "tag_ok": 3,
            "fresh_good_days": 60,
            "fresh_ok_days": 180,
            "inactive_penalty": 0.60,
        },
    }


def canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_profile_hash(profile: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(profile).encode("utf-8")).hexdigest()


def validate_profile(profile: Dict[str, Any]) -> None:
    required_top = {"profile_id", "profile_name", "created_at", "updated_at", "schema_version", "weights", "params"}
    if set(profile.keys()) != required_top:
        raise ValueError("Profile top-level keys must match schema exactly")
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported schema_version")

    weights = profile.get("weights")
    params = profile.get("params")
    if not isinstance(weights, dict) or not isinstance(params, dict):
        raise ValueError("weights and params must be objects")
    if set(weights.keys()) != set(REQUIRED_WEIGHTS):
        raise ValueError("weights keys must match schema exactly")
    if set(params.keys()) != set(REQUIRED_PARAMS):
        raise ValueError("params keys must match schema exactly")

    weight_total = 0.0
    for key in REQUIRED_WEIGHTS:
        value = float(weights[key])
        if value < 0:
            raise ValueError(f"weight {key} must be non-negative")
        weight_total += value
    if abs(weight_total - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")

    int_like = [k for k in REQUIRED_PARAMS if k != "inactive_penalty"]
    for key in int_like:
        if int(params[key]) < 0:
            raise ValueError(f"param {key} must be >= 0")
    penalty = float(params["inactive_penalty"])
    if not (0.0 <= penalty <= 1.0):
        raise ValueError("inactive_penalty must be in [0,1]")


def ensure_profile_store() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    default = default_profile()
    default_path = PROFILE_DIR / f"{default['profile_id']}.json"
    if not default_path.exists():
        save_profile(default)
    if not ACTIVE_PROFILE_PATH.exists():
        set_active_profile(default["profile_id"])


def profile_path(profile_id: str) -> Path:
    return PROFILE_DIR / f"{profile_id}.json"


def save_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    validate_profile(profile)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = profile_path(str(profile["profile_id"]))
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def load_profile(profile_id: str) -> Dict[str, Any]:
    path = profile_path(profile_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_profile(payload)
    return payload


def list_profiles() -> List[Dict[str, Any]]:
    ensure_profile_store()
    out: List[Dict[str, Any]] = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        if path.name == ACTIVE_PROFILE_PATH.name:
            continue
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            validate_profile(profile)
            out.append(profile)
        except Exception:
            continue
    return out


def set_active_profile(profile_id: str) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"profile_id": profile_id, "updated_at": _now_iso()}
    ACTIVE_PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_active_profile_id() -> str:
    ensure_profile_store()
    try:
        payload = json.loads(ACTIVE_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    profile_id = str(payload.get("profile_id") or "v1_completeness")
    if not profile_path(profile_id).exists():
        profile_id = "v1_completeness"
        set_active_profile(profile_id)
    return profile_id


def get_active_profile() -> Dict[str, Any]:
    return load_profile(get_active_profile_id())
