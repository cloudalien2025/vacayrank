from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from profiles import compute_profile_hash


@dataclass
class ScoreResult:
    total_score: float
    components: Dict[str, Dict[str, Any]]
    profile_id: str
    profile_hash: str


FIELD_MAPPING: Dict[str, str] = {
    "description": "group_desc",
    "tags": "post_tags",
    "lat": "lat",
    "lon": "lon",
    "status": "group_status",
    "name": "group_name",
    "address": "post_location",
}

GEO_RELATED_FIELDS: Set[str] = {
    "lat",
    "lon",
    "post_location",
    "location",
    "address",
    "address1",
    "address2",
    "city",
    "state",
    "zip",
    "postal_code",
    "country",
}


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> Optional[float]:
    try:
        text = _as_str(value)
        return float(text) if text else None
    except Exception:
        return None


def _days_ago(ts: Any) -> Optional[int]:
    text = _as_str(ts)
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 86400))


def _component_tier(value: int, ok: int, good: int) -> float:
    if value >= good:
        return 1.0
    if value >= ok:
        return 0.6
    if value > 0:
        return 0.3
    return 0.0


_DESCRIPTION_FIELDS: List[str] = [
    "group_desc",
    "group_description",
    "post_desc",
    "post_description",
    "description",
    "post_content",
    "content",
]


def strip_html(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", s)
    normalized_ws = re.sub(r"\s+", " ", without_tags)
    return normalized_ws.strip()


def _count_words(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    text = value.strip()
    if not text:
        return 0
    return len([token for token in re.split(r"\s+", text) if token])


def resolve_description_text(listing_obj: Dict[str, Any]) -> tuple[str, str]:
    for field_name in _DESCRIPTION_FIELDS:
        value = listing_obj.get(field_name)
        if isinstance(value, str) and value.strip():
            return value, field_name
    return "", ""


def compute_score(listing_norm: Dict[str, Any], profile: Dict[str, Any]) -> ScoreResult:
    params = profile["params"]
    weights = profile["weights"]

    desc_text_raw, desc_source_field = resolve_description_text(listing_norm)
    desc_visible_text = strip_html(desc_text_raw)
    desc_words = _count_words(desc_visible_text)
    image_count = _as_int(listing_norm.get("images_count") or listing_norm.get("image_count"))
    tags_value = _as_str(listing_norm.get("post_tags"))
    tags = [t.strip() for t in tags_value.split(",") if t.strip()] if tags_value else (listing_norm.get("tags_list") or [])
    tag_count = len(tags)
    updated_days_ago = _days_ago(listing_norm.get("revision_timestamp") or listing_norm.get("date_updated"))

    description_score = _component_tier(desc_words, int(params["desc_ok_words"]), int(params["desc_good_words"]))
    media_score = _component_tier(image_count, int(params["image_ok"]), int(params["image_good"]))
    tags_score = _component_tier(tag_count, int(params["tag_ok"]), int(params["tag_good"]))
    geo_score = 1.0 if _safe_float(listing_norm.get("lat")) is not None and _safe_float(listing_norm.get("lon")) is not None else 0.0
    if updated_days_ago is None:
        freshness_score = 0.3
    elif updated_days_ago <= int(params["fresh_good_days"]):
        freshness_score = 1.0
    elif updated_days_ago <= int(params["fresh_ok_days"]):
        freshness_score = 0.6
    else:
        freshness_score = 0.3
    status_score = 1.0 if _as_str(listing_norm.get("group_status")) == "1" else 0.0

    total_0_1 = (
        weights["description"] * description_score
        + weights["media"] * media_score
        + weights["tags"] * tags_score
        + weights["geo"] * geo_score
        + weights["freshness"] * freshness_score
        + weights["status"] * status_score
    )
    total_score = total_0_1 * 100.0
    if _as_str(listing_norm.get("group_status")) != "1":
        total_score = total_score * float(params["inactive_penalty"])

    components = {
        "description": {
            "score": description_score,
            "evidence": (
                f"desc_source_field={desc_source_field or 'none'} "
                f"desc_word_count={desc_words} "
                f"desc_raw_len={len(desc_text_raw)}"
            ),
        },
        "media": {"score": media_score, "evidence": f"image_count={image_count}"},
        "tags": {"score": tags_score, "evidence": f"tag_count={tag_count}"},
        "geo": {"score": geo_score, "evidence": f"lat={listing_norm.get('lat')} lon={listing_norm.get('lon')}"},
        "freshness": {
            "score": freshness_score,
            "evidence": f"updated_days_ago={updated_days_ago} revision_timestamp={listing_norm.get('revision_timestamp')}",
        },
        "status": {"score": status_score, "evidence": f"group_status={listing_norm.get('group_status')}"},
    }
    return ScoreResult(
        total_score=round(total_score, 3),
        components=components,
        profile_id=profile["profile_id"],
        profile_hash=compute_profile_hash(profile),
    )


def identify_weaknesses(score_result: ScoreResult, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    weaknesses: List[Dict[str, Any]] = []
    for name, component in score_result.components.items():
        score = float(component.get("score", 0.0))
        if score >= 1.0:
            continue
        impact = float(profile["weights"].get(name, 0.0)) * (1.0 - score)
        weaknesses.append({"component": name, "impact": round(impact, 4), "evidence": component.get("evidence", "")})
    return sorted(weaknesses, key=lambda item: item["impact"], reverse=True)


def _description_template(listing: Dict[str, Any]) -> str:
    name = _as_str(listing.get("group_name")) or "This property"
    category = _as_str(listing.get("group_category") or listing.get("category")) or "vacation rental"
    return (
        f"<p>{name} is a {category} designed for guests seeking a comfortable stay.</p>\n"
        "<ul>\n"
        "  <li>Clearly list sleeping arrangements and guest capacity.</li>\n"
        "  <li>Summarize key amenities and location highlights.</li>\n"
        "  <li>Include check-in details and house expectations.</li>\n"
        "</ul>"
    )


def _tag_template(listing: Dict[str, Any]) -> str:
    seeds = [
        _as_str(listing.get("group_category") or listing.get("category")),
        _as_str(listing.get("group_name")),
        "vacation rental",
        "family friendly",
        "mountain getaway",
        "weekend stay",
    ]
    unique: List[str] = []
    for item in seeds:
        if not item:
            continue
        clean = item.lower().replace("  ", " ").strip()
        if clean and clean not in unique:
            unique.append(clean)
        if len(unique) >= 6:
            break
    return ", ".join(unique[:6])


def _extract_dict(candidate: Any) -> Dict[str, Any]:
    return candidate if isinstance(candidate, dict) else {}


def get_writable_schema_for_listing(
    listing_snapshot: Optional[Dict[str, Any]],
    user_get_obj: Optional[Dict[str, Any]] = None,
    search_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = _extract_dict(listing_snapshot)
    user_get = _extract_dict(user_get_obj)
    search = _extract_dict(search_obj)

    snapshot_fields = set(snapshot.keys())
    user_get_fields = set(user_get.keys())
    search_fields = set(search.keys())
    allowed_fields: Set[str] = set().union(snapshot_fields, user_get_fields, search_fields)

    if user_get_fields or search_fields:
        schema_mode = "union(snapshot+user_get+search)"
    else:
        schema_mode = "snapshot"

    return {
        "allowed_fields": allowed_fields,
        "schema_mode": schema_mode,
        "field_presence": {
            "snapshot": snapshot_fields,
            "user_get": user_get_fields,
            "search": search_fields,
        },
    }


def _validate_writable_field(field: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
    allowed_fields = schema_info.get("allowed_fields") or set()
    field_presence = schema_info.get("field_presence") or {}
    in_snapshot = field in (field_presence.get("snapshot") or set())
    in_user_get = field in (field_presence.get("user_get") or set())
    in_search = field in (field_presence.get("search") or set())
    allowed = field in allowed_fields

    reason = "field present in schema set" if allowed else "field absent from writable schema set"
    return {
        "allowed": allowed,
        "field": field,
        "schema_mode": schema_info.get("schema_mode"),
        "in_snapshot": in_snapshot,
        "in_user_get": in_user_get,
        "in_search": in_search,
        "reason": reason,
    }


def _has_material_change(current_value: Any, proposed_value: Any) -> bool:
    current_text = _as_str(current_value)
    proposed_text = _as_str(proposed_value)
    if not proposed_text:
        return False
    return current_text != proposed_text


def compute_patch_eligibility(
    current_obj: Dict[str, Any],
    proposed_fixes: Dict[str, Any],
    schema_info: Dict[str, Any],
    *,
    has_geo_coordinates: bool,
) -> Dict[str, Any]:
    field_eligibility: Dict[str, Dict[str, Any]] = {}
    blocked_fields: List[Dict[str, str]] = []
    eligible_fields: List[str] = []
    apply_changes: Dict[str, Any] = {}

    for field, proposed_value in proposed_fixes.items():
        schema_validation = _validate_writable_field(field, schema_info)
        reason: Optional[str] = None
        if not schema_validation["allowed"]:
            reason = (
                f"schema={schema_validation['schema_mode']} snapshot={schema_validation['in_snapshot']} "
                f"user_get={schema_validation['in_user_get']} search={schema_validation['in_search']} reason={schema_validation['reason']}"
            )
        elif field in GEO_RELATED_FIELDS and not has_geo_coordinates:
            reason = "Missing coordinates source"
        elif not _has_material_change(current_obj.get(field), proposed_value):
            reason = "No material non-empty change from current value"

        if reason:
            blocked_fields.append({"field": field, "reason": reason})
            field_eligibility[field] = {"eligible": False, "reason": reason}
            continue

        eligible_fields.append(field)
        apply_changes[field] = proposed_value
        field_eligibility[field] = {"eligible": True, "reason": None}

    return {
        "field_eligibility": field_eligibility,
        "eligible_fields": eligible_fields,
        "blocked_fields": blocked_fields,
        "safe_to_apply": bool(eligible_fields),
        "proposed_changes": apply_changes,
    }


def generate_patch_plan(
    listing_norm: Dict[str, Any],
    score_result: ScoreResult,
    profile: Dict[str, Any],
    *,
    set_active_opt_in: bool = False,
) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "target_user_id": _as_str(listing_norm.get("user_id") or listing_norm.get("listing_id") or listing_norm.get("group_id")),
        "profile_id": score_result.profile_id,
        "profile_hash": score_result.profile_hash,
        "proposed_changes": {},
        "proposed_fixes": {},
        "rationales": {},
        "validation_warnings": [],
        "preview_before_after": {"before_total": score_result.total_score, "after_total": score_result.total_score},
        "safe_to_apply": False,
        "advisory_notes": [],
        "eligible_fields": [],
        "blocked_fields": [],
        "field_eligibility": {},
    }

    params = profile["params"]
    schema_info = get_writable_schema_for_listing(
        _extract_dict(listing_norm.get("listing_snapshot") or listing_norm.get("raw_user")),
        _extract_dict(listing_norm.get("user_get_obj")),
        _extract_dict(listing_norm.get("search_obj")),
    )
    plan["schema_validation"] = {
        "schema_mode": schema_info["schema_mode"],
        "snapshot_fields": len(schema_info["field_presence"]["snapshot"]),
        "user_get_fields": len(schema_info["field_presence"]["user_get"]),
        "search_fields": len(schema_info["field_presence"]["search"]),
    }

    desc_text_raw, _ = resolve_description_text(listing_norm)
    desc_words = _count_words(strip_html(desc_text_raw))
    if desc_words < int(params["desc_ok_words"]):
        field = FIELD_MAPPING["description"]
        proposed = _description_template(listing_norm)
        plan["proposed_fixes"][field] = proposed
        plan["rationales"][field] = {
            "why": "Description below threshold",
            "evidence": f"desc_word_count={desc_words} < desc_ok_words={params['desc_ok_words']}",
            "expected_component_lift": "description up to 0.6 or 1.0",
        }

    tags_value = _as_str(listing_norm.get("post_tags"))
    tags = [t.strip() for t in tags_value.split(",") if t.strip()] if tags_value else (listing_norm.get("tags_list") or [])
    if len(tags) < 3:
        field = FIELD_MAPPING["tags"]
        proposed_tags = _tag_template(listing_norm)
        plan["proposed_fixes"][field] = proposed_tags
        plan["rationales"][field] = {
            "why": "Insufficient tags",
            "evidence": f"tag_count={len(tags)} < 3",
            "expected_component_lift": "tags up to 0.6 or 1.0",
        }

    image_count = _as_int(listing_norm.get("images_count") or listing_norm.get("image_count"))
    if image_count < 5:
        plan["advisory_notes"].append("Media TODO: add at least 5 images (no upload/write in Milestone 4)")

    has_geo = _safe_float(listing_norm.get("lat")) is not None and _safe_float(listing_norm.get("lon")) is not None

    if _as_str(listing_norm.get("group_status")) != "1":
        status_field = FIELD_MAPPING["status"]
        if set_active_opt_in:
            plan["proposed_fixes"][status_field] = "1"
            plan["rationales"][status_field] = {
                "why": "Listing is inactive",
                "evidence": f"group_status={listing_norm.get('group_status')}",
                "expected_component_lift": "status to 1.0 and removes inactive penalty",
            }
        else:
            plan["advisory_notes"].append("Status is inactive; enable 'Set Active' option to propose activation")

    if not _as_str(listing_norm.get("revision_timestamp")):
        plan["advisory_notes"].append("Freshness recommendation: review and update listing content manually")

    geo_targets = [field for field in plan["proposed_fixes"] if field in GEO_RELATED_FIELDS]
    if not has_geo:
        if geo_targets:
            plan["validation_warnings"].append("Missing lat/lon; geo writeback blocked until coordinates are available")
        else:
            plan["advisory_notes"].append("Missing lat/lon; geo writeback blocked until coordinates are available")

    eligibility = compute_patch_eligibility(
        listing_norm,
        plan["proposed_fixes"],
        schema_info,
        has_geo_coordinates=has_geo,
    )
    plan["field_eligibility"] = eligibility["field_eligibility"]
    plan["eligible_fields"] = eligibility["eligible_fields"]
    plan["blocked_fields"] = eligibility["blocked_fields"]
    plan["safe_to_apply"] = eligibility["safe_to_apply"]
    plan["proposed_changes"] = eligibility["proposed_changes"]
    for blocked in plan["blocked_fields"]:
        plan["validation_warnings"].append(f"{blocked['field']} blocked: {blocked['reason']}")
        plan["rationales"].pop(blocked["field"], None)

    simulated = copy.deepcopy(listing_norm)
    simulated.update(plan["proposed_changes"])
    after = compute_score(simulated, profile)
    plan["preview_before_after"] = {
        "before_total": score_result.total_score,
        "after_total": after.total_score,
        "delta": round(after.total_score - score_result.total_score, 3),
        "before_components": score_result.components,
        "after_components": after.components,
    }

    if not plan["proposed_changes"]:
        plan["validation_warnings"].append("No safe field changes proposed")
    return plan
