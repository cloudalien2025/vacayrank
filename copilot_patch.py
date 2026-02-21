from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


def compute_score(listing_norm: Dict[str, Any], profile: Dict[str, Any]) -> ScoreResult:
    params = profile["params"]
    weights = profile["weights"]

    desc_words = _as_int(listing_norm.get("desc_word_count"))
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
        "description": {"score": description_score, "evidence": f"desc_word_count={desc_words}"},
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
        "rationales": {},
        "validation_warnings": [],
        "preview_before_after": {"before_total": score_result.total_score, "after_total": score_result.total_score},
        "safe_to_apply": True,
        "advisory_notes": [],
    }

    params = profile["params"]
    existing_keys = set((listing_norm.get("raw_user") or {}).keys())

    desc_words = _as_int(listing_norm.get("desc_word_count"))
    if desc_words < int(params["desc_ok_words"]):
        field = FIELD_MAPPING["description"]
        if field in existing_keys:
            proposed = _description_template(listing_norm)
            plan["proposed_changes"][field] = proposed
            plan["rationales"][field] = {
                "why": "Description below threshold",
                "evidence": f"desc_word_count={desc_words} < desc_ok_words={params['desc_ok_words']}",
                "expected_component_lift": "description up to 0.6 or 1.0",
            }

    tags_value = _as_str(listing_norm.get("post_tags"))
    tags = [t.strip() for t in tags_value.split(",") if t.strip()] if tags_value else (listing_norm.get("tags_list") or [])
    if len(tags) < 3:
        field = FIELD_MAPPING["tags"]
        if field in existing_keys:
            proposed_tags = _tag_template(listing_norm)
            plan["proposed_changes"][field] = proposed_tags
            plan["rationales"][field] = {
                "why": "Insufficient tags",
                "evidence": f"tag_count={len(tags)} < 3",
                "expected_component_lift": "tags up to 0.6 or 1.0",
            }
        else:
            plan["validation_warnings"].append("post_tags not present on user object; cannot safely write tags")

    image_count = _as_int(listing_norm.get("images_count") or listing_norm.get("image_count"))
    if image_count < 5:
        plan["advisory_notes"].append("Media TODO: add at least 5 images (no upload/write in Milestone 4)")

    has_geo = _safe_float(listing_norm.get("lat")) is not None and _safe_float(listing_norm.get("lon")) is not None
    if not has_geo:
        plan["safe_to_apply"] = False
        plan["validation_warnings"].append("Missing lat/lon; geo writeback blocked until coordinates are available")

    if _as_str(listing_norm.get("group_status")) != "1":
        if set_active_opt_in and FIELD_MAPPING["status"] in existing_keys:
            plan["proposed_changes"][FIELD_MAPPING["status"]] = "1"
            plan["rationales"][FIELD_MAPPING["status"]] = {
                "why": "Listing is inactive",
                "evidence": f"group_status={listing_norm.get('group_status')}",
                "expected_component_lift": "status to 1.0 and removes inactive penalty",
            }
        else:
            plan["advisory_notes"].append("Status is inactive; enable 'Set Active' option to propose activation")

    if not _as_str(listing_norm.get("revision_timestamp")):
        plan["advisory_notes"].append("Freshness recommendation: review and update listing content manually")

    simulated = dict(listing_norm)
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
