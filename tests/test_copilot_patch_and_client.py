from __future__ import annotations

from copilot_patch import compute_score, generate_patch_plan
from bd_copilot_client import validate_base_url_self_check, validate_form_payload_self_check
from profiles import default_profile


def test_copilot_plan_has_evidence_and_preview_delta():
    profile = default_profile()
    listing = {
        "listing_id": "77",
        "user_id": "77",
        "group_id": "77",
        "group_name": "River Cabin",
        "group_category": "Cabin",
        "desc_word_count": 20,
        "post_tags": "",
        "images_count": 1,
        "lat": "39.63",
        "lon": "-106.37",
        "group_status": "0",
        "raw_user": {
            "group_desc": "",
            "post_tags": "",
            "group_status": "0",
            "group_name": "River Cabin",
            "lat": "39.63",
            "lon": "-106.37",
        },
    }
    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile, set_active_opt_in=True)
    assert plan["target_user_id"] == "77"
    assert "group_desc" in plan["proposed_changes"]
    assert "post_tags" in plan["proposed_changes"]
    assert "group_status" in plan["proposed_changes"]
    assert "desc_word_count" in plan["rationales"]["group_desc"]["evidence"]
    assert plan["preview_before_after"]["after_total"] >= plan["preview_before_after"]["before_total"]


def test_copilot_self_checks():
    assert validate_base_url_self_check()[0] is True
    assert validate_form_payload_self_check()[0] is True
