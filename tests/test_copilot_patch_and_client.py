from __future__ import annotations

from copilot_patch import compute_score, generate_patch_plan, resolve_description_text
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


def test_resolve_description_text_prefers_group_desc():
    listing = {
        "group_desc": "<p>Primary description</p>",
        "group_description": "Backup description",
        "description": "Generic description",
    }

    text, source = resolve_description_text(listing)

    assert source == "group_desc"
    assert text == "<p>Primary description</p>"


def test_preview_after_uses_patched_group_desc_word_count():
    profile = default_profile()
    listing = {
        "listing_id": "177",
        "user_id": "177",
        "group_name": "Peak Cabin",
        "group_category": "Cabin",
        "group_desc": "tiny",
        "post_tags": "nature",
        "images_count": 3,
        "lat": "39.63",
        "lon": "-106.37",
        "group_status": "1",
        "listing_snapshot": {"post_tags": "nature", "group_desc": "tiny"},
    }

    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile)

    after_description_evidence = plan["preview_before_after"]["after_components"]["description"]["evidence"]
    assert "desc_source_field=group_desc" in after_description_evidence
    assert "desc_word_count=" in after_description_evidence
    assert after_description_evidence != score.components["description"]["evidence"]



def test_copilot_self_checks():
    assert validate_base_url_self_check()[0] is True
    assert validate_form_payload_self_check()[0] is True


def test_post_tags_allowed_from_listing_snapshot_schema():
    profile = default_profile()
    listing = {
        "listing_id": "88",
        "user_id": "88",
        "desc_word_count": 1,
        "post_tags": "",
        "images_count": 0,
        "lat": "40.0",
        "lon": "-105.0",
        "group_status": "1",
        "listing_snapshot": {"post_tags": "", "group_desc": ""},
        "search_obj": {"first_name": "Jane", "email": "jane@example.com"},
    }
    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile)
    assert "post_tags" in plan["proposed_changes"]
    assert plan["schema_validation"]["schema_mode"].startswith("union")


def test_unknown_fields_still_blocked_with_schema_audit_reason():
    profile = default_profile()
    listing = {
        "listing_id": "99",
        "user_id": "99",
        "desc_word_count": 1,
        "post_tags": "",
        "images_count": 0,
        "lat": "40.0",
        "lon": "-105.0",
        "group_status": "0",
        "listing_snapshot": {"post_tags": "", "group_desc": ""},
    }
    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile, set_active_opt_in=True)
    assert "group_status" not in plan["proposed_changes"]
    assert any("group_status blocked:" in warning for warning in plan["validation_warnings"])
