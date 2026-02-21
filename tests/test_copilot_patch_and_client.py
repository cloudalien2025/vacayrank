from __future__ import annotations

from copilot_patch import compute_patch_eligibility, compute_score, generate_patch_plan, resolve_description_text
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


def test_missing_geo_is_advisory_when_only_desc_and_tags_are_proposed():
    profile = default_profile()
    listing = {
        "listing_id": "vail-1",
        "user_id": "vail-1",
        "group_name": "Vail Film Festival",
        "group_desc": "",
        "post_tags": "film",
        "images_count": 2,
        "lat": "",
        "lon": "",
        "group_status": "1",
        "listing_snapshot": {"group_desc": "", "post_tags": "film"},
    }

    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile)

    assert plan["safe_to_apply"] is True
    assert sorted(plan["eligible_fields"]) == ["group_desc", "post_tags"]
    assert plan["blocked_fields"] == []
    assert plan["blocking_reasons_by_field"] == {}
    assert any("Missing lat/lon" in note for note in plan["advisory_notes"])


def test_all_proposed_fields_blocked_sets_safe_to_apply_false():
    profile = default_profile()
    listing = {
        "listing_id": "vail-3",
        "user_id": "vail-3",
        "group_name": "Vail Film Festival",
        "group_desc": "",
        "post_tags": "",
        "images_count": 2,
        "lat": "",
        "lon": "",
        "group_status": "1",
        "listing_snapshot": {},
        "search_obj": {"first_name": "Jane"},
    }

    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile)

    assert plan["safe_to_apply"] is False
    assert plan["eligible_fields"] == []
    assert set(plan["blocked_fields"]) == {"group_desc", "post_tags"}
    assert set(plan["blocking_reasons_by_field"].keys()) == {"group_desc", "post_tags"}


def test_geo_fields_are_blocked_when_coordinates_are_missing_but_other_fields_can_apply():
    schema_info = {
        "allowed_fields": {"group_desc", "lat", "lon"},
        "schema_mode": "snapshot",
        "field_presence": {
            "snapshot": {"group_desc", "lat", "lon"},
            "user_get": set(),
            "search": set(),
        },
    }
    current = {"group_desc": "old", "lat": "", "lon": ""}
    proposed = {"group_desc": "new", "lat": "39.63", "lon": "-106.37"}

    eligibility = compute_patch_eligibility(current, proposed, schema_info, has_geo_coordinates=False)

    assert eligibility["safe_to_apply"] is True
    assert eligibility["eligible_fields"] == ["group_desc"]
    assert set(eligibility["blocked_fields"]) == {"lat", "lon"}
    assert set(eligibility["blocking_reasons_by_field"].keys()) == {"lat", "lon"}


def test_patch_plan_eligibility_is_exhaustive_and_never_null():
    profile = default_profile()
    listing = {
        "listing_id": "vail-4",
        "user_id": "vail-4",
        "group_name": "Vail Film Festival",
        "group_desc": "",
        "post_tags": "",
        "images_count": 0,
        "lat": "",
        "lon": "",
        "group_status": "1",
        "listing_snapshot": {"group_desc": ""},
    }

    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile)

    assert isinstance(plan["patch_plan"], dict)
    assert isinstance(plan["eligible_fields"], list)
    assert isinstance(plan["blocked_fields"], list)
    assert isinstance(plan["blocking_reasons_by_field"], dict)
    assert isinstance(plan["safe_to_apply"], bool)

    all_fields = set(plan["patch_plan"].keys())
    partition = set(plan["eligible_fields"]) | set(plan["blocked_fields"])
    assert partition == all_fields
    assert set(plan["eligible_fields"]).isdisjoint(set(plan["blocked_fields"]))
    assert plan["safe_to_apply"] == (len(plan["eligible_fields"]) > 0)


def test_empty_patch_plan_has_never_null_eligibility_shapes():
    profile = default_profile()
    listing = {
        "listing_id": "vail-5",
        "user_id": "vail-5",
        "group_name": "Vail Film Festival",
        "group_desc": " ".join(["festival"] * 140),
        "post_tags": "film,festival,colorado",
        "images_count": 6,
        "lat": "39.6403",
        "lon": "-106.3742",
        "group_status": "1",
        "listing_snapshot": {"group_desc": "ready", "post_tags": "film,festival,colorado"},
    }

    score = compute_score(listing, profile)
    plan = generate_patch_plan(listing, score, profile)

    assert plan["patch_plan"] == {}
    assert plan["eligible_fields"] == []
    assert plan["blocked_fields"] == []
    assert plan["blocking_reasons_by_field"] == {}
    assert plan["safe_to_apply"] is False
