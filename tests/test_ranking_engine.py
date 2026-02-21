from __future__ import annotations

import json
from pathlib import Path

from profiles import default_profile, validate_profile
from ranking_engine import build_feature_frame, run_ranking, score_dataframe


def _fixture_records():
    return [
        {
            "listing_id": "1",
            "norm": {
                "group_id": "1",
                "group_name": "Alpha",
                "group_category": "Cabin",
                "desc_word_count": 300,
                "tags_list": ["a", "b", "c", "d", "e", "f"],
                "lat": "39.5",
                "lon": "-106.1",
                "revision_timestamp": "2026-01-01T00:00:00+00:00",
                "group_status": "1",
                "image_count": 8,
            },
        },
        {
            "listing_id": "2",
            "norm": {
                "group_id": "2",
                "group_name": "Beta",
                "group_category": "Cabin",
                "desc_word_count": 130,
                "tags_list": ["a", "b", "c"],
                "lat": "",
                "lon": "",
                "date_updated": "2025-01-01T00:00:00+00:00",
                "group_status": "1",
                "image_count": 3,
            },
            "images": {"image_count": 3},
        },
        {
            "listing_id": "3",
            "norm": {
                "group_id": "3",
                "group_name": "Gamma",
                "group_category": "Villa",
                "desc_word_count": 130,
                "tags_list": ["a", "b", "c"],
                "lat": "",
                "lon": "",
                "group_status": "0",
                "image_count": 3,
            },
            "images": {"image_count": 3},
        },
    ]


def test_profile_validation_strictness(tmp_path: Path):
    profile = default_profile()
    profile["weights"]["extra"] = 1
    try:
        validate_profile(profile)
        assert False, "expected validation error"
    except ValueError:
        pass


def test_deterministic_scoring_and_tie_break(tmp_path: Path):
    profile = default_profile()
    profile["profile_id"] = "test_profile"

    hydrated = tmp_path / "hydrated.jsonl"
    for row in _fixture_records():
        hydrated.write_text("", encoding="utf-8") if not hydrated.exists() else None
        with hydrated.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    run1 = run_ranking(profile, hydrated_path=hydrated)
    run2 = run_ranking(profile, hydrated_path=hydrated)

    order1 = [r["listing_id"] for r in run1["rows"]]
    order2 = [r["listing_id"] for r in run2["rows"]]
    assert order1 == order2
    assert order1[0] == "1"
    assert run1["rows"][-1]["inactive_penalty_applied"] is True


def test_score_dataframe_tiebreakers():
    profile = default_profile()
    frame = build_feature_frame(_fixture_records())
    ranked = score_dataframe(frame, profile)
    assert ranked.iloc[0]["listing_id"] == "1"
