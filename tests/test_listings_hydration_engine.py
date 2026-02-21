import unittest

from listings_hydration_engine import (
    LISTINGS_GET_ENDPOINT_TEMPLATE,
    build_get_endpoint_path,
    build_request_headers,
    extract_features,
    parse_success_listing_payload,
)


class ListingsHydrationEngineTests(unittest.TestCase):
    def test_request_builder_endpoint_and_headers(self):
        endpoint = build_get_endpoint_path("765")
        self.assertEqual(endpoint, LISTINGS_GET_ENDPOINT_TEMPLATE.format(group_id="765"))

        headers = build_request_headers("abc123")
        self.assertIn("X-Api-Key", headers)
        self.assertEqual(headers["X-Api-Key"], "abc123")
        self.assertNotIn("Content-Type", headers)

    def test_parser_accepts_success_and_extracts_group_id(self):
        payload = {
            "status": "success",
            "message": [{"group_id": 765, "group_name": "Unit 765"}],
        }
        ok, record, reason = parse_success_listing_payload(payload)
        self.assertTrue(ok)
        self.assertEqual(str(record["group_id"]), "765")
        self.assertEqual(reason, "")

    def test_feature_extractor_required_columns(self):
        rows = extract_features(
            [
                {
                    "listing_id": "765",
                    "norm": {
                        "listing_id": "765",
                        "canonical_url": "https://example.com/test",
                        "group_name": "Test Listing",
                        "group_category": "Vacation Rental",
                        "tags_list": ["ski", "family"],
                        "group_desc_text": "A nice place",
                        "desc_word_count": 3,
                        "desc_char_count": 12,
                        "post_location": "Vail, CO",
                        "lat": "39.64",
                        "lon": "-106.37",
                        "image_count": 2,
                        "cover_image_url": "https://example.com/a.webp",
                        "date_updated": "2024-01-01T00:00:00+00:00",
                        "revision_count": "4",
                        "group_status": "active",
                    },
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        required = {
            "listing_id",
            "canonical_url",
            "group_name",
            "category",
            "tag_count",
            "has_tags",
            "has_description",
            "desc_word_count",
            "desc_char_count",
            "has_location",
            "has_geo",
            "lat",
            "lon",
            "image_count",
            "cover_image_present",
            "updated_days_ago",
            "revision_count",
            "group_status",
            "source",
            "data_id",
        }
        self.assertTrue(required.issubset(set(rows[0].keys())))


if __name__ == "__main__":
    unittest.main()
