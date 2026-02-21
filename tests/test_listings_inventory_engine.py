import unittest
from pathlib import Path

from milestone3.bd_client import BDClient
from listings_inventory_engine import (
    LISTINGS_AUDIT_CACHE,
    LISTINGS_SEARCH_ENDPOINT,
    build_listings_search_payload,
    build_safe_api_url,
    canonical_listing_key,
    detect_repetition,
    parse_users_portfolio_groups_response,
    normalize_endpoint,
    probe_listings_endpoints,
)


class ListingsInventoryEngineTests(unittest.TestCase):
    def setUp(self):
        if LISTINGS_AUDIT_CACHE.exists():
            LISTINGS_AUDIT_CACHE.unlink()

    def test_discovery_does_not_crash_without_network(self):
        client = BDClient("https://example.com", "k")
        fixtures = {
            "GET:/api/v2/data_categories/get/75": {
                "status": "success",
                "message": [{"data_id": "75", "data_type": "4"}],
            },
            "POST:/api/v2/users_portfolio_groups/search": {
                "status": "success",
                "message": [{"group_id": 99, "group_name": "Sample", "data_id": 75}],
                "total_posts": "10",
                "total_pages": "10",
            },
        }
        decision = probe_listings_endpoints(client, dry_run=True, fixtures=fixtures)
        self.assertIsNotNone(decision.selected_endpoint)
        self.assertEqual(decision.selected_endpoint, LISTINGS_SEARCH_ENDPOINT)

    def test_dedupe_canonical_key_same_record(self):
        row = {"group_id": 123, "group_name": "A"}
        key1 = canonical_listing_key(row)
        key2 = canonical_listing_key(dict(row))
        dedupe = {key1: row, key2: dict(row)}
        self.assertEqual(len(dedupe), 1)

    def test_repetition_detection_aborts(self):
        self.assertTrue(detect_repetition(["a", "a", "a"], repeat_threshold=3))
        self.assertFalse(detect_repetition(["a", "b", "a", "b"], repeat_threshold=3))

    def test_streamlit_tab_and_buttons_exist(self):
        app_src = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("Listings Inventory (Discovery)", app_src)
        self.assertIn("Probe API for Listings Endpoints", app_src)
        self.assertIn("Build Listings (Scrape Fallback)", app_src)


    def test_endpoint_normalization_and_safe_url_builder(self):
        self.assertIsNone(normalize_endpoint(None))
        self.assertIsNone(normalize_endpoint(" none "))
        self.assertEqual(normalize_endpoint("api/v2/users_portfolio_groups/search"), "/api/v2/users_portfolio_groups/search")
        with self.assertRaises(ValueError):
            normalize_endpoint("https://evil.example/path")

        final_url = build_safe_api_url("https://www.vailvacay.com", "/api/v2/users_portfolio_groups/search")
        self.assertEqual(final_url, "https://www.vailvacay.com/api/v2/users_portfolio_groups/search")

        with self.assertRaises(ValueError):
            build_safe_api_url("https://www.vailvacay.com", None)
        with self.assertRaises(ValueError):
            build_safe_api_url("https://www.vailvacay.comnone", "/api/v2/users_portfolio_groups/search")

    def test_stress_sequence_clear_then_fetch_is_blocked(self):
        endpoint = normalize_endpoint(LISTINGS_SEARCH_ENDPOINT)
        self.assertEqual(endpoint, LISTINGS_SEARCH_ENDPOINT)

        cleared_endpoint = normalize_endpoint(None)
        self.assertIsNone(cleared_endpoint)
        with self.assertRaises(ValueError):
            build_safe_api_url("https://www.vailvacay.com", cleared_endpoint)

    def test_listings_request_builder_and_parser(self):
        payload = build_listings_search_payload(limit=10, page=2)
        self.assertEqual(payload["action"], "search")
        self.assertEqual(payload["output_type"], "array")
        self.assertEqual(payload["data_id"], 75)
        self.assertEqual(payload["limit"], 10)
        self.assertEqual(payload["page"], 2)

        parsed = parse_users_portfolio_groups_response(
            {
                "status": "success",
                "message": [{"group_id": 123, "group_name": "A"}],
                "total_posts": "370",
                "total_pages": "37",
            }
        )
        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["has_group_id"])
        self.assertEqual(parsed["total_posts"], 370)
        self.assertEqual(parsed["total_pages"], 37)


if __name__ == "__main__":
    unittest.main()
