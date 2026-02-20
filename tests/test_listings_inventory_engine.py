import unittest
from pathlib import Path

from milestone3.bd_client import BDClient
from listings_inventory_engine import (
    LISTINGS_AUDIT_CACHE,
    canonical_listing_key,
    detect_repetition,
    probe_listings_endpoints,
)


class ListingsInventoryEngineTests(unittest.TestCase):
    def setUp(self):
        if LISTINGS_AUDIT_CACHE.exists():
            LISTINGS_AUDIT_CACHE.unlink()

    def test_discovery_does_not_crash_without_network(self):
        client = BDClient("https://example.com", "k")
        fixtures = {
            "GET:/api/v2/users_portfolio_groups/search": {
                "status": "success",
                "message": [{"group_id": 99, "group_name": "Sample", "data_id": 75}],
            }
        }
        decision = probe_listings_endpoints(client, dry_run=True, fixtures=fixtures)
        self.assertIsNotNone(decision.selected_endpoint)
        self.assertEqual(decision.selected_endpoint["path"], "/api/v2/users_portfolio_groups/search")

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


if __name__ == "__main__":
    unittest.main()
