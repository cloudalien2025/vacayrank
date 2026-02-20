import unittest
from pathlib import Path

from bd_copilot import evaluate_rules
from inventory_engine import InventoryBundle, cache_inventory_to_disk, fetch_inventory_index, inventory_to_csv, load_inventory_from_cache, load_inventory_progress


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", content_type="application/json", headers=None):
        self.ok = status_code < 400
        self.status_code = status_code
        self.json_data = [] if json_data is None else json_data
        self.text = text
        self.content_type = content_type
        self.headers = headers or {}


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.base_url = "https://example.com"
        self.evidence_log = []

    def search_users(self, payload):
        response = self.responses.pop(0)
        self.evidence_log.append(
            {
                "label": "M1 Inventory Fetch",
                "status_code": response.status_code,
                "response_content_type": response.content_type,
                "response_text_snippet": response.text,
                "parse_result_summary": {"records_parsed": 0, "parse_errors": []},
            }
        )
        return response

    def annotate_last_evidence(self, records_parsed=0, parse_error=None):
        summary = self.evidence_log[-1]["parse_result_summary"]
        summary["records_parsed"] = records_parsed
        if parse_error:
            summary["parse_errors"].append(parse_error)


class InventoryEngineAndCopilotTests(unittest.TestCase):
    def test_wrapper_message_records_parse(self):
        client = FakeClient(
            [
                FakeResponse(json_data={"status": "success", "message": [{"id": 1, "name": "A"}]}),
                FakeResponse(json_data={"status": "success", "message": []}),
            ]
        )
        bundle = fetch_inventory_index(client, max_pages=2)
        self.assertEqual(len(bundle.records), 1)
        self.assertEqual(client.evidence_log[0]["parse_result_summary"]["records_parsed"], 1)
        self.assertEqual(client.evidence_log[0].get("bd_status"), "success")

    def test_cloudflare_classification(self):
        client = FakeClient([FakeResponse(status_code=403, text="Attention Required! | Cloudflare", content_type="text/html")])
        with self.assertRaises(Exception):
            fetch_inventory_index(client, max_pages=1)
        self.assertEqual(client.evidence_log[0].get("error_type"), "CLOUDFLARE_BLOCK")

    def test_cache_roundtrip_and_csv_count(self):
        bundle = InventoryBundle(records=[{"user_id": 1, "name": "A"}, {"user_id": 2, "name": "B"}], inventory_index={}, summary={"total_members": 2})
        cache_inventory_to_disk(bundle, "cache/test_inventory.json")
        loaded = load_inventory_from_cache("cache/test_inventory.json")
        self.assertEqual(len(loaded.records), 2)
        csv_data = inventory_to_csv(loaded.records)
        self.assertEqual(len([line for line in csv_data.splitlines() if line.strip()]), 3)

    def test_rate_limit_returns_partial_and_progress(self):
        cache_path = "cache/test_inventory_partial.json"
        progress_path = "cache/test_inventory_progress.json"
        client = FakeClient(
            [
                FakeResponse(json_data={"status": "success", "message": [{"id": 1, "name": "A"}]}),
                FakeResponse(status_code=429, json_data={"status": "error", "message": "Too many API requests per minute"}, text="Too many API requests per minute", headers={"Retry-After": "0"}),
                FakeResponse(status_code=429, json_data={"status": "error", "message": "Too many API requests per minute"}, text="Too many API requests per minute"),
                FakeResponse(status_code=429, json_data={"status": "error", "message": "Too many API requests per minute"}, text="Too many API requests per minute"),
                FakeResponse(status_code=429, json_data={"status": "error", "message": "Too many API requests per minute"}, text="Too many API requests per minute"),
                FakeResponse(status_code=429, json_data={"status": "error", "message": "Too many API requests per minute"}, text="Too many API requests per minute"),
                FakeResponse(status_code=429, json_data={"status": "error", "message": "Too many API requests per minute"}, text="Too many API requests per minute"),
            ]
        )
        bundle = fetch_inventory_index(client, max_pages=2, requests_per_minute=120, cache_path=cache_path, progress_path=progress_path)
        self.assertEqual(bundle.status, "partial")
        self.assertEqual((bundle.meta or {}).get("error_type"), "RATE_LIMIT")
        progress = load_inventory_progress(progress_path)
        self.assertGreaterEqual(int(progress.get("last_page", 0)), 1)
        self.assertTrue(Path(cache_path).exists())

    def test_copilot_rules_detect_cloudflare(self):
        rules_doc = {
            "rules": [
                {
                    "id": "CLOUDFLARE_BLOCK_DETECT",
                    "title": "t",
                    "severity": "blocker",
                    "classification": {},
                    "recommended_actions": [],
                    "playbook_ref": "p",
                }
            ]
        }
        playbooks_doc = {"playbooks": [{"id": "p", "title": "P", "steps": ["a"]}]}
        evidence = {
            "status_code": 403,
            "response_text_snippet": "Attention Required! | Cloudflare",
            "response_content_type": "text/html",
            "parse_result_summary": {"records_parsed": 0, "parse_errors": []},
        }
        findings = evaluate_rules(evidence, rules_doc, playbooks_doc)
        self.assertEqual(findings[0]["id"], "CLOUDFLARE_BLOCK_DETECT")

    def test_copilot_rate_limit_rule(self):
        rules_doc = {
            "rules": [
                {
                    "id": "RATE_LIMIT_DETECT",
                    "title": "Rate limit",
                    "severity": "warn",
                    "classification": {"error_type": "RATE_LIMIT"},
                    "recommended_actions": ["Lower rpm"],
                    "playbook_ref": "PLAYBOOK_RATE_LIMIT",
                }
            ]
        }
        playbooks_doc = {"playbooks": [{"id": "PLAYBOOK_RATE_LIMIT", "title": "Rate", "steps": ["a"]}]}
        evidence = {
            "status_code": 429,
            "attempt_no": 3,
            "response_text_snippet": "Too many API requests per minute",
            "response_content_type": "application/json",
            "parse_result_summary": {"records_parsed": 0, "parse_errors": []},
        }
        findings = evaluate_rules(evidence, rules_doc, playbooks_doc)
        self.assertEqual(findings[0]["id"], "RATE_LIMIT_DETECT")
        self.assertEqual(findings[0]["severity"], "blocker")


if __name__ == "__main__":
    unittest.main()
