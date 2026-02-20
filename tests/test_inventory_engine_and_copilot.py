import unittest

from bd_copilot import evaluate_rules
from inventory_engine import InventoryBundle, cache_inventory_to_disk, fetch_inventory_index, inventory_to_csv, load_inventory_from_cache


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", content_type="application/json"):
        self.ok = status_code < 400
        self.status_code = status_code
        self.json_data = [] if json_data is None else json_data
        self.text = text
        self.content_type = content_type


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
        bundle = fetch_inventory_index(client, max_pages=2, delay_seconds=0)
        self.assertEqual(len(bundle.records), 1)
        self.assertEqual(client.evidence_log[0]["parse_result_summary"]["records_parsed"], 1)
        self.assertEqual(client.evidence_log[0].get("bd_status"), "success")

    def test_cloudflare_classification(self):
        client = FakeClient([FakeResponse(status_code=403, text="Attention Required! | Cloudflare", content_type="text/html")])
        with self.assertRaises(Exception):
            fetch_inventory_index(client, max_pages=1, delay_seconds=0)
        self.assertEqual(client.evidence_log[0].get("error_type"), "CLOUDFLARE_BLOCK")

    def test_cache_roundtrip_and_csv_count(self):
        bundle = InventoryBundle(records=[{"user_id": 1, "name": "A"}, {"user_id": 2, "name": "B"}], inventory_index={}, summary={"total_members": 2})
        cache_inventory_to_disk(bundle, "cache/test_inventory.json")
        loaded = load_inventory_from_cache("cache/test_inventory.json")
        self.assertEqual(len(loaded.records), 2)
        csv_data = inventory_to_csv(loaded.records)
        self.assertEqual(len([line for line in csv_data.splitlines() if line.strip()]), 3)

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


if __name__ == "__main__":
    unittest.main()
