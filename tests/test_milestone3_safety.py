import unittest
from unittest.mock import patch

from milestone3.audit_log import AuditLog
from milestone3.bd_client import BDClient
from milestone3.patch_engine import compute_patch
from milestone3.write_queue import build_write_queue_from_m2, enforce_max_writes_per_session


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.ok = status < 400
        self.status_code = status
        self._payload = payload or {}
        self.text = "{}"

    def json(self):
        return self._payload


class Milestone3SafetyTests(unittest.TestCase):
    @patch("requests.post")
    def test_form_encoding_enforced_on_post(self, mock_post):
        mock_post.return_value = FakeResponse()
        client = BDClient("https://example.com", "k")
        client.request_post_form("/api/v2/user/search", {"search_keyword": "x"})
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/x-www-form-urlencoded")
        self.assertIn("data", kwargs)
        self.assertNotIn("json", kwargs)

    @patch("requests.put")
    def test_form_encoding_enforced_on_put(self, mock_put):
        mock_put.return_value = FakeResponse()
        client = BDClient("https://example.com", "k")
        client.update_user({"user_id": 1, "quote": "hello"})
        kwargs = mock_put.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/x-www-form-urlencoded")
        self.assertIn("data", kwargs)
        self.assertNotIn("json", kwargs)

    def test_patch_only_updates(self):
        current = {"company": "Old", "city": "A", "website": "w"}
        proposed = {"company": "New", "city": "A", "website": ""}
        patch = compute_patch(current, proposed, allow_clearing=False)
        self.assertEqual(patch, {"company": "New"})

    def test_audit_logs_preview_and_commit(self):
        audit = AuditLog(path="logs/test_audit.jsonl")
        audit.log_event({"action_type": "preview_update"})
        audit.log_event({"action_type": "update"})
        self.assertEqual(len(audit.entries), 2)

    def test_dry_run_style_guard_max_writes(self):
        self.assertTrue(enforce_max_writes_per_session(0, 1))
        self.assertFalse(enforce_max_writes_per_session(1, 1))

    def test_build_write_queue_from_m2_schema(self):
        import pandas as pd

        missing = pd.DataFrame([{"category": "Hotels", "city": "Vail", "candidate_name": "A", "website": "https://a.com"}])
        possible = pd.DataFrame([{"category": "Hotels", "city": "Vail", "candidate_name": "B", "website": "https://b.com", "best_match_user_id": 11}])
        duplicates = [{"candidate_name": "C", "bd_user_ids": [1, 2, 3]}]
        queue = build_write_queue_from_m2(pd.DataFrame([]), missing, possible, duplicates, {})
        self.assertIn("missing", queue)
        self.assertIn("possible_matches", queue)
        self.assertIn("duplicates", queue)
        self.assertEqual(queue["missing"][0]["status"], "pending")
        self.assertEqual(queue["possible_matches"][0]["best_match_user_id"], 11)


if __name__ == "__main__":
    unittest.main()
