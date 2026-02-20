import unittest
from unittest.mock import patch

from milestone3.bd_client import BDClient
from milestone3.listing_pipeline import (
    discover_data_posts_endpoint,
    map_listing_cards_to_true_post_ids,
    parse_listing_cards,
    update_data_post_fields,
)


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.ok = status < 400
        self.status_code = status
        self._payload = payload or {}
        self.text = "{}"
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._payload


class ListingPipelineTests(unittest.TestCase):
    def test_parse_listing_cards_reads_dom_values(self):
        html = """
        <div data-userid='1' data-dataid='75' data-datatype='4' data-postid='764'>
          <a href='/listings/yama-sushi'>Yama Sushi</a>
        </div>
        """
        cards = parse_listing_cards(html)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].slug, "/listings/yama-sushi")
        self.assertEqual(cards[0].dom_post_id, 764)
        self.assertEqual(cards[0].data_id, 75)

    @patch("requests.post")
    def test_discover_endpoint_skips_empty_results(self, mock_post):
        mock_post.side_effect = [
            FakeResponse(payload={"status": "ok", "data": []}),
            FakeResponse(payload={"status": "ok", "data": [{"post_id": 991, "slug": "/listings/yama-sushi"}]}),
        ]
        client = BDClient("https://example.com", "k")
        endpoint, rows = discover_data_posts_endpoint(client, data_id=75)
        self.assertEqual(endpoint, "/api/v2/data_post/search")
        self.assertEqual(rows[0]["post_id"], 991)

    def test_map_listing_cards_uses_slug_mapping(self):
        client = BDClient("https://example.com", "k")

        with patch("milestone3.listing_pipeline.fetch_listing_cards_html", return_value="""
            <div data-userid='1' data-dataid='75' data-datatype='4' data-postid='764'>
              <a href='/listings/yama-sushi'>Yama Sushi</a>
            </div>
        """), patch(
            "milestone3.listing_pipeline.discover_data_posts_endpoint",
            return_value=(
                "/api/v2/data_posts/search",
                [{"post_id": 991, "slug": "/listings/yama-sushi", "title": "Yama Sushi", "user_id": 1}],
            ),
        ):
            result = map_listing_cards_to_true_post_ids(client, data_id=75)

        self.assertEqual(result["endpoint_used"], "/api/v2/data_posts/search")
        self.assertEqual(result["mapped_count"], 1)
        self.assertEqual(result["listings"][0]["true_post_id"], 991)
        self.assertEqual(result["listings"][0]["mapping_key"], "slug")

    def test_update_data_post_fields_handles_record_cannot_update(self):
        client = BDClient("https://example.com", "k")
        with patch("milestone3.listing_pipeline.read_data_post", return_value=type("R", (), {"json_data": {"data": {"title": "Old"}}})()), patch(
            "milestone3.bd_client.BDClient.request_put_form",
            return_value=type(
                "U", (), {"ok": True, "status_code": 200, "json_data": {"status": "error", "message": "Record cannot be updated"}}
            )(),
        ):
            result = update_data_post_fields(client, 991, {"title": "New"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "record cannot be updated")


if __name__ == "__main__":
    unittest.main()
