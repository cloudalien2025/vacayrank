from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from milestone3.bd_client import BDClient, BDResponse
from milestone3.patch_engine import compute_patch


@dataclass
class ListingCard:
    slug: str
    title: str
    user_id: Optional[int]
    data_id: Optional[int]
    data_type: Optional[int]
    dom_post_id: Optional[int]


class _ListingCardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: List[ListingCard] = []
        self._current_attrs: Optional[Dict[str, str]] = None
        self._in_anchor = False
        self._current_href: Optional[str] = None
        self._title_chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        if tag == "div" and "data-postid" in attr_map:
            self._current_attrs = attr_map
            self._title_chunks = []
            self._current_href = None
            self._in_anchor = False
        elif tag == "a" and self._current_attrs is not None:
            href = attr_map.get("href", "")
            if "/listings/" in href and not self._current_href:
                self._current_href = href
            self._in_anchor = True

    def handle_data(self, data: str) -> None:
        if self._current_attrs is not None and self._in_anchor:
            chunk = (data or "").strip()
            if chunk:
                self._title_chunks.append(chunk)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_anchor = False
        elif tag == "div" and self._current_attrs is not None:
            slug = _normalize_slug(self._current_href or "")
            title = " ".join(self._title_chunks).strip()
            self.cards.append(
                ListingCard(
                    slug=slug,
                    title=title,
                    user_id=_to_int(self._current_attrs.get("data-userid")),
                    data_id=_to_int(self._current_attrs.get("data-dataid")),
                    data_type=_to_int(self._current_attrs.get("data-datatype")),
                    dom_post_id=_to_int(self._current_attrs.get("data-postid")),
                )
            )
            self._current_attrs = None
            self._title_chunks = []
            self._current_href = None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_slug(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(value)
    path = parsed.path or value
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/")


def parse_listing_cards(search_html: str) -> List[ListingCard]:
    parser = _ListingCardParser()
    parser.feed(search_html or "")
    return [card for card in parser.cards if card.slug]


def _extract_records(resp: BDResponse) -> List[Dict[str, Any]]:
    payload = resp.json_data if isinstance(resp.json_data, dict) else {}
    for key in ("data", "results", "posts", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _record_slug(record: Dict[str, Any]) -> str:
    for key in ("slug", "listing_slug", "url", "permalink", "link", "path"):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            return _normalize_slug(raw)
    return ""


def _record_title(record: Dict[str, Any]) -> str:
    for key in ("title", "name", "post_title", "listing_title", "company"):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _record_post_id(record: Dict[str, Any]) -> Optional[int]:
    for key in ("post_id", "id", "data_post_id"):
        parsed = _to_int(record.get(key))
        if parsed:
            return parsed
    return None


def discover_data_posts_endpoint(client: BDClient, data_id: int, page: int = 1, limit: int = 25) -> Tuple[str, List[Dict[str, Any]]]:
    candidates = [
        "/api/v2/data_posts/search",
        "/api/v2/data_post/search",
        "/api/v2/posts/search",
        "/api/v2/data_posts/list",
    ]
    payload = {"data_id": int(data_id), "page": int(page), "limit": int(limit), "output_type": "array"}
    for path in candidates:
        resp = client.request_post_form(path, payload, label=f"Discover listing endpoint: {path}")
        records = _extract_records(resp)
        if resp.ok and records:
            return path, records
    raise RuntimeError("Unable to discover a working data_posts search endpoint for the provided data_id")


def fetch_listing_cards_html(client: BDClient, data_id: int, page: int = 1, limit: int = 25) -> str:
    payload = {"data_id": int(data_id), "page": int(page), "limit": int(limit)}
    resp = client.request_post_form("/api/v2/users_portfolio_groups/search", payload, label="Listing cards HTML search")
    if isinstance(resp.json_data, dict):
        for key in ("html", "data", "results"):
            value = resp.json_data.get(key)
            if isinstance(value, str) and "data-postid" in value:
                return value
    return resp.text


def map_listing_cards_to_true_post_ids(
    client: BDClient,
    data_id: int,
    page: int = 1,
    limit: int = 25,
    listing_cards_html: Optional[str] = None,
) -> Dict[str, Any]:
    html = listing_cards_html or fetch_listing_cards_html(client, data_id, page=page, limit=limit)
    cards = parse_listing_cards(html)
    endpoint, records = discover_data_posts_endpoint(client, data_id=data_id, page=page, limit=limit)

    by_slug = {_record_slug(record): record for record in records if _record_slug(record)}
    by_title = {_record_title(record).lower(): record for record in records if _record_title(record)}

    mapped: List[Dict[str, Any]] = []
    for card in cards:
        match_key = "slug"
        record = by_slug.get(card.slug)
        if not record and card.title:
            record = by_title.get(card.title.lower())
            match_key = "title"

        matched_post_id = _record_post_id(record or {})
        mapped.append(
            {
                "slug": card.slug,
                "title": card.title,
                "true_post_id": matched_post_id,
                "dom_post_id": card.dom_post_id,
                "user_id": card.user_id,
                "data_id": card.data_id,
                "data_type": card.data_type,
                "mapping_key": match_key if matched_post_id else "unresolved",
            }
        )

    return {
        "data_id": data_id,
        "endpoint_used": endpoint,
        "mapped_count": sum(1 for row in mapped if row["true_post_id"]),
        "total_cards": len(mapped),
        "listings": mapped,
    }


def read_data_post(client: BDClient, true_post_id: int) -> BDResponse:
    return client.request_get(f"/api/v2/data_posts/get/{int(true_post_id)}", label="Get data_post")


def update_data_post_fields(client: BDClient, true_post_id: int, fields: Dict[str, Any]) -> Dict[str, Any]:
    current_resp = read_data_post(client, true_post_id)
    current = current_resp.json_data.get("data", current_resp.json_data) if isinstance(current_resp.json_data, dict) else {}
    patch = compute_patch(current if isinstance(current, dict) else {}, fields, allow_clearing=False)
    if not patch:
        return {"ok": True, "message": "No changes detected", "true_post_id": true_post_id, "payload": {"post_id": true_post_id}}

    payload = {"post_id": int(true_post_id), **patch}
    resp = client.request_put_form("/api/v2/data_posts/update", payload, label="Update data_post")
    message = ""
    if isinstance(resp.json_data, dict):
        message = str(resp.json_data.get("message") or "")
    diagnostics = {
        "ok": resp.ok and "record cannot be updated" not in message.lower(),
        "endpoint_used": "/api/v2/data_posts/update",
        "true_post_id": true_post_id,
        "payload": payload,
        "http_status": resp.status_code,
        "response": resp.json_data,
    }
    if "record cannot be updated" in message.lower():
        diagnostics["error"] = "record cannot be updated"
    return diagnostics
