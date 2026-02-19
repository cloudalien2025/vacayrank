from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import streamlit as st
import xml.etree.ElementTree as ET


APP_TITLE = "VacayRank v1 — Inventory + SERP Gap Engine"
CACHE_DIR = "vacayrank_cache"
DEFAULT_TIMEOUT_SECS = 20
SERP_API_URL = "https://serpapi.com/search.json"

XML_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

CATEGORY_QUERY_TEMPLATES = {
    "Hotels": [
        "hotels in {destination} {region}",
        "best hotels in {destination} {region}",
        "luxury hotels in {destination} {region}",
        "ski resort hotels in {destination} {region}",
        "{destination} {region} lodging",
    ],
    "Restaurants": [
        "restaurants in {destination} {region}",
        "best restaurants in {destination} {region}",
        "fine dining in {destination} {region}",
        "breakfast in {destination} {region}",
        "{destination} {region} dinner",
    ],
    "Activities": [
        "things to do in {destination} {region}",
        "best activities in {destination} {region}",
        "tours in {destination} {region}",
        "{destination} {region} attractions",
        "outdoor activities in {destination} {region}",
    ],
    "Nightlife": [
        "nightlife in {destination} {region}",
        "bars in {destination} {region}",
        "live music in {destination} {region}",
        "cocktail bars in {destination} {region}",
        "clubs in {destination} {region}",
    ],
    "Shopping": [
        "shopping in {destination} {region}",
        "best shops in {destination} {region}",
        "boutiques in {destination} {region}",
        "souvenir stores in {destination} {region}",
        "outlet stores in {destination} {region}",
    ],
}


@dataclass(frozen=True)
class UrlRow:
    url: str
    url_type: str
    source_sitemap: str
    lastmod: Optional[str] = None


@dataclass(frozen=True)
class FetchStats:
    sitemap_index_url: str
    fetched_at_utc: str
    child_sitemaps_found: int
    child_sitemaps_fetched: int
    urls_found: int
    elapsed_seconds: float
    cache_used: bool


@dataclass(frozen=True)
class InventoryEntity:
    url: str
    domain: str
    derived_name: str
    normalized_name: str


class FetchError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _append_debug(debug_log: Optional[List[str]], message: str) -> None:
    if debug_log is not None:
        debug_log.append(message)


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _alternate_host_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    swapped_host = host[4:] if host.startswith("www.") else f"www.{host}"
    netloc = f"{swapped_host}:{parsed.port}" if parsed.port else swapped_host
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _validate_xml_response(url: str, status_code: int, response_text: str) -> str:
    snippet = response_text[:200]
    if "<" not in snippet:
        raise FetchError(f"Invalid XML payload from {url} (HTTP {status_code}). First 200 chars: {snippet}")
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError as e:
        raise FetchError(f"Invalid XML returned by {url} (HTTP {status_code}): {e}. First 200 chars: {snippet}") from e
    root_tag = _strip_ns(root.tag).lower()
    if root_tag not in {"sitemapindex", "urlset"}:
        raise FetchError(f"Unexpected XML root '{root_tag}' from {url} (HTTP {status_code}). First 200 chars: {snippet}")
    return root_tag


def _fetch_once(session: requests.Session, url: str, timeout_secs: int, debug_log: Optional[List[str]]) -> str:
    retryable_statuses = {403, 429, 520, 521, 522, 523, 524}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/xml,text/xml;q=0.9,text/html;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = session.get(url, timeout=timeout_secs, headers=headers, allow_redirects=True)
    except requests.RequestException as e:
        _append_debug(debug_log, f"{url} -> request error: {e}")
        raise FetchError(f"Request failed for {url}: {e}", retryable=True) from e

    _append_debug(debug_log, f"{url} -> HTTP {resp.status_code}, bytes={len(resp.content or b'')}")
    if resp.status_code >= 500 or resp.status_code in retryable_statuses:
        raise FetchError(f"HTTP {resp.status_code} for {url}", retryable=True)
    if resp.status_code >= 400:
        raise FetchError(f"HTTP {resp.status_code} for {url}")

    resp.encoding = resp.encoding or "utf-8"
    text = resp.text
    _validate_xml_response(url, resp.status_code, text)
    return text


def _retry_request(session: requests.Session, url: str, timeout_secs: int, debug_log: Optional[List[str]]) -> str:
    backoffs = (0.5, 1.5, 3.5)
    last_err: Optional[FetchError] = None
    for attempt in range(1, 4):
        try:
            _append_debug(debug_log, f"Attempt {attempt}/3: {url}")
            return _fetch_once(session, url, timeout_secs, debug_log)
        except FetchError as err:
            last_err = err
            if not err.retryable or attempt == 3:
                break
            delay = backoffs[min(attempt - 1, len(backoffs) - 1)]
            _append_debug(debug_log, f"Retrying in {delay:.1f}s due to: {err}")
            time.sleep(delay)
    if last_err is None:
        raise FetchError(f"No request attempts were executed for {url}")
    raise last_err


def http_get_text(url: str, timeout_secs: int, debug_log: Optional[List[str]] = None) -> str:
    candidates = [url]
    fallback = _alternate_host_url(url)
    if fallback and fallback not in candidates:
        candidates.append(fallback)
    with requests.Session() as session:
        last_error: Optional[FetchError] = None
        for candidate in candidates:
            try:
                return _retry_request(session, candidate, timeout_secs, debug_log)
            except FetchError as e:
                last_error = e
                _append_debug(debug_log, f"Candidate failed: {candidate} -> {e}")
        if last_error:
            raise last_error
    raise FetchError(f"Unable to fetch sitemap from {url}")


def parse_sitemap_index(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError as e:
        raise FetchError(f"Invalid XML: {e}") from e

    tag = _strip_ns(root.tag).lower()
    if tag.endswith("sitemapindex"):
        sitemaps = []
        for sm_el in root.findall("sm:sitemap", XML_NS):
            loc_el = sm_el.find("sm:loc", XML_NS)
            if loc_el is not None and loc_el.text:
                sitemaps.append(loc_el.text.strip())
        if not sitemaps:
            for sm_el in root.iter():
                if _strip_ns(sm_el.tag).lower() == "loc" and sm_el.text:
                    sitemaps.append(sm_el.text.strip())
        if not sitemaps:
            raise FetchError("Sitemap index parsed but contained 0 child sitemaps.")
        return dedupe_preserve_order(sitemaps)
    if tag.endswith("urlset"):
        return []
    return []


def parse_urlset(xml_text: str, source_sitemap_url: str) -> List[UrlRow]:
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError as e:
        raise FetchError(f"Invalid XML in child sitemap: {e}") from e
    if _strip_ns(root.tag).lower() != "urlset":
        return []

    rows: List[UrlRow] = []
    for el in root.iter():
        if _strip_ns(el.tag).lower() != "url":
            continue
        loc, lastmod = None, None
        for child in el:
            cname = _strip_ns(child.tag).lower()
            if cname == "loc" and child.text and not loc:
                loc = child.text.strip()
            elif cname == "lastmod" and child.text and not lastmod:
                lastmod = child.text.strip()
        if loc:
            rows.append(UrlRow(url=loc, url_type="unclassified", source_sitemap=source_sitemap_url, lastmod=lastmod))
    return rows


@dataclass(frozen=True)
class ClassifierConfig:
    profile_patterns: Tuple[str, ...]
    blog_patterns: Tuple[str, ...]
    category_patterns: Tuple[str, ...]
    search_patterns: Tuple[str, ...]
    static_patterns: Tuple[str, ...]


DEFAULT_CLASSIFIER = ClassifierConfig(
    profile_patterns=(r"^/hotels?(/|$)", r"^/lodging(/|$)", r"^/stay(/|$)", r"^/restaurant(s)?(/|$)", r"^/dining(/|$)", r"^/activities?(/|$)", r"^/things-to-do(/|$)", r"^/shops?(/|$)", r"^/vacation-rentals?(/|$)"),
    blog_patterns=(r"^/blog(/|$)", r"^/posts?(/|$)", r"^/news(/|$)", r"^/\d{4}/\d{2}/\d{2}/"),
    category_patterns=(r"^/category(/|$)", r"^/tag(/|$)", r"^/topics?(/|$)", r"^/categories(/|$)"),
    search_patterns=(r"^/search(/|$)", r"^/\?s=", r"^/(\w+/)?\?q="),
    static_patterns=(r"^/$", r"^/about(/|$)", r"^/contact(/|$)", r"^/privacy(-policy)?(/|$)", r"^/terms(-of-service)?(/|$)"),
)


def classify_url(url: str, cfg: ClassifierConfig) -> str:
    try:
        parsed = urlparse(url)
        path = parsed.path or "/"
        combined = path + ("?" + parsed.query if parsed.query else "")
    except Exception:
        path, combined = url, url

    def _match(patterns: Tuple[str, ...], target: str) -> bool:
        return any(re.search(p, target, flags=re.IGNORECASE) for p in patterns)

    if _match(cfg.search_patterns, combined):
        return "search"
    if _match(cfg.static_patterns, path):
        return "static"
    if _match(cfg.category_patterns, path):
        return "categories"
    if _match(cfg.blog_patterns, path):
        return "blog_posts"
    if _match(cfg.profile_patterns, path):
        return "profiles"
    return "other"


def apply_classification(rows: List[UrlRow], cfg: ClassifierConfig) -> List[UrlRow]:
    return [UrlRow(url=r.url, url_type=classify_url(r.url, cfg), source_sitemap=r.source_sitemap, lastmod=r.lastmod) for r in rows]


def _disk_cache_paths_inventory(sitemap_index_url: str) -> Tuple[str, str]:
    _ensure_cache_dir()
    key = _stable_key(sitemap_index_url.strip())
    return os.path.join(CACHE_DIR, f"sitemap_inventory_{key}.json"), os.path.join(CACHE_DIR, f"sitemap_inventory_{key}.csv")


def load_inventory_disk_cache(sitemap_index_url: str, max_age_hours: float) -> Optional[Tuple[FetchStats, List[UrlRow]]]:
    json_path, _ = _disk_cache_paths_inventory(sitemap_index_url)
    if not os.path.exists(json_path):
        return None
    try:
        payload = json.loads(open(json_path, "r", encoding="utf-8").read())
        fetched_at = payload.get("stats", {}).get("fetched_at_utc")
        if not fetched_at:
            return None
        fetched_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 3600.0
        if age > max_age_hours:
            return None
        return FetchStats(**payload["stats"]), [UrlRow(**r) for r in payload["rows"]]
    except Exception:
        return None


def save_inventory_disk_cache(sitemap_index_url: str, stats: FetchStats, rows: List[UrlRow]) -> None:
    json_path, csv_path = _disk_cache_paths_inventory(sitemap_index_url)
    payload = {"stats": asdict(stats), "rows": [asdict(r) for r in rows]}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "url_type", "source_sitemap", "lastmod"])
        for r in rows:
            w.writerow([r.url, r.url_type, r.source_sitemap, r.lastmod or ""])


@st.cache_data(show_spinner=False)
def fetch_inventory_live(
    sitemap_index_url: str,
    timeout_secs: int,
    max_child_sitemaps: int,
    max_urls_per_child: int,
    debug_enabled: bool,
) -> Tuple[List[str], List[UrlRow], List[str]]:
    debug_log: List[str] = []
    dbg = debug_log if debug_enabled else None

    xml = http_get_text(sitemap_index_url, timeout_secs, dbg)
    child_sitemaps = parse_sitemap_index(xml)
    if not child_sitemaps:
        child_sitemaps = [sitemap_index_url]
    child_sitemaps = child_sitemaps[:max_child_sitemaps]

    rows: List[UrlRow] = []
    for sm_url in child_sitemaps:
        sm_xml = http_get_text(sm_url, timeout_secs, dbg)
        nested = parse_sitemap_index(sm_xml)
        if nested:
            for nested_url in nested[:max_child_sitemaps]:
                nested_xml = http_get_text(nested_url, timeout_secs, dbg)
                nested_rows = parse_urlset(nested_xml, nested_url)
                rows.extend(nested_rows[:max_urls_per_child] if max_urls_per_child > 0 else nested_rows)
        else:
            sm_rows = parse_urlset(sm_xml, sm_url)
            rows.extend(sm_rows[:max_urls_per_child] if max_urls_per_child > 0 else sm_rows)

    return child_sitemaps, rows, debug_log


def build_inventory(
    sitemap_index_url: str,
    timeout_secs: int,
    max_child_sitemaps: int,
    max_urls_per_child: int,
    disk_cache_max_age_hours: float,
    force_refresh: bool,
    debug_enabled: bool,
) -> Tuple[FetchStats, List[UrlRow], List[str]]:
    start = time.time()
    if not force_refresh:
        disk = load_inventory_disk_cache(sitemap_index_url, disk_cache_max_age_hours)
        if disk:
            stats, rows = disk
            rows = apply_classification(rows, DEFAULT_CLASSIFIER)
            stats = FetchStats(
                sitemap_index_url=stats.sitemap_index_url,
                fetched_at_utc=stats.fetched_at_utc,
                child_sitemaps_found=stats.child_sitemaps_found,
                child_sitemaps_fetched=stats.child_sitemaps_fetched,
                urls_found=stats.urls_found,
                elapsed_seconds=round(time.time() - start, 3),
                cache_used=True,
            )
            return stats, rows, ["Loaded inventory from disk cache."]

    child_sitemaps, unclassified, debug_log = fetch_inventory_live(
        sitemap_index_url, timeout_secs, max_child_sitemaps, max_urls_per_child, debug_enabled
    )
    rows = apply_classification(unclassified, DEFAULT_CLASSIFIER)
    stats = FetchStats(
        sitemap_index_url=sitemap_index_url,
        fetched_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        child_sitemaps_found=len(child_sitemaps),
        child_sitemaps_fetched=len(child_sitemaps),
        urls_found=len(rows),
        elapsed_seconds=round(time.time() - start, 3),
        cache_used=False,
    )
    if rows:
        save_inventory_disk_cache(sitemap_index_url, stats, rows)
    return stats, rows, debug_log


def _normalize_domain(url: str) -> str:
    if not url:
        return ""
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _slug_to_name(url: str) -> str:
    path = (urlparse(url).path or "").strip("/")
    if not path:
        return _normalize_domain(url)
    slug = path.split("/")[-1]
    slug = slug.replace("_", "-")
    slug = re.sub(r"[-]+", " ", slug)
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug or _normalize_domain(url)


def normalize_name(text: str, destination_word: str = "") -> str:
    stopwords = {"the", "a", "an", "hotel", "resort", "inn", "lodge", "suites"}
    if destination_word:
        stopwords.add(destination_word.lower())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    tokens = [t for t in cleaned.split() if t and t not in stopwords]
    return " ".join(tokens)


def build_inventory_entities(rows: List[UrlRow], destination: str) -> List[InventoryEntity]:
    entities = []
    for row in rows:
        name = _slug_to_name(row.url)
        entities.append(
            InventoryEntity(
                url=row.url,
                domain=_normalize_domain(row.url),
                derived_name=name,
                normalized_name=normalize_name(name, destination),
            )
        )
    return entities


def token_set_similarity(a: str, b: str) -> float:
    a_tokens, b_tokens = set(a.split()), set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    score = (2 * inter) / (len(a_tokens) + len(b_tokens))
    return round(score * 100, 2)


def fuzzy_score(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz  # type: ignore

        return float(fuzz.token_set_ratio(a, b))
    except Exception:
        return token_set_similarity(a, b)


def _serp_cache_paths(cache_key: str) -> Tuple[str, str]:
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"serp_{cache_key}.json"), os.path.join(CACHE_DIR, f"serp_{cache_key}_meta.json")


def _serp_cache_key(parts: Dict[str, str]) -> str:
    stable = json.dumps(parts, sort_keys=True)
    return _stable_key(stable)


def load_serp_disk_cache(parts: Dict[str, str], max_age_hours: float) -> Optional[dict]:
    key = _serp_cache_key(parts)
    json_path, meta_path = _serp_cache_paths(key)
    if not os.path.exists(json_path) or not os.path.exists(meta_path):
        return None
    try:
        payload = json.loads(open(json_path, "r", encoding="utf-8").read())
        meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
        fetched_at = datetime.fromisoformat(meta["fetched_at_utc"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0
        if age > max_age_hours:
            return None
        return payload
    except Exception:
        return None


def save_serp_disk_cache(parts: Dict[str, str], payload: dict) -> None:
    if not payload:
        return
    key = _serp_cache_key(parts)
    json_path, meta_path = _serp_cache_paths(key)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}, f, indent=2)


@st.cache_data(show_spinner=False)
def serpapi_search_cached(params: Dict[str, str], timeout_secs: int) -> dict:
    with requests.Session() as session:
        backoffs = (0.8, 1.6, 3.2)
        last_error = None
        for attempt in range(1, 4):
            try:
                resp = session.get(SERP_API_URL, params=params, timeout=timeout_secs)
                if resp.status_code >= 500 or resp.status_code in {429}:
                    raise FetchError(f"SerpAPI HTTP {resp.status_code}", retryable=True)
                if resp.status_code >= 400:
                    raise FetchError(f"SerpAPI HTTP {resp.status_code}: {resp.text[:250]}")
                data = resp.json()
                if data.get("error"):
                    raise FetchError(f"SerpAPI error: {data['error']}")
                return data
            except (requests.RequestException, ValueError, FetchError) as e:
                last_error = e
                retryable = isinstance(e, requests.RequestException) or (isinstance(e, FetchError) and e.retryable)
                if not retryable or attempt == 3:
                    break
                time.sleep(backoffs[min(attempt - 1, len(backoffs) - 1)])
        raise FetchError(f"SerpAPI request failed: {last_error}")


def run_serp_request(
    params: Dict[str, str],
    timeout_secs: int,
    cache_max_age_hours: float,
    force_refresh: bool,
    debug_log: List[str],
) -> dict:
    parts = {k: str(v) for k, v in params.items() if k != "api_key"}
    if not force_refresh:
        cached = load_serp_disk_cache(parts, cache_max_age_hours)
        if cached is not None:
            _append_debug(debug_log, f"SERP cache hit: engine={params.get('engine')} q={params.get('q')} start={params.get('start', 0)}")
            return cached

    _append_debug(debug_log, f"SERP query: engine={params.get('engine')} q={params.get('q')} start={params.get('start', 0)}")
    data = serpapi_search_cached(params, timeout_secs)
    if data:
        save_serp_disk_cache(parts, data)
    return data


def clean_query(template: str, destination: str, region: str, country: str) -> str:
    query = template.format(destination=destination.strip(), region=region.strip(), country=country.strip())
    return re.sub(r"\s+", " ", query).strip()


def build_query_seeds(destination: str, region: str, country: str, category: str, custom_text: str) -> List[str]:
    if custom_text.strip():
        return dedupe_preserve_order([line.strip() for line in custom_text.splitlines() if line.strip()])
    templates = CATEGORY_QUERY_TEMPLATES.get(category, [])
    return [clean_query(t, destination, region, country) for t in templates]


def extract_organic_candidates(payload: dict) -> List[dict]:
    out = []
    for item in payload.get("organic_results", []) or []:
        out.append(
            {
                "name": item.get("title", ""),
                "address": "",
                "phone": "",
                "rating": None,
                "reviews": None,
                "website": item.get("link", ""),
                "place_id": item.get("place_id") or item.get("result_id") or "",
                "source_engine": "google",
            }
        )
    return out


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    txt = re.sub(r"[^0-9]", "", str(value))
    return int(txt) if txt else None


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def extract_maps_candidates(payload: dict) -> List[dict]:
    results = payload.get("local_results") or payload.get("places_results") or []
    out = []
    for item in results:
        out.append(
            {
                "name": item.get("title", ""),
                "address": item.get("address", ""),
                "phone": item.get("phone", ""),
                "rating": _to_float(item.get("rating")),
                "reviews": _to_int(item.get("reviews") or item.get("reviews_original")),
                "website": item.get("website", ""),
                "place_id": item.get("place_id") or item.get("data_id") or item.get("data_cid") or "",
                "source_engine": "google_maps",
            }
        )
    return out


def aggregate_candidates(raw_candidates: List[dict], destination: str) -> List[dict]:
    by_key: Dict[str, dict] = {}
    for c in raw_candidates:
        domain = _normalize_domain(c.get("website", ""))
        norm_name = normalize_name(c.get("name", ""), destination)
        key = domain or f"{norm_name}|{(c.get('address') or '').lower()}"
        if key not in by_key:
            by_key[key] = {
                "candidate_name": c.get("name", ""),
                "normalized_name": norm_name,
                "address": c.get("address", ""),
                "phone": c.get("phone", ""),
                "rating": c.get("rating"),
                "reviews": c.get("reviews") or 0,
                "website": c.get("website", ""),
                "normalized_domain": domain,
                "place_id": c.get("place_id", ""),
                "engines": {c.get("source_engine", "")},
                "queries": set(),
            }
        else:
            item = by_key[key]
            item["engines"].add(c.get("source_engine", ""))
            item["reviews"] = max(item.get("reviews") or 0, c.get("reviews") or 0)
            if not item.get("website") and c.get("website"):
                item["website"] = c.get("website")
                item["normalized_domain"] = domain
            if (item.get("rating") or 0) < (c.get("rating") or 0):
                item["rating"] = c.get("rating")
    return list(by_key.values())


def apply_matching(candidates: List[dict], inventory_entities: List[InventoryEntity], threshold: int) -> Tuple[List[dict], List[dict], List[dict]]:
    missing, possible, listed = [], [], []

    by_domain: Dict[str, List[InventoryEntity]] = {}
    for inv in inventory_entities:
        if inv.domain:
            by_domain.setdefault(inv.domain, []).append(inv)

    for c in candidates:
        best_url = ""
        best_score = 0.0
        match_type = "none"

        c_domain = c.get("normalized_domain", "")
        if c_domain and c_domain in by_domain:
            best_url = by_domain[c_domain][0].url
            best_score = 100.0
            match_type = "domain"
        else:
            cname = c.get("normalized_name", "")
            for inv in inventory_entities:
                score = fuzzy_score(cname, inv.normalized_name)
                if score > best_score:
                    best_score = score
                    best_url = inv.url
            if best_score > 0:
                match_type = "name_fuzzy"

        c["best_match_url"] = best_url
        c["best_match_score"] = round(best_score, 2)
        c["match_type"] = match_type

        candidate_score = 0
        if "google_maps" in c.get("engines", set()):
            candidate_score += 3
        if len(c.get("queries", set())) >= 2:
            candidate_score += 2
        if c.get("website"):
            candidate_score += 1
        if (c.get("rating") or 0) >= 4.5:
            candidate_score += 1
        if (c.get("reviews") or 0) >= 200:
            candidate_score += 1
        c["candidate_score"] = candidate_score

        if best_score >= threshold:
            listed.append(c)
        elif threshold - 10 <= best_score < threshold:
            possible.append(c)
        else:
            missing.append(c)

    missing.sort(key=lambda x: (x.get("candidate_score", 0), x.get("reviews", 0)), reverse=True)
    possible.sort(key=lambda x: x.get("best_match_score", 0), reverse=True)
    listed.sort(key=lambda x: x.get("best_match_score", 0), reverse=True)
    return missing, possible, listed


def rows_to_csv_bytes(rows: List[Dict[str, object]]) -> bytes:
    if not rows:
        return b""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def render_inventory_page() -> None:
    st.subheader("Milestone 1 — Inventory")
    with st.sidebar:
        st.header("Inventory Settings")
        sitemap_index_url = st.text_input("Sitemap index URL", value="https://vailvacay.com/sitemap_index.xml").strip()
        timeout_secs = st.number_input("HTTP timeout (seconds)", min_value=5, max_value=60, value=DEFAULT_TIMEOUT_SECS)
        max_child_sitemaps = st.number_input("Max child sitemaps", min_value=1, max_value=500, value=150)
        max_urls_per_child = st.number_input("Max URLs per child sitemap (0=no limit)", min_value=0, max_value=200000, value=0)
        disk_cache_max_age_hours = st.number_input("Use disk cache if newer than (hours)", min_value=0.0, max_value=168.0, value=12.0)
        force_refresh = st.checkbox("Force refresh (ignore disk cache)", value=False)
        show_debug_log = st.checkbox("Show debug log", value=False)
        run = st.button("Run inventory scan", type="primary")

    if not run:
        st.info("Configure settings in the sidebar, then click **Run inventory scan**.")
        return

    try:
        with st.spinner("Fetching and parsing sitemaps..."):
            stats, rows, debug_log = build_inventory(
                sitemap_index_url,
                int(timeout_secs),
                int(max_child_sitemaps),
                int(max_urls_per_child),
                float(disk_cache_max_age_hours),
                bool(force_refresh),
                bool(show_debug_log),
            )
    except Exception as e:
        st.error(f"Inventory scan failed: {e}")
        st.stop()

    st.metric("Inventory URLs", stats.urls_found)
    st.caption(f"Fetched at: {stats.fetched_at_utc} • Cache used: {'Yes' if stats.cache_used else 'No'} • Elapsed: {stats.elapsed_seconds}s")

    table = [{"url": r.url, "url_type": r.url_type, "source_sitemap": r.source_sitemap, "lastmod": r.lastmod or ""} for r in rows]
    st.dataframe(table, use_container_width=True, height=500)
    st.download_button("Download inventory CSV", data=rows_to_csv_bytes(table), file_name="vacayrank_inventory.csv", mime="text/csv")

    if show_debug_log and debug_log:
        st.subheader("Debug log")
        st.code("\n".join(debug_log), language="text")


def render_serp_gap_page() -> None:
    st.subheader("Milestone 2 — SERP Gap")
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()

    with st.sidebar:
        st.header("SERP Gap Settings")
        sitemap_index_url = st.text_input("Sitemap index URL", value="https://vailvacay.com/sitemap_index.xml").strip()
        destination = st.text_input("Destination", value="Vail").strip()
        region = st.text_input("Region/State", value="CO").strip()
        country = st.text_input("Country", value="US").strip()
        category = st.selectbox("Category", options=["Hotels", "Restaurants", "Activities", "Nightlife", "Shopping"])
        custom_seeds = st.text_area("Custom query seeds (one per line)", value="", height=120)
        use_location_targeting = st.checkbox("Use SerpAPI location targeting (advanced)", value=False, disabled=True)
        if use_location_targeting:
            st.caption("Location targeting is currently unavailable in this release.")
        else:
            st.caption("Localization uses query text (destination + region) with hl/gl settings.")
        organic_pages = st.number_input("Organic pages to fetch", min_value=1, max_value=10, value=2)
        maps_pages = st.number_input("Maps pages to fetch", min_value=1, max_value=10, value=2)
        threshold = st.slider("Fuzzy threshold", min_value=0, max_value=100, value=88)
        serp_cache_age = st.number_input("Use disk cache if newer than (hours)", min_value=0.0, max_value=168.0, value=12.0)
        force_refresh = st.checkbox("Force refresh (ignore disk cache)", value=False)
        show_debug_log = st.checkbox("Show debug log", value=False)
        run_disabled = not bool(api_key)
        run = st.button("Run SERP Gap Scan", type="primary", disabled=run_disabled)

    if not api_key:
        st.error("SERPAPI_API_KEY is missing. Set it in your environment/secrets to run Milestone 2.")

    if not run:
        st.info("Configure settings in the sidebar, then click **Run SERP Gap Scan**.")
        return

    debug_log: List[str] = []

    try:
        with st.spinner("Loading directory inventory..."):
            _, inventory_rows, _ = build_inventory(
                sitemap_index_url=sitemap_index_url,
                timeout_secs=DEFAULT_TIMEOUT_SECS,
                max_child_sitemaps=200,
                max_urls_per_child=0,
                disk_cache_max_age_hours=float(serp_cache_age),
                force_refresh=bool(force_refresh),
                debug_enabled=False,
            )
    except Exception as e:
        st.error(f"Failed to load inventory for matching: {e}")
        st.stop()

    queries = build_query_seeds(destination, region, country, category, custom_seeds)
    raw_candidates: List[dict] = []

    try:
        for q in queries:
            for page_idx in range(int(organic_pages)):
                params = {
                    "engine": "google",
                    "q": q,
                    "api_key": api_key,
                    "hl": "en",
                    "gl": (country or "us").lower(),
                    "num": "10",
                    "start": str(page_idx * 10),
                }
                payload = run_serp_request(params, DEFAULT_TIMEOUT_SECS, float(serp_cache_age), bool(force_refresh), debug_log)
                organic = extract_organic_candidates(payload)
                _append_debug(debug_log, f"Organic extracted: query='{q}' page={page_idx + 1} count={len(organic)}")
                for c in organic:
                    c["query"] = q
                raw_candidates.extend(organic)

            maps_query = f"{category.lower()} in {destination} {region}".strip()
            for page_idx in range(int(maps_pages)):
                params = {
                    "engine": "google_maps",
                    "q": maps_query,
                    "api_key": api_key,
                    "hl": "en",
                    "gl": (country or "us").lower(),
                    "start": str(page_idx * 20),
                }
                payload = run_serp_request(params, DEFAULT_TIMEOUT_SECS, float(serp_cache_age), bool(force_refresh), debug_log)
                maps = extract_maps_candidates(payload)
                _append_debug(debug_log, f"Maps extracted: query='{maps_query}' page={page_idx + 1} count={len(maps)}")
                for c in maps:
                    c["query"] = maps_query
                raw_candidates.extend(maps)
    except Exception as e:
        st.error(f"SERP scan failed: {e}")
        if show_debug_log and debug_log:
            st.code("\n".join(debug_log), language="text")
        st.stop()

    candidates = aggregate_candidates(raw_candidates, destination)
    for raw in raw_candidates:
        domain = _normalize_domain(raw.get("website", ""))
        norm = normalize_name(raw.get("name", ""), destination)
        key = domain or f"{norm}|{(raw.get('address') or '').lower()}"
        for c in candidates:
            ckey = c.get("normalized_domain") or f"{c.get('normalized_name')}|{(c.get('address') or '').lower()}"
            if ckey == key:
                c["queries"].add(raw.get("query", ""))
                break

    inventory_entities = build_inventory_entities(inventory_rows, destination)
    missing, possible, listed = apply_matching(candidates, inventory_entities, int(threshold))

    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Inventory count", len(inventory_rows))
    c2.metric("SERP candidates discovered", len(candidates))
    c3.metric("Missing", len(missing))
    c4.metric("Possible matches", len(possible))
    c5.metric("Already listed", len(listed))

    def _display_rows(items: List[dict]) -> List[Dict[str, object]]:
        display = []
        for x in items:
            display.append(
                {
                    "candidate_name": x.get("candidate_name", ""),
                    "address": x.get("address", ""),
                    "phone": x.get("phone", ""),
                    "rating": x.get("rating"),
                    "reviews": x.get("reviews"),
                    "website": x.get("website", ""),
                    "candidate_score": x.get("candidate_score", 0),
                    "best_match_url": x.get("best_match_url", ""),
                    "best_match_score": x.get("best_match_score", 0),
                    "match_type": x.get("match_type", "none"),
                    "queries_count": len(x.get("queries", set())),
                    "engines": ",".join(sorted([e for e in x.get("engines", set()) if e])),
                    "place_id": x.get("place_id", ""),
                }
            )
        return display

    tab1, tab2, tab3 = st.tabs(["Missing", "Possible Matches", "Already Listed"])
    with tab1:
        missing_rows = _display_rows(missing)
        st.dataframe(missing_rows, use_container_width=True, height=450)
        st.download_button("Download Missing CSV", data=rows_to_csv_bytes(missing_rows), file_name="vacayrank_missing.csv", mime="text/csv")
    with tab2:
        possible_rows = _display_rows(possible)
        st.dataframe(possible_rows, use_container_width=True, height=450)
        st.download_button("Download Possible Matches CSV", data=rows_to_csv_bytes(possible_rows), file_name="vacayrank_possible_matches.csv", mime="text/csv")
    with tab3:
        listed_rows = _display_rows(listed)
        st.dataframe(listed_rows, use_container_width=True, height=450)
        st.download_button("Download Already Listed CSV", data=rows_to_csv_bytes(listed_rows), file_name="vacayrank_already_listed.csv", mime="text/csv")

    if show_debug_log:
        st.subheader("Debug log")
        st.code("\n".join(debug_log) if debug_log else "No debug entries.", language="text")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("Milestone 1 provides sitemap inventory. Milestone 2 discovers SERP candidates and computes listing gaps.")

    with st.sidebar:
        page = st.radio("Navigation", options=["Milestone 1 — Inventory", "Milestone 2 — SERP Gap"])

    if page == "Milestone 1 — Inventory":
        render_inventory_page()
    else:
        render_serp_gap_page()


if __name__ == "__main__":
    _ensure_cache_dir()
    main()
