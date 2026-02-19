from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import streamlit as st

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    fuzz = None
    _HAS_RAPIDFUZZ = False

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
        "bars in {destination} {region}",
        "best bars in {destination} {region}",
        "nightlife in {destination} {region}",
        "live music in {destination} {region}",
        "cocktail bar in {destination} {region}",
    ],
    "Shopping": [
        "shopping in {destination} {region}",
        "best shops in {destination} {region}",
        "boutiques in {destination} {region}",
        "souvenir shop in {destination} {region}",
        "outlet in {destination} {region}",
    ],
}

GENERIC_LODGING_TOKENS = {
    "hotel",
    "resort",
    "lodge",
    "inn",
    "suites",
    "collection",
    "luxury",
    "mountain",
    "retreat",
    "spa",
    "accommodations",
    "club",
}
STOPWORDS = {"the", "a", "an", "and", "of", "at", "in", "on"}


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
class ClassifierConfig:
    profile_patterns: Tuple[str, ...]
    blog_patterns: Tuple[str, ...]
    category_patterns: Tuple[str, ...]
    search_patterns: Tuple[str, ...]
    static_patterns: Tuple[str, ...]


@dataclass(frozen=True)
class InventoryEntity:
    url: str
    domain: str
    name_key: str


class FetchError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


DEFAULT_CLASSIFIER = ClassifierConfig(
    profile_patterns=(
        r"^/hotels?(/|$)",
        r"^/lodging(/|$)",
        r"^/stay(/|$)",
        r"^/restaurant(s)?(/|$)",
        r"^/dining(/|$)",
        r"^/activities?(/|$)",
        r"^/things-to-do(/|$)",
        r"^/shops?(/|$)",
        r"^/vacation-rentals?(/|$)",
        r"/listings/",
    ),
    blog_patterns=(r"^/blog(/|$)", r"^/posts?(/|$)", r"^/news(/|$)", r"^/\d{4}/\d{2}/\d{2}/"),
    category_patterns=(r"^/category(/|$)", r"^/tag(/|$)", r"^/topics?(/|$)", r"^/categories(/|$)"),
    search_patterns=(r"^/search(/|$)", r"^/\?s=", r"^/(\w+/)?\?q="),
    static_patterns=(r"^/$", r"^/about(/|$)", r"^/contact(/|$)", r"^/privacy(-policy)?(/|$)", r"^/terms(-of-service)?(/|$)"),
)


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _append_debug(debug_log: Optional[List[str]], message: str) -> None:
    if debug_log is not None:
        debug_log.append(message)


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
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


def _request_with_retries(session: requests.Session, url: str, timeout_secs: int, debug_log: Optional[List[str]]) -> str:
    backoffs = (0.5, 1.5, 3.0)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/xml,text/xml;q=0.9,text/html;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
    }
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=timeout_secs, headers=headers, allow_redirects=True)
            _append_debug(debug_log, f"Attempt {attempt}/3 {url} -> HTTP {resp.status_code}")
            if resp.status_code >= 500 or resp.status_code in {403, 429, 520, 521, 522, 523, 524}:
                raise FetchError(f"HTTP {resp.status_code} for {url}", retryable=True)
            if resp.status_code >= 400:
                raise FetchError(f"HTTP {resp.status_code} for {url}")
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except (requests.RequestException, FetchError) as err:
            last_error = err
            retryable = isinstance(err, requests.RequestException) or (isinstance(err, FetchError) and err.retryable)
            if not retryable or attempt == 3:
                break
            time.sleep(backoffs[min(attempt - 1, len(backoffs) - 1)])
    raise FetchError(f"Request failed for {url}: {last_error}")


def http_get_text(url: str, timeout_secs: int, debug_log: Optional[List[str]] = None) -> str:
    with requests.Session() as session:
        errors = []
        for candidate in [url, _alternate_host_url(url)]:
            if not candidate:
                continue
            try:
                return _request_with_retries(session, candidate, timeout_secs, debug_log)
            except FetchError as err:
                errors.append(str(err))
        raise FetchError("; ".join(errors) if errors else f"Unable to fetch {url}")


def parse_sitemap_index(xml_text: str) -> List[str]:
    root = ET.fromstring(xml_text.strip())
    if _strip_ns(root.tag).lower() != "sitemapindex":
        return []
    sitemaps = []
    for node in root.findall("sm:sitemap", XML_NS):
        loc = node.find("sm:loc", XML_NS)
        if loc is not None and loc.text:
            sitemaps.append(loc.text.strip())
    return dedupe_preserve_order(sitemaps)


def parse_urlset(xml_text: str, source_sitemap_url: str) -> List[UrlRow]:
    root = ET.fromstring(xml_text.strip())
    if _strip_ns(root.tag).lower() != "urlset":
        return []
    rows: List[UrlRow] = []
    for node in root.findall("sm:url", XML_NS):
        loc = node.find("sm:loc", XML_NS)
        lastmod = node.find("sm:lastmod", XML_NS)
        if loc is not None and loc.text:
            rows.append(
                UrlRow(
                    url=loc.text.strip(),
                    url_type="unclassified",
                    source_sitemap=source_sitemap_url,
                    lastmod=lastmod.text.strip() if (lastmod is not None and lastmod.text) else None,
                )
            )
    return rows


def classify_url(url: str, cfg: ClassifierConfig) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    combined = path + ("?" + parsed.query if parsed.query else "")

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
        fetched_dt = datetime.fromisoformat(payload["stats"]["fetched_at_utc"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        return FetchStats(**payload["stats"]), [UrlRow(**row) for row in payload["rows"]]
    except Exception:
        return None


def save_inventory_disk_cache(sitemap_index_url: str, stats: FetchStats, rows: List[UrlRow]) -> None:
    json_path, csv_path = _disk_cache_paths_inventory(sitemap_index_url)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"stats": asdict(stats), "rows": [asdict(r) for r in rows]}, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "url_type", "source_sitemap", "lastmod"])
        for row in rows:
            writer.writerow([row.url, row.url_type, row.source_sitemap, row.lastmod or ""])


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
            parsed = parse_urlset(sm_xml, sm_url)
            rows.extend(parsed[:max_urls_per_child] if max_urls_per_child > 0 else parsed)
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
    started = time.time()
    if not force_refresh:
        disk = load_inventory_disk_cache(sitemap_index_url, disk_cache_max_age_hours)
        if disk is not None:
            stats, rows = disk
            return (
                FetchStats(
                    sitemap_index_url=stats.sitemap_index_url,
                    fetched_at_utc=stats.fetched_at_utc,
                    child_sitemaps_found=stats.child_sitemaps_found,
                    child_sitemaps_fetched=stats.child_sitemaps_fetched,
                    urls_found=stats.urls_found,
                    elapsed_seconds=round(time.time() - started, 3),
                    cache_used=True,
                ),
                apply_classification(rows, DEFAULT_CLASSIFIER),
                ["Loaded inventory from disk cache."],
            )

    child_sitemaps, rows, debug_log = fetch_inventory_live(
        sitemap_index_url=sitemap_index_url,
        timeout_secs=timeout_secs,
        max_child_sitemaps=max_child_sitemaps,
        max_urls_per_child=max_urls_per_child,
        debug_enabled=debug_enabled,
    )
    rows = apply_classification(rows, DEFAULT_CLASSIFIER)
    stats = FetchStats(
        sitemap_index_url=sitemap_index_url,
        fetched_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        child_sitemaps_found=len(child_sitemaps),
        child_sitemaps_fetched=len(child_sitemaps),
        urls_found=len(rows),
        elapsed_seconds=round(time.time() - started, 3),
        cache_used=False,
    )
    save_inventory_disk_cache(sitemap_index_url, stats, rows)
    return stats, rows, debug_log


def rows_to_csv_bytes(rows: List[Dict[str, object]]) -> bytes:
    if not rows:
        return b""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _clean_text(text: str) -> str:
    text = text.lower().replace("—", "-").replace("–", "-")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return [x for x in _clean_text(text).split() if x]


def _region_tokens(region: str) -> Set[str]:
    tokens = set(_tokenize(region))
    reg = region.strip().lower()
    if reg in {"co", "colorado"}:
        tokens.update({"co", "colorado"})
    return tokens


def clean_candidate_name(name: str, destination: str, region: str) -> str:
    txt = name.lower().replace("—", "-").replace("–", "-")
    if " - " in txt:
        txt = txt.rsplit(" - ", 1)[-1]
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    tokens = [t for t in txt.split() if t]
    remove = set(_tokenize(destination)) | _region_tokens(region) | GENERIC_LODGING_TOKENS
    kept = [t for t in tokens if t not in remove]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def inventory_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    segment = (parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else "")
    segment = segment.replace("-", " ").replace("_", " ")
    segment = re.sub(r"[^a-z0-9\s]", " ", segment.lower())
    tokens = [t for t in segment.split() if t and t not in {"hotel", "resort", "lodge", "inn", "suites", "collection", "club"}]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def match_score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if _HAS_RAPIDFUZZ and fuzz is not None:
        base = int(round(float(fuzz.token_set_ratio(a, b))))
    else:
        a_set, b_set = set(a.split()), set(b.split())
        if not a_set or not b_set:
            return 0
        jacc = len(a_set & b_set) / len(a_set | b_set)
        base = int(round(jacc * 100))
        if a.startswith(b) or b.startswith(a):
            base += 6
        if " ".join(sorted(a_set)).startswith(" ".join(sorted(b_set))) or " ".join(sorted(b_set)).startswith(" ".join(sorted(a_set))):
            base += 4
    a_set, b_set = set(a.split()), set(b.split())
    if a_set and b_set and (a_set.issubset(b_set) or b_set.issubset(a_set)):
        base += 8
    return max(0, min(base, 100))


def _normalize_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def build_inventory_entities(rows: List[UrlRow], destination: str, region: str) -> List[InventoryEntity]:
    entities: List[InventoryEntity] = []
    for row in rows:
        name_key = inventory_name_from_url(row.url)
        name_key = clean_candidate_name(name_key, destination, region)
        entities.append(InventoryEntity(url=row.url, domain=_normalize_domain(row.url), name_key=name_key))
    return entities


def _serp_cache_paths(cache_key: str) -> str:
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"serp_{cache_key}.json")


def _serp_cache_key(parts: Dict[str, str]) -> str:
    return _stable_key(json.dumps(parts, sort_keys=True))


def load_serp_disk_cache(parts: Dict[str, str], max_age_hours: float) -> Optional[dict]:
    cache_path = _serp_cache_paths(_serp_cache_key(parts))
    if not os.path.exists(cache_path):
        return None
    try:
        payload = json.loads(open(cache_path, "r", encoding="utf-8").read())
        fetched = datetime.fromisoformat(payload["metadata"]["fetched_at_utc"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        return payload["raw_json"]
    except Exception:
        return None


def save_serp_disk_cache(parts: Dict[str, str], response_json: dict) -> None:
    if not response_json:
        return
    cache_path = _serp_cache_paths(_serp_cache_key(parts))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "params": parts,
                },
                "raw_json": response_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


@st.cache_data(show_spinner=False)
def serpapi_search_cached(params: Dict[str, str], timeout_secs: int) -> dict:
    backoffs = (0.8, 1.6, 3.2)
    with requests.Session() as session:
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = session.get(SERP_API_URL, params=params, timeout=timeout_secs)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise FetchError(f"SerpAPI HTTP {resp.status_code}", retryable=True)
                if resp.status_code >= 400:
                    raise FetchError(f"SerpAPI HTTP {resp.status_code}: {resp.text[:260]}")
                data = resp.json()
                if data.get("error"):
                    raise FetchError(f"SerpAPI error: {data['error']}")
                return data
            except (requests.RequestException, ValueError, FetchError) as err:
                last_err = err
                retryable = isinstance(err, requests.RequestException) or (isinstance(err, FetchError) and err.retryable)
                if not retryable or attempt == 3:
                    break
                time.sleep(backoffs[min(attempt - 1, len(backoffs) - 1)])
    raise FetchError(f"SerpAPI request failed: {last_err}")


def run_serp_request(params: Dict[str, str], timeout_secs: int, cache_max_age_hours: float, force_refresh: bool, debug_log: List[str]) -> dict:
    cache_parts = {k: str(v) for k, v in params.items() if k != "api_key"}
    if not force_refresh:
        cached = load_serp_disk_cache(cache_parts, cache_max_age_hours)
        if cached is not None:
            _append_debug(debug_log, f"cache-hit engine={params.get('engine')} start={params.get('start', '0')} q={params.get('q')}")
            return cached
    _append_debug(debug_log, f"fetch engine={params.get('engine')} start={params.get('start', '0')} q={params.get('q')}")
    payload = serpapi_search_cached(params, timeout_secs)
    save_serp_disk_cache(cache_parts, payload)
    return payload


def build_query_text(template: str, destination: str, region: str) -> str:
    query = template.format(destination=destination.strip(), region=region.strip())
    query = re.sub(r"\s+", " ", query).strip()
    return query


def build_query_seeds(destination: str, region: str, category: str, custom_seeds: str) -> List[str]:
    if custom_seeds.strip():
        return dedupe_preserve_order([line.strip() for line in custom_seeds.splitlines() if line.strip()])
    templates = CATEGORY_QUERY_TEMPLATES.get(category, [])
    return [build_query_text(t, destination, region) for t in templates]


def extract_organic_candidates(payload: dict, source_query: str) -> List[dict]:
    output = []
    for item in payload.get("organic_results", []) or []:
        output.append(
            {
                "candidate_name": item.get("title", ""),
                "address": "",
                "phone": "",
                "rating": None,
                "reviews": None,
                "website": item.get("link", ""),
                "place_id": item.get("place_id") or item.get("result_id") or "",
                "rank": item.get("position"),
                "source_query": source_query,
                "source_engine": "google",
            }
        )
    return output


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    txt = re.sub(r"[^0-9]", "", str(value))
    return int(txt) if txt else None


def extract_maps_candidates(payload: dict, source_query: str) -> List[dict]:
    output = []
    for item in payload.get("local_results", []) or []:
        output.append(
            {
                "candidate_name": item.get("title") or item.get("name") or "",
                "address": item.get("address", ""),
                "phone": item.get("phone", ""),
                "rating": _to_float(item.get("rating")),
                "reviews": _to_int(item.get("reviews") or item.get("reviews_original")),
                "website": item.get("website", ""),
                "place_id": item.get("place_id") or item.get("data_id") or item.get("cid") or item.get("data_cid") or "",
                "rank": item.get("position"),
                "source_query": source_query,
                "source_engine": "google_maps",
            }
        )
    return output


def aggregate_candidates(raw_candidates: List[dict], destination: str, region: str) -> List[dict]:
    merged: Dict[str, dict] = {}
    for item in raw_candidates:
        normalized_name = clean_candidate_name(item.get("candidate_name", ""), destination, region)
        domain = _normalize_domain(item.get("website", ""))
        key = domain or f"{normalized_name}|{(item.get('address') or '').lower()}"
        if key not in merged:
            merged[key] = {
                **item,
                "name_key": normalized_name,
                "domain": domain,
                "engines": {item.get("source_engine", "")},
                "queries": {item.get("source_query", "")},
                "reviews": item.get("reviews") or 0,
            }
        else:
            cur = merged[key]
            cur["engines"].add(item.get("source_engine", ""))
            cur["queries"].add(item.get("source_query", ""))
            cur["reviews"] = max(cur.get("reviews") or 0, item.get("reviews") or 0)
            if not cur.get("website") and item.get("website"):
                cur["website"] = item.get("website")
                cur["domain"] = _normalize_domain(item.get("website", ""))
            if (cur.get("rating") or 0) < (item.get("rating") or 0):
                cur["rating"] = item.get("rating")
    return list(merged.values())


def score_and_partition(candidates: List[dict], inventory_entities: List[InventoryEntity], threshold: int) -> Tuple[List[dict], List[dict], List[dict]]:
    by_domain: Dict[str, List[InventoryEntity]] = {}
    for entity in inventory_entities:
        if entity.domain:
            by_domain.setdefault(entity.domain, []).append(entity)

    missing: List[dict] = []
    possible: List[dict] = []
    listed: List[dict] = []

    for candidate in candidates:
        best_score = 0
        best_url = ""
        best_name = ""

        c_domain = candidate.get("domain", "")
        if c_domain and c_domain in by_domain:
            best = by_domain[c_domain][0]
            best_score = 100
            best_url = best.url
            best_name = best.name_key
        else:
            c_name = candidate.get("name_key", "")
            for entity in inventory_entities:
                score = match_score(c_name, entity.name_key)
                if c_domain and c_domain and c_domain in entity.url:
                    score = min(100, score + 5)
                if score > best_score:
                    best_score = score
                    best_url = entity.url
                    best_name = entity.name_key

        candidate["best_match_url"] = best_url
        candidate["best_match_score"] = best_score
        candidate["best_match_name_key"] = best_name

        cscore = 0
        if "google_maps" in candidate.get("engines", set()):
            cscore += 3
        if len(candidate.get("queries", set())) >= 2:
            cscore += 2
        if candidate.get("website"):
            cscore += 1
        if (candidate.get("rating") or 0) >= 4.5:
            cscore += 1
        if (candidate.get("reviews") or 0) >= 200:
            cscore += 1
        candidate["candidate_score"] = cscore

        if best_score >= threshold:
            listed.append(candidate)
        elif threshold - 10 <= best_score < threshold:
            possible.append(candidate)
        else:
            missing.append(candidate)

    missing.sort(key=lambda x: (x.get("candidate_score", 0), x.get("reviews", 0)), reverse=True)
    possible.sort(key=lambda x: x.get("best_match_score", 0), reverse=True)
    listed.sort(key=lambda x: x.get("best_match_score", 0), reverse=True)
    return missing, possible, listed


def listing_display_name(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else ""
    slug = slug.replace("-", " ").replace("_", " ")
    slug = re.sub(r"[^a-z0-9\s]", " ", slug.lower())
    return re.sub(r"\s+", " ", slug).strip()


def listing_name_key(url: str, destination: str, region: str) -> str:
    tokens = [t for t in listing_display_name(url).split() if t]
    remove = STOPWORDS | GENERIC_LODGING_TOKENS | set(_tokenize(destination)) | _region_tokens(region)
    kept = [t for t in tokens if t not in remove]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def choose_canonical_name(display_names: List[str]) -> str:
    scored = sorted(display_names, key=lambda n: (len(n.split()), len(n), n))
    return scored[0] if scored else ""


def _listing_subset(rows: List[UrlRow], listings_only: bool) -> List[UrlRow]:
    if not listings_only:
        return rows
    return [r for r in rows if "/listings/" in r.url.lower() or r.url_type == "profiles"]


def detect_duplicates(rows: List[UrlRow], destination: str, region: str, threshold: int, min_cluster_size: int, listings_only: bool) -> Tuple[List[dict], List[dict], int]:
    listing_rows = _listing_subset(rows, listings_only)
    items = []
    for row in listing_rows:
        items.append(
            {
                "url": row.url,
                "display_name": listing_display_name(row.url),
                "name_key": listing_name_key(row.url, destination, region),
            }
        )

    clusters: List[dict] = []
    members: List[dict] = []
    used_urls: Set[str] = set()

    exact_groups: Dict[str, List[dict]] = {}
    for item in items:
        exact_groups.setdefault(item["name_key"], []).append(item)

    cluster_id = 1
    for key, group in exact_groups.items():
        if key and len(group) >= min_cluster_size:
            cid = f"DUP-{cluster_id:04d}"
            cluster_id += 1
            urls = [g["url"] for g in group]
            canonical = choose_canonical_name([g["display_name"] for g in group])
            clusters.append(
                {
                    "cluster_id": cid,
                    "confidence": "HIGH",
                    "canonical_suggested_name": canonical,
                    "cluster_size": len(group),
                    "urls": "\n".join(urls),
                    "example_pair_score": 100,
                }
            )
            for g in group:
                used_urls.add(g["url"])
                members.append({"cluster_id": cid, "confidence": "HIGH", **g})

    remaining = [item for item in items if item["url"] not in used_urls and item["name_key"]]
    if len(remaining) > 2000:
        blocks: Dict[str, List[dict]] = {}
        for item in remaining:
            first = item["name_key"].split()[0][0] if item["name_key"].split() else "#"
            blocks.setdefault(first, []).append(item)
        compare_groups = list(blocks.values())
    else:
        compare_groups = [remaining]

    seen_in_medium: Set[str] = set()
    for group in compare_groups:
        for i in range(len(group)):
            a = group[i]
            if a["url"] in seen_in_medium:
                continue
            cluster_items = [a]
            best_example = 0
            for j in range(i + 1, len(group)):
                b = group[j]
                if b["url"] in seen_in_medium:
                    continue
                score = match_score(a["name_key"], b["name_key"])
                if score >= threshold:
                    cluster_items.append(b)
                    best_example = max(best_example, score)
            if len(cluster_items) >= min_cluster_size:
                cid = f"DUP-{cluster_id:04d}"
                cluster_id += 1
                canonical = choose_canonical_name([x["display_name"] for x in cluster_items])
                clusters.append(
                    {
                        "cluster_id": cid,
                        "confidence": "MEDIUM",
                        "canonical_suggested_name": canonical,
                        "cluster_size": len(cluster_items),
                        "urls": "\n".join([x["url"] for x in cluster_items]),
                        "example_pair_score": best_example,
                    }
                )
                for x in cluster_items:
                    seen_in_medium.add(x["url"])
                    members.append({"cluster_id": cid, "confidence": "MEDIUM", **x})

    clusters.sort(key=lambda x: (x["confidence"] != "HIGH", -x["cluster_size"], x["cluster_id"]))
    return clusters, members, len(listing_rows)


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
                sitemap_index_url=sitemap_index_url,
                timeout_secs=int(timeout_secs),
                max_child_sitemaps=int(max_child_sitemaps),
                max_urls_per_child=int(max_urls_per_child),
                disk_cache_max_age_hours=float(disk_cache_max_age_hours),
                force_refresh=bool(force_refresh),
                debug_enabled=bool(show_debug_log),
            )
    except Exception as err:
        st.error(f"Inventory scan failed: {err}")
        return

    st.metric("Inventory URLs", stats.urls_found)
    st.caption(f"Fetched at: {stats.fetched_at_utc} • Cache used: {'Yes' if stats.cache_used else 'No'} • Elapsed: {stats.elapsed_seconds}s")

    table = [{"url": r.url, "url_type": r.url_type, "source_sitemap": r.source_sitemap, "lastmod": r.lastmod or ""} for r in rows]
    st.dataframe(table, use_container_width=True, height=460)
    st.download_button("Download inventory CSV", data=rows_to_csv_bytes(table), file_name="vacayrank_inventory.csv", mime="text/csv")

    if show_debug_log:
        st.subheader("Debug log")
        st.code("\n".join(debug_log) if debug_log else "No debug entries", language="text")


def _extract_serp_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("SERPAPI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("SERPAPI_API_KEY", "")
    return key.strip()


def render_serp_gap_page() -> None:
    st.subheader("Milestone 2 — SERP Gap")
    api_key = _extract_serp_api_key()

    with st.sidebar:
        st.header("SERP Gap Settings")
        sitemap_index_url = st.text_input("Sitemap index URL", value="https://vailvacay.com/sitemap_index.xml").strip()
        destination = st.text_input("Destination", value="Vail").strip()
        region = st.text_input("Region/State", value="CO").strip()
        country = st.text_input("Country", value="US").strip()
        category = st.selectbox("Category", options=["Hotels", "Restaurants", "Activities", "Nightlife", "Shopping"], index=0)
        custom_query_seeds = st.text_area("Custom query seeds (one per line)", value="", height=120)
        organic_pages = st.number_input("Organic pages", min_value=1, max_value=5, value=2)
        maps_pages = st.number_input("Maps pages", min_value=1, max_value=5, value=2)
        fuzzy_threshold = st.slider("Fuzzy threshold", min_value=0, max_value=100, value=88)
        cache_hours = st.number_input("Use disk cache if newer than (hours)", min_value=0.0, max_value=168.0, value=12.0)
        force_refresh = st.checkbox("Force refresh (ignore disk cache)", value=False)
        show_debug_log = st.checkbox("Show debug log", value=False)
        use_location_targeting = st.checkbox("Use SerpAPI location targeting (advanced)", value=False)
        st.caption("Location targeting requires SerpAPI Locations API resolution and is intentionally disabled in this patch.")
        run_disabled = not bool(api_key)
        run = st.button("Run SERP Gap Scan", type="primary", disabled=run_disabled)

    if not api_key:
        st.error("SERPAPI_API_KEY is missing. Set it in Streamlit secrets or environment variables to enable Milestone 2.")

    if use_location_targeting:
        st.info("Advanced location targeting was selected, but this release still runs query-based localization and omits the location parameter.")

    if not run:
        return

    debug_log: List[str] = []
    queries = build_query_seeds(destination, region, category, custom_query_seeds)
    if not queries:
        st.error("No queries were generated. Add a destination/category or custom query seeds.")
        return

    try:
        _, inventory_rows, _ = build_inventory(
            sitemap_index_url=sitemap_index_url,
            timeout_secs=DEFAULT_TIMEOUT_SECS,
            max_child_sitemaps=200,
            max_urls_per_child=0,
            disk_cache_max_age_hours=float(cache_hours),
            force_refresh=bool(force_refresh),
            debug_enabled=False,
        )
    except Exception as err:
        st.error(f"Failed to load inventory: {err}")
        return

    raw_candidates: List[dict] = []
    gl_code = (country or "US").strip().lower()

    try:
        for query in queries:
            for page_idx in range(int(organic_pages)):
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": api_key,
                    "hl": "en",
                    "gl": gl_code,
                    "num": "10",
                    "start": str(page_idx * 10),
                }
                payload = run_serp_request(params, DEFAULT_TIMEOUT_SECS, float(cache_hours), bool(force_refresh), debug_log)
                extracted = extract_organic_candidates(payload, query)
                _append_debug(debug_log, f"engine=google page_start={page_idx * 10} candidates={len(extracted)} query={query}")
                raw_candidates.extend(extracted)

            for page_idx in range(int(maps_pages)):
                params = {
                    "engine": "google_maps",
                    "q": query,
                    "api_key": api_key,
                    "hl": "en",
                    "gl": gl_code,
                    "start": str(page_idx * 20),
                }
                payload = run_serp_request(params, DEFAULT_TIMEOUT_SECS, float(cache_hours), bool(force_refresh), debug_log)
                extracted = extract_maps_candidates(payload, query)
                _append_debug(debug_log, f"engine=google_maps page_start={page_idx * 20} candidates={len(extracted)} query={query}")
                raw_candidates.extend(extracted)
    except Exception as err:
        st.error(f"SERP scan failed: {err}")
        if show_debug_log:
            st.code("\n".join(debug_log), language="text")
        return

    candidates = aggregate_candidates(raw_candidates, destination, region)
    inventory_entities = build_inventory_entities(inventory_rows, destination, region)
    missing, possible, listed = score_and_partition(candidates, inventory_entities, int(fuzzy_threshold))

    listings_inventory = [r for r in inventory_rows if "/listings/" in r.url.lower() or r.url_type == "profiles"]
    st.subheader("Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Inventory count", len(listings_inventory))
    m2.metric("SERP candidates discovered", len(candidates))
    m3.metric("Missing", len(missing))
    m4.metric("Possible matches", len(possible))
    m5.metric("Already listed", len(listed))

    def to_display_rows(items: List[dict]) -> List[Dict[str, object]]:
        display = []
        for item in items:
            display.append(
                {
                    "candidate_name": item.get("candidate_name", ""),
                    "name_key": item.get("name_key", ""),
                    "address": item.get("address", ""),
                    "phone": item.get("phone", ""),
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "website": item.get("website", ""),
                    "candidate_score": item.get("candidate_score", 0),
                    "best_match_url": item.get("best_match_url", ""),
                    "best_match_score": item.get("best_match_score", 0),
                    "best_match_name_key": item.get("best_match_name_key", ""),
                    "queries_count": len(item.get("queries", set())),
                    "engines": ",".join(sorted([e for e in item.get("engines", set()) if e])),
                    "place_id": item.get("place_id", ""),
                }
            )
        return display

    dup_listings_only = st.checkbox("Duplicates: Listings-only", value=True)
    dup_threshold = st.slider("Duplicates fuzzy threshold", min_value=80, max_value=100, value=92)
    dup_min_size = st.number_input("Duplicates min cluster size", min_value=2, max_value=20, value=2)

    dup_clusters, dup_members, dup_scanned = detect_duplicates(
        rows=inventory_rows,
        destination=destination,
        region=region,
        threshold=int(dup_threshold),
        min_cluster_size=int(dup_min_size),
        listings_only=bool(dup_listings_only),
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Missing", "Possible Matches", "Already Listed", "Duplicates"])
    with tab1:
        rows = to_display_rows(missing)
        st.dataframe(rows, use_container_width=True, height=460)
        st.download_button("Download Missing CSV", data=rows_to_csv_bytes(rows), file_name="vacayrank_missing.csv", mime="text/csv")
    with tab2:
        rows = to_display_rows(possible)
        st.dataframe(rows, use_container_width=True, height=460)
        st.download_button("Download Possible Matches CSV", data=rows_to_csv_bytes(rows), file_name="vacayrank_possible_matches.csv", mime="text/csv")
    with tab3:
        rows = to_display_rows(listed)
        st.dataframe(rows, use_container_width=True, height=460)
        st.download_button("Download Already Listed CSV", data=rows_to_csv_bytes(rows), file_name="vacayrank_already_listed.csv", mime="text/csv")
    with tab4:
        d1, d2, d3 = st.columns(3)
        d1.metric("Listing pages scanned", dup_scanned)
        d2.metric("Clusters found", len(dup_clusters))
        d3.metric("Total URLs clustered", len(dup_members))
        if not dup_clusters:
            st.info("No duplicates found for the selected settings.")
        else:
            st.dataframe(dup_clusters, use_container_width=True, height=350)
            for cluster in dup_clusters:
                cid = cluster["cluster_id"]
                with st.expander(f"{cid} — {cluster['confidence']} ({cluster['cluster_size']} URLs)"):
                    cluster_members = [m for m in dup_members if m["cluster_id"] == cid]
                    st.dataframe(cluster_members, use_container_width=True)
        st.download_button(
            "Download Duplicate Clusters CSV",
            data=rows_to_csv_bytes(dup_clusters),
            file_name="vacayrank_duplicate_clusters.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Duplicate Members CSV",
            data=rows_to_csv_bytes(dup_members),
            file_name="vacayrank_duplicate_members.csv",
            mime="text/csv",
        )

    if not candidates:
        st.warning("SERP scan returned 0 candidates. Try broader queries or inspect the debug log.")

    if show_debug_log:
        st.subheader("Debug log")
        st.code("\n".join(debug_log) if debug_log else "No debug entries.", language="text")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("Milestone 1 provides sitemap inventory. Milestone 2 discovers SERP candidates, computes SERP gaps, and surfaces duplicates.")

    with st.sidebar:
        page = st.radio("Navigation", options=["Milestone 1 — Inventory", "Milestone 2 — SERP Gap"])

    if page == "Milestone 1 — Inventory":
        render_inventory_page()
    else:
        render_serp_gap_page()


if __name__ == "__main__":
    _ensure_cache_dir()
    main()
