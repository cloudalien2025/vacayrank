from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import string
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import streamlit as st
import pandas as pd

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

CATEGORY_CONFIG = {
    "Hotels": {
        "inventory_categories": ["hotel", "hotels", "lodging", "stay"],
        "inventory_category_field": "category",
        "synonyms": ["resort", "inn"],
        "strategic_weight": 1.2,
        "query_templates": [
            "hotels in {destination} {region}",
            "best hotels in {destination} {region}",
            "luxury hotels in {destination} {region}",
            "{destination} {region} lodging",
        ],
        "inventory_url_contains_any": ["/listings/", "/hotels/", "/lodging/", "/stay/"],
        "serp_entity_stopwords": ["hotel", "resort", "lodge", "inn", "suites", "collection", "club", "spa", "accommodations"],
    },
    "Restaurants": {
        "inventory_categories": ["restaurant", "restaurants", "dining"],
        "inventory_category_field": "category",
        "synonyms": ["eatery", "bistro", "cafe"],
        "strategic_weight": 1.1,
        "query_templates": [
            "restaurants in {destination} {region}",
            "best restaurants in {destination} {region}",
            "fine dining in {destination} {region}",
            "{destination} {region} dinner",
        ],
        "inventory_url_contains_any": ["/listings/", "/restaurants/", "/restaurant/", "/dining/"],
        "serp_entity_stopwords": ["restaurant", "grill", "kitchen", "eatery", "bar", "bistro", "cafe"],
    },
    "Bars": {
        "inventory_categories": ["bar", "bars", "nightlife"],
        "inventory_category_field": "category",
        "synonyms": ["pub", "lounge", "tavern"],
        "strategic_weight": 1.0,
        "query_templates": [
            "bars in {destination} {region}",
            "best bars in {destination} {region}",
            "cocktail bars in {destination} {region}",
            "apres ski bars in {destination} {region}",
        ],
        "inventory_url_contains_any": ["/listings/", "/bars/", "/nightlife/"],
        "serp_entity_stopwords": ["bar", "pub", "lounge", "tavern", "nightclub", "cocktail"],
    },
    "Attractions": {
        "inventory_categories": ["attraction", "attractions", "activities", "things to do"],
        "inventory_category_field": "category",
        "synonyms": ["tour", "experience"],
        "strategic_weight": 1.4,
        "query_templates": [
            "attractions in {destination} {region}",
            "things to do in {destination} {region}",
            "best activities in {destination} {region}",
            "{destination} {region} sightseeing",
        ],
        "inventory_url_contains_any": ["/listings/", "/activities/", "/things-to-do/", "/attractions/"],
        "serp_entity_stopwords": ["attraction", "adventure", "tour", "experience", "activity", "park"],
    },
    "Shops": {
        "inventory_categories": ["shop", "shops", "shopping"],
        "inventory_category_field": "category",
        "synonyms": ["store", "boutique", "market"],
        "strategic_weight": 0.9,
        "query_templates": [
            "shops in {destination} {region}",
            "shopping in {destination} {region}",
            "boutiques in {destination} {region}",
            "gift shops in {destination} {region}",
        ],
        "inventory_url_contains_any": ["/listings/", "/shops/", "/shopping/"],
        "serp_entity_stopwords": ["shop", "store", "boutique", "market", "outlet"],
    },
    "Ski Rentals": {
        "inventory_categories": ["ski rentals", "rental", "rentals", "equipment rentals"],
        "inventory_category_field": "category",
        "synonyms": ["snowboard rentals", "ski shop"],
        "strategic_weight": 1.5,
        "query_templates": [
            "ski rentals in {destination} {region}",
            "snowboard rentals in {destination} {region}",
            "ski shop rentals in {destination} {region}",
            "equipment rentals in {destination} {region}",
        ],
        "inventory_url_contains_any": ["/listings/", "/rentals/", "/ski-rentals/"],
        "serp_entity_stopwords": ["rental", "rentals", "ski", "snowboard", "equipment", "shop"],
    },
}

GENERIC_LODGING_TOKENS = {"hotel", "resort", "lodge", "inn", "suites", "collection", "club"}
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
            rows.append(UrlRow(url=loc.text.strip(), url_type="unclassified", source_sitemap=source_sitemap_url, lastmod=lastmod.text.strip() if (lastmod is not None and lastmod.text) else None))
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
def fetch_inventory_live(sitemap_index_url: str, timeout_secs: int, max_child_sitemaps: int, max_urls_per_child: int, debug_enabled: bool) -> Tuple[List[str], List[UrlRow], List[str]]:
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


def build_inventory(sitemap_index_url: str, timeout_secs: int, max_child_sitemaps: int, max_urls_per_child: int, disk_cache_max_age_hours: float, force_refresh: bool, debug_enabled: bool) -> Tuple[FetchStats, List[UrlRow], List[str]]:
    started = time.time()
    if not force_refresh:
        disk = load_inventory_disk_cache(sitemap_index_url, disk_cache_max_age_hours)
        if disk is not None:
            stats, rows = disk
            return FetchStats(
                sitemap_index_url=stats.sitemap_index_url,
                fetched_at_utc=stats.fetched_at_utc,
                child_sitemaps_found=stats.child_sitemaps_found,
                child_sitemaps_fetched=stats.child_sitemaps_fetched,
                urls_found=stats.urls_found,
                elapsed_seconds=round(time.time() - started, 3),
                cache_used=True,
            ), apply_classification(rows, DEFAULT_CLASSIFIER), ["Loaded inventory from disk cache."]

    child_sitemaps, rows, debug_log = fetch_inventory_live(sitemap_index_url, timeout_secs, max_child_sitemaps, max_urls_per_child, debug_enabled)
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


def get_inventory_for_category(category_key: str, inventory_df: pd.DataFrame, category_cfg: dict) -> tuple[pd.DataFrame, dict]:
    report = {
        "category": category_key,
        "inventory_total_rows": int(len(inventory_df)),
        "inventory_rows_for_category": 0,
        "inventory_filter_field_used": None,
        "inventory_filter_labels_used": [],
        "inventory_filter_strategy": "none",
        "fallback": "none",
        "reason": "",
    }
    if inventory_df.empty:
        report["reason"] = "inventory empty"
        return inventory_df.iloc[0:0], report

    labels = [category_key] + list(category_cfg.get("inventory_categories", []) or []) + list(category_cfg.get("synonyms", []) or [])
    labels_norm = sorted({_clean_text(x) for x in labels if str(x).strip()})
    report["inventory_filter_labels_used"] = labels_norm

    configured_field = category_cfg.get("inventory_category_field")
    candidate_fields = [configured_field] if configured_field else []
    candidate_fields += ["category", "categories", "type"]
    field_used = next((f for f in candidate_fields if f and f in inventory_df.columns), None)
    report["inventory_filter_field_used"] = field_used

    if not field_used:
        report["reason"] = "no usable category field"
        return inventory_df.iloc[0:0], report

    def _values(cell: object) -> List[str]:
        if isinstance(cell, list):
            vals = cell
        elif isinstance(cell, str):
            vals = [x.strip() for x in re.split(r"[,|;/]", cell) if x.strip()]
        else:
            vals = [str(cell)] if cell is not None else []
        return [_clean_text(x) for x in vals if str(x).strip()]

    series = inventory_df[field_used]
    has_multi = series.apply(lambda x: isinstance(x, list)).any() or series.astype(str).str.contains(r"[,|;/]", na=False, regex=True).any()
    strategy = "list-membership" if has_multi else "exact"

    label_set = set(labels_norm)
    mask = inventory_df[field_used].apply(lambda x: bool(set(_values(x)) & label_set))
    filtered = inventory_df[mask].copy()
    report["inventory_filter_strategy"] = strategy
    report["inventory_rows_for_category"] = int(len(filtered))
    if filtered.empty:
        report["reason"] = "no match"
    return filtered, report


def _milestone3_cache_path(cache_key: str) -> str:
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"milestone3_{cache_key}.json")


def _milestone3_cache_key(parts: dict) -> str:
    return _stable_key(json.dumps(parts, sort_keys=True))


def load_milestone3_disk_cache(parts: dict, max_age_hours: float) -> Optional[dict]:
    path = _milestone3_cache_path(_milestone3_cache_key(parts))
    if not os.path.exists(path):
        return None
    try:
        payload = json.loads(open(path, "r", encoding="utf-8").read())
        fetched_dt = datetime.fromisoformat(payload["fetched_at_utc"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 3600
        return None if age_hours > max_age_hours else payload.get("rows", {})
    except Exception:
        return None


def save_milestone3_disk_cache(parts: dict, rows: dict) -> None:
    path = _milestone3_cache_path(_milestone3_cache_key(parts))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "parts": parts, "rows": rows}, f, ensure_ascii=False, indent=2)


def _intent_signal_count(queries: Iterable[str]) -> int:
    words = ["best", "top", "near me", "book", "reservations", "rental", "rentals", "tickets", "tour", "tours", "luxury", "family", "cheap", "deals"]
    found = set()
    for query in queries:
        q = (query or "").lower()
        for w in words:
            if w in q:
                found.add(w)
    return len(found)


def _strategic_score(weight: float) -> int:
    return max(0, min(10, round(5 * float(weight))))


def build_milestone3_scores(results_by_category: Dict[str, dict], selected_categories: List[str], category_config: dict, include_possible: bool) -> Tuple[List[dict], List[dict], dict]:
    category_stats = {}
    for category in selected_categories:
        cat = results_by_category.get(category, {})
        serp_found = len(cat.get("candidates", []))
        missing_count = len(cat.get("missing", []))
        gap_ratio = missing_count / max(1, serp_found)
        category_stats[category] = {
            "serp_found": serp_found,
            "missing_count": missing_count,
            "gap_ratio": gap_ratio,
            "gap_score": round(20 * max(0, min(1, gap_ratio))),
            "strategic_score": _strategic_score(category_config.get(category, {}).get("strategic_weight", 1.0)),
        }

    scored_rows = []
    for category in selected_categories:
        cat_results = results_by_category.get(category, {})
        source_rows = [{**x, "candidate_status": "Missing"} for x in cat_results.get("missing", [])]
        if include_possible:
            source_rows += [{**x, "candidate_status": "Possible"} for x in cat_results.get("possible", [])]
        duplicate_urls = {m.get("url", "") for m in cat_results.get("duplicates", ([], [], 0))[1]}
        for candidate in source_rows:
            rank = candidate.get("rank") or 999
            appearances = len(candidate.get("queries", set()))
            if rank <= 3:
                visibility = 35
            elif rank <= 10:
                visibility = 25
            elif rank <= 20:
                visibility = 15
            else:
                visibility = 8
            visibility = min(35, visibility + min(10, 2 * appearances))
            demand = min(20, 5 * _intent_signal_count(candidate.get("queries", set())))
            gap_score = category_stats[category]["gap_score"]
            conf = 0
            if candidate.get("website"):
                conf += 5
            if len((candidate.get("snippet") or "").strip()) > 30:
                conf += 3
            if candidate.get("candidate_status") == "Missing" and 0 < (candidate.get("best_match_score") or 0) < 70:
                conf += 4
            is_duplicate = candidate.get("best_match_url") in duplicate_urls
            if not is_duplicate:
                conf += 3
            conf = min(15, conf)
            strategic = category_stats[category]["strategic_score"]

            penalties = 0
            if candidate.get("geo_bucket") == "Unknown Location":
                penalties += 25
            if candidate.get("geo_bucket") == "Out of Scope":
                penalties += 50
            if is_duplicate:
                penalties += 30
            if candidate.get("candidate_status") == "Possible":
                penalties += 10

            score = max(0, min(100, visibility + demand + gap_score + conf + strategic - penalties))
            reason = f"vis={visibility}, demand={demand}, gap={gap_score}, conf={conf}, strategic={strategic}, penalties={penalties}"
            scored_rows.append({
                "category": category,
                "candidate_name": candidate.get("candidate_name", ""),
                "opportunity_score": int(score),
                "visibility_score": visibility,
                "demand_score": demand,
                "gap_score": gap_score,
                "confidence_score": conf,
                "strategic_score": strategic,
                "penalties": penalties,
                "reason": reason,
                "best_source_query": sorted(candidate.get("queries", set()))[0] if candidate.get("queries") else candidate.get("source_query", ""),
                "serp_rank": candidate.get("rank"),
                "candidate_domain": candidate.get("domain", ""),
                "location_status": candidate.get("geo_bucket", "In Scope"),
                "candidate_status": candidate.get("candidate_status", "Missing"),
                "gap_ratio": round(category_stats[category]["gap_ratio"], 3),
                "is_duplicate": is_duplicate,
            })

    scored_rows.sort(key=lambda x: x["opportunity_score"], reverse=True)
    summary_rows = []
    for category in selected_categories:
        rows = [r for r in scored_rows if r["category"] == category and r["candidate_status"] == "Missing"]
        summary_rows.append({
            "category": category,
            "avg_score (missing)": round(sum(r["opportunity_score"] for r in rows) / max(1, len(rows)), 2),
            "top_score": max([r["opportunity_score"] for r in rows], default=0),
            "count_scored": len(rows),
            "gap_ratio": round(category_stats[category]["gap_ratio"], 3),
        })
    debug_stats = {
        "rows_scored": len(scored_rows),
        "score_min": min([r["opportunity_score"] for r in scored_rows], default=0),
        "score_mean": round(sum(r["opportunity_score"] for r in scored_rows) / max(1, len(scored_rows)), 2),
        "score_max": max([r["opportunity_score"] for r in scored_rows], default=0),
    }
    return scored_rows, summary_rows, debug_stats


def run_self_check() -> List[dict]:
    inventory_df = pd.DataFrame([
        {"url": "https://example.com/hotel-a", "category": "Hotels"},
        {"url": "https://example.com/food-a", "category": "Restaurants"},
        {"url": "https://example.com/food-b", "category": "Restaurants"},
    ])
    hotels_df, _ = get_inventory_for_category("Hotels", inventory_df, {"inventory_category_field": "category", "inventory_categories": ["hotels"]})
    restaurants_df, _ = get_inventory_for_category("Restaurants", inventory_df, {"inventory_category_field": "category", "inventory_categories": ["restaurants"]})
    score_rows, _, _ = build_milestone3_scores({"Hotels": {"candidates": [{"id": 1}], "missing": [{"candidate_name": "X", "rank": 2, "queries": {"best hotels"}, "website": "https://x.com", "snippet": "great hotel great hotel great hotel great hotel", "geo_bucket": "Unknown Location"}], "possible": [], "duplicates": ([], [], 0)}}, ["Hotels"], {"Hotels": {"strategic_weight": 1.0}}, False)
    return [
        {"check": "inventory_counts_differ", "passed": len(hotels_df) != len(restaurants_df)},
        {"check": "score_within_bounds", "passed": bool(score_rows) and 0 <= score_rows[0]["opportunity_score"] <= 100},
        {"check": "unknown_location_penalty_applied", "passed": bool(score_rows) and score_rows[0]["penalties"] >= 25},
    ]


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


def _country_tokens(country: str) -> Set[str]:
    c = country.strip().lower()
    if c in {"us", "usa", "united states", "united states of america"}:
        return {"us", "usa", "united", "states", "america"}
    return set(_tokenize(country))


def clean_candidate_name(name: str, destination: str, region: str, country: str, category_stopwords: List[str]) -> str:
    txt = name.lower().replace("—", "-").replace("–", "-")
    if " - " in txt:
        txt = txt.rsplit(" - ", 1)[-1]
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    tokens = [t for t in txt.split() if t]
    remove = set(_tokenize(destination)) | _region_tokens(region) | _country_tokens(country) | set(category_stopwords)
    kept = [t for t in tokens if t not in remove]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def inventory_name_from_url(url: str, category_stopwords: List[str], destination: str, region: str, country: str) -> str:
    parsed = urlparse(url)
    segment = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else ""
    segment = segment.replace("-", " ").replace("_", " ")
    segment = re.sub(r"[^a-z0-9\s]", " ", segment.lower())
    tokens = [t for t in segment.split() if t]
    remove = GENERIC_LODGING_TOKENS | set(category_stopwords) | set(_tokenize(destination)) | _region_tokens(region) | _country_tokens(country)
    kept = [t for t in tokens if t not in remove]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def match_score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if _HAS_RAPIDFUZZ and fuzz is not None:
        base = int(round(float(fuzz.token_set_ratio(a, b))))
    else:
        a_set, b_set = set(a.split()), set(b.split())
        if not a_set or not b_set:
            return 0
        jaccard = len(a_set & b_set) / len(a_set | b_set)
        base = int(round(jaccard * 100))
        if a_set.issubset(b_set) or b_set.issubset(a_set):
            base += 8
        elif a in b or b in a:
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


def build_inventory_entities(rows: List[UrlRow], destination: str, region: str, country: str, category_cfg: dict) -> List[InventoryEntity]:
    entities: List[InventoryEntity] = []
    stopwords = category_cfg.get("serp_entity_stopwords", [])
    for row in rows:
        name_key = inventory_name_from_url(row.url, stopwords, destination, region, country)
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
        json.dump({"metadata": {"fetched_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "params": parts}, "raw_json": response_json}, f, ensure_ascii=False, indent=2)


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


def build_query_seeds(destination: str, region: str, category_cfg: dict, custom_seeds: str) -> List[str]:
    if custom_seeds.strip():
        return dedupe_preserve_order([line.strip() for line in custom_seeds.splitlines() if line.strip()])
    templates = category_cfg.get("query_templates", [])
    return [build_query_text(t, destination, region) for t in templates]


def extract_organic_candidates(payload: dict, source_query: str) -> List[dict]:
    output = []
    for item in payload.get("organic_results", []) or []:
        output.append({"candidate_name": item.get("title", ""), "address": "", "phone": "", "rating": None, "reviews": None, "website": item.get("link", ""), "place_id": item.get("place_id") or item.get("result_id") or "", "rank": item.get("position"), "source_query": source_query, "source_engine": "google"})
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
    rows = payload.get("local_results", []) or payload.get("places_results", []) or []
    output = []
    for item in rows:
        output.append({
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
        })
    return output


def extract_city_from_address(address: str) -> Optional[str]:
    if not address or not address.strip():
        return None
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 3:
        city_candidate = parts[-2]
    elif len(parts) == 2:
        first, second = parts
        if any(ch.isdigit() for ch in first):
            city_candidate = second
        else:
            city_candidate = first
    else:
        return None
    city_candidate = re.sub(f"[{re.escape(string.punctuation)}]", " ", city_candidate.lower())
    city_candidate = re.sub(r"\b[a-z]{2}\s*\d{4,}\b", " ", city_candidate)
    city_candidate = re.sub(r"\s+", " ", city_candidate).strip()
    if not city_candidate:
        return None
    words = [w for w in city_candidate.split() if not re.fullmatch(r"[a-z]{2}", w) and not w.isdigit()]
    if not words:
        return None
    return " ".join(words)


def aggregate_candidates(raw_candidates: List[dict], destination: str, region: str, country: str, category_cfg: dict) -> List[dict]:
    merged: Dict[str, dict] = {}
    stopwords = category_cfg.get("serp_entity_stopwords", [])
    for item in raw_candidates:
        normalized_name = clean_candidate_name(item.get("candidate_name", ""), destination, region, country, stopwords)
        domain = _normalize_domain(item.get("website", ""))
        key = domain or f"{normalized_name}|{(item.get('address') or '').lower()}"
        if key not in merged:
            merged[key] = {**item, "name_key": normalized_name, "domain": domain, "engines": {item.get("source_engine", "")}, "queries": {item.get("source_query", "")}, "reviews": item.get("reviews") or 0}
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


def apply_geo_filter(candidates: List[dict], destination: str, strict_city_match: bool, allowed_cities: Set[str]) -> Tuple[List[dict], List[dict], List[dict]]:
    if not strict_city_match:
        for candidate in candidates:
            candidate["geo_bucket"] = "In Scope"
            candidate["candidate_city"] = extract_city_from_address(candidate.get("address", ""))
        return candidates, [], []

    destination_city = _clean_text(destination)
    in_scope: List[dict] = []
    out_scope: List[dict] = []
    unknown: List[dict] = []
    for candidate in candidates:
        city = extract_city_from_address(candidate.get("address", ""))
        candidate["candidate_city"] = city or ""
        if city is None:
            candidate["geo_bucket"] = "Unknown Location"
            unknown.append(candidate)
        elif city == destination_city or city in allowed_cities:
            candidate["geo_bucket"] = "In Scope"
            in_scope.append(candidate)
        else:
            candidate["geo_bucket"] = "Out of Scope"
            out_scope.append(candidate)
    return in_scope, out_scope, unknown


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
        c_domain = candidate.get("domain", "")
        if c_domain and c_domain in by_domain:
            best = by_domain[c_domain][0]
            best_score = 100
            best_url = best.url
        else:
            c_name = candidate.get("name_key", "")
            for entity in inventory_entities:
                score = match_score(c_name, entity.name_key)
                if score > best_score:
                    best_score = score
                    best_url = entity.url

        candidate["best_match_url"] = best_url
        candidate["best_match_score"] = best_score

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


def listing_name_key(url: str, destination: str, region: str, category_stopwords: List[str]) -> str:
    tokens = [t for t in listing_display_name(url).split() if t]
    remove = STOPWORDS | GENERIC_LODGING_TOKENS | set(category_stopwords) | set(_tokenize(destination)) | _region_tokens(region)
    kept = [t for t in tokens if t not in remove]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def choose_canonical_name(display_names: List[str]) -> str:
    scored = sorted(display_names, key=lambda n: (len(n.split()), len(n), n))
    return scored[0] if scored else ""


def category_inventory_rows(rows: List[UrlRow], category_cfg: dict) -> List[UrlRow]:
    patterns = [p.lower() for p in category_cfg.get("inventory_url_contains_any", [])]
    out = []
    for r in rows:
        u = r.url.lower()
        if r.url_type == "profiles" or any(p in u for p in patterns):
            out.append(r)
    return out


def detect_duplicates(rows: List[UrlRow], destination: str, region: str, category_cfg: dict, threshold: int = 92, min_cluster_size: int = 2) -> Tuple[List[dict], List[dict], int]:
    listing_rows = category_inventory_rows(rows, category_cfg)
    items = [{"url": r.url, "display_name": listing_display_name(r.url), "name_key": listing_name_key(r.url, destination, region, category_cfg.get("serp_entity_stopwords", []))} for r in listing_rows]

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
            clusters.append({"cluster_id": cid, "confidence": "HIGH", "canonical_suggested_name": choose_canonical_name([g["display_name"] for g in group]), "cluster_size": len(group), "urls": "\n".join([g["url"] for g in group]), "example_pair_score": 100})
            for g in group:
                used_urls.add(g["url"])
                members.append({"cluster_id": cid, "confidence": "HIGH", **g})

    remaining = [item for item in items if item["url"] not in used_urls and item["name_key"]]
    compare_groups: List[List[dict]]
    if len(remaining) > 2000:
        blocks: Dict[str, List[dict]] = {}
        for item in remaining:
            k = item["name_key"].split()[0][0] if item["name_key"].split() else "#"
            blocks.setdefault(k, []).append(item)
        compare_groups = list(blocks.values())
    else:
        compare_groups = [remaining]

    seen: Set[str] = set()
    for group in compare_groups:
        for i in range(len(group)):
            base = group[i]
            if base["url"] in seen:
                continue
            cluster_items = [base]
            best_example = 0
            for j in range(i + 1, len(group)):
                other = group[j]
                if other["url"] in seen:
                    continue
                score = match_score(base["name_key"], other["name_key"])
                if score >= threshold:
                    cluster_items.append(other)
                    best_example = max(best_example, score)
            if len(cluster_items) >= min_cluster_size:
                cid = f"DUP-{cluster_id:04d}"
                cluster_id += 1
                clusters.append({"cluster_id": cid, "confidence": "MEDIUM", "canonical_suggested_name": choose_canonical_name([x["display_name"] for x in cluster_items]), "cluster_size": len(cluster_items), "urls": "\n".join([x["url"] for x in cluster_items]), "example_pair_score": best_example})
                for x in cluster_items:
                    seen.add(x["url"])
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
            stats, rows, debug_log = build_inventory(sitemap_index_url=sitemap_index_url, timeout_secs=int(timeout_secs), max_child_sitemaps=int(max_child_sitemaps), max_urls_per_child=int(max_urls_per_child), disk_cache_max_age_hours=float(disk_cache_max_age_hours), force_refresh=bool(force_refresh), debug_enabled=bool(show_debug_log))
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


def _safe_category_config() -> dict:
    cfg = CATEGORY_CONFIG
    with st.sidebar:
        load_json_cfg = st.checkbox("Load category config from JSON (advanced)", value=False)
        if load_json_cfg:
            raw = st.text_area("Category config JSON", value=json.dumps(CATEGORY_CONFIG, indent=2), height=220)
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict) or not parsed:
                    raise ValueError("JSON must be a non-empty object of category definitions")
                cfg = parsed
                st.success("Using JSON category config for this run.")
            except Exception as err:
                st.error(f"Invalid category config JSON; using built-in defaults. Error: {err}")
                cfg = CATEGORY_CONFIG
    return cfg


def _display_candidate_rows(items: List[dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in items:
        rows.append({
            "candidate_name": item.get("candidate_name", ""),
            "name_key": item.get("name_key", ""),
            "candidate_city": item.get("candidate_city", ""),
            "address": item.get("address", ""),
            "phone": item.get("phone", ""),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "website": item.get("website", ""),
            "candidate_score": item.get("candidate_score", 0),
            "best_match_url": item.get("best_match_url", ""),
            "best_match_score": item.get("best_match_score", 0),
            "queries_count": len(item.get("queries", set())),
            "engines": ",".join(sorted([e for e in item.get("engines", set()) if e])),
            "place_id": item.get("place_id", ""),
        })
    return rows


def render_serp_gap_page() -> None:
    st.subheader("Milestone 2 — SERP Gap")
    api_key = _extract_serp_api_key()
    category_config = _safe_category_config()
    category_keys = list(category_config.keys())

    with st.sidebar:
        st.header("SERP Gap Settings")
        sitemap_index_url = st.text_input("Sitemap index URL", value="https://vailvacay.com/sitemap_index.xml").strip()
        destination = st.text_input("Destination", value="Vail").strip()
        region = st.text_input("Region/State", value="CO").strip()
        country = st.text_input("Country", value="US").strip()
        if st.button("Select all categories"):
            st.session_state["selected_categories"] = category_keys
        selected_categories = st.multiselect("Categories to scan", options=category_keys, default=st.session_state.get("selected_categories", category_keys))
        st.session_state["selected_categories"] = selected_categories
        custom_query_seeds = st.text_area("Custom query seeds (one per line, applies to all selected categories)", value="", height=120)
        organic_pages = st.number_input("Organic pages to fetch", min_value=1, max_value=5, value=2)
        maps_pages = st.number_input("Maps pages to fetch", min_value=1, max_value=5, value=2)
        fuzzy_threshold = st.slider("Matching threshold", min_value=0, max_value=100, value=88)
        cache_hours = st.number_input("Use disk cache if newer than (hours)", min_value=0.0, max_value=168.0, value=12.0)
        force_refresh = st.checkbox("Force refresh", value=False)
        show_debug_log = st.checkbox("Show debug log", value=False)
        strict_city_match = st.checkbox("Strict city match", value=True)
        allowed_cities_raw = st.text_input("Allowed cities (comma separated, optional)", value="")
        use_location_targeting = st.checkbox("Use SerpAPI location targeting (advanced)", value=False)
        st.caption("Advanced location targeting is intentionally not used here; localization is query + gl/hl only.")
        run = st.button("Run SERP Gap Scan", type="primary", disabled=not bool(api_key))

    if not api_key:
        st.error("SERPAPI_API_KEY is missing. Set it in Streamlit secrets or environment variables to enable Milestone 2.")
    if use_location_targeting:
        st.info("Advanced location targeting selected. This build intentionally omits the SerpAPI location parameter to avoid unsupported location errors.")
    if not run:
        return
    if not selected_categories:
        st.error("Select at least one category to run the matrix scan.")
        return

    allowed_cities = {_clean_text(x) for x in allowed_cities_raw.split(",") if x.strip()}
    debug_log: List[str] = []
    gl_code = (country or "US").strip().lower()

    try:
        _, inventory_rows, _ = build_inventory(sitemap_index_url=sitemap_index_url, timeout_secs=DEFAULT_TIMEOUT_SECS, max_child_sitemaps=200, max_urls_per_child=0, disk_cache_max_age_hours=float(cache_hours), force_refresh=bool(force_refresh), debug_enabled=False)
    except Exception as err:
        st.error(f"Failed to load inventory: {err}")
        return

    inventory_df = pd.DataFrame([asdict(row) for row in inventory_rows])
    results_by_category: Dict[str, dict] = {}
    inventory_counts: List[int] = []
    missing_category_mapping_count = 0

    for category in selected_categories:
        cfg = category_config.get(category, {})
        category_inventory_df, inventory_report = get_inventory_for_category(category, inventory_df, cfg)
        category_inventory = [UrlRow(**row) for row in category_inventory_df.to_dict(orient="records")]
        inventory_counts.append(len(category_inventory))
        if not inventory_report.get("inventory_filter_field_used"):
            missing_category_mapping_count += 1
            _append_debug(debug_log, f"WARNING {category} inventory has no usable category field. per-category inventory count forced to 0")

        _append_debug(debug_log, (
            f"{category} inventory_total_rows={inventory_report['inventory_total_rows']} "
            f"inventory_rows_for_category={inventory_report['inventory_rows_for_category']} "
            f"inventory_filter_field_used={inventory_report['inventory_filter_field_used']} "
            f"inventory_filter_labels_used={inventory_report['inventory_filter_labels_used']} "
            f"inventory_filter_strategy={inventory_report['inventory_filter_strategy']} "
            f"fallback={inventory_report['fallback']} reason={inventory_report['reason']}"
        ))

        queries = build_query_seeds(destination, region, cfg, custom_query_seeds)
        if not queries:
            results_by_category[category] = {"inventory_rows": category_inventory, "candidates": [], "in_scope": [], "missing": [], "possible": [], "listed": [], "out_scope": [], "unknown": [], "duplicates": ([], [], 0)}
            continue

        raw_candidates: List[dict] = []
        for query in queries:
            for page_idx in range(int(organic_pages)):
                params = {"engine": "google", "q": query, "api_key": api_key, "hl": "en", "gl": gl_code, "num": "10", "start": str(page_idx * 10)}
                try:
                    payload = run_serp_request(params, DEFAULT_TIMEOUT_SECS, float(cache_hours), bool(force_refresh), debug_log)
                    extracted = extract_organic_candidates(payload, query)
                    raw_candidates.extend(extracted)
                    _append_debug(debug_log, f"{category} query={query} engine=google start={page_idx * 10} extracted={len(extracted)}")
                except Exception as err:
                    _append_debug(debug_log, f"{category} query={query} engine=google error={err}")
            for page_idx in range(int(maps_pages)):
                params = {"engine": "google_maps", "q": query, "api_key": api_key, "hl": "en", "gl": gl_code, "start": str(page_idx * 20)}
                try:
                    payload = run_serp_request(params, DEFAULT_TIMEOUT_SECS, float(cache_hours), bool(force_refresh), debug_log)
                    extracted = extract_maps_candidates(payload, query)
                    raw_candidates.extend(extracted)
                    _append_debug(debug_log, f"{category} query={query} engine=google_maps start={page_idx * 20} extracted={len(extracted)}")
                except Exception as err:
                    _append_debug(debug_log, f"{category} query={query} engine=google_maps error={err}")

        candidates = aggregate_candidates(raw_candidates, destination, region, country, cfg)
        in_scope, out_scope, unknown = apply_geo_filter(candidates, destination, strict_city_match, allowed_cities)
        _append_debug(debug_log, f"{category} geo_stats total={len(candidates)} in_scope={len(in_scope)} out_scope={len(out_scope)} unknown={len(unknown)}")

        inventory_entities = build_inventory_entities(category_inventory, destination, region, country, cfg)
        missing, possible, listed = score_and_partition(in_scope, inventory_entities, int(fuzzy_threshold))
        dup_clusters, dup_members, dup_scanned = detect_duplicates(inventory_rows, destination, region, cfg)

        results_by_category[category] = {
            "inventory_rows": category_inventory,
            "candidates": candidates,
            "in_scope": in_scope,
            "missing": missing,
            "possible": possible,
            "listed": listed,
            "out_scope": out_scope,
            "unknown": unknown,
            "duplicates": (dup_clusters, dup_members, dup_scanned),
        }

    if len(set(inventory_counts)) == 1 and inventory_counts and inventory_counts[0] > 0:
        st.warning("All category inventory counts are identical. Category-to-inventory mapping may not be applied.")
    if len(inventory_rows) > 0 and sum(inventory_counts) > len(inventory_rows) * 1.2:
        _append_debug(debug_log, "Multi-category inventory detected; counts may overlap.")

    matrix_rows = []
    for category in selected_categories:
        r = results_by_category[category]
        matrix_rows.append({
            "Category": category,
            "Inventory": len(r["inventory_rows"]),
            "SERP Found": len(r["candidates"]),
            "In Scope": len(r["in_scope"]),
            "Missing": len(r["missing"]),
            "Possible": len(r["possible"]),
            "Already": len(r["listed"]),
            "Out of Scope": len(r["out_scope"]),
            "Unknown Location": len(r["unknown"]),
        })

    st.subheader("Category Matrix Summary")
    st.dataframe(matrix_rows, use_container_width=True)
    totals = {
        "total_candidates": sum(x["SERP Found"] for x in matrix_rows),
        "in_scope": sum(x["In Scope"] for x in matrix_rows),
        "out_scope": sum(x["Out of Scope"] for x in matrix_rows),
        "unknown": sum(x["Unknown Location"] for x in matrix_rows),
    }
    a, b, c, d = st.columns(4)
    a.metric("Total candidates discovered", totals["total_candidates"])
    b.metric("In scope", totals["in_scope"])
    c.metric("Out of scope", totals["out_scope"])
    d.metric("Unknown location", totals["unknown"])

    for category in selected_categories:
        r = results_by_category[category]
        title = f"{category} — Missing {len(r['missing'])} / Possible {len(r['possible'])} / Listed {len(r['listed'])}"
        with st.expander(title, expanded=False):
            tabs = st.tabs(["Missing", "Possible Matches", "Already Listed", "Out of Scope", "Unknown Location", "Duplicates"])
            with tabs[0]:
                rows = _display_candidate_rows(r["missing"])
                st.dataframe(rows, use_container_width=True, height=320)
                st.download_button(f"Download {category} Missing CSV", data=rows_to_csv_bytes(rows), file_name=f"vacayrank_{category.lower().replace(' ', '_')}_missing.csv", mime="text/csv")
            with tabs[1]:
                rows = _display_candidate_rows(r["possible"])
                st.dataframe(rows, use_container_width=True, height=320)
                st.download_button(f"Download {category} Possible CSV", data=rows_to_csv_bytes(rows), file_name=f"vacayrank_{category.lower().replace(' ', '_')}_possible.csv", mime="text/csv")
            with tabs[2]:
                rows = _display_candidate_rows(r["listed"])
                st.dataframe(rows, use_container_width=True, height=320)
                st.download_button(f"Download {category} Already CSV", data=rows_to_csv_bytes(rows), file_name=f"vacayrank_{category.lower().replace(' ', '_')}_already.csv", mime="text/csv")
            with tabs[3]:
                rows = _display_candidate_rows(r["out_scope"])
                st.dataframe(rows, use_container_width=True, height=320)
                st.download_button(f"Download {category} Out-of-Scope CSV", data=rows_to_csv_bytes(rows), file_name=f"vacayrank_{category.lower().replace(' ', '_')}_out_of_scope.csv", mime="text/csv")
            with tabs[4]:
                rows = _display_candidate_rows(r["unknown"])
                st.dataframe(rows, use_container_width=True, height=320)
                st.download_button(f"Download {category} Unknown CSV", data=rows_to_csv_bytes(rows), file_name=f"vacayrank_{category.lower().replace(' ', '_')}_unknown_location.csv", mime="text/csv")
            with tabs[5]:
                clusters, members, scanned = r["duplicates"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Listing pages scanned", scanned)
                c2.metric("Clusters found", len(clusters))
                c3.metric("Total URLs clustered", len(members))
                if not clusters:
                    st.info("No duplicates found for this category.")
                else:
                    st.dataframe(clusters, use_container_width=True, height=260)
                    st.dataframe(members, use_container_width=True, height=260)
                st.download_button(f"Download {category} Duplicate Clusters CSV", data=rows_to_csv_bytes(clusters), file_name=f"vacayrank_{category.lower().replace(' ', '_')}_duplicate_clusters.csv", mime="text/csv")
                st.download_button(f"Download {category} Duplicate Members CSV", data=rows_to_csv_bytes(members), file_name=f"vacayrank_{category.lower().replace(' ', '_')}_duplicate_members.csv", mime="text/csv")

    st.subheader("Milestone 3 — Opportunities")
    self_check = st.button("Run Milestone 3 self-check")
    if self_check:
        st.dataframe(run_self_check(), use_container_width=True)

    include_possible = st.checkbox("Include Possible candidates", value=False)
    min_score = st.slider("Minimum opportunity score", min_value=0, max_value=100, value=35)
    exclude_unknown = st.checkbox("Exclude Unknown Location", value=True)
    exclude_oos = st.checkbox("Exclude Out of Scope", value=True)
    exclude_dupes = st.checkbox("Exclude Duplicates", value=True)
    category_filter = st.selectbox("Category filter", options=["All"] + selected_categories, index=0)

    cache_parts = {
        "destination": destination,
        "region": region,
        "country": country,
        "allowed_cities": sorted(list(allowed_cities)),
        "strict_city_match": bool(strict_city_match),
        "selected_categories": selected_categories,
        "category_config_hash": _stable_key(json.dumps(category_config, sort_keys=True)),
        "fuzzy_threshold": int(fuzzy_threshold),
        "cache_ttl_hours": float(cache_hours),
        "include_possible": bool(include_possible),
    }
    m3_payload = None if force_refresh else load_milestone3_disk_cache(cache_parts, float(cache_hours))
    if m3_payload is None:
        scored_rows, summary_rows, m3_debug = build_milestone3_scores(results_by_category, selected_categories, category_config, include_possible)
        save_milestone3_disk_cache(cache_parts, {"scored_rows": scored_rows, "summary_rows": summary_rows, "debug": m3_debug})
    else:
        scored_rows = m3_payload.get("scored_rows", [])
        summary_rows = m3_payload.get("summary_rows", [])
        m3_debug = m3_payload.get("debug", {})

    filtered_rows = []
    excluded_unknown = excluded_oos = excluded_dupes = 0
    for row in scored_rows:
        if category_filter != "All" and row["category"] != category_filter:
            continue
        if row["opportunity_score"] < min_score:
            continue
        if exclude_unknown and row.get("location_status") == "Unknown Location":
            excluded_unknown += 1
            continue
        if exclude_oos and row.get("location_status") == "Out of Scope":
            excluded_oos += 1
            continue
        if exclude_dupes and row.get("is_duplicate"):
            excluded_dupes += 1
            continue
        filtered_rows.append(row)

    display_cols = ["category", "candidate_name", "opportunity_score", "visibility_score", "demand_score", "gap_score", "confidence_score", "strategic_score", "penalties", "reason", "best_source_query", "serp_rank", "candidate_domain", "location_status"]
    st.write("Top Opportunities")
    st.dataframe([{k: row.get(k) for k in display_cols} for row in filtered_rows], use_container_width=True, height=320)
    st.download_button("Download CSV: top opportunities", data=rows_to_csv_bytes([{k: row.get(k) for k in display_cols} for row in filtered_rows]), file_name="vacayrank_milestone3_top_opportunities.csv", mime="text/csv")

    st.write("Per-Category Opportunity Summary")
    st.dataframe(summary_rows, use_container_width=True, height=220)
    st.download_button("Download CSV: per-category opportunity summary", data=rows_to_csv_bytes(summary_rows), file_name="vacayrank_milestone3_summary.csv", mime="text/csv")

    if show_debug_log:
        _append_debug(debug_log, f"Milestone3 scoring inputs summary rows={len(scored_rows)} include_possible={include_possible} category_filter={category_filter} min_score={min_score}")
        _append_debug(debug_log, f"Milestone3 score ranges min/mean/max={m3_debug.get('score_min', 0)}/{m3_debug.get('score_mean', 0)}/{m3_debug.get('score_max', 0)}")
        _append_debug(debug_log, f"Milestone3 excluded counts unknown={excluded_unknown} out_of_scope={excluded_oos} duplicates={excluded_dupes} missing_mapping_categories={missing_category_mapping_count}")
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
