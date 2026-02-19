# app.py — VacayRank v1 (Milestone 1: Read-Only Inventory Engine)
# Streamlit + Python + GitHub. Reads sitemaps, classifies URLs, shows counts, exports CSV, caches locally.
#
# How to run:
#   pip install -r requirements.txt  (see suggested requirements at bottom)
#   streamlit run app.py
#
# Notes:
# - No write operations. No BD API calls in this milestone.
# - Caching:
#   1) Streamlit in-memory cache (st.cache_data)
#   2) Local disk cache under ./vacayrank_cache/
#
# Suggested requirements.txt:
#   streamlit
#   requests

from __future__ import annotations

import csv
import hashlib
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


# -----------------------------
# Config & Constants
# -----------------------------

APP_TITLE = "VacayRank v1 — Milestone 1 (Read-Only Inventory Engine)"
CACHE_DIR = "vacayrank_cache"
DEFAULT_TIMEOUT_SECS = 20

XML_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    # Some WP plugins add extra namespaces; we only need the sitemap core schema.
}


# -----------------------------
# Data Models
# -----------------------------

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


# -----------------------------
# Disk Cache Utilities
# -----------------------------

def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _disk_cache_paths(sitemap_index_url: str) -> Tuple[str, str]:
    """
    Returns (json_path, csv_path) for a given sitemap index URL.
    """
    _ensure_cache_dir()
    key = _stable_key(sitemap_index_url.strip())
    return (
        os.path.join(CACHE_DIR, f"sitemap_inventory_{key}.json"),
        os.path.join(CACHE_DIR, f"sitemap_inventory_{key}.csv"),
    )


def load_disk_cache(sitemap_index_url: str, max_age_hours: float) -> Optional[Tuple[FetchStats, List[UrlRow]]]:
    json_path, _ = _disk_cache_paths(sitemap_index_url)
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    fetched_at = payload.get("stats", {}).get("fetched_at_utc")
    if not fetched_at:
        return None

    try:
        fetched_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except Exception:
        return None

    age_hours = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return None

    try:
        stats = FetchStats(**payload["stats"])
        rows = [UrlRow(**r) for r in payload["rows"]]
        return stats, rows
    except Exception:
        return None


def save_disk_cache(sitemap_index_url: str, stats: FetchStats, rows: List[UrlRow]) -> None:
    json_path, csv_path = _disk_cache_paths(sitemap_index_url)

    payload = {
        "stats": asdict(stats),
        "rows": [asdict(r) for r in rows],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Also save CSV for convenience (matches Streamlit export contents)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "url_type", "source_sitemap", "lastmod"])
        for r in rows:
            writer.writerow([r.url, r.url_type, r.source_sitemap, r.lastmod or ""])


def purge_disk_cache() -> int:
    _ensure_cache_dir()
    removed = 0
    for name in os.listdir(CACHE_DIR):
        if name.startswith("sitemap_inventory_") and (name.endswith(".json") or name.endswith(".csv")):
            try:
                os.remove(os.path.join(CACHE_DIR, name))
                removed += 1
            except Exception:
                pass
    return removed


# -----------------------------
# HTTP & XML Parsing
# -----------------------------

class FetchError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _alternate_host_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return None

    if host.startswith("www."):
        swapped_host = host[4:]
    else:
        swapped_host = f"www.{host}"

    netloc = swapped_host
    if parsed.port:
        netloc = f"{swapped_host}:{parsed.port}"
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
        raise FetchError(
            f"Unexpected XML root '{root_tag}' from {url} (HTTP {status_code}). First 200 chars: {snippet}"
        )
    return root_tag


def _append_debug(debug_log: Optional[List[str]], message: str) -> None:
    if debug_log is not None:
        debug_log.append(message)


def _fetch_once(session: requests.Session, url: str, timeout_secs: int, debug_log: Optional[List[str]]) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "application/xml,text/xml;q=0.9,text/html;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.vailvacay.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    retryable_statuses: Set[int] = {403, 429, 520, 521, 522, 523, 524}

    try:
        resp = session.get(url, timeout=timeout_secs, headers=headers, allow_redirects=True)
    except requests.RequestException as e:
        _append_debug(debug_log, f"{url} -> request error: {e}")
        raise FetchError(f"Request failed for {url}: {e}", retryable=True) from e

    content_len = len(resp.content or b"")
    _append_debug(debug_log, f"{url} -> HTTP {resp.status_code}, bytes={content_len}")

    if resp.status_code >= 500 or resp.status_code in retryable_statuses:
        raise FetchError(f"HTTP {resp.status_code} for {url}", retryable=True)
    if resp.status_code >= 400:
        raise FetchError(f"HTTP {resp.status_code} for {url}")

    resp.encoding = resp.encoding or "utf-8"
    text = resp.text
    root_tag = _validate_xml_response(url, resp.status_code, text)
    _append_debug(debug_log, f"{url} -> XML root={root_tag}")
    return text


def _retry_request(
    session: requests.Session,
    url: str,
    timeout_secs: int,
    max_attempts: int,
    backoffs: Tuple[float, ...],
    debug_log: Optional[List[str]],
) -> str:
    last_err: Optional[FetchError] = None
    for attempt in range(1, max_attempts + 1):
        try:
            _append_debug(debug_log, f"Attempt {attempt}/{max_attempts}: {url}")
            return _fetch_once(session, url, timeout_secs, debug_log)
        except FetchError as err:
            last_err = err
            if not err.retryable or attempt == max_attempts:
                break
            delay = backoffs[min(attempt - 1, len(backoffs) - 1)]
            _append_debug(debug_log, f"Retrying {url} in {delay:.1f}s because: {err}")
            time.sleep(delay)
    if last_err is None:
        raise FetchError(f"No request attempts were executed for {url}")
    raise last_err


def http_get_text(url: str, timeout_secs: int, debug_log: Optional[List[str]] = None) -> str:
    candidate_urls = [url]
    fallback_url = _alternate_host_url(url)
    if fallback_url and fallback_url not in candidate_urls:
        candidate_urls.append(fallback_url)

    backoffs = (0.5, 1.5, 3.5)
    max_attempts = 3
    last_error: Optional[FetchError] = None

    with requests.Session() as session:
        for candidate in candidate_urls:
            try:
                return _retry_request(session, candidate, timeout_secs, max_attempts, backoffs, debug_log)
            except FetchError as err:
                last_error = err
                _append_debug(debug_log, f"Candidate failed: {candidate} -> {err}")

    if last_error is None:
        raise FetchError(f"Unable to fetch sitemap from {url}")
    raise last_error


def parse_sitemap_index(xml_text: str) -> List[str]:
    """
    Parses a sitemap index and returns child sitemap URLs.
    Supports both <sitemapindex> and a plain <urlset> (in case user points directly to a child sitemap).
    """
    xml_text = xml_text.strip()
    try:
        root = ET.fromstring(xml_text)
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
                if _strip_ns(sm_el.tag).lower() != "sitemap":
                    continue
                for child in sm_el:
                    if _strip_ns(child.tag).lower() == "loc" and child.text:
                        sitemaps.append(child.text.strip())
                        break

        if not sitemaps:
            raise FetchError("Sitemap index parsed but contained 0 child sitemaps (loc entries).")

        return dedupe_preserve_order(sitemaps)

    # If it's a urlset, treat it as a "single sitemap" input (no children)
    if tag.endswith("urlset"):
        return []

    # Fallback: try loc elements generically
    locs = []
    for loc_el in root.findall(".//sm:loc", XML_NS) + root.findall(".//loc"):
        if loc_el is not None and loc_el.text:
            locs.append(loc_el.text.strip())
    return dedupe_preserve_order(locs)


def parse_urlset(xml_text: str, source_sitemap_url: str) -> List[UrlRow]:
    """
    Parses a child sitemap <urlset> and returns UrlRow list (url, lastmod).
    """
    xml_text = xml_text.strip()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise FetchError(f"Invalid XML in child sitemap: {e}") from e

    root_localname = _strip_ns(root.tag).lower()
    if root_localname != "urlset":
        # Some sites nest indexes; if so, parse as index and return rows empty here.
        return []

    rows: List[UrlRow] = []
    for el in root.iter():
        if _strip_ns(el.tag).lower() != "url":
            continue

        loc: Optional[str] = None
        lastmod: Optional[str] = None
        for child in el:
            child_localname = _strip_ns(child.tag).lower()
            if child_localname == "loc" and child.text and not loc:
                loc = child.text.strip()
            elif child_localname == "lastmod" and child.text and not lastmod:
                lastmod = child.text.strip()

        if not loc:
            continue

        rows.append(UrlRow(url=loc, url_type="unclassified", source_sitemap=source_sitemap_url, lastmod=lastmod))

    return rows


def _xml_root_localname(xml_text: str) -> str:
    root = ET.fromstring(xml_text.strip())
    return _strip_ns(root.tag).lower()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------------
# URL Classification
# -----------------------------

@dataclass(frozen=True)
class ClassifierConfig:
    # Path regexes; evaluated in order.
    profile_patterns: Tuple[str, ...]
    blog_patterns: Tuple[str, ...]
    category_patterns: Tuple[str, ...]
    search_patterns: Tuple[str, ...]
    static_patterns: Tuple[str, ...]


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
    ),
    blog_patterns=(
        r"^/blog(/|$)",
        r"^/posts?(/|$)",
        r"^/news(/|$)",
        r"^/\d{4}/\d{2}/\d{2}/",  # WP style
    ),
    category_patterns=(
        r"^/category(/|$)",
        r"^/tag(/|$)",
        r"^/topics?(/|$)",
        r"^/categories(/|$)",
    ),
    search_patterns=(
        r"^/search(/|$)",
        r"^/\?s=",
        r"^/(\w+/)?\?q=",
    ),
    static_patterns=(
        r"^/$",
        r"^/about(/|$)",
        r"^/contact(/|$)",
        r"^/privacy(-policy)?(/|$)",
        r"^/terms(-of-service)?(/|$)",
        r"^/sitemap(_index)?\.xml$",
        r"^/robots\.txt$",
    ),
)


def classify_url(url: str, cfg: ClassifierConfig) -> str:
    """
    Classify a URL into one of:
      profiles, blog_posts, categories, static, search, other
    """
    try:
        parsed = urlparse(url)
        path = parsed.path or "/"
        query = parsed.query or ""
        combined_for_search = path + ("?" + query if query else "")
    except Exception:
        # If parsing fails, do best-effort string checks
        path = url
        combined_for_search = url

    def _matches_any(patterns: Tuple[str, ...], target: str) -> bool:
        for pat in patterns:
            if re.search(pat, target, flags=re.IGNORECASE):
                return True
        return False

    if _matches_any(cfg.search_patterns, combined_for_search):
        return "search"

    if _matches_any(cfg.static_patterns, path):
        return "static"

    if _matches_any(cfg.category_patterns, path):
        return "categories"

    if _matches_any(cfg.blog_patterns, path):
        return "blog_posts"

    if _matches_any(cfg.profile_patterns, path):
        return "profiles"

    return "other"


def apply_classification(rows: List[UrlRow], cfg: ClassifierConfig) -> List[UrlRow]:
    out: List[UrlRow] = []
    for r in rows:
        out.append(
            UrlRow(
                url=r.url,
                url_type=classify_url(r.url, cfg),
                source_sitemap=r.source_sitemap,
                lastmod=r.lastmod,
            )
        )
    return out


# -----------------------------
# Inventory Engine
# -----------------------------

def _parse_manual_child_sitemap_urls(raw_text: str) -> List[str]:
    urls = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line:
            urls.append(line)
    return dedupe_preserve_order(urls)


@st.cache_data(show_spinner=False)
def _fetch_inventory_uncached(
    sitemap_index_url: str,
    timeout_secs: int,
    max_child_sitemaps: int,
    max_urls_per_child: int,
    manual_mode: bool,
    manual_index_xml: str,
    manual_child_urls_text: str,
    debug_enabled: bool,
) -> Tuple[List[str], List[UrlRow], List[str], Dict[str, str]]:
    """
    Fetch sitemap index, parse child sitemaps, fetch child urlsets, return (child_sitemaps, rows_unclassified).
    This function is wrapped by cache; disk-cache is handled outside.
    """
    debug_log: List[str] = []
    child_failures: Dict[str, str] = {}
    active_debug_log: Optional[List[str]] = debug_log if debug_enabled else None

    if manual_mode:
        manual_child_sitemaps = _parse_manual_child_sitemap_urls(manual_child_urls_text)
        if manual_child_sitemaps:
            child_sitemaps = manual_child_sitemaps
        else:
            if not manual_index_xml.strip():
                raise FetchError("Manual XML mode is enabled. Paste sitemap index XML or provide manual child sitemap URLs.")
            child_sitemaps = parse_sitemap_index(manual_index_xml)
            if not child_sitemaps:
                raise FetchError("Manual sitemap XML did not produce any child sitemap URLs.")
    else:
        xml = http_get_text(sitemap_index_url, timeout_secs=timeout_secs, debug_log=active_debug_log)
        child_sitemaps = parse_sitemap_index(xml)

        # If user gave a direct child sitemap (urlset), treat index as the only sitemap
        if not child_sitemaps:
            child_sitemaps = [sitemap_index_url]

    child_sitemaps = child_sitemaps[:max_child_sitemaps]

    rows: List[UrlRow] = []
    for sm_url in child_sitemaps:
        try:
            sm_xml = http_get_text(sm_url, timeout_secs=timeout_secs, debug_log=active_debug_log)
            child_failures[sm_url] = "OK"
        except FetchError as e:
            child_failures[sm_url] = str(e)
            _append_debug(active_debug_log, f"Child sitemap failed: {sm_url} -> {e}")
            continue

        # Some "child sitemaps" are themselves sitemap indexes (nested). Support one level of nesting.
        nested = parse_sitemap_index(sm_xml)
        if nested:
            nested = nested[:max_child_sitemaps]
            for nested_url in nested:
                try:
                    nested_xml = http_get_text(nested_url, timeout_secs=timeout_secs, debug_log=active_debug_log)
                    child_failures[nested_url] = "OK"
                except FetchError as e:
                    child_failures[nested_url] = str(e)
                    _append_debug(active_debug_log, f"Nested child sitemap failed: {nested_url} -> {e}")
                    continue
                nested_root = _xml_root_localname(nested_xml)
                nested_rows = parse_urlset(nested_xml, source_sitemap_url=nested_url)
                _append_debug(
                    active_debug_log,
                    f"Child sitemap parsed: url={nested_url}, root={nested_root}, urls_extracted={len(nested_rows)}",
                )
                if nested_root == "urlset" and not nested_rows:
                    snippet = nested_xml.strip().replace("\n", " ")[:200]
                    _append_debug(
                        active_debug_log,
                        f"WARNING: urlset sitemap returned 0 urls: {nested_url}; first_200_chars={snippet}",
                    )
                if max_urls_per_child > 0:
                    nested_rows = nested_rows[:max_urls_per_child]
                rows.extend(nested_rows)
            continue

        sm_root = _xml_root_localname(sm_xml)
        sm_rows = parse_urlset(sm_xml, source_sitemap_url=sm_url)
        _append_debug(
            active_debug_log,
            f"Child sitemap parsed: url={sm_url}, root={sm_root}, urls_extracted={len(sm_rows)}",
        )
        if sm_root == "urlset" and not sm_rows:
            snippet = sm_xml.strip().replace("\n", " ")[:200]
            _append_debug(
                active_debug_log,
                f"WARNING: urlset sitemap returned 0 urls: {sm_url}; first_200_chars={snippet}",
            )
        if max_urls_per_child > 0:
            sm_rows = sm_rows[:max_urls_per_child]
        rows.extend(sm_rows)

    return child_sitemaps, rows, debug_log, child_failures


def build_inventory(
    sitemap_index_url: str,
    classifier: ClassifierConfig,
    timeout_secs: int,
    max_child_sitemaps: int,
    max_urls_per_child: int,
    disk_cache_max_age_hours: float,
    force_refresh: bool,
    manual_mode: bool,
    manual_index_xml: str,
    manual_child_urls_text: str,
    debug_enabled: bool,
) -> Tuple[FetchStats, List[UrlRow], List[str], Dict[str, str]]:
    """
    Primary orchestration: uses disk cache unless forced, else fetches and saves to disk.
    """
    start = time.time()

    # Manual mode is always live input; skip disk cache read/write.
    if not manual_mode and not force_refresh:
        disk = load_disk_cache(sitemap_index_url, max_age_hours=disk_cache_max_age_hours)
        if disk is not None:
            stats, rows = disk
            # Re-classify with current patterns (so changes in patterns update without re-fetch)
            rows = apply_classification(rows, classifier)
            stats = FetchStats(
                sitemap_index_url=stats.sitemap_index_url,
                fetched_at_utc=stats.fetched_at_utc,
                child_sitemaps_found=stats.child_sitemaps_found,
                child_sitemaps_fetched=stats.child_sitemaps_fetched,
                urls_found=stats.urls_found,
                elapsed_seconds=round(time.time() - start, 3),
                cache_used=True,
            )
            return stats, rows, ["Loaded inventory from disk cache; no HTTP fetch attempts were made."], {}

    child_sitemaps, rows_unclassified, debug_log, child_failures = _fetch_inventory_uncached(
        sitemap_index_url=sitemap_index_url,
        timeout_secs=timeout_secs,
        max_child_sitemaps=max_child_sitemaps,
        max_urls_per_child=max_urls_per_child,
        manual_mode=manual_mode,
        manual_index_xml=manual_index_xml,
        manual_child_urls_text=manual_child_urls_text,
        debug_enabled=debug_enabled,
    )

    rows = apply_classification(rows_unclassified, classifier)

    stats = FetchStats(
        sitemap_index_url=sitemap_index_url,
        fetched_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        child_sitemaps_found=len(child_sitemaps),
        child_sitemaps_fetched=len(child_sitemaps),
        urls_found=len(rows),
        elapsed_seconds=round(time.time() - start, 3),
        cache_used=False,
    )

    if not manual_mode and rows:
        save_disk_cache(sitemap_index_url, stats, rows)
    return stats, rows, debug_log, child_failures


# -----------------------------
# CSV Export Helpers
# -----------------------------

def rows_to_csv_bytes(rows: List[UrlRow]) -> bytes:
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["url", "url_type", "source_sitemap", "lastmod"])
    for r in rows:
        writer.writerow([r.url, r.url_type, r.source_sitemap, r.lastmod or ""])
    return buf.getvalue().encode("utf-8")


def summarize_counts(rows: List[UrlRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.url_type] = counts.get(r.url_type, 0) + 1
    # ensure stable keys even if empty
    for k in ("profiles", "blog_posts", "categories", "static", "search", "other"):
        counts.setdefault(k, 0)
    return counts


# -----------------------------
# Streamlit UI
# -----------------------------

def _patterns_editor(title: str, patterns: Tuple[str, ...], help_text: str) -> Tuple[str, ...]:
    st.caption(help_text)
    text = st.text_area(title, value="\n".join(patterns), height=140)
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned.append(line)
    return tuple(cleaned)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write(
        "This milestone reads your sitemap(s), classifies URLs, shows inventory counts, and exports CSV. "
        "It performs **no write operations**."
    )

    with st.sidebar:
        st.header("Settings")

        sitemap_index_url = st.text_input(
            "Sitemap index URL",
            value="https://vailvacay.com/sitemap_index.xml",
            help="Point to your sitemap index. If you paste a single child sitemap (urlset), it still works.",
        ).strip()

        manual_mode = st.checkbox("Manual XML mode (paste sitemap index XML)", value=False)
        show_debug_log = st.checkbox("Show debug log", value=False)
        manual_index_xml = ""
        manual_child_urls_text = ""
        if manual_mode:
            manual_index_xml = st.text_area(
                "Manual sitemap index XML",
                value="",
                height=180,
                help="Paste sitemap index XML directly when origin blocks automated fetches.",
            )
            manual_child_urls_text = st.text_area(
                "Manual child sitemap URLs (one per line)",
                value="",
                height=120,
                help="Optional last-resort override: if provided, these URLs are used instead of sitemap URLs parsed from manual XML.",
            )

        timeout_secs = st.number_input(
            "HTTP timeout (seconds)",
            min_value=5,
            max_value=60,
            value=DEFAULT_TIMEOUT_SECS,
            step=1,
        )

        max_child_sitemaps = st.number_input(
            "Max child sitemaps to fetch",
            min_value=1,
            max_value=5000,
            value=200,
            step=10,
            help="Safety limit. Increase if your sitemap index is large.",
        )

        max_urls_per_child = st.number_input(
            "Max URLs per child sitemap (0 = no limit)",
            min_value=0,
            max_value=500000,
            value=0,
            step=1000,
            help="Safety limit for very large sitemaps. Use 0 to fetch all URLs.",
        )

        st.divider()
        st.subheader("Local disk cache")
        disk_cache_max_age_hours = st.number_input(
            "Use disk cache if newer than (hours)",
            min_value=0.0,
            max_value=720.0,
            value=12.0,
            step=1.0,
            help="If a cached run exists on disk newer than this age, it will load instantly.",
        )

        colA, colB = st.columns(2)
        with colA:
            force_refresh = st.checkbox("Force refresh (ignore disk cache)", value=False)
        with colB:
            if st.button("Purge disk cache"):
                removed = purge_disk_cache()
                st.success(f"Removed {removed} cached file(s).")

        st.divider()
        st.subheader("Classification patterns")
        st.caption(
            "Patterns are evaluated in this order: **search → static → categories → blog_posts → profiles → other**. "
            "Regex is supported."
        )

        profiles = _patterns_editor(
            "Profiles patterns (regex, one per line)",
            DEFAULT_CLASSIFIER.profile_patterns,
            "Example: /hotels, /restaurants, /activities",
        )
        blog_posts = _patterns_editor(
            "Blog patterns (regex, one per line)",
            DEFAULT_CLASSIFIER.blog_patterns,
            "Example: /blog or WordPress /YYYY/MM/DD/",
        )
        categories = _patterns_editor(
            "Category patterns (regex, one per line)",
            DEFAULT_CLASSIFIER.category_patterns,
            "Example: /category, /tag",
        )
        search = _patterns_editor(
            "Search patterns (regex, one per line)",
            DEFAULT_CLASSIFIER.search_patterns,
            "Example: /search or ?s= query strings",
        )
        static = _patterns_editor(
            "Static patterns (regex, one per line)",
            DEFAULT_CLASSIFIER.static_patterns,
            "Example: /about, /contact, /privacy-policy, /",
        )

        classifier = ClassifierConfig(
            profile_patterns=profiles,
            blog_patterns=blog_posts,
            category_patterns=categories,
            search_patterns=search,
            static_patterns=static,
        )

        st.divider()
        run = st.button("Run inventory scan", type="primary")

    if not sitemap_index_url:
        st.warning("Enter a sitemap index URL to begin.")
        return

    if not run:
        st.info("Configure settings on the left, then click **Run inventory scan**.")
        return

    # Execution
    try:
        with st.spinner("Fetching and parsing sitemaps..."):
            stats, rows, debug_log, child_failures = build_inventory(
                sitemap_index_url=sitemap_index_url,
                classifier=classifier,
                timeout_secs=int(timeout_secs),
                max_child_sitemaps=int(max_child_sitemaps),
                max_urls_per_child=int(max_urls_per_child),
                disk_cache_max_age_hours=float(disk_cache_max_age_hours),
                force_refresh=bool(force_refresh),
                manual_mode=bool(manual_mode),
                manual_index_xml=manual_index_xml,
                manual_child_urls_text=manual_child_urls_text,
                debug_enabled=bool(show_debug_log),
            )
    except FetchError as e:
        st.error(f"Fetch/parse error: {e}")
        if "vailvacay.com" in sitemap_index_url.lower() or "www.vailvacay.com" in sitemap_index_url.lower():
            st.warning("If this looks like bot/WAF blocking, use **Show debug log** to inspect attempts, then use **Manual XML mode** only as a last resort.")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    counts = summarize_counts(rows)

    if stats.urls_found == 0:
        st.warning(
            "0 URLs were discovered. The origin may be blocking automated fetches (bot/WAF behavior). "
            "See the debug diagnostics below for fetch attempts and failing child sitemap URLs."
        )
        if child_failures:
            failed_only = {k: v for k, v in child_failures.items() if v != "OK"}
            if failed_only:
                st.error("Child sitemap failures:")
                st.json(failed_only)

    if show_debug_log or stats.urls_found == 0:
        st.subheader("Debug log")
        if debug_log:
            st.code("\n".join(debug_log), language="text")
        else:
            st.caption("No debug lines were captured for this run.")

    # Top stats
    st.subheader("Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Profiles", counts["profiles"])
    c2.metric("Blog posts", counts["blog_posts"])
    c3.metric("Categories", counts["categories"])
    c4.metric("Static", counts["static"])
    c5.metric("Search", counts["search"])
    c6.metric("Other", counts["other"])

    st.caption(
        f"Sitemap: {stats.sitemap_index_url}  •  "
        f"Fetched at (UTC): {stats.fetched_at_utc}  •  "
        f"Child sitemaps fetched: {stats.child_sitemaps_fetched}  •  "
        f"URLs found: {stats.urls_found}  •  "
        f"Elapsed: {stats.elapsed_seconds}s  •  "
        f"Disk cache used: {'Yes' if stats.cache_used else 'No'}"
    )

    # Download export
    st.subheader("Export")
    csv_bytes = rows_to_csv_bytes(rows)
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="vacayrank_inventory.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Child sitemap list + details
    st.subheader("Details")
    tab1, tab2, tab3 = st.tabs(["URL Table", "By Type", "Cache Files"])

    with tab1:
        st.caption("All discovered URLs (classified). Use the table filters/sort.")
        st.dataframe(
            [{"url": r.url, "url_type": r.url_type, "source_sitemap": r.source_sitemap, "lastmod": r.lastmod} for r in rows],
            use_container_width=True,
            height=520,
        )

    with tab2:
        left, right = st.columns([1, 2])
        with left:
            st.caption("Counts by type")
            st.json(counts)
        with right:
            selected = st.selectbox("Show URLs for type", options=["profiles", "blog_posts", "categories", "static", "search", "other"])
            subset = [r for r in rows if r.url_type == selected]
            st.write(f"Found **{len(subset)}** URLs for **{selected}**.")
            st.dataframe(
                [{"url": r.url, "source_sitemap": r.source_sitemap, "lastmod": r.lastmod} for r in subset],
                use_container_width=True,
                height=520,
            )

    with tab3:
        _ensure_cache_dir()
        json_path, csv_path = _disk_cache_paths(sitemap_index_url)
        st.write("Disk cache locations for this sitemap index URL:")
        st.code(json_path)
        st.code(csv_path)
        st.caption("Tip: Commit **nothing** from vacayrank_cache to GitHub (add to .gitignore).")
        if manual_mode:
            st.info("Manual XML mode is enabled; disk cache read/write is skipped for this run.")

    # Lightweight recommendation block (non-placeholder, actionable)
    st.subheader("Next Step (Milestone 2 preview)")
    st.write(
        "Once inventory is stable, the next milestone typically adds **diffing** (today vs prior run), "
        "**per-type sampling/validation**, and **BD API mapping** (read-only) so we can prepare safe write operations later."
    )


if __name__ == "__main__":
    main()
