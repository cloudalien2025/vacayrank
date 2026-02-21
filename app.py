from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

from bd_copilot_client import (
    bd_get_user,
    bd_update_user,
    build_audit_entry,
    normalize_base_url as normalize_copilot_base_url,
    validate_base_url_self_check,
    validate_form_payload_self_check,
)
from bd_write_engine import SafeWriteEngine
from copilot_patch import compute_score, generate_patch_plan, identify_weaknesses
from identity_resolution_engine import resolve_identity
import listings_inventory_engine as lie
import listings_hydration_engine as lhe
import profiles
import ranking_engine
from inventory_engine import (
    InventoryBundle,
    build_canonical_member_set,
    cache_inventory_to_disk,
    fetch_inventory_index,
    inventory_to_csv,
    load_inventory_from_cache,
    load_inventory_progress,
)
from milestone3.bd_client import BDClient, normalize_base_url
import serp_gap_engine as sge
from structural_audit_engine import run_structural_audit

CACHE_PATH = "cache/inventory_index.json"
PROGRESS_PATH = "cache/inventory_progress.json"


def _missing_engine_fn(module_name: str, func_name: str):
    def _missing(*_args, **_kwargs):
        raise RuntimeError(f"{module_name}.{func_name} is unavailable in this deployment")

    return _missing


build_listings_via_scrape = getattr(lie, "build_listings_via_scrape", _missing_engine_fn("listings_inventory_engine", "build_listings_via_scrape"))
clear_listings_inventory_cache = getattr(lie, "clear_listings_inventory_cache", _missing_engine_fn("listings_inventory_engine", "clear_listings_inventory_cache"))
fetch_listings_via_api = getattr(lie, "fetch_listings_via_api", _missing_engine_fn("listings_inventory_engine", "fetch_listings_via_api"))
listings_to_csv = getattr(lie, "listings_to_csv", _missing_engine_fn("listings_inventory_engine", "listings_to_csv"))
load_audit_rows = getattr(lie, "load_audit_rows", _missing_engine_fn("listings_inventory_engine", "load_audit_rows"))
load_listings_inventory = getattr(lie, "load_listings_inventory", lambda: {"records": [], "selected_endpoint": None})
load_listings_progress = getattr(lie, "load_listings_progress", lambda: {"last_completed_page": 0, "total_pages": 0, "total_posts": 0})
normalize_endpoint = getattr(lie, "normalize_endpoint", lambda value: value)
probe_listings_endpoints = getattr(lie, "probe_listings_endpoints", _missing_engine_fn("listings_inventory_engine", "probe_listings_endpoints"))
run_serp_gap_analysis = getattr(sge, "run_serp_gap_analysis", _missing_engine_fn("serp_gap_engine", "run_serp_gap_analysis"))

st.set_page_config(page_title="VacayRank BD-Native", layout="wide")
st.title("VacayRank — BD-Native Milestones 1-4")

if "inventory_bundle" not in st.session_state:
    st.session_state.inventory_bundle = None
if "structural_audit" not in st.session_state:
    st.session_state.structural_audit = None
if "serp_rows" not in st.session_state:
    st.session_state.serp_rows = []
if "write_engine_log" not in st.session_state:
    st.session_state.write_engine_log = []
if "api_evidence" not in st.session_state:
    st.session_state.api_evidence = []
if "run_audit_log" not in st.session_state:
    st.session_state.run_audit_log = []
if "inventory_fetch_in_progress" not in st.session_state:
    st.session_state.inventory_fetch_in_progress = False
if "canonical_members" not in st.session_state:
    st.session_state.canonical_members = {}
if "listings_inventory" not in st.session_state:
    st.session_state.listings_inventory = load_listings_inventory()
if "listings_endpoint_decision" not in st.session_state:
    st.session_state.listings_endpoint_decision = None
if "listings_progress" not in st.session_state:
    st.session_state.listings_progress = load_listings_progress()
if "hydration_capability" not in st.session_state:
    st.session_state.hydration_capability = None
if "hydration_last_result" not in st.session_state:
    st.session_state.hydration_last_result = None
if "ranking_result" not in st.session_state:
    st.session_state.ranking_result = None
if "copilot_patch_plans" not in st.session_state:
    st.session_state.copilot_patch_plans = {}
if "copilot_audit_log" not in st.session_state:
    st.session_state.copilot_audit_log = []

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("BD Base URL", "https://www.vailvacay.com")
    api_key = st.text_input("BD API Key", type="password")
    serp_api_key = st.text_input("SerpAPI Key", type="password")
    dry_run = st.toggle("Dry Run", value=True)
    typed_confirm = st.text_input("Typed Confirm", value="")
    page_size = st.number_input("Inventory page size", min_value=1, max_value=200, value=100, step=1)
    inventory_max_pages = st.number_input("Inventory max pages", min_value=1, max_value=500, value=200, step=1)
    inventory_rpm = st.slider("Inventory Requests/Minute", min_value=5, max_value=120, value=30, step=1)
    max_pages_per_run = st.number_input("Max pages per run", min_value=1, max_value=int(inventory_max_pages), value=min(20, int(inventory_max_pages)), step=1)
    resume_mode = st.toggle("Resume from last page if progress exists", value=True)

normalized_base_url = normalize_base_url(base_url)
client = BDClient(base_url=normalized_base_url, api_key=api_key)
write_engine = SafeWriteEngine(client)

def inventory_fingerprint(base_url: str, limit: int) -> dict:
    return {
        "base_url": normalize_base_url(base_url),
        "endpoint": "/api/v2/user/search",
        "limit": int(limit),
        "output_type": "array",
    }


def sync_evidence() -> None:
    st.session_state.api_evidence = list(client.evidence_log)


def append_run_audit(action: str, records_fetched: int, pages_fetched: int, outcome: str) -> None:
    run_id = str(uuid.uuid4())
    evidence_path = f"session://api_evidence/{run_id}"
    st.session_state.run_audit_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "action": action,
            "records_fetched": records_fetched,
            "pages_fetched": pages_fetched,
            "status_outcome": outcome,
            "evidence_ref": evidence_path,
        }
    )


tab_inventory, tab_listings, tab_ranking, tab_copilot, tab_audit, tab_serp, tab_write, tab_log = st.tabs(
    [
        "Inventory (API Index)",
        "Listings Inventory (Discovery)",
        "Ranking (Milestone 3)",
        "BD Co-Pilot",
        "Structural Audit",
        "SERP Gap Analysis",
        "Write Queue",
        "Audit Log",
    ]
)

with tab_inventory:
    st.subheader("Milestone 1 — API Member Inventory")
    if st.button("Resolve BD Base URL"):
        resolved_base = client.resolve_base_url()
        if client.evidence_log:
            client.evidence_log[-1]["base_url"] = normalized_base_url
            client.evidence_log[-1]["resolved_base_url"] = resolved_base
        sync_evidence()
        st.info(f"Resolved base URL: {resolved_base}")

    st.caption(f"Final resolved base URL in use: {client.base_url}")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Fetch from /api/v2/user/search"):
        if st.session_state.inventory_fetch_in_progress:
            st.warning("Inventory fetch is already in progress. Please wait for it to finish.")
        else:
            st.session_state.inventory_fetch_in_progress = True
            try:
                st.session_state.canonical_members = {}
                progress = load_inventory_progress(PROGRESS_PATH)
                active_fingerprint = inventory_fingerprint(client.base_url, int(page_size))
                progress_fingerprint = progress.get("fingerprint", {}) if progress else {}

                start_page = 1
                if not resume_mode and progress:
                    st.info("Resume is disabled; starting from page 1.")
                elif progress and progress_fingerprint == active_fingerprint:
                    start_page = int(progress.get("last_page", 0)) + 1
                elif progress:
                    st.info("Progress fingerprint changed; resetting inventory fetch to page 1.")

                if start_page > 1 and Path(CACHE_PATH).exists():
                    cached_bundle = load_inventory_from_cache(CACHE_PATH)
                    known_total_pages = int((cached_bundle.meta or {}).get("total_pages", 0) or 0)
                    if known_total_pages and start_page > known_total_pages:
                        st.info("Saved resume page exceeds known total_pages; restarting from page 1.")
                        start_page = 1

                if start_page > 1:
                    st.info(f"Resuming inventory fetch from page {start_page}.")

                bundle = fetch_inventory_index(
                    client,
                    page_size=int(page_size),
                    max_pages=int(max_pages_per_run),
                    requests_per_minute=int(inventory_rpm),
                    start_page=start_page,
                    cache_path=CACHE_PATH,
                    progress_path=PROGRESS_PATH,
                )
                st.session_state.inventory_bundle = bundle
                st.session_state.canonical_members = build_canonical_member_set(bundle.records)
                cache_inventory_to_disk(
                    bundle,
                    CACHE_PATH,
                    base_url_used=client.base_url,
                    payload_defaults={"output_type": "array", "limit": int(page_size)},
                )
                pages_fetched = sum(1 for x in client.evidence_log if x.get("label") == "M1 Inventory Fetch")
                outcome = "partial" if bundle.status == "partial" else "success"
                append_run_audit("M1 Inventory Fetch", len(bundle.records), pages_fetched, outcome)
                if bundle.status == "partial" and (bundle.meta or {}).get("error_type") == "RATE_LIMIT":
                    st.error("Rate limited by BD API (HTTP 429). Cooldown and resume supported.")
                    st.write("Next actions: reduce rpm slider, reduce page batch size, wait and press Resume/Continue.")
                    st.info((bundle.meta or {}).get("message", "Rate limited; resume available."))
            except Exception as exc:
                append_run_audit("M1 Inventory Fetch", 0, 0, f"error: {exc}")
                st.error(str(exc))
            finally:
                st.session_state.inventory_fetch_in_progress = False
                sync_evidence()

    if col2.button("Single Page Test"):
        try:
            single_page_bundle = fetch_inventory_index(client, page_size=int(page_size), max_pages=1, requests_per_minute=int(inventory_rpm), start_page=1)
            parsed_count = len(single_page_bundle.records)
            append_run_audit("M1 Single Page Test", parsed_count, 1, "success")
            st.success(f"Single page parsed records: {parsed_count}")
        except Exception as exc:
            append_run_audit("M1 Single Page Test", 0, 1, f"error: {exc}")
            st.error(str(exc))
        finally:
            sync_evidence()

    if col3.button("Load cached inventory"):
        if Path(CACHE_PATH).exists():
            st.session_state.inventory_bundle = load_inventory_from_cache(CACHE_PATH)
            st.session_state.canonical_members = build_canonical_member_set(st.session_state.inventory_bundle.records)
        else:
            st.warning("No cache file found")

    if col4.button("Clear inventory"):
        st.session_state.inventory_bundle = None
        st.session_state.canonical_members = {}
        for target_path in (Path(CACHE_PATH), Path(PROGRESS_PATH)):
            if target_path.exists():
                target_path.unlink()

    bundle: InventoryBundle | None = st.session_state.inventory_bundle
    if bundle:
        if bundle.status == "partial":
            resume_page = (bundle.meta or {}).get("resume_from_page")
            st.warning(f"Rate limited; resume from page {resume_page} after cooldown" if resume_page else "Rate limited; resume supported")
        canonical_members = st.session_state.get("canonical_members") or build_canonical_member_set(bundle.records)
        st.session_state.canonical_members = canonical_members
        canonical_records = list(canonical_members.values())
        st.metric("Total members (API inventory)", len(canonical_members))
        run_meta = bundle.meta or {}
        st.caption(
            "New members added this run: "
            f"{run_meta.get('new_members_added', 0)} | "
            f"Duplicates skipped: {run_meta.get('duplicates_skipped', 0)} | "
            f"Pages fetched: {run_meta.get('pages_fetched', 0)} | "
            f"Stopped because: {run_meta.get('stopped_reason', 'n/a')}"
        )
        progress = load_inventory_progress(PROGRESS_PATH) if resume_mode else {}
        if resume_mode and int(progress.get("last_page", 0) or 0) > 1:
            st.caption(f"Resume fingerprint: {json.dumps(progress.get('fingerprint', {}), sort_keys=True)}")
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(bundle.summary.get("by_category", {}).items(), columns=["member_category", "count"]))
        c2.dataframe(pd.DataFrame(bundle.summary.get("by_geo", {}).items(), columns=["member_geo", "count"]))
        c3, c4 = st.columns(2)
        c3.dataframe(pd.DataFrame(bundle.summary.get("status_distribution", {}).items(), columns=["member_status", "count"]))
        c4.dataframe(pd.DataFrame(bundle.summary.get("membership_plan_distribution", {}).items(), columns=["member_plan", "count"]))
        st.download_button("Export Member Inventory CSV", data=inventory_to_csv(canonical_records), file_name="member_inventory_api_index.csv", mime="text/csv")

    st.markdown("### API Evidence Panel")
    if st.session_state.api_evidence:
        evidence_df = pd.DataFrame(st.session_state.api_evidence)
        st.dataframe(evidence_df)
        rate_limit_rows = [e for e in st.session_state.api_evidence if int(e.get("status_code", 0) or 0) == 429]
        if rate_limit_rows:
            first_429_time = rate_limit_rows[0].get("timestamp", "")
            total_429s = len(rate_limit_rows)
            suggested_rpm = max(20, min(30, int(inventory_rpm * 0.75)))
            st.markdown("#### Rate Limit Summary")
            s1, s2, s3 = st.columns(3)
            s1.metric("First 429 Time", first_429_time or "n/a")
            s2.metric("Total 429s", total_429s)
            s3.metric("Suggested RPM", suggested_rpm)
        st.json(st.session_state.api_evidence[-1])
    else:
        st.info("No API evidence yet.")


with tab_listings:
    st.subheader("Listings Inventory (Discovery)")
    seed_url = st.text_input("Seed URL", value=f"{client.base_url}/listings")
    listings_limit = st.number_input("Listings page size", min_value=1, max_value=200, value=10, step=1)
    listings_max_pages = st.number_input("Listings max pages per run", min_value=1, max_value=500, value=1, step=1)
    listings_rpm = st.slider("Listings Requests/Minute", min_value=5, max_value=120, value=30, step=1)
    listings_resume = st.toggle("Resume listings from last completed page", value=True)

    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("Probe API for Listings Endpoints"):
        decision = probe_listings_endpoints(client, dry_run=dry_run)
        st.session_state.listings_endpoint_decision = decision
        inventory = st.session_state.listings_inventory or {}
        inventory["selected_endpoint"] = normalize_endpoint(decision.selected_endpoint)
        st.session_state.listings_inventory = inventory
        if decision.selected_endpoint:
            st.success(f"Listings API usable. Selected endpoint: {decision.selected_endpoint}")
        else:
            st.error(
                "Probe failed. Endpoint/method/http/response details are in Evidence Panel. "
                "Next action: scrape fallback is allowed only after this probe failure."
            )

    endpoint = normalize_endpoint(st.session_state.listings_endpoint_decision.selected_endpoint) if st.session_state.listings_endpoint_decision else None

    if b2.button("Fetch Listings (API)", disabled=endpoint is None):
        if endpoint is None:
            st.warning("Run 'Probe API for Listings Endpoints' first.")
        else:
            try:
                progress = st.session_state.listings_progress if st.session_state.listings_progress else load_listings_progress()
                st.session_state.listings_progress = progress
                start_page = int(progress.get("last_completed_page", 0)) + 1 if listings_resume else 1
                existing = (st.session_state.listings_inventory or {}).get("records", []) if listings_resume else []
                result = fetch_listings_via_api(
                    client,
                    endpoint=endpoint,
                    page_limit=int(listings_limit),
                    max_pages=int(listings_max_pages),
                    rpm=int(listings_rpm),
                    start_page=start_page,
                    existing_records=existing,
                    dry_run=dry_run,
                )
                st.session_state.listings_inventory = {"records": result.records, "selected_endpoint": normalize_endpoint(endpoint)}
                st.session_state.listings_progress = {"last_completed_page": result.last_completed_page, "total_pages": result.total_pages, "total_posts": result.total_posts}
                st.success(
                    f"Listings fetch complete. last_page={result.last_completed_page}, total_pages={result.total_pages}, total_posts={result.total_posts}"
                )
            except Exception as exc:
                st.error(str(exc))

    probe_decision = st.session_state.listings_endpoint_decision
    scrape_allowed = probe_decision is not None and probe_decision.selected_endpoint is None
    if b3.button("Build Listings (Scrape Fallback)", disabled=not scrape_allowed):
        result = build_listings_via_scrape(client, seed_url=seed_url, max_pages=int(listings_max_pages), rpm=int(listings_rpm))
        st.session_state.listings_inventory = {"records": result.records, "selected_endpoint": None}

    if b4.button("Clear Listings Inventory"):
        clear_listings_inventory_cache()
        st.session_state.listings_inventory = {"records": [], "selected_endpoint": None}
        st.session_state.listings_endpoint_decision = None
        st.session_state.listings_progress = {"last_completed_page": 0, "total_pages": 0, "total_posts": 0}
        st.session_state.pop("resolved_base_url", None)


    listings_records = (st.session_state.listings_inventory or {}).get("records", [])
    if b5.download_button("Export Listings CSV", data=listings_to_csv(listings_records), file_name="listings_inventory.csv", mime="text/csv"):
        pass

    progress = st.session_state.listings_progress if st.session_state.listings_progress else load_listings_progress()
    st.metric("Total Listings (unique)", len(listings_records))
    st.caption(
        f"Last page fetched: {progress.get('last_completed_page', 0)} | Total pages: {progress.get('total_pages', 0)} | Total posts: {progress.get('total_posts', 0)}"
    )
    if listings_records:
        listings_df = pd.DataFrame(listings_records)
        st.dataframe(listings_df.head(25))
        cat_col, geo_col = st.columns(2)
        cat_col.dataframe(listings_df["category"].fillna("Unknown").value_counts().reset_index(name="count").rename(columns={"index": "category"}))
        listings_df["geo_key"] = listings_df[["city", "state", "country"]].fillna("").agg(", ".join, axis=1).str.strip(", ").replace("", "Unknown")
        geo_col.dataframe(listings_df["geo_key"].value_counts().reset_index(name="count").rename(columns={"index": "geo"}))

    st.markdown("### Endpoint Decision")
    decision = st.session_state.listings_endpoint_decision
    if decision:
        st.json({
            "selected_endpoint": decision.selected_endpoint,
            "why_accepted": decision.accepted_reason,
            "why_rejected": decision.rejected_reasons[-10:],
            "probe_details": decision.probe_details,
            "status": "Listings API usable" if decision.selected_endpoint else "Listings API not usable (scrape fallback allowed)",
        })
    else:
        st.info("No endpoint decision yet.")

    st.markdown("### Evidence Panel")
    audit_rows = load_audit_rows()
    if audit_rows:
        st.dataframe(pd.DataFrame(audit_rows))
    else:
        st.info("No listings evidence yet.")

    st.markdown("---")
    st.subheader("Hydrate Listings (API)")
    hydration_rpm = st.slider("Hydration Requests/Minute", min_value=1, max_value=120, value=30, step=1)
    hydration_max_per_run = st.number_input("Hydration max listings per run", min_value=1, max_value=500, value=25, step=1)
    hydration_resume = st.toggle("Resume hydration from first incomplete listing", value=True)
    h1, h2, h3, h4 = st.columns(4)
    worklist = lhe.build_worklist(listings_records)

    if h1.button("Probe GET-by-ID Capability"):
        if not worklist:
            st.warning("No listings inventory records found. Fetch Listings first.")
        elif not api_key:
            st.warning("API key required for capability probe.")
        else:
            probe_listing_id = worklist[0]
            st.session_state.hydration_capability = lhe.probe_get_by_id_capability(client, probe_listing_id)

    capability = st.session_state.hydration_capability or {}
    get_available = bool(capability.get("available"))
    endpoint_used = "GET /api/v2/users_portfolio_groups/get/{group_id}" if get_available else "POST /api/v2/users_portfolio_groups/search"

    if h2.button("Hydrate Listings (API)"):
        if not listings_records:
            st.warning("No listings inventory records found.")
        elif not api_key:
            st.warning("API key required for hydration.")
        else:
            result = lhe.hydrate_listings(
                client,
                listings_records,
                site_base=client.base_url,
                rpm=int(hydration_rpm),
                max_listings_per_run=int(hydration_max_per_run),
                resume=bool(hydration_resume),
                get_by_id_available=get_available,
            )
            st.session_state.hydration_last_result = {
                "processed": result.processed_count,
                "success": result.success_count,
                "failures": result.failures_count,
                "completed": result.completed_count,
                "last_listing_id": result.last_listing_id,
                "last_listing_name": result.last_listing_name,
                "last_errors": result.last_errors,
            }
            st.success(f"Hydration run complete: success={result.success_count}, failures={result.failures_count}")

    if h3.button("Clear Hydration Cache"):
        lhe.clear_hydration_cache()
        st.session_state.hydration_last_result = None

    hydrated_records = lhe.load_hydrated_records()
    features_rows = lhe.extract_features(hydrated_records)
    jsonl_export = lhe.export_hydrated_jsonl_text()
    csv_export = lhe.features_to_csv(features_rows)

    h4.download_button("Export Hydrated JSONL", data=jsonl_export, file_name="listings_hydrated.jsonl", mime="application/json")
    st.download_button("Export Features CSV", data=csv_export, file_name="listings_features.csv", mime="text/csv")

    st.markdown("### Hydration Capability")
    st.json(
        {
            "get_by_id_available": get_available,
            "hydration_endpoint_used": endpoint_used,
            "probe_result": capability,
            "proof_note": "GET /api/v2/users_portfolio_groups/get/{id} must return HTTP 200 and matching group_id to be considered available.",
        }
    )

    checkpoint = lhe.load_checkpoint()
    progress_cols = st.columns(3)
    progress_cols[0].metric("Hydrated completed / worklist", f"{checkpoint.get('completed_ids_count', 0)} / {len(worklist)}")
    progress_cols[1].metric("Hydration failures (checkpoint)", int(checkpoint.get("failures_count", 0) or 0))
    last_label = f"{checkpoint.get('last_completed_listing_id') or 'n/a'}"
    progress_cols[2].metric("Last hydrated listing_id", last_label)
    if st.session_state.hydration_last_result:
        st.caption(
            f"Last run: processed={st.session_state.hydration_last_result['processed']}, "
            f"success={st.session_state.hydration_last_result['success']}, "
            f"failures={st.session_state.hydration_last_result['failures']}"
        )
        if st.session_state.hydration_last_result.get("last_listing_id"):
            st.write(
                f"Last hydrated: {st.session_state.hydration_last_result.get('last_listing_id')} — "
                f"{st.session_state.hydration_last_result.get('last_listing_name') or ''}"
            )
        for err in st.session_state.hydration_last_result.get("last_errors", [])[-3:]:
            st.warning(err)

    if hydrated_records:
        st.markdown("### Hydrated Norm Preview")
        norm_preview = [r.get("norm", {}) for r in hydrated_records[:20]]
        st.dataframe(pd.DataFrame(norm_preview))
        st.markdown("### Features Preview")
        st.dataframe(pd.DataFrame(features_rows[:20]))


with tab_ranking:
    st.subheader("Ranking (Milestone 3)")
    profiles.ensure_profile_store()
    hydrated_path = Path("data/listings_hydrated.jsonl")
    if not hydrated_path.exists():
        st.info("Run Milestone 2 hydration first.")
    else:
        st.markdown("### Profile Manager")
        all_profiles = profiles.list_profiles()
        profile_ids = [p["profile_id"] for p in all_profiles]
        active_profile_id = profiles.get_active_profile_id()
        selected_profile_id = st.selectbox(
            "Select profile",
            options=profile_ids,
            index=profile_ids.index(active_profile_id) if active_profile_id in profile_ids else 0,
            key="ranking_selected_profile",
        )
        selected_profile = profiles.load_profile(selected_profile_id)
        selected_hash = profiles.compute_profile_hash(selected_profile)
        st.caption(f"Profile hash: `{selected_hash}` | Updated: {selected_profile.get('updated_at')}")

        with st.expander("Create / Edit Profile", expanded=False):
            new_profile_id = st.text_input("Profile ID", value=selected_profile["profile_id"], key="ranking_profile_id")
            new_profile_name = st.text_input("Profile Name", value=selected_profile["profile_name"], key="ranking_profile_name")
            st.markdown("**Weights (must total 1.0)**")
            weight_values = {}
            for key in profiles.REQUIRED_WEIGHTS:
                weight_values[key] = st.number_input(
                    f"weight_{key}",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    value=float(selected_profile["weights"][key]),
                    key=f"weight_{key}",
                )
            st.write(f"Weight sum: {sum(weight_values.values()):.6f}")
            st.markdown("**Parameters**")
            params = {}
            for key in profiles.REQUIRED_PARAMS:
                if key == "inactive_penalty":
                    params[key] = st.number_input(
                        key,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        value=float(selected_profile["params"][key]),
                        key=f"param_{key}",
                    )
                else:
                    params[key] = st.number_input(
                        key,
                        min_value=0,
                        step=1,
                        value=int(selected_profile["params"][key]),
                        key=f"param_{key}",
                    )

            csave, cactive = st.columns(2)
            if csave.button("Save Profile"):
                now = datetime.now(timezone.utc).isoformat()
                draft = {
                    "profile_id": new_profile_id.strip(),
                    "profile_name": new_profile_name.strip() or new_profile_id.strip(),
                    "created_at": selected_profile.get("created_at") or now,
                    "updated_at": now,
                    "schema_version": 1,
                    "weights": weight_values,
                    "params": params,
                }
                try:
                    profiles.save_profile(draft)
                    st.success(f"Saved profile {draft['profile_id']}")
                except Exception as exc:
                    st.error(f"Profile save failed: {exc}")
            if cactive.button("Set Active"):
                profiles.set_active_profile(selected_profile_id)
                st.success(f"Active profile set to {selected_profile_id}")

        st.markdown("### Ranking Runner")
        if st.button("Run Ranking"):
            try:
                run = ranking_engine.run_ranking(selected_profile, hydrated_path=hydrated_path)
                st.session_state.ranking_result = run
                st.success(f"Ranking run complete: {run['run_id']}")
            except Exception as exc:
                st.error(f"Ranking failed: {exc}")

        run = st.session_state.ranking_result
        if run and run.get("rows"):
            rows_df = pd.DataFrame(run["rows"])
            st.markdown("### Ranked Table")
            categories = sorted(set(rows_df["group_category"].fillna("Uncategorized")))
            category = st.selectbox("Category filter", options=["All"] + categories, key="rank_category")
            query = st.text_input("Search by listing name", value="", key="rank_search")
            filtered = rows_df.copy()
            if category != "All":
                filtered = filtered[filtered["group_category"] == category]
            if query.strip():
                filtered = filtered[filtered["group_name"].str.contains(query.strip(), case=False, na=False)]
            top_n = st.number_input("Top N", min_value=1, max_value=max(1, len(filtered)), value=min(20, len(filtered)), step=1)
            st.dataframe(filtered.head(int(top_n)))

            st.markdown("### Listing Explainer")
            listing_options = [str(x) for x in rows_df["listing_id"].tolist()]
            selected_listing = st.selectbox("Select listing_id", options=listing_options, key="rank_listing")
            selected_row = next((r for r in run["rows"] if str(r.get("listing_id")) == selected_listing), None)
            if selected_row:
                st.json(selected_row.get("explanation", {}))

            st.markdown("### Exports")
            meta = run.get("meta", {})
            csv_path = Path(meta.get("csv_path", ""))
            json_path = Path(meta.get("json_path", ""))
            profile_path = profiles.profile_path(selected_profile_id)
            if csv_path.exists():
                st.download_button("Download Ranked CSV", data=csv_path.read_text(encoding="utf-8"), file_name=csv_path.name, mime="text/csv")
            if json_path.exists():
                st.download_button("Download Ranked JSON", data=json_path.read_text(encoding="utf-8"), file_name=json_path.name, mime="application/json")
            if profile_path.exists():
                st.download_button("Download Profile JSON", data=profile_path.read_text(encoding="utf-8"), file_name=profile_path.name, mime="application/json")


with tab_copilot:
    st.subheader("BD Co-Pilot (Milestone 4)")
    hydrated_records = lhe.load_hydrated_records()
    if not hydrated_records:
        st.info("Hydrate listings first to use BD Co-Pilot.")
    else:
        profiles.ensure_profile_store()
        all_profiles = profiles.list_profiles()
        profile_ids = [p["profile_id"] for p in all_profiles]
        active_profile_id = profiles.get_active_profile_id()

        left_panel, main_panel = st.columns([1, 2])
        with left_panel:
            search_q = st.text_input("Search listing name/category", value="", key="copilot_search")
            listing_options = []
            listing_map = {}
            for record in hydrated_records:
                norm = record.get("norm") or {}
                listing_id = str(norm.get("listing_id") or norm.get("group_id") or record.get("listing_id") or "")
                group_name = str(norm.get("group_name") or "")
                category = str(norm.get("group_category") or norm.get("category") or "")
                label = f"{listing_id} · {group_name} · {category}"
                if search_q.strip() and search_q.lower() not in label.lower():
                    continue
                listing_options.append(label)
                listing_map[label] = record

            selected_label = st.selectbox("Listing", options=listing_options, key="copilot_listing") if listing_options else None
            selected_profile_id = st.selectbox(
                "Score Profile",
                options=profile_ids,
                index=profile_ids.index(active_profile_id) if active_profile_id in profile_ids else 0,
                key="copilot_profile",
            )
            selected_profile = profiles.load_profile(selected_profile_id)
            selected_profile_hash = profiles.compute_profile_hash(selected_profile)
            copilot_dry_run = st.toggle("Dry Run", value=True, key="copilot_dry_run")
            set_active_opt_in = st.checkbox("Set Active (group_status=1)", value=False, key="copilot_set_active")
            irreversible_ack = st.checkbox("Writes are irreversible", value=False, key="copilot_irreversible_ack")
            typed_live_confirm = st.text_input("Typed Confirm (UPDATE)", value="", key="copilot_typed_confirm")
            generate_clicked = st.button("Generate Patch Plan", key="copilot_generate")

            selected_record = listing_map.get(selected_label) if selected_label else None
            selected_norm = copy.deepcopy((selected_record or {}).get("norm") or {})
            if selected_norm:
                selected_norm["raw_user"] = copy.deepcopy((selected_record or {}).get("raw") or {})
                selected_norm["listing_snapshot"] = copy.deepcopy((selected_record or {}).get("raw") or {})
                selected_norm.setdefault("user_id", selected_norm.get("listing_id") or selected_norm.get("group_id"))
                selected_norm.setdefault("listing_id", selected_norm.get("group_id") or selected_norm.get("user_id"))

            plan_key = None
            if selected_norm:
                plan_key = f"{selected_norm.get('listing_id')}::{selected_profile_hash}"

            if generate_clicked and selected_norm:
                score_result = compute_score(selected_norm, selected_profile)
                plan = generate_patch_plan(selected_norm, score_result, selected_profile, set_active_opt_in=set_active_opt_in)
                st.session_state.copilot_patch_plans[plan_key] = {
                    "plan": plan,
                    "listing_norm": selected_norm,
                    "score_result": score_result,
                    "profile": selected_profile,
                }

            current_bundle = st.session_state.copilot_patch_plans.get(plan_key) if plan_key else None
            can_apply = bool(
                current_bundle
                and not copilot_dry_run
                and irreversible_ack
                and typed_live_confirm.strip() == "UPDATE"
                and current_bundle["plan"].get("safe_to_apply")
                and current_bundle["plan"].get("proposed_changes")
            )
            apply_clicked = st.button("Apply Patch to BD", key="copilot_apply", disabled=not can_apply)

            if apply_clicked and current_bundle:
                listing_norm = current_bundle["listing_norm"]
                plan = current_bundle["plan"]
                user_id = plan.get("target_user_id")
                if not api_key:
                    st.error("API key required for live update")
                else:
                    try:
                        normalized = normalize_copilot_base_url(base_url)
                        update_response = bd_update_user(
                            base_url=normalized,
                            api_key=api_key,
                            user_id=user_id,
                            changes_dict=plan["proposed_changes"],
                        )
                        verify_response = bd_get_user(normalized, api_key, user_id)
                        audit_entry = build_audit_entry(
                            str(listing_norm.get("listing_id")),
                            str(user_id),
                            request_fields={"user_id": user_id, **plan["proposed_changes"]},
                            response={
                                "ok": update_response.get("ok"),
                                "status_code": update_response.get("status_code"),
                                "json": {
                                    "update": update_response.get("json"),
                                    "verify": verify_response.get("json"),
                                },
                            },
                        )
                        st.session_state.copilot_audit_log.append(audit_entry)
                        if update_response.get("ok"):
                            st.success(f"BD update success (HTTP {update_response.get('status_code')})")
                        else:
                            st.error(f"BD update failed (HTTP {update_response.get('status_code')}): {update_response.get('text')}")
                    except Exception as exc:
                        failure_entry = build_audit_entry(
                            str(listing_norm.get("listing_id")),
                            str(user_id),
                            request_fields={"user_id": user_id, **plan["proposed_changes"]},
                            response={"ok": False, "status_code": 0, "json": {"error": str(exc)}},
                        )
                        st.session_state.copilot_audit_log.append(failure_entry)
                        st.error(str(exc))

        with main_panel:
            current_bundle = st.session_state.copilot_patch_plans.get(plan_key) if plan_key else None
            if not current_bundle:
                st.info("Select a listing and click Generate Patch Plan.")
            else:
                listing_norm = current_bundle["listing_norm"]
                score_result = current_bundle["score_result"]
                plan = current_bundle["plan"]
                profile = current_bundle["profile"]

                st.markdown("### Listing Snapshot")
                st.json(
                    {
                        "listing_id": listing_norm.get("listing_id"),
                        "user_id": listing_norm.get("user_id"),
                        "group_name": listing_norm.get("group_name"),
                        "group_category": listing_norm.get("group_category") or listing_norm.get("category"),
                        "current_total_score": score_result.total_score,
                        "component_breakdown": score_result.components,
                    }
                )

                st.markdown("### Weaknesses (weighted impact)")
                weakness_rows = identify_weaknesses(score_result, profile)
                st.dataframe(pd.DataFrame(weakness_rows))

                st.markdown("### Proposed Fixes")
                updated_changes = {}
                proposed_fixes = plan.get("patch_plan") or plan.get("proposed_fixes") or plan.get("proposed_changes", {})
                eligible_fields = set(plan.get("eligible_fields") or [])
                blocked_fields = set(plan.get("blocked_fields") or [])
                blocking_reasons = plan.get("blocking_reasons_by_field") or {}
                for field, proposed in proposed_fixes.items():
                    current_value = listing_norm.get(field)
                    rationale = plan.get("rationales", {}).get(field, {})
                    if field in eligible_fields:
                        eligible = True
                        reason = None
                    elif field in blocked_fields:
                        eligible = False
                        reason = blocking_reasons.get(field, "Field not in writable schema union")
                    else:
                        eligible = False
                        reason = "Eligibility missing from backend"
                    badge = "🟢 Will apply" if eligible else "🔴 Blocked"
                    st.markdown(f"**{field}** · {badge}")
                    st.caption(f"Current: {str(current_value)[:140]}")
                    if not eligible and reason:
                        st.caption(f"Block reason: {reason}")
                    edited_value = st.text_area(
                        f"Proposed value · {field}",
                        value=str(proposed),
                        key=f"copilot_edit_{plan_key}_{field}",
                        height=120,
                    )
                    if eligible:
                        updated_changes[field] = edited_value
                    st.write(
                        f"Rationale: {rationale.get('why', '')} | Evidence: {rationale.get('evidence', '')} | Expected lift: {rationale.get('expected_component_lift', '')}"
                    )

                plan["proposed_changes"] = updated_changes
                st.markdown("### Preview score after patch")
                simulated = copy.deepcopy(listing_norm)
                simulated.update(updated_changes)
                preview = compute_score(simulated, profile)
                st.json(
                    {
                        "before_total": score_result.total_score,
                        "after_total": preview.total_score,
                        "delta": round(preview.total_score - score_result.total_score, 3),
                        "before_components": score_result.components,
                        "after_components": preview.components,
                        "patch_plan": plan.get("patch_plan") or {},
                        "safe_to_apply": bool(plan.get("safe_to_apply")),
                        "eligible_fields": plan.get("eligible_fields") or [],
                        "blocked_fields": plan.get("blocked_fields") or [],
                        "blocking_reasons_by_field": plan.get("blocking_reasons_by_field") or {},
                        "validation_warnings": plan.get("validation_warnings"),
                        "advisory_notes": plan.get("advisory_notes"),
                    }
                )

                st.markdown("### Audit log (last 20 actions)")
                audit_rows = st.session_state.copilot_audit_log[-20:]
                if audit_rows:
                    st.dataframe(pd.DataFrame(audit_rows))
                    st.download_button(
                        "Export Co-Pilot Audit JSON",
                        data=json.dumps(audit_rows, indent=2),
                        file_name="bd_copilot_audit.json",
                        mime="application/json",
                    )
                else:
                    st.caption("No write attempts yet.")

                st.markdown("### Self-checks")
                url_check_ok, url_check_msg = validate_base_url_self_check()
                payload_check_ok, payload_check_msg = validate_form_payload_self_check()
                st.write(
                    {
                        "validate_base_url": {"ok": url_check_ok, "detail": url_check_msg},
                        "validate_form_payload": {"ok": payload_check_ok, "detail": payload_check_msg},
                    }
                )


with tab_audit:
    st.subheader("Milestone 2.5 — BD Structural Audit")
    override_text = st.text_area("Optional settings/category override JSON", value="{}")
    if st.button("Run Structural Audit"):
        bundle = st.session_state.inventory_bundle
        if not bundle:
            st.error("Fetch inventory first.")
        else:
            overrides = json.loads(override_text or "{}")
            st.session_state.structural_audit = run_structural_audit(client, bundle.records, overrides)

    audit = st.session_state.structural_audit
    if audit:
        st.metric("Risk Level", audit.risk_level)
        st.json(audit.report)

with tab_serp:
    st.subheader("Milestone 2 — SERP + Category/Geo Gap Engine")
    bundle = st.session_state.inventory_bundle
    if not bundle:
        st.info("Load member inventory index first.")
    else:
        default_pairs = [f"{k[0]}|||{k[1]}" for k in list(bundle.inventory_index.keys())[:30]]
        selected_pairs = st.multiselect("Category + Geo pairs", options=default_pairs, default=default_pairs[:5])
        allow_secondary = st.checkbox("Allow secondary categories", value=False)
        if st.button("Run SERP Gap Analysis"):
            pairs: List[Tuple[str, str]] = [tuple(pair.split("|||", 1)) for pair in selected_pairs]
            result = run_serp_gap_analysis(serp_api_key, bundle, pairs, allow_secondary_categories=allow_secondary)
            st.session_state.serp_rows = result.rows

        if st.session_state.serp_rows:
            serp_df = pd.DataFrame(st.session_state.serp_rows)
            for (category, geo), group in serp_df.groupby(["category", "geo"]):
                st.markdown(f"### {category} / {geo}")
                st.dataframe(group)

with tab_write:
    st.subheader("Milestone 3 — Identity Resolution + Safe Write Queue")
    audit = st.session_state.structural_audit
    if not audit:
        st.warning("Run structural audit before enabling writes.")
    elif audit.risk_level == "High":
        st.error("Writes blocked because structural audit risk is HIGH.")
    else:
        serp_rows = st.session_state.serp_rows
        bundle = st.session_state.inventory_bundle
        if not serp_rows or not bundle:
            st.info("Run inventory and SERP gap analysis first.")
        else:
            serp_df = pd.DataFrame(serp_rows)
            add_missing = serp_df[serp_df["status"] == "Missing"]
            possible = serp_df[serp_df["status"].isin(["Possible Match", "Category Misalignment"])]
            duplicates = serp_df[serp_df.duplicated(["website"], keep=False) & serp_df["website"].astype(bool)]

            add_tab, fix_tab, dup_tab = st.tabs(["Add Missing", "Fix Possible Matches", "Resolve Duplicates"])

            with add_tab:
                st.dataframe(add_missing)
                for idx, row in add_missing.iterrows():
                    inventory_rows = bundle.inventory_index.get((row["category"].lower(), row["geo"].lower()), [])
                    decision = resolve_identity(row.to_dict(), inventory_rows)
                    st.write(f"{row['business_name']} → suggested action: {decision.action} (score={decision.score})")
                    if decision.action == "CREATE" and st.button(f"Preview Create {idx}"):
                        payload = {
                            "email": f"{row['business_name'].lower().replace(' ', '-')}-{idx}@example.invalid",
                            "password": "TempPass#123456",
                            "subscription_id": 1,
                            "company": row["business_name"],
                            "address1": row.get("address", ""),
                            "website": row.get("website", ""),
                            "city": row.get("geo", ""),
                        }
                        result = write_engine.commit_create(payload, typed_confirm, dry_run)
                        st.json(result.__dict__)

            with fix_tab:
                st.dataframe(possible)
                for idx, row in possible.iterrows():
                    if not row.get("best_match_user_id"):
                        continue
                    if st.button(f"Preview Update {idx}"):
                        current = client.get_user(int(row["best_match_user_id"]))
                        current_data = current.json_data.get("data", current.json_data)
                        proposed = {
                            "company": row["business_name"],
                            "address1": row.get("address", ""),
                            "website": row.get("website", ""),
                        }
                        result = write_engine.commit_update(
                            user_id=int(row["best_match_user_id"]),
                            current=current_data,
                            proposed=proposed,
                            typed_confirm=typed_confirm,
                            dry_run=dry_run,
                        )
                        st.json(result.__dict__)

            with dup_tab:
                st.dataframe(duplicates)

            st.session_state.write_engine_log = write_engine.audit_log

with tab_log:
    st.subheader("Audit Log")
    st.markdown("#### Write Engine Audit")
    entries = st.session_state.write_engine_log
    st.dataframe(pd.DataFrame(entries))
    st.download_button("Download write audit JSON", data=json.dumps(entries, indent=2), file_name="write_audit_log.json", mime="application/json")

    st.markdown("#### API Run Audit")
    run_entries = st.session_state.run_audit_log
    st.dataframe(pd.DataFrame(run_entries))
    st.download_button("Download API run audit JSON", data=json.dumps(run_entries, indent=2), file_name="api_run_audit_log.json", mime="application/json")
