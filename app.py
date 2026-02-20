from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

from bd_copilot import evaluate_rules, load_bd_core
from bd_write_engine import SafeWriteEngine
from identity_resolution_engine import resolve_identity
from inventory_engine import (
    InventoryBundle,
    build_canonical_member_set,
    cache_inventory_to_disk,
    fetch_inventory_index,
    inventory_to_csv,
    load_inventory_from_cache,
    load_inventory_progress,
)
from milestone3.bd_client import BDClient
from serp_gap_engine import run_serp_gap_analysis
from structural_audit_engine import run_structural_audit

CACHE_PATH = "cache/inventory_index.json"
PROGRESS_PATH = "cache/inventory_progress.json"

st.set_page_config(page_title="VacayRank BD-Native", layout="wide")
st.title("VacayRank — BD-Native Milestones 1-3")

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

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("BD Base URL", "https://example.com")
    api_key = st.text_input("BD API Key", type="password")
    serp_api_key = st.text_input("SerpAPI Key", type="password")
    dry_run = st.toggle("Dry Run", value=True)
    typed_confirm = st.text_input("Typed Confirm", value="")
    page_size = st.number_input("Inventory page size", min_value=1, max_value=200, value=100, step=1)
    inventory_max_pages = st.number_input("Inventory max pages", min_value=1, max_value=500, value=200, step=1)
    inventory_rpm = st.slider("Inventory Requests/Minute", min_value=5, max_value=120, value=30, step=1)
    max_pages_per_run = st.number_input("Max pages per run", min_value=1, max_value=int(inventory_max_pages), value=min(20, int(inventory_max_pages)), step=1)
    resume_mode = st.toggle("Resume from last page if progress exists", value=True)

normalized_base_url = base_url.strip().rstrip("/")
client = BDClient(base_url=normalized_base_url, api_key=api_key)
write_engine = SafeWriteEngine(client)

def inventory_fingerprint(base_url: str, limit: int) -> dict:
    return {
        "base_url": base_url.strip().rstrip("/"),
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


tab_inventory, tab_copilot, tab_audit, tab_serp, tab_write, tab_log = st.tabs(
    ["Inventory (API Index)", "BD Co-Pilot", "Structural Audit", "SERP Gap Analysis", "Write Queue", "Audit Log"]
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

with tab_copilot:
    st.subheader("BD Co-Pilot — Intelligence Schema v1")
    try:
        bd_core = load_bd_core()
        st.caption(
            f"Schema {bd_core['version'].get('schema_version')} · KB {bd_core['version'].get('kb_version')} · Updated {bd_core['version'].get('updated_at')}"
        )

        latest_evidence = st.session_state.api_evidence[-1] if st.session_state.api_evidence else {}
        if latest_evidence:
            findings = evaluate_rules(latest_evidence, bd_core["rules"], bd_core["playbooks"])
            grouped = {"blocker": [], "warn": [], "info": []}
            for finding in findings:
                grouped.setdefault(finding.get("severity", "info"), []).append(finding)

            for severity in ["blocker", "warn", "info"]:
                items = grouped.get(severity, [])
                if not items:
                    continue
                st.markdown(f"### {severity.upper()}")
                for item in items:
                    st.markdown(f"**{item['id']}** — {item['title']}")
                    if item.get("recommended_actions"):
                        st.write("Recommended actions:")
                        for action in item["recommended_actions"]:
                            st.markdown(f"- {action}")
                    playbook = item.get("playbook") or {}
                    if playbook.get("steps"):
                        st.write(f"Playbook: {playbook.get('title', playbook.get('id', 'N/A'))}")
                        for step in playbook["steps"]:
                            st.markdown(f"  - {step}")

            debug_bundle = {
                "evidence": latest_evidence,
                "findings": findings,
                "versions": bd_core["version"],
            }
            st.download_button(
                "Export Debug Bundle",
                data=json.dumps(debug_bundle, indent=2),
                file_name="bd_copilot_debug_bundle.json",
                mime="application/json",
            )
        else:
            st.info("No API evidence yet. Run Single Page Test or Inventory Fetch first.")
    except Exception as exc:
        st.error(f"Unable to load BD Co-Pilot files: {exc}")

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
