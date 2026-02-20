from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from bd_write_engine import SafeWriteEngine
from identity_resolution_engine import resolve_identity
from inventory_engine import InventoryBundle, cache_inventory_to_disk, fetch_inventory_index, inventory_to_csv, load_inventory_from_cache
from milestone3.bd_client import BDClient, BDClientError
from serp_gap_engine import run_serp_gap_analysis
from structural_audit_engine import run_structural_audit

CACHE_PATH = "cache/inventory_index.json"

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

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("BD Base URL", "https://example.com")
    api_key = st.text_input("BD API Key", type="password")
    serp_api_key = st.text_input("SerpAPI Key", type="password")
    dry_run = st.toggle("Dry Run", value=True)
    typed_confirm = st.text_input("Typed Confirm", value="")
    page_size = st.number_input("Inventory page size", min_value=1, max_value=200, value=100, step=1)
    max_pages = st.number_input("Inventory max pages", min_value=1, max_value=500, value=200, step=1)

normalized_base_url = base_url.strip().rstrip("/")
client = BDClient(base_url=normalized_base_url, api_key=api_key)
write_engine = SafeWriteEngine(client)

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


tab_inventory, tab_audit, tab_serp, tab_write, tab_log = st.tabs(
    ["Inventory (API Index)", "Structural Audit", "SERP Gap Analysis", "Write Queue", "Audit Log"]
)

with tab_inventory:
    st.subheader("Milestone 1 — API Member Inventory")
    if st.button("Resolve BD Base URL"):
        resolved_base = client.resolve_base_url()
        sync_evidence()
        st.info(f"Resolved base URL: {resolved_base}")

    st.caption(f"Final resolved base URL in use: {client.base_url}")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Fetch from /api/v2/user/search"):
        try:
            bundle = fetch_inventory_index(client, page_size=int(page_size), max_pages=int(max_pages))
            st.session_state.inventory_bundle = bundle
            cache_inventory_to_disk(
                bundle,
                CACHE_PATH,
                base_url_used=client.base_url,
                payload_defaults={"output_type": "array", "limit": int(page_size)},
            )
            pages_fetched = sum(1 for x in client.evidence_log if x.get("label") == "M1 Inventory Fetch")
            append_run_audit("M1 Inventory Fetch", len(bundle.records), pages_fetched, "success")
        except Exception as exc:
            append_run_audit("M1 Inventory Fetch", 0, 0, f"error: {exc}")
            st.error(str(exc))
        finally:
            sync_evidence()

    if col2.button("Single Page Test"):
        try:
            payload = {"output_type": "array", "page": 1, "limit": int(page_size)}
            response = client.search_users(payload, label="M1 Single Page Test")
            content_type = (response.content_type or "").lower()
            if "text/html" in content_type or response.text.lstrip().startswith("<"):
                client.annotate_last_evidence(parse_error="HTML response detected; output_type=array missing or endpoint returned HTML")
                raise BDClientError("Single page test returned HTML. output_type=array may be missing or ignored.")
            if not response.ok:
                client.annotate_last_evidence(parse_error=f"HTTP {response.status_code}")
                raise BDClientError(f"Single page test failed with HTTP {response.status_code}")
            parsed = response.json_data if isinstance(response.json_data, list) else response.json_data.get("data", [])
            parsed_count = len(parsed) if isinstance(parsed, list) else 0
            client.annotate_last_evidence(records_parsed=parsed_count)
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
        else:
            st.warning("No cache file found")

    if col4.button("Clear inventory"):
        st.session_state.inventory_bundle = None

    bundle: InventoryBundle | None = st.session_state.inventory_bundle
    if bundle:
        st.metric("Total members (API inventory)", bundle.summary.get("total_members", 0))
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(bundle.summary.get("by_category", {}).items(), columns=["member_category", "count"]))
        c2.dataframe(pd.DataFrame(bundle.summary.get("by_geo", {}).items(), columns=["member_geo", "count"]))
        c3, c4 = st.columns(2)
        c3.dataframe(pd.DataFrame(bundle.summary.get("status_distribution", {}).items(), columns=["member_status", "count"]))
        c4.dataframe(pd.DataFrame(bundle.summary.get("membership_plan_distribution", {}).items(), columns=["member_plan", "count"]))
        st.download_button("Export Member Inventory CSV", data=inventory_to_csv(bundle.records), file_name="member_inventory_api_index.csv", mime="text/csv")

    st.markdown("### API Evidence Panel")
    if st.session_state.api_evidence:
        evidence_df = pd.DataFrame(st.session_state.api_evidence)
        st.dataframe(evidence_df)
        st.json(st.session_state.api_evidence[-1])
    else:
        st.info("No API evidence yet.")

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
