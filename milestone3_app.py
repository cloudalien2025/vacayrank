from __future__ import annotations

import getpass
import json
import secrets
import string
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from milestone3.audit_log import AuditLog
from milestone3.bd_client import BDClient
from milestone3.bd_search import build_search_payload, cache_search_pattern, pick_best_match
from milestone3.patch_engine import compute_patch, encode_form, human_diff_table
from milestone3.verifier import verify_user_fields
from milestone3.write_queue import build_queue_tables, enforce_max_writes_per_session, load_m2_artifacts, update_row_status


st.set_page_config(page_title="VacayRank Milestone 3 — BD Write Engine", layout="wide")
st.title("VacayRank Milestone 3 — Controlled BD Write Queue")

if "writes_done" not in st.session_state:
    st.session_state.writes_done = 0
if "queue_rows" not in st.session_state:
    st.session_state.queue_rows = []
if "audit" not in st.session_state:
    st.session_state.audit = AuditLog()

with st.sidebar:
    st.header("Global Controls")
    base_url = st.text_input("BD Base URL", value="https://example.com")
    api_key = st.text_input("BD API Key", type="password")
    operator = st.text_input("Operator", value=getpass.getuser())
    dry_run = st.toggle("Dry Run Mode", value=True)
    strict_city = st.toggle("Strict City Geo Filter", value=True)
    require_typed_confirm = st.toggle("Require typed CONFIRM", value=True)
    allow_clearing = st.toggle("Allow clearing fields", value=False)
    auto_verify = st.toggle("Auto-verify after write", value=True)
    max_writes = st.number_input("Max writes per session", min_value=1, max_value=500, value=10)

    st.subheader("Milestone 2 Artifact Paths")
    inventory_path = st.text_input("inventory.csv", value="inventory.csv")
    missing_path = st.text_input("serp_gap_results.csv|json", value="serp_gap_results.csv")
    possible_path = st.text_input("possible_matches.csv|json", value="possible_matches.csv")
    duplicates_path = st.text_input("duplicates.json|csv", value="duplicates.json")

    if st.button("Load Milestone 2 Artifacts"):
        artifacts = load_m2_artifacts(inventory_path, missing_path, possible_path, duplicates_path)
        st.session_state.queue_rows = build_queue_tables(artifacts)
        st.success(f"Loaded {len(st.session_state.queue_rows)} queue rows")

client = BDClient(base_url=base_url, api_key=api_key)
audit: AuditLog = st.session_state.audit


def _default_password(length: int = 16) -> str:
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(chars) for _ in range(length))


def _confirm_ui(row_id: str) -> bool:
    if require_typed_confirm:
        typed = st.text_input("Type CONFIRM to continue", key=f"confirm_{row_id}")
        return typed.strip() == "CONFIRM"
    return st.checkbox("I confirm this exact request payload", key=f"confirmcb_{row_id}")


def _log_preview(action_type: str, endpoint: str, payload: Dict[str, Any], row_id: str) -> None:
    audit.log_event(
        {
            "operator": operator,
            "action_type": action_type,
            "endpoint": endpoint,
            "headers_redacted": {"X-Api-Key": "***REDACTED***", "Content-Type": "application/x-www-form-urlencoded"},
            "payload_form_encoded": encode_form(payload),
            "http_status": None,
            "response_json": None,
            "user_id": payload.get("user_id"),
            "verification_status": "not_run",
            "verification_details": {},
            "source_row_id": row_id,
        }
    )


def _commit(action_type: str, endpoint: str, payload: Dict[str, Any], row: Dict[str, Any], request_fn):
    if not enforce_max_writes_per_session(st.session_state.writes_done, int(max_writes)):
        st.error("Max writes per session reached")
        return
    if dry_run:
        st.warning("Dry Run is ON; commit blocked")
        return
    if not _confirm_ui(row["row_id"]):
        st.info("Confirmation required")
        return

    resp = request_fn(payload)
    user_id = payload.get("user_id")
    if not user_id:
        user_id = resp.json_data.get("user_id") or resp.json_data.get("data", {}).get("user_id")

    verification_status = "not_run"
    verification_details: Dict[str, Any] = {}
    if auto_verify and user_id:
        passed, verification_details, _ = verify_user_fields(client, int(user_id), {k: v for k, v in payload.items() if k != "user_id"})
        verification_status = "passed" if passed else "failed"

    audit.log_event(
        {
            "operator": operator,
            "action_type": action_type,
            "endpoint": endpoint,
            "headers_redacted": {"X-Api-Key": "***REDACTED***", "Content-Type": "application/x-www-form-urlencoded"},
            "payload_form_encoded": encode_form(payload),
            "http_status": resp.status_code,
            "response_json": resp.json_data,
            "user_id": user_id,
            "verification_status": verification_status,
            "verification_details": verification_details,
            "source_row_id": row["row_id"],
        }
    )
    st.session_state.writes_done += 1
    update_row_status(st.session_state.queue_rows, row["row_id"], "verified" if verification_status == "passed" else "committed")
    st.success(f"Committed {action_type} for row {row['row_id']}")


def _extract_results(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("data", "users", "results"):
        if isinstance(resp_json.get(key), list):
            return resp_json[key]
    return []


tab_add, tab_fix, tab_dup, tab_log = st.tabs(["Add Missing", "Fix Possible Matches", "Resolve Duplicates", "Audit Log"])

with tab_add:
    missing_rows = [r for r in st.session_state.queue_rows if r["match_type"] == "missing"]
    st.write(f"Rows: {len(missing_rows)}")
    for row in missing_rows:
        with st.expander(f"{row['candidate_name']} ({row['category_key']})"):
            st.write({k: row.get(k) for k in ["candidate_address", "candidate_website", "match_score", "status"]})
            cfg = {
                "use_search_keyword": st.checkbox("Use search_keyword", value=True, key=f"kw_{row['row_id']}"),
                "use_website": st.checkbox("Use website", value=True, key=f"ws_{row['row_id']}"),
                "use_email": st.checkbox("Use email", value=False, key=f"em_{row['row_id']}"),
                "strict_city_geo_filter": strict_city,
            }
            if st.button("Preview Create", key=f"preview_create_{row['row_id']}"):
                cache_search_pattern(base_url, cfg)
                payload = build_search_payload(row, cfg)
                search_resp = client.search_users(payload)
                results = _extract_results(search_resp.json_data)
                best_id, best_score = pick_best_match(results, row)
                st.write("Search payload", payload)
                st.write("Search results", results[:5])
                if best_id and best_score >= 80:
                    st.warning(f"Possible existing listing found user_id={best_id} score={best_score:.1f}. Offer update instead of create.")
                create_payload = {
                    "email": row.get("email") or f"{row['candidate_name'].lower().replace(' ', '-')}-{row['city_key']}@example.invalid",
                    "password": _default_password(),
                    "subscription_id": row.get("subscription_id") or 1,
                    "company": row.get("candidate_name"),
                    "address1": row.get("candidate_address"),
                    "website": row.get("candidate_website"),
                    "phone_number": row.get("candidate_phone"),
                    "city": row.get("city_key"),
                    "send_email_notifications": 0,
                }
                create_payload = {k: v for k, v in create_payload.items() if str(v).strip()}
                st.code(f"POST /api/v2/user/create\n{encode_form(create_payload)}")
                _log_preview("preview_create", "/api/v2/user/create", create_payload, row["row_id"])
                if st.button("Commit Create", key=f"commit_create_{row['row_id']}"):
                    _commit("create", "/api/v2/user/create", create_payload, row, client.create_user)

with tab_fix:
    possible_rows = [r for r in st.session_state.queue_rows if r["match_type"] == "possible_match"]
    st.write(f"Rows: {len(possible_rows)}")
    for row in possible_rows:
        with st.expander(f"{row['candidate_name']} score={row['match_score']}"):
            user_id = int(row.get("best_match_user_id") or 0)
            st.write(f"best_match_user_id: {user_id}")
            if st.button("Preview Update", key=f"preview_update_{row['row_id']}"):
                if user_id <= 0:
                    st.error("No valid user_id")
                else:
                    current_resp = client.get_user(user_id)
                    current = current_resp.json_data.get("data", current_resp.json_data)
                    proposed = {
                        "company": row.get("candidate_name"),
                        "address1": row.get("candidate_address"),
                        "website": row.get("candidate_website"),
                        "phone_number": row.get("candidate_phone"),
                    }
                    patch = compute_patch(current, proposed, allow_clearing=allow_clearing)
                    payload = {"user_id": user_id, **patch}
                    st.dataframe(pd.DataFrame(human_diff_table(current, patch)))
                    st.code(f"PUT /api/v2/user/update\n{encode_form(payload)}")
                    _log_preview("preview_update", "/api/v2/user/update", payload, row["row_id"])
                    if st.button("Commit Update", key=f"commit_update_{row['row_id']}"):
                        _commit("update", "/api/v2/user/update", payload, row, client.update_user)

with tab_dup:
    dup_rows = [r for r in st.session_state.queue_rows if r["match_type"] == "duplicate"]
    st.write(f"Rows: {len(dup_rows)}")
    for row in dup_rows:
        with st.expander(f"Cluster {row['row_id']} - {row['candidate_name']}"):
            ids = row.get("bd_user_ids")
            if isinstance(ids, str):
                try:
                    ids = json.loads(ids)
                except Exception:
                    ids = [v.strip() for v in ids.split(",") if v.strip()]
            ids = [int(x) for x in (ids or [])]
            st.write("Cluster IDs", ids)
            if not ids:
                st.info("No user ids in cluster")
                continue
            canonical = st.selectbox("Canonical user_id", options=ids, key=f"canon_{row['row_id']}")
            deactivate_targets = [x for x in ids if x != canonical]
            st.write("Deactivate targets", deactivate_targets)
            for target in deactivate_targets:
                payload = {"user_id": target, "active": 4}
                st.code(f"PUT /api/v2/user/update\n{encode_form(payload)}")
                _log_preview("preview_deactivate", "/api/v2/user/update", payload, row["row_id"])
                if st.button(f"Commit Deactivate {target}", key=f"deact_{row['row_id']}_{target}"):
                    _commit("deactivate", "/api/v2/user/update", payload, row, client.update_user)

            st.caption("Delete endpoint exists but is intentionally not exposed for automatic actions.")

with tab_log:
    st.subheader("Audit Entries")
    st.dataframe(pd.DataFrame(audit.entries))
    st.download_button("Download JSON", data=audit.export_json(), file_name="m3_audit_log.json", mime="application/json")
    st.download_button("Download CSV", data=audit.export_csv(), file_name="m3_audit_log.csv", mime="text/csv")
