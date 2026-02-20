from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


BD_CORE_DIR = Path("bd_core")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bd_core() -> Dict[str, Any]:
    return {
        "schema": _read_json(BD_CORE_DIR / "bd_core_schema_v1.json"),
        "kb": _read_json(BD_CORE_DIR / "bd_core_kb_v1.json"),
        "rules": _read_json(BD_CORE_DIR / "bd_core_rules_v1.json"),
        "playbooks": _read_json(BD_CORE_DIR / "bd_core_playbooks_v1.json"),
        "version": _read_json(BD_CORE_DIR / "bd_core_VERSION.json"),
    }


def evaluate_rules(evidence: Dict[str, Any], rules_doc: Dict[str, Any], playbooks_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    rules = rules_doc.get("rules", [])
    playbooks = {item.get("id"): item for item in playbooks_doc.get("playbooks", [])}

    status_code = int(evidence.get("status_code", 0) or 0)
    snippet = str(evidence.get("response_text_snippet", "") or "")
    snippet_lower = snippet.lower()
    content_type = str(evidence.get("response_content_type", "") or "").lower()
    parse_summary = evidence.get("parse_result_summary", {}) or {}
    parse_errors = [str(x) for x in parse_summary.get("parse_errors", [])]
    records_parsed = int(parse_summary.get("records_parsed", 0) or 0)
    has_cloudflare = "cloudflare" in snippet_lower or "attention required" in snippet_lower
    has_non_human_ua = any(token in snippet_lower for token in ["python", "curl", "wget", "headless", "non-human browsers"])
    has_wrapper = str(evidence.get("bd_status", "")).lower() == "success" and records_parsed >= 0 and not parse_errors
    has_encoded = "%3a" in snippet_lower or "%2f" in snippet_lower

    for rule in rules:
        rid = rule.get("id")
        matched = False

        if rid == "CLOUDFLARE_BLOCK_DETECT":
            matched = status_code == 403 and has_cloudflare
        elif rid == "NON_HUMAN_BROWSER_RULE_BLOCK":
            matched = status_code == 403 and has_non_human_ua and has_cloudflare
        elif rid == "AUTH_FORBIDDEN_DETECT":
            matched = status_code in {401, 403} and not has_cloudflare
        elif rid == "BD_WRAPPER_JSON_DETECT":
            matched = str(evidence.get("bd_status", "")).lower() == "success" and isinstance(evidence.get("response_json_type"), str) and evidence.get("response_json_type") == "dict"
        elif rid == "BD_HTML_RESPONSE_DETECT":
            matched = status_code == 200 and "text/html" in content_type
        elif rid == "BASE_URL_REDIRECT_RISK":
            matched = bool(evidence.get("base_url") and evidence.get("resolved_base_url") and evidence.get("base_url") != evidence.get("resolved_base_url"))
        elif rid == "SILENT_EMPTY_GUARD":
            matched = records_parsed == 0 and len(parse_errors) == 0
        elif rid == "URLENCODED_FIELDS_PRESENT":
            matched = has_encoded
        elif rid == "RATE_LIMIT_DETECT":
            matched = status_code == 429 or "too many api requests" in snippet_lower
            if matched and int(evidence.get("attempt_no", 1) or 1) >= 3:
                rule = dict(rule)
                rule["severity"] = "blocker"

        if matched:
            playbook = playbooks.get(rule.get("playbook_ref"), {})
            findings.append(
                {
                    "id": rid,
                    "title": rule.get("title", rid),
                    "severity": rule.get("severity", "info"),
                    "classification": rule.get("classification", {}),
                    "recommended_actions": rule.get("recommended_actions", []),
                    "playbook": playbook,
                }
            )

    if has_wrapper:
        # lightweight signal for successful wrapper handling
        findings.append(
            {
                "id": "BD_WRAPPER_JSON_OK",
                "title": "Wrapper JSON parsed successfully",
                "severity": "info",
                "classification": {"signal_type": "PARSING_OK"},
                "recommended_actions": ["No action required; wrapper response normalization is active."],
                "playbook": {},
            }
        )
    return findings
