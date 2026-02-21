from __future__ import annotations

from brainpack.utils import now_iso


def extract_brain_core_records(keyword: str, transcripts: list[dict]) -> list[dict]:
    created_at = now_iso()
    return [
        {
            "id": f"bd-{keyword.lower().replace(' ', '-')}-rule-001",
            "brain": "BD_Brain",
            "version_introduced": "v1.0",
            "topic": "Guardrails",
            "type": "rule",
            "assertion": "Validate all generated records against schema before writing any pack outputs.",
            "confidence": 0.94,
            "status": "active",
            "evidence": [{"source_id": transcripts[0]["source_id"], "kind": "transcript"}] if transcripts else [{"source_id": "n/a", "kind": "placeholder"}],
            "created_at": created_at,
            "tags": ["validation", "pipeline"]
        },
        {
            "id": f"bd-{keyword.lower().replace(' ', '-')}-fact-001",
            "brain": "BD_Brain",
            "version_introduced": "v1.0",
            "topic": "SEO",
            "type": "fact",
            "assertion": "Keyword-specific source clustering improves repeatable extraction quality over random sampling.",
            "confidence": 0.77,
            "status": "experimental",
            "evidence": [{"source_id": transcripts[0]["source_id"], "kind": "transcript"}] if transcripts else [],
            "created_at": created_at,
            "tags": ["keyword", "clustering"]
        },
        {
            "id": f"bd-{keyword.lower().replace(' ', '-')}-tactic-001",
            "brain": "BD_Brain",
            "version_introduced": "v1.0",
            "topic": "Templates",
            "type": "tactic",
            "assertion": "Generate a compact additions markdown file that summarizes newly extracted assertions for rapid review.",
            "confidence": 0.84,
            "status": "active",
            "evidence": [{"source_id": transcripts[-1]["source_id"], "kind": "transcript"}] if transcripts else [],
            "created_at": created_at,
            "tags": ["review", "workflow"]
        }
    ]
