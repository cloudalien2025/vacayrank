from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import traceback

import streamlit as st

from adapters.youtube_discovery import discover_videos
from adapters.transcription import transcribe_source
from adapters.extraction import extract_brain_core_records
from brainpack.exporters import write_json, write_jsonl, write_sources_csv
from brainpack.utils import now_iso, slugify
from brainpack.validators import SchemaValidationError, build_validator, validate_or_raise


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "Schemas"
PACKS_ROOT = REPO_ROOT / "BD_Brain" / "Packs"


st.set_page_config(page_title="Brains Ingestion MVP", layout="wide")
st.title("Brains Ingestion MVP")

keyword = st.text_input("Keyword", placeholder="e.g., vacation rental SEO")
max_videos = st.number_input("Max videos", min_value=1, max_value=100, value=25)
discovery_only = st.toggle("Discovery only", value=False)

if "run_log" not in st.session_state:
    st.session_state["run_log"] = []

if st.button("Generate Brain Pack", type="primary"):
    st.session_state["run_log"] = []

    try:
        if not keyword.strip():
            raise ValueError("Keyword is required.")

        started_at = now_iso()
        videos = discover_videos(keyword, int(max_videos))
        st.session_state["discovery_results"] = videos
        st.session_state["run_log"].append(f"Discovered {len(videos)} sources")

        queue = videos if not discovery_only else []
        transcripts = []
        for source in queue:
            transcripts.append({"source_id": source["source_id"], "text": transcribe_source(source)})
        st.session_state["queue"] = queue

        records = [] if discovery_only else extract_brain_core_records(keyword, transcripts)
        st.session_state["records"] = records

        ended_at = now_iso()
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

        run_metadata = {
            "run_id": run_id,
            "keyword": keyword,
            "started_at": started_at,
            "ended_at": ended_at,
            "config": {"max_videos": int(max_videos), "discovery_only": discovery_only},
            "errors": [],
            "env": {"mode": "mocked"},
        }

        pack_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pack_slug = slugify(keyword)
        pack_dir = PACKS_ROOT / f"{pack_date}__{pack_slug}__pack_v1"

        outputs = {
            "Brain_Additions.md": "Brain_Additions.md",
            "Brain_Core.jsonl": "Brain_Core.jsonl",
            "Brain_Diff.md": "Brain_Diff.md",
            "Sources.csv": "Sources.csv",
            "Run_Metadata.json": "Run_Metadata.json",
        }

        pack_metadata = {
            "pack_id": pack_dir.name,
            "brain": "BD_Brain",
            "keyword": keyword,
            "created_at": ended_at,
            "sources": videos,
            "outputs": outputs,
            "stats": {
                "discovered": len(videos),
                "processed": len(queue),
                "records_generated": len(records),
                "errors": 0,
            },
        }

        # Validate all payloads before writing anything.
        core_validator = build_validator(SCHEMAS_DIR, "brain_core.schema.json")
        source_validator = build_validator(SCHEMAS_DIR, "sources.schema.json")
        run_validator = build_validator(SCHEMAS_DIR, "run_metadata.schema.json")
        pack_validator = build_validator(SCHEMAS_DIR, "brain_pack.schema.json")

        for source in videos:
            validate_or_raise(source, source_validator, "source")
        for record in records:
            validate_or_raise(record, core_validator, "brain_core record")
        validate_or_raise(run_metadata, run_validator, "run_metadata")
        validate_or_raise(pack_metadata, pack_validator, "brain_pack")

        pack_dir.mkdir(parents=True, exist_ok=True)

        additions_md = pack_dir / "Brain_Additions.md"
        additions_md.write_text(
            "# Brain Additions\n\n" + "\n".join(f"- {r['id']}: {r['assertion']}" for r in records),
            encoding="utf-8",
        )
        (pack_dir / "Brain_Diff.md").write_text("# Brain Diff\n\nInitial diff scaffold.", encoding="utf-8")
        write_jsonl(pack_dir / "Brain_Core.jsonl", records)
        write_sources_csv(pack_dir / "Sources.csv", videos)
        write_json(pack_dir / "Run_Metadata.json", run_metadata)
        write_json(pack_dir / "Pack_Metadata.json", pack_metadata)

        st.success(f"Brain Pack generated: {pack_dir}")
        st.session_state["run_log"].append("Validation passed; outputs written")

    except (SchemaValidationError, ValueError) as exc:
        st.error(str(exc))
        st.session_state["run_log"].append(f"Error: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
        st.session_state["run_log"].append(traceback.format_exc())


with st.expander("Discovery results", expanded=True):
    st.json(st.session_state.get("discovery_results", []))

with st.expander("Queue"):
    st.json(st.session_state.get("queue", []))

with st.expander("Run log"):
    st.code("\n".join(st.session_state.get("run_log", [])) or "No run yet")

with st.expander("Outputs"):
    st.json(st.session_state.get("records", []))
