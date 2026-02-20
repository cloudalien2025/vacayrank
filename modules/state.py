from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


@dataclass
class VrState:
    inventory_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory_meta: Dict[str, Any] = field(default_factory=dict)
    serp_gap_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    possible_matches_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicates_obj: List[Dict[str, Any]] = field(default_factory=list)
    write_queue: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {"missing": [], "possible_matches": [], "duplicates": []}
    )
    audit_log_entries: List[Dict[str, Any]] = field(default_factory=list)
    writes_committed_this_session: int = 0


def init_state() -> VrState:
    if "vr_state" not in st.session_state:
        st.session_state.vr_state = VrState()
    return st.session_state.vr_state
