from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from listings_hydration_engine import load_hydrated_records
from profiles import compute_profile_hash

RANKINGS_DIR = Path("data/rankings")


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> Optional[float]:
    try:
        text = _as_str(value)
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _days_ago(ts: Any) -> Optional[int]:
    text = _as_str(ts)
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 86400))


def build_feature_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        norm = record.get("norm") if isinstance(record.get("norm"), dict) else {}
        images = record.get("images") if isinstance(record.get("images"), dict) else {}
        tags_list = norm.get("tags_list") if isinstance(norm.get("tags_list"), list) else []
        image_count = _as_int(images.get("image_count") if images else norm.get("image_count"))
        listing_id = _as_str(record.get("listing_id") or norm.get("listing_id") or norm.get("group_id") or record.get("group_id"))
        updated_days = _days_ago(norm.get("revision_timestamp"))
        source_timestamp = "revision_timestamp"
        if updated_days is None:
            updated_days = _days_ago(norm.get("date_updated"))
            source_timestamp = "date_updated"
        if updated_days is None:
            source_timestamp = "unknown"
        rows.append(
            {
                "listing_id": listing_id,
                "group_name": _as_str(norm.get("group_name")),
                "group_category": _as_str(norm.get("group_category")) or "Uncategorized",
                "desc_word_count": _as_int(norm.get("desc_word_count")),
                "image_count": image_count,
                "tag_count": len(tags_list),
                "lat": norm.get("lat"),
                "lon": norm.get("lon"),
                "updated_days_ago": updated_days,
                "updated_source": source_timestamp,
                "group_status": _as_str(norm.get("group_status")),
            }
        )
    return pd.DataFrame(rows)


def _component_tier(value: int, ok: int, good: int) -> float:
    if value >= good:
        return 1.0
    if value >= ok:
        return 0.6
    if value > 0:
        return 0.3
    return 0.0


def _geo_score(lat: Any, lon: Any) -> float:
    return 1.0 if _safe_float(lat) is not None and _safe_float(lon) is not None else 0.0


def _freshness_score(updated_days_ago: Optional[int], good: int, ok: int) -> float:
    if updated_days_ago is None:
        return 0.3
    if updated_days_ago <= good:
        return 1.0
    if updated_days_ago <= ok:
        return 0.6
    return 0.3


def score_dataframe(df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
    weights = profile["weights"]
    params = profile["params"]
    out = df.copy()

    out["description_score"] = out["desc_word_count"].apply(lambda v: _component_tier(_as_int(v), int(params["desc_ok_words"]), int(params["desc_good_words"])))
    out["media_score"] = out["image_count"].apply(lambda v: _component_tier(_as_int(v), int(params["image_ok"]), int(params["image_good"])))
    out["tags_score"] = out["tag_count"].apply(lambda v: _component_tier(_as_int(v), int(params["tag_ok"]), int(params["tag_good"])))
    out["geo_score"] = out.apply(lambda r: _geo_score(r.get("lat"), r.get("lon")), axis=1)
    out["freshness_score"] = out["updated_days_ago"].apply(lambda v: _freshness_score(v if pd.notna(v) else None, int(params["fresh_good_days"]), int(params["fresh_ok_days"])))
    out["status_score"] = out["group_status"].apply(lambda s: 1.0 if _as_str(s) == "1" else 0.0)

    out["total_0_1"] = (
        weights["description"] * out["description_score"]
        + weights["media"] * out["media_score"]
        + weights["tags"] * out["tags_score"]
        + weights["geo"] * out["geo_score"]
        + weights["freshness"] * out["freshness_score"]
        + weights["status"] * out["status_score"]
    )
    out["inactive_penalty_applied"] = out["group_status"].apply(lambda s: _as_str(s) != "1")
    out["total_score"] = (out["total_0_1"] * 100.0).round(3)
    out.loc[out["inactive_penalty_applied"], "total_score"] = (
        out.loc[out["inactive_penalty_applied"], "total_score"] * float(params["inactive_penalty"])
    ).round(3)

    out["updated_days_sort"] = out["updated_days_ago"].apply(lambda v: int(v) if pd.notna(v) else 10**9)
    out = out.sort_values(
        by=["total_score", "desc_word_count", "image_count", "tag_count", "updated_days_sort", "listing_id"],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["overall_rank"] = out.index + 1
    out["category_rank"] = out.groupby("group_category").cumcount() + 1
    return out


def build_explanation(row: Dict[str, Any], profile: Dict[str, Any], profile_hash: str) -> Dict[str, Any]:
    components = {
        "description": {
            "score": row["description_score"],
            "evidence": f"desc_word_count={row['desc_word_count']}",
        },
        "media": {
            "score": row["media_score"],
            "evidence": f"image_count={row['image_count']}",
        },
        "tags": {
            "score": row["tags_score"],
            "evidence": f"tag_count={row['tag_count']}",
        },
        "geo": {
            "score": row["geo_score"],
            "evidence": f"lat={row.get('lat')} lon={row.get('lon')}",
        },
        "freshness": {
            "score": row["freshness_score"],
            "evidence": f"updated_days_ago={row.get('updated_days_ago')} source={row.get('updated_source')}; unknown dates score as 0.3",
        },
        "status": {
            "score": row["status_score"],
            "evidence": f"group_status={row.get('group_status')}",
        },
    }
    strengths = [name for name, value in components.items() if float(value["score"]) >= 0.9]
    weaknesses = [name for name, value in components.items() if float(value["score"]) <= 0.35]
    note = "Inactive penalty applied" if bool(row.get("inactive_penalty_applied")) else "No status penalty"
    return {
        "profile_id": profile["profile_id"],
        "profile_hash": profile_hash,
        "total_score": row["total_score"],
        "components": components,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "status_note": note,
    }


def run_ranking(profile: Dict[str, Any], hydrated_path: Path = Path("data/listings_hydrated.jsonl")) -> Dict[str, Any]:
    records = load_hydrated_records(hydrated_path)
    frame = build_feature_frame(records)
    if frame.empty:
        return {"rows": [], "frame": frame, "run_id": None, "meta": {"error": "no_records"}}

    ranked = score_dataframe(frame, profile)
    profile_hash = compute_profile_hash(profile)
    ranked["profile_id"] = profile["profile_id"]
    ranked["profile_hash"] = profile_hash
    ranked["component_summary"] = ranked.apply(
        lambda r: f"d={r['description_score']:.1f}|m={r['media_score']:.1f}|t={r['tags_score']:.1f}|g={r['geo_score']:.1f}|f={r['freshness_score']:.1f}|s={r['status_score']:.1f}",
        axis=1,
    )

    rows = ranked.to_dict(orient="records")
    for row in rows:
        row["explanation"] = build_explanation(row, profile, profile_hash)

    run_prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_prefix}_{profile_hash[:8]}"
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RANKINGS_DIR / f"{run_id}_ranked.csv"
    json_path = RANKINGS_DIR / f"{run_id}_ranked.json"
    meta_path = RANKINGS_DIR / f"{run_id}_meta.json"

    pd.DataFrame(rows).drop(columns=["explanation"]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": profile["profile_id"],
        "profile_hash": profile_hash,
        "hydrated_path": str(hydrated_path),
        "record_count": len(rows),
        "artifacts": {
            "ranked_csv": str(csv_path),
            "ranked_json": str(json_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "rows": rows,
        "frame": pd.DataFrame(rows),
        "run_id": run_id,
        "meta": {**meta, "meta_path": str(meta_path), "csv_path": str(csv_path), "json_path": str(json_path)},
    }
