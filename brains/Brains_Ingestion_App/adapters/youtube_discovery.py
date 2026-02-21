from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta


def discover_videos(keyword: str, max_videos: int) -> list[dict]:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        now = datetime.now(timezone.utc)
        return [
            {
                "source_id": f"yt_mock_{idx+1}",
                "source_type": "youtube",
                "title": f"{keyword.title()} Strategy Session #{idx+1}",
                "channel": "Mock Channel",
                "url": f"https://youtube.com/watch?v=mock{idx+1}",
                "published_at": (now - timedelta(days=idx + 1)).isoformat(),
                "duration_seconds": 600 + idx * 30,
            }
            for idx in range(min(max_videos, 5))
        ]

    # API-backed discovery is intentionally left as a scaffold for MVP.
    return []
