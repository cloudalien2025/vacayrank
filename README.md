# VacayRank

## Milestone 3 — BD API Write Engine

Run the Milestone 3 app:

```bash
streamlit run milestone3_app.py
```

### Required inputs / env
- **BD Base URL** (sidebar input)
- **BD API Key** (sidebar input, sent as `X-Api-Key`)
- Optional operator value for audit logs.

### Milestone 2 artifacts
Drop (or point to) files in sidebar fields:
- `inventory.csv`
- `serp_gap_results.csv` or `.json`
- `possible_matches.csv` or `.json`
- `duplicates.json` or `.csv`

Then click **Load Milestone 2 Artifacts**.

### Safety model
- **Dry Run Mode defaults ON** and blocks commits.
- **Typed CONFIRM** can be required before any commit.
- Writes are limited by **Max writes per session**.
- Updates send only `user_id` + changed fields via patch computation.
- All writes use `application/x-www-form-urlencoded` (never JSON).

### Audit logs
- Every preview + commit is logged to in-memory session and append-only disk JSONL:
  - `logs/m3_audit.jsonl`
- Audit tab supports download as JSON and CSV.

### Verification
- When auto-verify is enabled (default ON), each commit is verified by:
  - `GET /api/v2/user/get/{user_id}`
  - Expected patch fields are compared to actual values.
