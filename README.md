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

## Milestone 1 inventory safeguards

The `/api/v2/user/search` endpoint can return repeated member rows across pages. To keep inventory totals accurate, Milestone 1 now:
- Deduplicates by normalized `user_id` before counting/storing records.
- Stops pagination immediately when a page adds no new unique members.
- Stores a resume fingerprint (`base_url`, endpoint, page size, `output_type`) in progress state so stale cursors are not reused after config changes.

## Listings Inventory (Discovery)

The Streamlit app now includes a **Listings Inventory (Discovery)** tab with deterministic API discovery + scrape fallback.

### Workflow
1. **Probe API for Listings Endpoints**
   - Probes a constrained endpoint list (`classified`, `portfolio`, `users_portfolio_groups`, etc.).
   - Uses `X-Api-Key` only for auth.
   - Uses form-encoded POST bodies (`application/x-www-form-urlencoded`) with `limit/page/output_type` keys.
   - Accepts an endpoint only when a `200` response returns records with listing-like fields (`group_id`, `group_filename`, `group_name`, etc.).
2. **Fetch Listings (API)**
   - Enabled only after an endpoint is accepted.
   - Pages results with rate limiting.
   - Deduplicates listing rows by canonical key:
     - `group_id`, else `user_id:group_filename`, else stable SHA1 hash.
   - Detects bad pagination by aborting when first record key repeats for 3 consecutive pages.
3. **Build Listings (Scrape Fallback)**
   - If no endpoint is found, builds inventory from sitemap/listings page scraping.
   - Extracts `data-userid`, `data-postid` (group candidate), listing title, and category breadcrumbs.

### Storage and evidence
- Inventory cache: `cache/listings_inventory.json`
- Progress cache: `cache/listings_progress.json`
- Audit log (append + capped): `cache/audit_log.jsonl`
- UI includes endpoint decision details, total listings, category/geo counts, table preview, and CSV export.
