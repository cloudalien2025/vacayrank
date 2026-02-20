# BD Listing ID Mapping

## DOM IDs vs API IDs
- Listing cards returned by `/api/v2/users_portfolio_groups/search` expose `data-postid` in HTML.
- That `data-postid` is a DOM/UI identifier and is **not** guaranteed to be the `data_posts.post_id` used by `/api/v2/data_posts/get/{id}`.
- True API IDs must be resolved from a data-posts API record set (mapped by canonical slug first, then title fallback).

## Discovered endpoints (runtime discovery flow)
- Card HTML source: `POST /api/v2/users_portfolio_groups/search` (form-encoded).
- Data posts list/search endpoint: discovered by probing, in order:
  1. `POST /api/v2/data_posts/search`
  2. `POST /api/v2/data_post/search`
  3. `POST /api/v2/posts/search`
  4. `POST /api/v2/data_posts/list`
- True single-record read: `GET /api/v2/data_posts/get/{true_post_id}`.
- Update endpoint: `PUT /api/v2/data_posts/update` (form-encoded, patch-only fields + `post_id`).

## Mapping contract
Input:
- `data_id` (example `75`)
- `page`, `limit`

Output per listing:
- `slug` (`/listings/<slug>`)
- `title`
- `true_post_id` (API-valid ID)
- `dom_post_id` (for diagnostics only)
- `user_id`, `data_id`, `data_type`
- `mapping_key` (`slug`, `title`, or `unresolved`)

## Postman-ready requests

### 1) List/search posts for `data_id=75`
```
POST {{base_url}}/api/v2/data_posts/search
X-Api-Key: {{api_key}}
Content-Type: application/x-www-form-urlencoded

data_id=75&page=1&limit=25&output_type=array
```

### 2) Get single post by true ID
```
GET {{base_url}}/api/v2/data_posts/get/{{true_post_id}}
X-Api-Key: {{api_key}}
```

### 3) Update one field safely
```
PUT {{base_url}}/api/v2/data_posts/update
X-Api-Key: {{api_key}}
Content-Type: application/x-www-form-urlencoded

post_id={{true_post_id}}&short_description=Updated%20description
```

If response includes `Record cannot be updated`, treat as a hard failure and inspect permissions/field constraints for that record.
