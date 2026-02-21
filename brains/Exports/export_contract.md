# BD_Brain Export Contract

## Export Artifact
- **Filename:** `bd_brain_export.json`

## Required Structure
```json
{
  "brain": "BD_Brain",
  "version": "vX.X",
  "generated_at": "2026-02-21T00:00:00Z",
  "records": [],
  "index": {
    "by_topic": {},
    "by_type": {},
    "by_status": {},
    "by_tag": {}
  }
}
```

## Rules
- Include records with `status` = `active` and `experimental`.
- Exclude `deprecated` records by default.
- Include `disputed` records only when explicitly requested by the export run configuration.
- Consumers **must version-pin** to a specific export version.
