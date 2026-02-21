# brains

`brains` is a standalone repository for the BD_Brain authoritative baseline, machine-readable schemas, export contract, and a Streamlit ingestion skeleton.

## Scope
- **Included:** BD_Brain baseline artifacts, governance docs, schema contracts, and ingestion tooling.
- **Excluded:** Any VacayRank application/business logic.

## Quickstart
```bash
cd Brains_Ingestion_App
pip install -r requirements.txt  # optional convenience if you add one
streamlit run app.py
```

## Layout
- `BD_Brain/` — human + machine baseline
- `Governance/` — standards and hard rules
- `Schemas/` — JSON Schema Draft 2020-12 contracts
- `Exports/` — downstream consumer export contract
- `Brains_Ingestion_App/` — executable ingestion MVP
