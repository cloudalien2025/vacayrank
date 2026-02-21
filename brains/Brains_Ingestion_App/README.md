# Brains Ingestion App (MVP)

Streamlit app for discovery-first ingestion and Brain Pack generation.

## Run
```bash
pip install streamlit jsonschema
streamlit run app.py
```

## MVP behavior
- Keyword + max videos + discovery-only controls
- Mocked YouTube discovery unless API key exists
- Dummy transcription placeholder
- Extraction generates 3–5 schema-valid `brain_core` records
- Validation is blocking: pack write stops on schema errors
