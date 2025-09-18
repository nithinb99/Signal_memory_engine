# Changelog — Signal Memory Engine

All notable changes to this project are documented here.

---

## [Unreleased]
- Possible future changes:
  - Route versioning (`/api/v1/...`) for public endpoints.
  - Config centralization into a `config.py` with validated env vars.

  ---

## [v2.0.0] — 2025-09-11
### Added
- **Modular API structure**
  - Split large `main.py` into route modules:
    - `api/routes/query.py` — single-agent query (`/query`)
    - `api/routes/multi.py` — multi-agent fan-out (`/multi_query`)
    - `api/routes/memory.py` — memory log endpoint (`/memory_log`)
    - `api/routes/signal.py` — signal logging (`/signal`) and drift history (`/drift/{user_id}`)
    - `api/routes/search.py` — Pinecone memory search (`/memory/search`)
  - Centralized **schemas** into `api/models.py`.
  - Centralized **dependencies** (QA chain, embeddings, Pinecone init, agents) into `api/deps.py`.

- **New utils**
  - `utils/tracing.py` — JSONL trace logging and tail helper.
  - `utils.llm.py` — safe invocation wrapper with 429 → 503 handling.

- **Documentation & artifacts**
  - Added `.env.example` with all required env vars (`PINECONE_API_KEY`, `OPENAI_API_KEY`, `MLFLOW_EXPERIMENT_NAME`, `SME_DB_PATH`, `SME_TEST_MODE`, etc.).
  - Updated README with:
    - Environment variable documentation.
    - Expanded endpoint documentation with consistent format.
    - Updated project structure.
  - Generated **App Flow PDFs**:
    - `query_endpoint_flow.pdf` — request lifecycle for `/query`.
    - `ingestion_coherence_storage_flow.pdf` — pipeline stages for ingestion, coherence, storage.
  - Created mermaid diagrams

- **Lightweight Testing**
  - Added basic tests for each stage (ingestion, coherence, storage) to ensure end-to-end reliability.

- **Conda Compatibility**
  - Added environment.yml for conda package installation

- **MLflow Bootstrapping & Observability**
  - Falls back to a Windows-safe local `file://` tracking URI when `MLFLOW_TRACKING_URI` is unset.
  - Creates/uses a named experiment (`signal-memory-engine`) instead of relying on experiment ID `0`.
  - Logs useful params/metrics/artifacts:
    - Prompt, answer, latency.
    - Per-agent flags/scores.
    - Biometrics (`hrv`, `temp`, `blink_rate`, `gsr`).
    - Text artifacts (prompt, answer).
  - `sanitize_key()` used to ensure valid MLflow param names.

- **LLM Invocation & Error Semantics**
  - Replaced deprecated `.run(...)` with `.invoke(...)` via `_invoke_qa(...)`.
  - Translates OpenAI 429/quota errors into HTTP `503 Service Unavailable`.
  - Allows `HTTPException` to propagate unchanged.
  - Wraps unknown errors as HTTP `500 Internal Server Error`.

### Changed
- Improved logging with structured trace entries (`utils/tracing.py`).
- Consolidated duplicate memory/search endpoints into a consistent `GET /memory/search`.
- Environment variables normalized:
  - Added `SME_TEST_MODE` flag for lightweight test runs.
- MLflow experiment bootstrap now uses experiment name instead of numeric IDs.
- Moved heavy initialization (Pinecone, QA chains) into `deps.py` instead of importing in every route.
- Added dynamic backend URL loading depending on if application is started via docker or directly in console with uvicorn.
- Moved mlflow docker service setup to dockerfile.mlflow

### Fixed
- Pydantic v2 compatibility warnings.
- Consistent import style across modules (`api`, `agents`, `utils`).
- Logs and storage now handle JSON payloads safely (normalize stringified payloads).
- Autologging for LangChain, OpenAI, and transformers integrated with MLflow.
- Dependency mismatch