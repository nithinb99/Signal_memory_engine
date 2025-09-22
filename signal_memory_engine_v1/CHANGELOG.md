# Changelog — Signal Memory Engine

All notable changes introduced in **PR A — Core/API**.

## [v2.0.0] — 2025-09-21

### Added
- **Endpoints**
  - `GET /agents` — lists configured agents and whether each is enabled.
  - `POST /score` — lightweight trust score + flag for a query (no LLM call).
- **LLM invocation wrapper**
  - `utils/llm.invoke_chain` uses **`.invoke()` only** and normalizes outputs to a string.
  - Stable error mapping for upstream failures:
    - **429 / quota** → **503 Service Unavailable**
    - **timeouts** → **504 Gateway Timeout**
    - **connection / HTTP errors** → **502 Bad Gateway**
    - Unknowns → **500 Internal Server Error**
  - Pass-through of existing `HTTPException`.
- **Trace logging utilities**
  - `utils/tracing.py` with:
    - `trace_log(agent, query, flag, score, request_id)` — append a JSONL record to `trace.log` at the repo root.
    - `read_trace_tail(limit)` — read the last N records (used by `/memory_log`).

### Changed
- **Modular API structure**
  - Routes split and organized under `api/routes/`:
    - `query.py` → `POST /query`
    - `multi.py` → `POST /multi_query`
    - `memory.py` → `GET /memory_log`
    - `signal.py` → `POST /signal`, `GET /drift/{user_id}`
    - `search.py` → `GET /memory/search` (+ `GET /memory/vector_query` alias)
    - `agents.py` → `GET /agents`
- **Shared deps consolidated in `api/deps.py`**
  - Centralizes Pinecone init, embeddings, default QA chain, and agent stores.
  - Exposes `AGENTS`, `SUGGESTIONS`, `flag_from_score`, and `sanitize_key`.
  - Validates required env (`PINECONE_API_KEY`, `OPENAI_API_KEY`) at startup; fails fast with a clear message.
- **Event mapping**
- **Packaging (semantic, not Docker)**
  - **`setup.py` simplified** to only **direct runtime dependencies** with **`~=` pin ranges**.
  - Rationale: keep runtime deps lean; Docker lockfiles and image parity will be handled in a later PR.
- Running FastAPI server live reload
  - Run Uvicorn with explicit excludes so reload only happens on source changes (command included in README.md)
- Load dotenv in agent files

### Fixed
- **Multi-agent metrics**: remove reliance on an undefined `events` var; aggregate via `all_events` for cross-agent metrics.
- **Resilience**: per-agent failures (e.g., LLM 503) no longer take down the whole `/multi_query`; loop logs and continues.
- **Consistency**: imports normalized across `api/`, `agents/`, and `utils`.

### Removed
- **Duplicate `/query`** path that lived inside the old `memory.py` variant.  
  The canonical single-agent endpoint is `POST /query` in `api/routes/query.py`.
- **Inline trace logging in `api/main.py`** — replaced by centralized helpers in `utils/tracing.py`.

### Migration Notes
- **API consumers**: No breaking changes to existing public endpoints. New endpoints `/agents` and `/score` are additive. Expect more precise error codes:
  - 429/“rate limit” conditions now surface as **503**,
  - timeouts as **504**,
  - network/HTTP upstream issues as **502**.

### Semantic Versioning
- Bumped to **`2.0.0`**: new endpoints, new utils, updated core.py, stronger error handling, and packaging cleanup.
