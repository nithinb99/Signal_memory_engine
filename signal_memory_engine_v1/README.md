# Signal Memory Engine v1

A conversational Retrieval-Augmented Generation (RAG) microservice powered by FastAPI, OpenAI, Pinecone, and LangChain. Features multi-agent handoff logic, biometric signal context, event-to-memory mapping, and a Streamlit UI for visualization.

---

## Features

* **Single-Agent RAG**: Query a memory index via the `/query` endpoint, using a RetrievalQA chain with OpenAI’s GPT models and Pinecone vector store.
* **Multi-Agent Fan-Out**: Route queries to three specialized agents—Axis™ Relationship Architect, Oria™ HR Oracle, and M™ Shadow Sentinel—via the `/multi_query` endpoint. Each agent returns an answer, top memory chunks, a stability flag, and a suggestion.
* **Biometric Context**: Simulate HRV, temperature, and blink rate readings and inject them into queries to enrich system context.
* **Event-to-Memory Mapping**: Normalize raw RAG hits into structured events (ID, content, score, timestamp, metadata) via Coherence Commons.
* **Agent-to-Agent Handoff**: Automatically escalate high-severity flags by notifying a human-in-the-loop endpoint.
* **Signal Logging**: Persist signals and drift events to SQLite with `/signal` and query user drift history with `/drift/{user_id}`.
* **Memory Log**: Tail recent trace logs via `/memory_log`.
* **Search API**: Query Pinecone directly with `/memory/search`.
* **Streamlit UI**: Interactive frontend (`streamlit_app.py`) that shows answers, memory chunks, flags, suggestions, and drift visualizations per agent.

---

## 📘 Documentation

* [Query Endpoint Flow (PDF)](./docs/query_endpoint_flow.pdf) — step-by-step walkthrough of how a request to `/query` is processed, including biometrics, retrieval, LLM, logging, and response.
* [Ingestion, Coherence, and Storage Flow (PDF)](./docs/ingestion_coherence_storage_flow.pdf) — overview of how raw inputs are ingested, normalized into coherent events, and persisted in storage.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Configuration](#configuration)  
4. [Running the API](#running-the-api)  
5. [Running with Docker](#running-with-docker)  
6. [Running with core.py](#running-with-corepy) 
7. [Endpoints](#endpoints)  
8. [Pipeline Diagram](#pipeline-diagram)  
9. [Streamlit UI](#streamlit-ui)  
10. [Project Structure](#project-structure)

---

## Prerequisites

* Python 3.9+  
* Pinecone account (API key & environment)  
* OpenAI account (API key)  

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/signal_memory_engine_v1.git
cd signal_memory_engine_v1

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```ini
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_ENV=<your-pinecone-environment>          # e.g. us-west1-gcp
PINECONE_INDEX=<your-pinecone-index-name>
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=<your-openai-model>                  # e.g. gpt-4o-mini
MLFLOW_TRACKING_URI=                              # optional; falls back to ./mlruns
MLFLOW_EXPERIMENT_NAME=<your-experiment-name>
ENABLED_AGENTS=<your-agents>                      # e.g. AXIS,ORIA,SENTINEL
SME_DB_PATH=<your-SME_DB_PATH>
SME_TEST_MODE=<0-or-1>                            # optional test switch; default 0 (off)
```

---

## Running the API

```bash
# Start FastAPI server with live reload
uvicorn api.main:app --reload
```

The API will run at `http://127.0.0.1:8000`.

---

## Running with Docker

> Ensure your `.env` is present in the repo root (see [Configuration](#configuration)).

### Using Docker Compose

```yaml
# docker-compose.yml (example)
services:
  api:
    build: .
    env_file: .env
    ports: ["8000:8000"]
    volumes:
      - ./mlruns:/app/mlruns
      - ./data:/app/data
    command: >
      uvicorn signal_memory_engine_v1.api.main:app
      --host 0.0.0.0 --port 8000 --log-level debug
```

Run:

```bash
docker compose up --build
```

> Tip: set `SME_DB_PATH=/app/data/signal.db` in `.env` so SQLite persists to the mounted volume.

---

## Running with `core.py`

`core.py` builds a **RetrievalQA** chain and **Pinecone** vectorstore **without** the FastAPI server—useful for quick experiments.

> Ensure `.env` is populated (OpenAI + Pinecone vars).  

Run it:

```bash
python core.py
```

---

## Endpoints

### Single-Agent RAG

**POST** `/query`

**Request Body**:

```json
{
  "query": "Your question here",
  "k": 3
}
```

**Response**:

```json
{
  "answer": "...",
  "chunks": [
    {"content": "...", "score": 0.72},
    ...
  ],
  "flag": "stable",    
  "suggestion": "No action needed."
}
```

### Multi-Agent Fan-Out

**POST** `/multi_query`

**Request Body**:

```json
{
  "query": "Your question here",
  "k": 3
}
```

**Response**:

```json
{
  "agents": {
    "Axis™ Relationship Architect": { /* AgentResponse */ },
    "Oria™ HR Oracle":           { /* AgentResponse */ },
    "M™ Shadow Sentinel":         { /* AgentResponse */ }
  }
}
```

Each `AgentResponse` matches the single-agent response schema.

### Memory Log

**GET** `/memory_log`

**Query Parameters**:

* `limit` (int, default = 20) — number of recent records to return.

**Response**:

```json
[
  {
    "timestamp": "2025-09-11T22:39:54.123Z",
    "request_id": "abc123",
    "agent": "single-agent",
    "query": "What is emotional recursion?",
    "flag": "stable",
    "trust_score": 0.72
  }
]
```

### Signal Logging

**POST** `/signal`

**Request Body**:

```json
{
  "user_id": "u123",
  "user_query": "Hello world",
  "signal_type": "relational",
  "drift_score": 0.3,
  "emotional_tone": 0.5,
  "payload": {"foo": "bar"},
  "relationship_context": "manager",
  "diagnostic_notes": "sample note"
}
```

**Response**:

```json
{
  "id": 1,
  "timestamp": "2025-09-11T22:39:54.123Z",
  "user_id": "u123",
  "user_query": "Hello world",
  "signal_type": "relational",
  "drift_score": 0.3,
  "emotional_tone": 0.5,
  "agent_id": "Selah",
  "payload": {"foo": "bar"},
  "relationship_context": "manager",
  "diagnostic_notes": "sample note",
  "escalate_flag": 0
}
```

### Memory Search

**GET** `/memory/search`

**Query Parameters**:

* `q` (string, required) — natural language query.
* `top_k` (int, default = 3) — number of results to return.

**Response**:

```json
[
  {
    "id": "doc123",
    "score": 0.82,
    "text": "Sample content",
    "agent": "Axis™ Relationship Architect",
    "tags": ["tag1", "tag2"],
    "metadata": {"source": "pinecone"}
  }
]
```

### Agents

**GET** `/agents`

Lists agents known to the service and whether each is **enabled**, based on the `ENABLED_AGENTS` environment variable (comma-separated).

**Response**:

```json
{
  "agents": [
    {"role": "Axis", "enabled": true},
    {"role": "Oria", "enabled": true},
    {"role": "Sentinel", "enabled": false}
  ]
}
```

### Trust Score

**POST** `/score`

Accepts the same body as `/query` but returns only the derived trust score and flag, for lightweight health/sanity checks.

**Request Body**:

```json
{
  "query": "Your question here",
  "k": 3
}
```

**Response**:

```json
{
  "trust_score": 0.72,
  "flag": "drifting"
}
```

---

## Pipeline Diagram

### High-level request path (API)

```mermaid
flowchart LR
    subgraph Client
      U[User / UI / cURL]
    end

    subgraph API[FastAPI]
      A[Collect biometrics] --> B[Build prompt]
      B --> C[LLM (invoke)]
      B --> D[Vector search (Pinecone)]
      D --> E[map_events_to_memory (Coherence Commons)]
      C --> F[Compose response]
      E --> F
      F --> G[Trace log + MLflow]
    end

    U -->|/query or /multi_query| A
```

### Multi-agent fan-out + router

```mermaid
flowchart TB
    Q[Request] --> R[router_stub.route_agent]
    R -->|advisory| AX[Axis QA] & OR[Oria QA] & MS[M Sentinel QA]
    R -->|enforced (ROUTER_ENFORCE=1)| CH[Chosen agent only]

    AX --> VS1[(Axis store)] --> M1[map_events_to_memory]
    OR --> VS2[(Oria store)] --> M2[map_events_to_memory]
    MS --> VS3[(Sentinel store)] --> M3[map_events_to_memory]

    M1 --> AGG[Aggregate events/scores]
    M2 --> AGG
    M3 --> AGG

    AGG --> FLAGS{flag_from_score / suggestions}
    FLAGS --> RESP[Response + escalation if concern]
```

### Ingestion → Coherence → Storage

```mermaid
flowchart LR
    IN[Raw inputs (docs/json)] --> SPLIT[Chunking / normalizer]
    SPLIT --> EMB[Embeddings]
    EMB --> PIN[(Pinecone Index)]
    PIN -.-> RET[Retriever]
    RET -.-> API[/query, /multi_query/]
    IN --> COH[Coherence Commons (map, tag, suggest)]
    COH --> DB[(SQLite)]
```

---

## Streamlit UI

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The sidebar allows you to configure:

* **Backend URL** (e.g. `http://localhost:8000`)
* **Mode**: Single-Agent vs. Multi-Agent
* **Number of chunks (k)**

Submit a query to see answers, chunks, flags, suggestions, and drift visualizations.

---

## Project Structure

```
	signal_memory_engine_v1/
	├── setup.py                          # Package install script
	├── __init__.py                       # Top-level package marker
	├── README.md                         # Project documentation
	├── requirements.txt                  # Python dependencies
	├── pytest.ini                        # Pytest configuration
	├── .env.example                      # Example environment variables (copy to .env)
	├── starter.sh                        # Helper script to launch the service
	├── streamlit_app.py                  # Streamlit frontend
	├── core.py                           # Legacy RAG builder (optional)
	├── data/                             # Local data & SQLite DBs
	│   └── (gitignored runtime files)    # e.g., signal.db
	├── mlruns/                           # MLflow tracking dir (created at runtime)
	│   └── (experiment runs)
	├── api/
	│   ├── main.py                       # FastAPI application (includes routers)
	│   ├── models.py                     # Pydantic schemas
	│   ├── deps.py                       # Shared dependencies (QA chains, vectorstore, config)
	│   └── routes/
	│       ├── __init__.py
	│       ├── query.py                  # POST /query (single-agent RAG)
	│       ├── multi.py                  # POST /multi_query (multi-agent fan-out)
	│       ├── memory.py                 # GET /memory_log (trace JSONL tail)
	│       ├── search.py                 # GET /memory/search (Pinecone search)
	│       └── signal.py                 # POST /signal, GET /drift/{user_id}
	├── agents/
	│   ├── __init__.py
	│   ├── axis_agent.py                 # Axis™ chain & vectorstore
	│   ├── oria_agent.py                 # Oria™ chain & vectorstore
	│   ├── m_agent.py                    # M™ chain & vectorstore
	│   └── router_stub.py                # Lightweight agent router prototype
	├── coherence/
	│   └── commons.py                    # Event-to-memory mapping utilities
	├── ingestion/
	│   ├── __init__.py
	│   └── batch_loader.py               # Data ingestion helpers
	├── processing/
	│   ├── __init__.py
	│   ├── normalizer.py                 # Stream processing helper
	│   └── stream_processor.py           # Real-time event processor
	├── sensors/
	│   └── biometric.py                  # Simulated biometric readings
	├── scripts/
	│   ├── __init__.py
	│   ├── langchain_retrieval.py        # build_qa_chain & vectorstore setup
	│   ├── smoke_test.py                 # Quick pipeline sanity check
	│   ├── seed_data.py                  # Sample data seeding script
	│   ├── drift_monitor.py              # Drift monitoring job / CLI
	│   └── ingest_memories.py            # One-off or batch memory ingestion
	├── storage/
	│   └── sqlite_store.py               # SQLite persistence (init_db, insert_event, list_by_user)
	├── utils/
	│   ├── llm.py                        # Safe LLM invocation helper (timeouts, 429→503)
	│   ├── tracing.py                    # trace.log append + tail helpers
	│   └── dashboard.py                  # Dashboard hook stub (send_to_dashboard)
	└── vector_store/
		├── __init__.py
		├── embeddings.py                 # Embedding factory
		└── pinecone_index.py             # Pinecone index initialization
```

---

Contributions, issues, and PRs are welcome!

