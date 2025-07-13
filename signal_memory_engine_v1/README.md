# Signal Memory Engine v1

A conversational Retrieval-Augmented Generation (RAG) microservice powered by FastAPI, OpenAI, Pinecone, and LangChain. Features multi-agent handoff logic, biometric signal context, event-to-memory mapping, and a Streamlit UI for visualization.

---

## Features

* **Single-Agent RAG**: Query a memory index via the `/query` endpoint, using a RetrievalQA chain with OpenAI’s GPT models and Pinecone vector store.
* **Multi-Agent Fan-Out**: Route queries to three specialized agents—Axis™ Relationship Architect, Oria™ HR Oracle, and M™ Shadow Sentinel—via the `/multi_query` endpoint. Each agent returns an answer, top memory chunks, a stability flag, and a suggestion.
* **Biometric Context**: Simulate HRV, temperature, and blink rate readings and inject them into queries to enrich system context.
* **Event-to-Memory Mapping**: Normalize raw RAG hits into structured events (ID, content, score, timestamp, metadata) via Coherence Commons.
* **Agent-to-Agent Handoff**: Automatically escalate high-severity flags by notifying a human-in-the-loop endpoint.
* **Streamlit UI**: Interactive frontend (`streamlit_app.py`) that shows answers, memory chunks, flags, suggestions, and drift visualizations per agent.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the API](#running-the-api)
5. [Endpoints](#endpoints)
6. [Streamlit UI](#streamlit-ui)
7. [Project Structure](#project-structure)

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
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```ini
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_ENV=<your-pinecone-environment>
PINECONE_INDEX=<your-pinecone-index-name>
OPENAI_API_KEY=<your-openai-api-key>
```

* `PINECONE_INDEX` defaults to `signal-engine` if unset.

---

## Running the API

```bash
# Start FastAPI server with live reload
uvicorn api.main:app --reload
```

The API will run at `http://127.0.0.1:8000`.

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
	├── api/
	│   ├── routes/
	│   │   ├── memory.py          # Memory query router
	│   │   ├── search.py          # Generic search router
	│   │   └── __init__.py
	│   └── main.py                # FastAPI application
	├── agents/
	│   ├── axis_agent.py          # Axis™ Relationship Architect chain & store
	│   ├── oria_agent.py          # Oria™ HR Oracle chain & store
	│   ├── m_agent.py             # M™ Shadow Sentinel chain & store
	│   └── __init__.py
	├── coherence/
	│   └── commons.py             # Event-to-memory mapping utilities
	├── ingestion/
	│   ├── batch_loader.py        # Data ingestion helpers
	│   └── __init__.py
	├── processing/
	│   ├── normalizer.py          # Stream processing helper
	│   ├── stream_processor.py    # Real-time event processor
	│   └── __init__.py
	├── sensors/
	│   └── biometric.py           # Simulated biometric readings
	├── scripts/
	│   ├── langchain_retrieval.py # build_qa_chain & vectorstore setup
	│   ├── smoke_test.py          # Quick pipeline sanity check
	│   ├── seed_data.py           # Sample data seeding script
	│   └── __init__.py
	├── vector_store/
	│   ├── embeddings.py          # Embedding factory
	│   ├── pinecone_index.py      # Pinecone index initialization
	│   └── __init__.py
	├── core.py                    # Legacy RAG builder (optional)
	├── requirements.txt           # Python dependencies
	├── starter.sh                 # Helper script to launch the service
	├── streamlit_app.py           # Streamlit frontend
	├── generate_structure.sh      # Project scaffolding script
	└── README.md                  # Project documentation
```

---

Contributions, issues, and PRs are welcome!
