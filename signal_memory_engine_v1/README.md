### Signal Memory Engine

A conversational Retrieval-Augmented Generation (RAG) microservice powered by FastAPI, OpenAI, Pinecone, and LangChain, with multi-agent handoff functionality and a Streamlit UI.

⸻

#### Features
	•	Single-agent RAG: Query your memory index via /query endpoint, powered by a RetrievalQA chain using OpenAI’s GPT-3.5-Turbo and Pinecone vector store.
	•	Multi-agent fan-out: Automatically route queries to three specialized agents—Axis™ Relationship Architect, Oria™ HR Oracle, and M™ Shadow Sentinel—via /multi_query endpoint, each returning an answer, top memory chunks, a stability flag, and a suggestion.
	•	Agent-to-Agent Handoff: High-severity flags can trigger custom workflows or escalate to human-in-the-loop endpoints.
	•	Streamlit UI: Lightweight frontend to interact with both single- and multi-agent endpoints.

⸻

#### Table of Contents
	•	Prerequisites
	•	Installation
	•	Configuration
	•	Running the API
	•	Endpoints
	•	Streamlit UI
	•	Project Structure

⸻

#### Prerequisites
	•	Python 3.9+
	•	A Pinecone account (API key & environment)
	•	An OpenAI account (API key)

⸻

#### Installation
	1.	Clone this repository: git clone https://github.com/your-org/signal-memory-engine.git
      cd signal-memory-engine
	2.	Create and activate a virtual environment:
      python -m venv venv
      source venv/bin/activate
	3.	Install dependencies:
      pip install -r requirements.txt
⸻

#### Configuration

Create a .env file in the project root with the following variables:

	PINECONE_API_KEY=<your-pinecone-api-key>
	PINECONE_ENV=<your-pinecone-environment>
	PINECONE_INDEX=<your-index-name>
	OPENAI_API_KEY=<your-openai-api-key>

PINECONE_INDEX default is signal-engine.

⸻

#### Running the API

Start the FastAPI server:

uvicorn api.main:app --reload

The API will be available at http://127.0.0.1:8000.

⸻

#### Endpoints

Single-Agent RAG
POST /query
Body:

		{
		  "query": "Your question here",
		  "k": 3
		}

Response:

	{
	  "answer": "...GPT-generated answer...",
	  "chunks": [
	    {"content": "chunk text...", "score": 0.73},
	    ...
	  ],
	  "flag": "stable", // or "drifting" / "concern"
	  "suggestion": "No action needed."
	}



Multi-Agent Fan-Out
POST /multi_query
Body:

	{
	  "query": "Your question here",
	  "k": 3
	}


Response:

	{
	  "agents": {
	    "Axis™ Relationship Architect": { /* AgentResponse */ },
	    "Oria™ HR Oracle": { /* AgentResponse */ },
	    "M™ Shadow Sentinel": { /* AgentResponse */ }
	  }
	}



Each AgentResponse contains the same fields as the single-agent response.

⸻

#### Streamlit UI

A sample streamlit_app.py is provided in the project root. To run the UI:

streamlit run streamlit_app.py

Use the sidebar to configure your backend URL, endpoint (/query vs. /multi_query), and k. Enter your query in the main panel and submit to see answers, memory chunks, flags, and suggestions.

⸻

#### Project Structure

├── api
│   └── main.py             # FastAPI application
├── scripts
│   └── langchain_retrieval.py  # build_qa_chain helper
├── vector_store
│   ├── embeddings.py       # get_embedder utility
│   ├── pinecone_index.py   # init_pinecone_index helper
│   └── __init__.py
├── agents
│   ├── axis_agent.py       # Axis™ agent chain & store
│   ├── oria_agent.py       # Oria™ agent chain & store
│   └── m_agent.py          # Shadow Sentinel agent chain & store
├── streamlit_app.py        # Sample Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md               # This file
