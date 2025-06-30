import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1) Load environment (for local dev)
load_dotenv()

# 2) Initialize Pinecone (must happen before any index imports)
from vector_store.pinecone_index import pc, index
# Already created in pinecone_index.py

# 3) (Optional) Initialize embedder if you have one in embeddings.py
# from vector_store.embeddings import init_embedder
# init_embedder()

# 4) Create FastAPI app
app = FastAPI(title="Signal Memory Engine API")

# 5) CORS (allow your Streamlit or other frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6) Include your routers
from api.routes.memory import router as memory_router
app.include_router(memory_router, prefix="/memory", tags=["memory"])

# 7) (Optional) health check
@app.get("/health")
def health():
    return {"status": "ok"}