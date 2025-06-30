# vector_store/__init__.py

from .pinecone_index import pc, index
from .embeddings import get_embedding

__all__ = ["pc", "index", "get_embedding"]