# vector_store/__init__.py

from .pinecone_index import init_pinecone_index
from .embeddings       import get_embedder

__all__ = ["init_pinecone_index", "get_embedder"]