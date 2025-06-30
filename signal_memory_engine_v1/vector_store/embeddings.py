# vector_store/embeddings.py

import hashlib
from textwrap import wrap
from typing import List

from sentence_transformers import SentenceTransformer

# Load the embedding model once at import time
_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> List[float]:
    """
    Embed a single text string into a vector.
    """
    return _model.encode(text, show_progress_bar=False).tolist()

def process_text_to_embeddings(text: str, width: int = 512) -> List[List[float]]:
    """
    Split longer text into fixed-size chunks and embed each chunk.
    Returns a list of embedding vectors.
    """
    chunks = wrap(text, width)
    return [
        _model.encode(chunk, show_progress_bar=False).tolist()
        for chunk in chunks
    ]

def generate_id_from_text(text: str) -> str:
    """
    Deterministically generate a unique ID for a chunk of text.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()