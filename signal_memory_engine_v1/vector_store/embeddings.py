# vector_store/embeddings.py

import hashlib
from textwrap import wrap
from typing import List

# --- LangChain-compatible OpenAI Embeddings Builder ---
from langchain_community.embeddings import OpenAIEmbeddings

def get_embedder(model: str, openai_api_key: str):
    """
    Return a LangChain-compatible OpenAIEmbeddings instance.

    Parameters:
        model: the OpenAI embedding model name, e.g. 'text-embedding-ada-002'.
        openai_api_key: your OpenAI API key.
    """
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=openai_api_key
    )

# --- Legacy local embeddings (optional) ---
from sentence_transformers import SentenceTransformer

_default_model = SentenceTransformer("all-MiniLM-L6-v2")  # dim=384


def get_local_embedding(text: str) -> List[float]:
    """
    Embed a single text string locally using SentenceTransformer.
    """
    return _default_model.encode(text, show_progress_bar=False).tolist()


def process_text_to_embeddings(text: str, width: int = 512) -> List[List[float]]:
    """
    Split longer text into fixed-size chunks and embed each chunk locally.
    """
    chunks = wrap(text, width)
    return [
        _default_model.encode(chunk, show_progress_bar=False).tolist()
        for chunk in chunks
    ]


def generate_id_from_text(text: str) -> str:
    """
    Deterministically generate a unique ID for a chunk of text.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()