#!/usr/bin/env python
"""
scripts/langchain_retrieval.py

Module to build and return a RetrievalQA chain and Pinecone vectorstore,
with helper functions for signal-flag scoring and suggestions.
"""

import pinecone
import logging
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# ── Configure logging ─────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def flag_from_score(score: float) -> str:
    """
    Convert a similarity score (0–1) into a signal flag.
    """
    if score > 0.8:
        return "concern"
    elif score > 0.5:
        return "drifting"
    else:
        return "stable"


SUGGESTIONS = {
    "stable":   "No action needed.",
    "drifting": "Consider sending a check-in message.",
    "concern":  "Recommend escalation or a one-on-one conversation."
}


def build_qa_chain(
    pinecone_api_key: str,
    pinecone_env: str,
    index_name: str,
    openai_api_key: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "gpt-3.5-turbo",
    k: int = 3,
) -> tuple[RetrievalQA, LC_Pinecone]:
    """
    Initialize Pinecone, embeddings, vectorstore, and a RetrievalQA chain using OpenAI or HF-ST.

    Returns:
        qa_chain: LangChain RetrievalQA
        vectorstore: Pinecone vector store client
    """
    # 1) Monkey-patch Pinecone
    if not hasattr(pinecone, "__version__"):
        pinecone.__version__ = "3.0.0"
    pc = pinecone.Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    from pinecone.db_data.index import Index as PineconeIndexClass
    pinecone.Index = PineconeIndexClass

    # 2) Set up embeddings: HF-ST for sentence-transformers, OpenAI for ADA
    if embed_model.startswith("sentence-transformers/") or embed_model.startswith("all-"):
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        embeddings = OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=openai_api_key,
            encode_kwargs={"normalize_embeddings": True},
        )

    # 3) Connect to existing Pinecone index
    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
        text_key="content"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 4) Initialize OpenAI LLM
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=llm_model,
        temperature=0.7,
        max_tokens=256,
    )

    # 5) Build a standard RetrievalQA chain (default prompt only needs a single string)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    logger.debug("QA chain initialized with default prompt template")

    return qa_chain, vectorstore
