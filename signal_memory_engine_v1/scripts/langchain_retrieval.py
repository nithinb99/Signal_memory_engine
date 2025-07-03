#!/usr/bin/env python
"""
scripts/langchain_retrieval.py

Interactive RetrievalQA pipeline using LangChain, a local HuggingFace LLM,
and your Pinecone vector store, with compatibility patches for the community adapter.
Includes memory-chunk scoring, stable/drifting/concern flags, and simple suggestions.
"""

import os
import torch
from dotenv import load_dotenv

import pinecone
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def flag_from_score(score: float) -> str:
    """
    Convert a similarity score (0–1) into a flag.
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

def format_and_print(query: str, vectorstore, k: int = 3):
    """
    Retrieve top-k chunks + scores, then print them along with a flag and suggestion.
    """
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    if not docs_and_scores:
        print("⚠️  No memories found for that query.")
        return

    # Determine overall flag from the highest score
    top_score = max(score for (_, score) in docs_and_scores)
    flag      = flag_from_score(top_score)
    suggestion = SUGGESTIONS[flag]

    # Print the raw hits
    print(f"\nTop {k} memory chunks (score):")
    for doc, score in docs_and_scores:
        print(f" • [{score:.3f}] {doc.page_content.strip()}")

    # Print flag & suggestion
    print(f"\n>>> Flag:       {flag.upper()}")
    print(f">>> Suggestion: {suggestion}\n{'-'*40}")


def main():
    # 1) Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
    INDEX_NAME       = os.getenv("PINECONE_INDEX", "signal-engine")
    HF_TOKEN         = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY environment variable")

    # 2) Monkey-patch Pinecone for LangChain compatibility
    if not hasattr(pinecone, "__version__"):
        pinecone.__version__ = "3.0.0"
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    from pinecone.db_data.index import Index as PineconeIndexClass

    pinecone.Index = PineconeIndexClass

    # 3) Prepare your embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    # 4) Connect to your existing Pinecone index
    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key="content"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

   # 5) Load Flan-T5 (fully open & instruction-tuned)
    MODEL_NAME = "google-t5/t5-base"
    device = "cpu"  # or 0 for your GPU

    # a) Load the seq2seq model & tokenizer once at startup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    text_gen = pipeline(
        "text2text-generation",   # NOTE: use text2text for T5
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256,
        temperature=0.7,
    )

    # b) Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=text_gen)
    # 7) Build the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # 8) Interactive prompt loop with flags & suggestions
    print("Local RAG + Signal Flags ready. Type 'exit' to quit.")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        # 8a) LLM answer
        answer = qa.invoke(query)
        print(f"\nLLM Answer:\n{answer}\n{'='*40}")

        # 8b) Raw memory hits + flag & suggestion
        format_and_print(query, vectorstore, k=3)

if __name__ == "__main__":
    main()