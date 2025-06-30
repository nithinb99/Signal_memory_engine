#!/usr/bin/env python
"""
scripts/langchain_retrieval.py

Interactive RetrievalQA pipeline using LangChain, a local HuggingFace LLM,
and your Pinecone vector store, with compatibility patches for the community adapter.
"""

import os
import torch
from dotenv import load_dotenv

import pinecone
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from transformers import pipeline

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

    # 5) Load the gated Llama-2 model once (auth only at load time)
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    text_gen = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device="cpu",               # or 0 for GPU
        # use_auth_token=HF_TOKEN     # **only** at load time
    )

    # 6) Wrap in LangChain, supplying **only** generation defaults here
    llm = HuggingFacePipeline(
        pipeline=text_gen,
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 256,
        }
    )

    # 7) Build the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # 8) Interactive prompt loop
    print("Local RAG ready. Type 'exit' to quit.")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        # Use invoke() or __call__; either will apply your model_kwargs correctly
        answer = qa.invoke(query)
        print(f"\nAnswer:\n{answer}\n{'-'*40}")

if __name__ == "__main__":
    main()