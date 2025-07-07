import os
import torch
import pinecone
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain.chat_models import ChatOpenAI

from transformers import pipeline

# -- Environment & Pinecone patch
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")

if not hasattr(pinecone, "__version__"):
    pinecone.__version__ = "3.0.0"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
from pinecone.db_data.index import Index as PineconeIndexClass
pinecone.Index = PineconeIndexClass

# -- Embeddings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE}
)

# -- Retriever builder
def build_retriever(index_name: str, k: int = 3):
    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
        text_key="content"
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

# -- LLM builder
def build_qa_chain(
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "gpt-3.5-turbo",
    index_name_env: str = "PINECONE_INDEX",
    text_key: str = "content",
    k: int = 3,
) -> RetrievalQA:
    """
    Build a RetrievalQA chain using Pinecone + HuggingFace embeddings + OpenAI chat LLM.

    Loads PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME, and OPENAI_API_KEY from env.
    """
    # load and validate env vars
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
    INDEX_NAME = os.getenv(index_name_env)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not PINECONE_API_KEY or not INDEX_NAME or not OPENAI_API_KEY:
        raise RuntimeError("Missing one of PINECONE_API_KEY, PINECONE_INDEX, or OPENAI_API_KEY")

    # monkey-patch Pinecone
    if not hasattr(pinecone, "__version__"):
        pinecone.__version__ = "3.0.0"
    client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    from pinecone.db_data.index import Index as PineconeIndexClass
    pinecone.Index = PineconeIndexClass

    # embeddings + vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key=text_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # chat LLM
    llm = ChatOpenAI(
        model_name=llm_model,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=256,
    )

    # build RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa


if __name__ == "__main__":
    # quick sanity check
    qa = build_qa_chain()
    print("QA chain successfully created:", qa)