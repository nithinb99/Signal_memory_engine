import os
import pinecone
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain.chains import RetrievalQA

def build_qa_chain(
    pinecone_api_key: str,
    pinecone_env: str,
    index_name: str,
    openai_api_key: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "gpt-3.5-turbo",
    k: int = 3,
    text_key: str = "content",
) -> tuple[RetrievalQA, LC_Pinecone]:
    """
    Build and return a RetrievalQA chain + Pinecone vectorstore client.

    Returns:
        qa_chain: LangChain RetrievalQA
        vectorstore: Pinecone vectorstore (for similarity_search_with_score)
    """
    # 1) Validate env vars
    load_dotenv()
    if not pinecone_api_key or not index_name or not openai_api_key:
        raise RuntimeError("Missing one of PINECONE_API_KEY, index_name, or OPENAI_API_KEY")

    # 2) Monkey-patch Pinecone
    if not hasattr(pinecone, "__version__"):
        pinecone.__version__ = "3.0.0"
    pc = pinecone.Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    from pinecone.db_data.index import Index as PineconeIndexClass
    pinecone.Index = PineconeIndexClass

    # 3) Embeddings & Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
        text_key=text_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 4) LLM
    llm = ChatOpenAI(
        model_name=llm_model,
        openai_api_key=openai_api_key,
        temperature=0.7,
        max_tokens=256,
    )

    # 5) Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    return qa_chain, vectorstore
