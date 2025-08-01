# setup.py
from setuptools import setup, find_packages

setup(
    name="signal_memory_engine_v1",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "annotated-types==0.7.0",
        "anyio==4.8.0",
        "certifi==2025.1.31",
        "charset-normalizer==3.4.1",
        "click==8.1.8",
        "fastapi==0.115.11",
        "fsspec==2025.3.0",
        "h11==0.14.0",
        "httpcore==1.0.7",
        "httpx==0.28.1",
        "idna==3.10",
        "MarkupSafe==3.0.2",
        "networkx==3.4.2",
        "numpy==2.2.3",
        "packaging==24.2",
        "pydantic==2.10.6",
        "pydantic_core==2.27.2",
        "PyYAML==6.0.2",
        "regex==2024.11.6",
        "requests==2.32.3",
        "safetensors==0.5.3",
        "sniffio==1.3.1",
        "starlette==0.46.1",
        "sympy==1.13.1",
        "tokenizers==0.21.1",
        "torch==2.6.0",
        "tqdm==4.67.1",
        "transformers==4.49.0",
        "typing_extensions==4.12.2",
        "uvicorn==0.34.0",
        "langchain-pinecone>=0.2.0",
        "sentence-transformers>=2.2.2",
        "pinecone>=2.0.0",
        "langchain>=0.0.285",
        "openai>=0.27.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.30.0",
        "pytest>=7.4.0",
        "huggingface-hub>=0.14.0",
    ],
)