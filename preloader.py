import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_and_index_text(path="data/quran_english.pdf"):
    # Load PDF document
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split into chunks for better indexing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Read OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")

    # Generate embeddings using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db
