from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_and_index_text(path="data/textbook.txt"):
    # Load raw textbook content
    loader = TextLoader(path)
    documents = loader.load()

    # Split into chunks for better indexing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Generate embeddings and index in FAISS
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db