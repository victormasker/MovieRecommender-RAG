"""
One-time script to build and persist the FAISS vector store.
Run this whenever the source data changes:
    python build_vectorstore.py
"""
import os
import shutil
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import huggingface_instruct_embedding

DB_PATH = '../vectorstore_faiss'
DATA_PATH = '../data/IMDB Dataset.csv'
NUM_DOCS = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def build():
    # Clean up old store
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Removed old vector store at {DB_PATH}")

    # Load documents
    print(f"Loading documents from {DATA_PATH}...")
    loader = CSVLoader(DATA_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # Clean HTML tags
    for doc in docs:
        doc.page_content = doc.page_content.replace('<br /><br />', '\n').replace('<br />', '\n')

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs[:NUM_DOCS])
    print(f"Split into {len(chunks)} chunks (from first {NUM_DOCS} docs)")

    # Build embeddings and vector store
    print("Creating embeddings and building vector store (this may take a few minutes)...")
    embeddings = huggingface_instruct_embedding()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    print(f"Vector store saved to {DB_PATH}")
    print("Done! You can now run the Streamlit app.")


if __name__ == '__main__':
    build()
