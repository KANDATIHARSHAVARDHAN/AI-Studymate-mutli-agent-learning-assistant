"""
Vector Store Module — Handles PDF processing, embedding, and Pinecone vector store management.
"""
import os
import hashlib
import time
from datetime import datetime
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize Pinecone index name
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "multi-agentic-rag")


def get_embedding_model(api_key: str, retries: int = 3, backoff_factor: int = 2):
    """Create embedding model with retry logic."""
    for attempt in range(retries):
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )
            _ = embedding_model.embed_query("connectivity probe")
            return embedding_model
        except Exception as e:
            if attempt < retries - 1:
                delay = backoff_factor ** attempt
                time.sleep(delay)
            else:
                raise Exception(f"Failed to initialize embedding model after {retries} attempts: {e}")


def get_pdf_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def init_pinecone_index():
    """Ensure the Pinecone index exists, create if not."""
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    existing_indexes = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768, # Gemini embeddings dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc


def process_pdf(file_path: str, original_filename: str, embedding_model) -> PineconeVectorStore:
    """Process a single PDF and upsert to Pinecone."""
    pdf_hash = get_pdf_hash(file_path)
    
    # We use the hash as a namespace to isolate documents
    namespace = pdf_hash

    # Ensure index exists
    init_pinecone_index()

    # Check if already processed (could check if namespace has vectors, but for simplicity, we just upsert)
    
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata.update({
            "source": original_filename,
            "hash": pdf_hash,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    
    # Upsert to Pinecone (default namespace)
    vectorstore = PineconeVectorStore.from_documents(
        docs, 
        embedding_model, 
        index_name=PINECONE_INDEX_NAME
    )
    return vectorstore


def build_retriever(file_paths: List[dict], embedding_model):
    """
    Build a Pinecone retriever from multiple PDFs.
    file_paths: list of {"path": str, "name": str}
    """
    namespaces = []
    for fp in file_paths:
        process_pdf(fp["path"], fp["name"], embedding_model)
        namespaces.append(get_pdf_hash(fp["path"]))

    if not namespaces:
        return None

    # Return a retriever that searches across all namespaces uploaded
    # Note: If searching across multiple namespaces is not supported in a single retriever call easily,
    # we can just use the default namespace for all, and filter by hash instead.
    # For now, we will just use the index and let it retrieve from the entire index (ignoring namespace in retriever).
    # To be more precise, let's just initialize a single retriever.
    
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, 
        embedding=embedding_model
    )
    
    # Filter by the specific hashes uploaded in this session
    return vectorstore.as_retriever(search_kwargs={
        "k": 10,
        "filter": {"hash": {"$in": namespaces}}
    })

def get_global_retriever(embedding_model):
    """Return a retriever that searches the entire index without filtering."""
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, 
        embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})
