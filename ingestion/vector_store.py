"""Create / reuse Qdrant collections backed by a local Qdrant server."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import COLLECTIONS, EMBEDDING_MODEL, QDRANT_URL
from .data_setup import load_and_split


def get_embeddings() -> Embeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_store(
    client: QdrantClient,
    embeddings: Embeddings,
    collection_name: str,
    filename: str,
    reindex: bool = False,
) -> QdrantVectorStore:
    """Return a vector store for a collection, ingesting documents only if needed."""
    if reindex and client.collection_exists(collection_name):
        print(f"[{collection_name}] Deleting existing collection...")
        client.delete_collection(collection_name)

    if client.collection_exists(collection_name):
        print(f"[{collection_name}] Using existing collection.")
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=QDRANT_URL,
        )

    chunks = load_and_split(filename, doc_type=collection_name)
    print(f"[{collection_name}] Indexing {len(chunks)} chunks from '{filename}'...")
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )


def build_vector_stores(reindex: bool = False) -> dict[str, QdrantVectorStore]:
    """Build one vector store per configured collection."""
    client = QdrantClient(url=QDRANT_URL)
    embeddings = get_embeddings()
    return {
        name: get_vector_store(client, embeddings, name, filename, reindex)
        for name, filename in COLLECTIONS.items()
    }