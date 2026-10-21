"""Central configuration for the RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

#docker run -p 6333:6333 -p 6334:6334 \
    # -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    # qdrant/qdrant
load_dotenv()

# --- Paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

# --- Qdrant --------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# collection name -> source file inside DOCUMENTS_DIR
COLLECTIONS = {
    "product_docs": "product_docs.txt",
    "policy_docs": "policy_docs.txt",
}

# --- Text splitting ------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Models --------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_MODEL = "claude-haiku-4-5"
LLM_MAX_NEW_TOKENS = 100
LLM_TEMPERATURE = 0.7

# --- Retrieval -----------------------------------------------------------
RETRIEVER_K = 4  # candidates fetched from the vector stores (across all collections)
RERANK_TOP_N = 2  # chunks kept after reranking and sent to the LLM