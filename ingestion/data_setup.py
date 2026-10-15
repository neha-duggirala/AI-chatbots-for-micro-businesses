'''
initialize:
An open source vector embedding model to create embeddings for text data. 

use langchain for the following
1. chunk text using langchain
2. setup some overlap
3. index them or something
4. create a vector embedding
'''
"""Load .txt files and split them into chunks."""

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR

def load_document(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")  # raises FileNotFoundError itself
    return [Document(page_content=text, metadata={"source": str(file_path)})]


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def load_and_split(filename: str, doc_type: str) -> list[Document]:
    """Load a file from DOCUMENTS_DIR, split it, and tag chunks with doc_type."""
    docs = load_document(DOCUMENTS_DIR / filename)
    chunks = split_documents(docs)
    for chunk in chunks:
        chunk.metadata["doc_type"] = doc_type
    return chunks