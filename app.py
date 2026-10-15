"""Streamlit chat UI for the RAG pipeline.

Run from the project root:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from rag_chain import build_llm, build_rag_chain
from ingestion.vector_store import build_vector_stores


# --- Cached resources (loaded once per server process) --------------------
@st.cache_resource(show_spinner="Connecting to Qdrant and indexing documents...")
def load_stores():
    return build_vector_stores()


@st.cache_resource(show_spinner="Loading language model...")
def load_llm():
    return build_llm()


@st.cache_resource
def load_chain():
    """RAG chain that searches every collection."""
    return build_rag_chain(list(load_stores().values()), llm=load_llm())


# --- UI helpers -----------------------------------------------------------
def render_context(docs: list[Document]) -> None:
    """Show retrieved chunks inside a collapsible section."""
    with st.expander(f"Retrieved context ({len(docs)} chunks)"):
        for i, doc in enumerate(docs, start=1):
            source = Path(doc.metadata.get("source", "unknown")).name
            doc_type = doc.metadata.get("doc_type", "unknown")
            st.markdown(f"**{i}. {doc_type}** · `{source}`")
            st.text(doc.page_content)
            if i < len(docs):
                st.divider()


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("docs"):
            render_context(message["docs"])


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Settings")

        if st.button("Re-index documents"):
            with st.spinner("Rebuilding collections..."):
                load_stores.clear()
                load_chain.clear()
                build_vector_stores(reindex=True)
            st.success("Collections rebuilt.")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


# --- App ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Docs Chat", page_icon="💬")
    st.title("💬 Chat with your docs")

    render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    query = st.chat_input("Ask about the product or policy docs...")
    if not query:
        return

    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = load_chain().invoke(query)

        answer = result["answer"].strip() or "_No answer generated._"
        st.markdown(answer)
        render_context(result["docs"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "docs": result["docs"]}
    )


main()