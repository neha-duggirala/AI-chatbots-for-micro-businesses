"""Streamlit chat UI for the RAG pipeline.

Run from the project root:
    streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from guardrails import AbhiMilkGuardrails  # NEW
from ingestion.plotting import plot_guardrail_stages, plot_score_answer  # CHANGED
from ingestion.reranker import RerankResult
from ingestion.vector_store import build_vector_stores
from rag_chain import build_llm, build_rag_chain, build_claude_llm


# --- Cached resources (loaded once per server process) --------------------
@st.cache_resource(show_spinner="Connecting to Qdrant and indexing documents...")
def load_stores():
    return build_vector_stores()


@st.cache_resource(show_spinner="Loading language model...")
def load_llm():
    return build_claude_llm()


def load_chain():
    """RAG chain that searches every collection, with guardrails on.

    Not cached on purpose: building it is cheap (the slow parts, stores and LLM,
    are cached above), and caching it would keep running stale chain code after
    you edit rag_chain.py.
    """
    return build_rag_chain(
        list(load_stores().values()),
        llm=load_llm(),
        guard=AbhiMilkGuardrails(),  # NEW
    )


# --- Helpers --------------------------------------------------------------
def build_history(messages: list[dict]) -> list[BaseMessage]:
    """Convert stored chat messages into LangChain messages for the chain."""
    return [
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in messages
    ]


def make_rerank_figure(rerank: RerankResult | None):
    """Bar plot of each retrieved chunk's relevance probability (None if not reranked)."""
    if rerank is None or rerank.answer is None:  # CHANGED: rerank is None when input is blocked
        return None
    labels = {
        i: f"Chunk {i + 1}\n{doc.metadata.get('doc_type', '')}"
        for i, doc in enumerate(rerank.candidates)
    }
    fig = plot_score_answer(
        rerank.answer, title="Chunk relevance to the question", labels=labels
    )
    plt.close(fig)  # st.pyplot still renders it; this avoids leaking open figures
    return fig


def make_guardrail_figures(guardrails: dict) -> dict:  # NEW
    """One bar plot per guardrail stage: {"input": fig, "output": fig}."""
    figs = plot_guardrail_stages(guardrails)
    for fig in figs.values():
        plt.close(fig)
    return figs


# --- UI components --------------------------------------------------------
def render_context(docs: list[Document], fig=None, guard_figs: dict | None = None) -> None:
    """Show the rerank plot, guardrail plots and the chunks used as context."""
    label = f"Retrieved context ({len(docs)} chunks)" if docs else "Guardrail checks"
    with st.expander(label):
        if fig is not None:
            st.pyplot(fig)
            st.caption(
                "Chunk numbers refer to retrieval order; the top chunks by relevance are used."
            )
        for guard_fig in (guard_figs or {}).values():  # NEW
            st.pyplot(guard_fig)
        if guard_figs:
            st.caption(
                "Guardrail scores from the classifier. Dashed line = threshold; "
                "pink = violation; blue = informational."
            )
        if (fig is not None or guard_figs) and docs:
            st.divider()
        for i, doc in enumerate(docs, start=1):
            source = Path(doc.metadata.get("source", "unknown")).name
            doc_type = doc.metadata.get("doc_type", "unknown")
            relevance = doc.metadata.get("rerank_score")
            similarity = doc.metadata.get("score")

            details = []
            if relevance is not None:
                details.append(f"relevance {relevance:.0%}")
            if similarity is not None:
                details.append(f"similarity {similarity}")
            details_text = "".join(f" · {d}" for d in details)

            st.markdown(f"**Source {i} · {doc_type}** · `{source}`{details_text}")
            st.text(doc.page_content)
            if i < len(docs):
                st.divider()


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("searched_as"):
            st.caption(f"Searched for: *{message['searched_as']}*")
        st.markdown(message["content"])
        if message.get("docs") or message.get("guard_figs"):  # CHANGED
            render_context(message.get("docs") or [], message.get("fig"), message.get("guard_figs"))


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Settings")

        if st.button("Re-index documents"):
            with st.spinner("Rebuilding collections..."):
                load_stores.clear()
                build_vector_stores(reindex=True)
            st.success("Collections rebuilt.")

        if st.button("Clear chat"):
            st.session_state.messages = []  # also resets the model's memory
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

    # History must be built BEFORE the new question is appended.
    history = build_history(st.session_state.messages)

    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.spinner("Thinking..."):
        result = load_chain().invoke({"question": query, "history": history})

    answer = result["answer"].strip() or "_No answer generated._"
    searched_as = (
        result["standalone_question"] if result["standalone_question"] != query else None
    )
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "docs": result["docs"],
        "fig": make_rerank_figure(result["rerank"]),
        "guard_figs": make_guardrail_figures(result.get("guardrails", {})),  # NEW
        "searched_as": searched_as,
    }

    # Render and store the reply with the same code path used for history.
    render_message(assistant_message)
    st.session_state.messages.append(assistant_message)


main()