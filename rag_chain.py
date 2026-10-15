"""LLM, retriever and LCEL chain construction."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_qdrant import QdrantVectorStore
from transformers import pipeline

from config import LLM_MAX_NEW_TOKENS, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n" \
    # "please be aware to return only the answer, dont return any other text, dont return the context, dont return the question just return the answer.\n\n" \
    "Context:\n{context}"
)
 

def format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved document contents."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_llm() -> HuggingFacePipeline:
    hf_pipeline = pipeline(
        "text-generation",
        model=LLM_MODEL,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=LLM_TEMPERATURE,
        return_full_text=False,
        clean_up_tokenization_spaces=False
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


def build_retriever(stores: list[QdrantVectorStore], k: int = RETRIEVER_K) -> Runnable:
    """Retrieve the top-k chunks across ALL given stores (not k per store).
 
    Each store returns its own top-k with similarity scores; the results are
    merged and the best k overall are kept. Scores are comparable because every
    collection uses the same embedding model and distance metric.
    """
 
    def retrieve(query: str) -> list[Document]:
        scored = [
            (doc, score)
            for store in stores
            for doc, score in store.similarity_search_with_score(query, k=k)
        ]
        top = sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]
        for doc, score in top:
            doc.metadata["score"] = round(score, 4)
        return [doc for doc, _ in top]
 
    return RunnableLambda(retrieve)

def build_rag_chain(
    stores: list[QdrantVectorStore],
    llm: BaseChatModel | None = None,
) -> Runnable:
    """Build a chain that returns {"question", "docs", "context", "answer"}.
 
    Pass an already-loaded `llm` to avoid reloading the model for every chain.
    """
    retriever = build_retriever(stores)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    answer_chain = prompt | (llm or build_llm()) | StrOutputParser()
 
    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        .assign(context=lambda x: format_docs(x["docs"]))
        .assign(answer=answer_chain)
    )