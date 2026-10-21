"""LLM, retriever and LCEL chain construction (with conversation memory + guardrails)."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from transformers import pipeline
from langchain_anthropic import ChatAnthropic

from config import LLM_MAX_NEW_TOKENS, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K
from ingestion.reranker import rerank_docs
from guardrails import AbhiMilkGuardrails  # <-- NEW (move the file wherever suits your project)

# Number of past messages (user + assistant) sent to the model. 6 = last 3 turns.
MAX_HISTORY_MESSAGES = 6

# If even the best retrieved chunk has a similarity below this, the LLM is not
# called and NO_ANSWER_REPLY is returned. Tune it using the "similarity" values
# shown in the app for on-topic vs off-topic questions.
MIN_SIMILARITY = 0.30

NOT_FOUND_REPLY = "I am sorry, but I cannot find the answer in the provided documents."

NO_ANSWER_REPLY = (
    "I'm the Abhi Milk Dairy assistant. I can answer questions about our products, "
    "delivery and policies, but I couldn't find anything about that in our documents. "
    "Please try rephrasing, or ask me about our products or policies."
)

# Canned replies need no output check (saves one guardrail call).
_CANNED_REPLIES = {NOT_FOUND_REPLY, NO_ANSWER_REPLY}

# Small models follow concrete examples far better than abstract rules, so the
# prompt is short, uses imperative rules, and includes two worked examples.
SYSTEM_PROMPT = f"""You are the customer support assistant for Abhi Milk Dairy. You answer questions using ONLY the Context given in the user's message.

RULES:
1. If the Context does not contain the answer, reply exactly: "{NOT_FOUND_REPLY}"
2. If the user only greets you or asks who you are, reply in one sentence: say you are the Abhi Milk Dairy assistant and can help with questions about our products, delivery and policies.
"""

HUMAN_PROMPT = "Context:\n{context}\n\nQuestion: {question}"

CONDENSE_PROMPT = """Given the conversation history and the user's latest question, rewrite the latest question as a standalone question that can be understood without the history.
- Do NOT answer the question.
- If the question is already standalone, return it unchanged.
- Output only the rewritten question."""


def format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved chunks, labelled so the model can cite [Source N]."""
    return "\n\n".join(
        f"[Source {i}] ({Path(doc.metadata.get('source', 'unknown')).name})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def build_llm() -> BaseChatModel:
    """Local instruct model wrapped as a chat model.

    Greedy decoding (do_sample=False) keeps factual answers stable, and a mild
    repetition penalty stops small models from looping. ChatHuggingFace applies
    the model's own chat template; return_full_text=False returns only the answer.
    """
    hf_pipeline = pipeline(
        "text-generation",
        model=LLM_MODEL,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=hf_pipeline))


def build_claude_llm() -> BaseChatModel:
    """Anthropic Claude model wrapped as a chat model.

    Temperature=0.0 mimics greedy decoding for stable, factual responses.
    Requires ANTHROPIC_API_KEY to be set in your environment variables.
    """
    return ChatAnthropic(
        model=LLM_MODEL
    )


def build_retriever(
    stores: list[QdrantVectorStore],
    k: int = RETRIEVER_K,
    guard: AbhiMilkGuardrails | None = None,  # <-- NEW
) -> Runnable:
    """Retrieve the top-k chunks across ALL given stores (not k per store).

    Each store returns its own top-k with similarity scores; the results are
    merged and the best k overall are kept. Scores are comparable because every
    collection uses the same embedding model and distance metric.

    RETRIEVAL GUARDRAIL: when `guard` is given, chunks from blocked / non-allowed /
    sensitive sources are dropped BEFORE the top-k cut, so they can never reach
    the reranker, the prompt, or the user.
    """

    def retrieve(query: str) -> list[Document]:
        scored = [
            (doc, score)
            for store in stores
            for doc, score in store.similarity_search_with_score(query, k=k)
        ]
        if guard is not None:  # <-- NEW
            scored = [(d, s) for d, s in scored if guard.is_doc_allowed(d)[0]]
        top = sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]
        for doc, score in top:
            doc.metadata["score"] = round(score, 4)
        return [doc for doc, _ in top]

    return RunnableLambda(retrieve)


def _normalize_input(value: str | dict) -> dict:
    """Accept a plain question string or {"question", "history"}; trim old history."""
    if isinstance(value, str):
        value = {"question": value}
    history: list[BaseMessage] = value.get("history") or []
    return {
        "question": value["question"],
        "history": history[-MAX_HISTORY_MESSAGES:],
    }


def _has_relevant_context(candidates: list[Document]) -> bool:
    """True if at least one retrieved chunk is similar enough to the question."""
    return any(doc.metadata.get("score", 0) >= MIN_SIMILARITY for doc in candidates)


def _print_prompt(prompt_value):
    """Debug: print the exact messages sent to the LLM, then pass them through unchanged."""
    print("\n" + "=" * 80)
    print("FINAL PROMPT SENT TO LLM")
    print("=" * 80)
    for msg in prompt_value.to_messages():
        print(f"[{msg.type.upper()}]\n{msg.content}\n")
    print("=" * 80 + "\n")
    return prompt_value


def build_rag_chain(
    stores: list[QdrantVectorStore],
    llm: BaseChatModel | None = None,
    guard: AbhiMilkGuardrails | None = None,  # <-- NEW (None = guardrails off)
) -> Runnable:
    """Build a chain: input guard -> condense -> retrieve -> rerank -> answer -> output guard.

    Input: a question string, or {"question": str, "history": [HumanMessage/AIMessage, ...]}
    where history holds the PREVIOUS turns (not the current question).

    Returns {"question", "history", "standalone_question", "candidates", "rerank",
    "docs", "context", "answer", "blocked_at", "violations"}.
      - `blocked_at` is None, "input" or "output" (retrieval filtering drops chunks silently
        and falls through to NO_ANSWER_REPLY if nothing is left).
      - `violations` lists the guardrail checks that fired.
      - If blocked at input, retrieval is skipped: `rerank` is None and `docs` is [].
    If no chunk reaches MIN_SIMILARITY, the LLM is skipped and `answer` is NO_ANSWER_REPLY.

    Pass an already-loaded `llm` to avoid reloading the model for every chain.
    """
    llm = llm or build_claude_llm()
    retriever = build_retriever(stores, guard=guard)

    condense_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", CONDENSE_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        | RunnableLambda(_print_prompt)
        | llm
        | StrOutputParser()
    )

    def condense(inputs: dict) -> str:
        """Rewrite a follow-up into a standalone question (skipped on the first turn)."""
        if not inputs["history"]:
            return inputs["question"]
        return condense_chain.invoke(inputs).strip() or inputs["question"]

    answer_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("history"),
                ("human", HUMAN_PROMPT),
            ]
        )
        | llm
        | StrOutputParser()
    )

    def answer(inputs: dict) -> str:
        """Call the LLM only when retrieval found something relevant."""
        if not _has_relevant_context(inputs["candidates"]):
            return NO_ANSWER_REPLY
        return answer_chain.invoke(inputs).strip()

    # ------------------------------------------------------------------ #
    # NEW: guardrail gates
    # ------------------------------------------------------------------ #
    def input_gate(inputs: dict) -> dict:
        """Check the RAW question (before condense) so malicious text never reaches the LLM."""
        if guard is None:
            return {**inputs, "blocked_at": None, "violations": []}
        res = guard.check_input(inputs["question"], history=inputs["history"])
        if res.passed:
            return {**inputs, "blocked_at": None, "violations": []}
        return {
            **inputs,
            "blocked_at": "input",
            "violations": res.violations,
            "answer": res.message,
        }

    def blocked_result(x: dict) -> dict:
        """Fill the keys downstream code expects when the input guard short-circuits."""
        return {
            **x,
            "standalone_question": x["question"],
            "candidates": [],
            "rerank": None,
            "docs": [],
            "context": "",
        }

    def output_gate(x: dict) -> dict:
        """Redact contact details / block PII leaks, hallucinations and brand-rule breaks."""
        if guard is None or x["answer"] in _CANNED_REPLIES:
            return x
        res = guard.check_output(x["standalone_question"], x["answer"], x["docs"])
        if not res.passed:
            return {**x, "answer": res.message, "blocked_at": "output", "violations": res.violations}
        return {**x, "answer": res.sanitized_text, "violations": res.violations}

    # ------------------------------------------------------------------ #
    # Main (unchanged) pipeline, plus the output gate at the end
    # ------------------------------------------------------------------ #
    main_chain = (
        RunnablePassthrough.assign(standalone_question=RunnableLambda(condense))
        | RunnablePassthrough.assign(
            candidates=lambda x: retriever.invoke(x["standalone_question"])
        )
        | RunnablePassthrough.assign(
            rerank=lambda x: rerank_docs(x["standalone_question"], x["candidates"])
        )
        | RunnablePassthrough.assign(docs=lambda x: x["rerank"].docs)
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(answer=RunnableLambda(answer))
        | RunnableLambda(output_gate)
    )

    return (
        RunnableLambda(_normalize_input)
        | RunnableLambda(input_gate)
        | RunnableBranch(
            (lambda x: x["blocked_at"] is not None, RunnableLambda(blocked_result)),
            main_chain,
        )
    )