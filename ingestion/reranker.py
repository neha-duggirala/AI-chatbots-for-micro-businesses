"""Rerank retrieved chunks by relevance to the question."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_typesafe import Score, TypeSafeClassifier

from config import RERANK_TOP_N

QUESTION_KEY = "relevance"
INSTRUCTIONS = "Rank the following answers based on their relevance to the question."


@dataclass
class RerankResult:
    candidates: list[Document]  # all retrieved chunks, in retrieval order
    docs: list[Document]  # top-n chunks after reranking, best first
    answer: Any | None  # ScoreAnswer used for plotting (None if nothing to rerank)


@lru_cache(maxsize=1)
def get_classifier() -> TypeSafeClassifier:
    return TypeSafeClassifier(name="Reranker")


def rerank_docs(
    question: str,
    candidates: list[Document],
    top_n: int = RERANK_TOP_N,
) -> RerankResult:
    """Score every candidate against the question and keep the best `top_n`."""
    if len(candidates) < 2:
        return RerankResult(candidates, candidates[:top_n], answer=None)

    response = get_classifier().invoke(
        {
            "state": question,
            "questions": {
                QUESTION_KEY: Score(
                    instructions=INSTRUCTIONS,
                    criteria=[doc.page_content for doc in candidates],
                )
            },
        }
    )
    answer = response.answers[QUESTION_KEY]

    # answer.probabilities maps candidate index -> relevance probability
    order = sorted(
        range(len(candidates)),
        key=lambda i: answer.probabilities[i],
        reverse=True,
    )
    for i in order:
        candidates[i].metadata["rerank_score"] = round(answer.probabilities[i], 4)

    return RerankResult(
        candidates=candidates,
        docs=[candidates[i] for i in order[:top_n]],
        answer=answer,
    )