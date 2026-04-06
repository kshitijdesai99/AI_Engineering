"""ZeroEntropy reranker — scores (query, chunk) pairs via API, returns top-k chunks."""
import os
from dotenv import load_dotenv
from zeroentropy import ZeroEntropy

load_dotenv()

MODEL_NAME     = "zerank-2"  # ZeroEntropy reranker model
MAX_CANDIDATES = 50           # Cap BM25 candidates before sending to reranker
TOP_K          = 10           # Number of top chunks to return after reranking

_client: ZeroEntropy | None = None  # Lazy singleton client


def _get_client() -> ZeroEntropy:
    global _client
    if _client is None:
        _client = ZeroEntropy()  # Reads ZEROENTROPY_API_KEY from env
    return _client


def rerank(query: str, candidates: list[dict], topk: int = TOP_K) -> list[dict]:
    if not candidates:
        return []

    candidates = candidates[:MAX_CANDIDATES]                    # Limit to MAX_CANDIDATES before API call
    documents  = [c.get("text", "") for c in candidates]       # Extract raw text for the reranker

    response = _get_client().models.rerank(
        model=MODEL_NAME,
        query=query,
        documents=documents,
    )

    # Map result indices back to original candidate dicts, attach relevance score, return top-k
    return [
        {**candidates[r.index], "rerank_score": round(r.relevance_score, 6)}
        for r in sorted(response.results, key=lambda x: x.relevance_score, reverse=True)[:topk]
    ]