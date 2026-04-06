import os
from dotenv import load_dotenv
from zeroentropy import ZeroEntropy

load_dotenv()

MODEL_NAME     = "zerank-2"
MAX_CANDIDATES = 50
TOP_K          = 10

_client: ZeroEntropy | None = None


def _get_client() -> ZeroEntropy:
    global _client
    if _client is None:
        _client = ZeroEntropy()
    return _client


def rerank(query: str, candidates: list[dict], topk: int = TOP_K) -> list[dict]:
    if not candidates:
        return []

    candidates = candidates[:MAX_CANDIDATES]
    documents  = [c.get("text", "") for c in candidates]

    response = _get_client().models.rerank(
        model=MODEL_NAME,
        query=query,
        documents=documents,
    )

    return [
        {**candidates[r.index], "rerank_score": round(r.relevance_score, 6)}
        for r in sorted(response.results, key=lambda x: x.relevance_score, reverse=True)[:topk]
    ]