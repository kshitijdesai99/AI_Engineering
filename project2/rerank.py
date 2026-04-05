from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("zeroentropy/zerank-1-small", trust_remote_code=True)
    return _model


def rerank(query: str, candidates: list[dict], topk: int) -> list[dict]:
    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [{**c, "rerank_score": round(float(s), 6)} for s, c in ranked[:topk]]
