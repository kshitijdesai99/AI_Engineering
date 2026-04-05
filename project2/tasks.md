# Project 2 — Re-Ranking Pipeline Tasks

## Goal
Extend the existing BM25-only retrieval into a full two-stage retrieve-then-rerank pipeline, then pass the top results to an LLM for answer generation.

## Current State
- `build_cache.py` — extracts PDF pages into `corpus.json` (733 pages)
- `keyword_search.py` — BM25 retrieval eval (top-k=5), no LLM, 94.4% / 100% accuracy

---

## Tasks

### 1. Expand BM25 retrieval to top-100
- In `keyword_search.py`, the current default `topk=5` is the final output
- For Stage 1, BM25 needs to return a wider candidate set (e.g. top-100) before re-ranking narrows it down
- Add a `--retrieval_k` arg (default 100) separate from `--topk` (final output size, default 5)

### 2. Build `rerank.py` — Cross-encoder re-ranker (Stage 2)
- Use `sentence-transformers` cross-encoder: `zeroentropy/zerank-1-small` with `trust_remote_code=True`
- Input: query string + list of top-100 BM25 candidates (text + metadata)
- Output: same list re-ordered by cross-encoder relevance score, truncated to top-k (e.g. 5)
- Keep it a pure function: `rerank(query: str, candidates: list[dict], topk: int) -> list[dict]`

### 3. Build `rerank_search.py` — Two-stage eval script
- Mirror `keyword_search.py` structure but with two-stage retrieval:
  1. BM25 fetches top-100 candidates
  2. Cross-encoder re-ranks → top-5
- Reuse `load_corpus`, `build_bm25`, `tokenize`, `is_hit` from `keyword_search.py`
- Args: `--file`, `--out`, `--retrieval_k` (default 100), `--topk` (default 5)
- Output JSON: `evals/results_{file}_rerank.json` — same schema as keyword results for direct comparison

### 4. Add cross-encoder dependency to `pyproject.toml`
- Add `sentence-transformers` (includes cross-encoder support)

### 5. Build `answer.py` — LLM answer generation (Stage 3)
- Input: query + top-5 re-ranked pages (text + source + page)
- Prompt: system prompt with retrieved context, user query
- Use Gemini or OpenAI via env var `ANSWER_MODEL` (default `gemini/gemini-2.0-flash`)
- Output: printed answer + source citations
- Args: `--query`, `--topk` (default 5), `--retrieval_k` (default 100)

### 6. Eval comparison
- Run both `keyword_search.py` and `rerank_search.py` against `evals/easy.csv` and `evals/hard.csv`
- Compare accuracy: BM25-only vs BM25 + cross-encoder
- Results saved to `evals/results_{file}_rerank.json`
