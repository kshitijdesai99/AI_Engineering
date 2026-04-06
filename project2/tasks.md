# Project 2 — Re-Ranking Pipeline Tasks

## Goal
Extend the existing BM25-only retrieval into a full two-stage retrieve-then-rerank pipeline, then pass the top results to an LLM for answer generation.

## Current State
- `build_cache.py` — fixed-size chunking (800 chars, 200 overlap) → `corpus.json` (3,241 chunks from 37 PDFs)
- `keyword_search.py` — BM25 retrieval, `--retrieval_k` (default 100), `--topk` (default 5)
- `rerank.py` — ZeroEntropy API reranker (`zerank-2`), `MAX_CANDIDATES=50`, `TOP_K=5`
- `rerank_search.py` — two-stage eval: BM25 → rerank, 17/17 (100%) on `evals/easy.csv`

---

## Tasks

### ~~1. Expand BM25 retrieval to top-100~~ ✅
### ~~2. Build `rerank.py` — Cross-encoder re-ranker (Stage 2)~~ ✅
### ~~3. Build `rerank_search.py` — Two-stage eval script~~ ✅
### ~~4. Add cross-encoder dependency to `pyproject.toml`~~ ✅ (switched to `zeroentropy` API)

### 5. Build `answer.py` — LLM answer generation (Stage 3)
- Input: query + top-5 re-ranked chunks (text + source + page + chunk_index)
- Prompt: system prompt with retrieved context, user query
- Use Gemini or OpenAI via env var `ANSWER_MODEL` (default `gemini/gemini-2.0-flash`)
- Output: printed answer + source citations
- Args: `--query`, `--topk` (default 5), `--retrieval_k` (default 100)

### 6. Eval comparison
- Run both `keyword_search.py` and `rerank_search.py` against `evals/easy.csv` and `evals/hard.csv`
- Compare accuracy: BM25-only vs BM25 + ZeroEntropy reranker
- Results saved to `evals/results_{file}_rerank.json`
