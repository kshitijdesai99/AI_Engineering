# Project 2 — BM25 + Rerank + LLM Answer Pipeline

Two-stage retrieval (BM25 → ZeroEntropy reranker) with LLM answer generation over a chunked PDF corpus.

## Pipeline

```
Question
    ↓
[BM25]  — retrieve top-100 candidate chunks by keyword overlap
    ↓
[ZeroEntropy Reranker (zerank-2)]  — score & re-rank, keep top-10
    ↓
[Expand Context]  — add all chunks from prev/next page of each hit
    ↓
[LLM (Gemini / OpenAI)]  — generate answer with source citations
    ↓
Answer
```

## Eval Results

All rerank results use `topk=10`.

| File | Questions | BM25 Only | BM25 + Rerank (top 10) |
|---|---|---|---|
| `evals/easy.csv` | 17 | **17/17 (100%)** | **17/17 (100%)** |
| `evals/hard.csv` | 9 | **9/9 (100%)** | **9/9 (100%)** |

## Files

| File | Description |
|---|---|
| `build_cache.py` | Extracts PDF text, splits into 800-char chunks (200 overlap) → `corpus.json` |
| `retrieval.py` | Shared BM25 utilities: `load_corpus`, `build_bm25`, `tokenize`, `search`, `is_hit` |
| `eval_keyword.py` | BM25-only retrieval eval — checks if correct page is in top-k |
| `rerank.py` | ZeroEntropy API reranker — scores `(query, chunk)` pairs, returns top-k |
| `eval_rerank.py` | Two-stage eval: BM25 → rerank |
| `answer.py` | Full pipeline: BM25 → rerank → expand context → LLM answer |
| `corpus.json` | Pre-built chunk index (3,241 chunks from 37 PDFs) |
| `evals/easy.csv` | 17 factual questions with source PDF + page |
| `evals/hard.csv` | 9 harder questions with source PDF + page |

## Setup

```bash
uv sync
```

Required env vars in `.env`:

| Key | Used by |
|---|---|
| `ZEROENTROPY_API_KEY` | `rerank.py` |
| `GEMINI_API_KEY` | `answer.py` (default provider) |
| `OPENAI_API_KEY` | `answer.py` (optional) |

## Usage

### Build the corpus

```bash
uv run python build_cache.py --force
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `input/` | Directory to scan for PDFs (recursive) |
| `--output` | `corpus.json` | Output path |
| `--force` | off | Overwrite existing `corpus.json` |

### Run evals

```bash
uv run python eval_keyword.py --file evals/easy.csv
uv run python eval_rerank.py --file evals/hard.csv
```

| Argument | Default | Description |
|---|---|---|
| `--file` | `evals/easy.csv` | Eval CSV |
| `--out` | auto | Output JSON path |
| `--retrieval_k` | `100` | BM25 candidate pool |
| `--topk` | `5` / `10` | Final chunks returned |

### Ask a question

Edit `query.txt` with your question, then run:

```bash
uv run python answer.py
```

Or pass the query directly:

```bash
uv run python answer.py --query "What is Coulomb's law?"
uv run python answer.py --query "What is Coulomb's law?" --provider openai
```

| Argument | Default | Description |
|---|---|---|
| `--query` | `query.txt` | Question to answer (overrides `query.txt`) |
| `--topk` | `10` | Reranked chunks passed to expand + LLM |
| `--retrieval_k` | `100` | BM25 candidate pool |
| `--provider` | `gemini` | `gemini` or `openai` |
