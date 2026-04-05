# Project 2 — BM25 Document Retrieval Eval

Builds a searchable text corpus from a directory of PDFs and evaluates BM25-based keyword retrieval against ground-truth question/answer pairs. No LLM involved — pure lexical search.

## Eval Results

| File | Questions | Score |
|---|---|---|
| `evals/easy.csv` | 18 | **17/18 (94.4%)** |
| `evals/hard.csv` | 9 | **9/9 (100%)** |

## Files

| File | Description |
|---|---|
| `build_cache.py` | Extracts text from every PDF in `input/` and saves it to `corpus.json` |
| `keyword_search.py` | BM25 retrieval eval — scores corpus pages per question, checks if the correct page is in top-k |
| `corpus.json` | Pre-built text index (page-level chunks from all PDFs) |
| `evals/easy.csv` | 18 factual questions with source PDF, page, and expected answer |
| `evals/hard.csv` | 9 harder questions with source PDF, page, and expected answer |

## Setup

```bash
uv sync
```

No API keys required.

## Usage

### Build the corpus

Run once before using `keyword_search.py`. Skip if `corpus.json` already exists.

```bash
python build_cache.py
```

```bash
# Force rebuild
python build_cache.py --force

# Custom paths
python build_cache.py --input input/ --output corpus.json
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `input/` | Directory to scan for PDFs (recursive) |
| `--output` | `corpus.json` | Output path for the text index |
| `--force` | off | Overwrite existing `corpus.json` |

### Run the retrieval eval

```bash
python keyword_search.py --file evals/easy.csv
python keyword_search.py --file evals/hard.csv
```

```bash
# Custom top-k and output path
python keyword_search.py --file evals/easy.csv --topk 10 --out evals/my_results.json
```

| Argument | Default | Description |
|---|---|---|
| `--file` | `evals/easy.csv` | Eval CSV (columns: `Question`, ` PDF`, ` Page`, ` Answer`) |
| `--out` | auto-generated | Output JSON path (defaults to `evals/results_{file}_keyword.json`) |
| `--topk` | `5` | Number of top pages to retrieve per question |

### Interpreting eval output

**Per-question line:**

```
[1/18] ✅  expected=leph101.pdf:3  top=leph101.pdf:p3  | What simple apparatus is used to detect charge on a body?
[2/18] ❌  expected=leph103.pdf:2  top=keph101.pdf:p2  | What is the SI unit of current?
```

| Symbol | Meaning |
|---|---|
| `✅` | Correct source PDF + page found within top-k results |
| `❌` | Expected page not found in top-k |
| `expected=` | Ground-truth source and page from the CSV |
| `top=` | Highest-scored result returned by BM25 |

**Summary line:**

```
17/18 correct (94.4%) in 0.01s — evals/results_easy_keyword.json
```

## How It Works

```
Question
    ↓
[extract_keywords]  — lowercase, tokenize, remove stop words
    ↓
[BM25Okapi.get_scores]  — score all 733 corpus pages
    ↓
[top-k results]  — sort by score, take top-k
    ↓
[is_hit]  — check if expected source + page appears in top-k
    ↓
✅ / ❌
```

**Key design decisions:**

- **BM25Okapi** (via `rank_bm25`) — weights rare terms higher and penalizes long pages, outperforming simple hit-count scoring
- **Index built once per eval run** — `build_bm25()` tokenizes all 733 pages upfront, then each query is a single `get_scores()` call (~0.01s total)
- **Page-level retrieval** — corpus chunks are individual PDF pages, so retrieval is precise to the page
- **No LLM** — entirely deterministic; same query always returns the same ranked list
