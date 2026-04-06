# Corpus Statistics

## Chunking Config
- `CHUNK_SIZE = 800` chars, `CHUNK_OVERLAP = 200` chars

| Metric | Value (chars) |
|--------|--------------|
| Chunks | 3,241        |
| Avg    | 681          |
| Median | 800          |
| Max    | 800          |
| Min    | 2            |

## Pipeline Context Budget

| Stage | Count |
|---|---|
| BM25 candidates | 100 |
| Reranker input cap (`MAX_CANDIDATES`) | 50 |
| Reranker output (`TOP_K`) | 10 chunks |
| Context expansion (prev + hit + next page) | up to 3× unique pages per hit |
| **Worst case pages sent to LLM** | **30** (10 hits × 3 pages, all different sources) |
| **Typical pages sent to LLM** | 10–20 (hits cluster on same pages, dedup reduces count) |

Deduplication is by `(source, page)` — if multiple top-10 chunks share a page, that page's chunks are only included once.