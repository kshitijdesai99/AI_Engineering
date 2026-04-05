# Tasks

## Task 1 - Keywords Based Search in corpus.json

Retrieval eval using literal keyword matching against corpus.json — no LLM involved.

- Extracts keywords from each question (stop-word filtered)
- Scores corpus pages by keyword hit count (substring match)
- Reports whether the correct source PDF + page appears in top-k results

**Usage**: `python project2/keyword_search.py [--file evals/easy.csv] [--topk 5]`
