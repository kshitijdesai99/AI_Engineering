# Project 2 — Rebuild Plan (Eval-Driven)

Build one layer at a time. Measure before adding the next.

```
Stage 0  Eval harness        → retrieval accuracy, no LLM
Stage 1  Retrieval baseline  → tune grep until top-3 ≥ 80% on easy
Stage 2  Single-page answer  → grep → read → LLM extract, one shot
Stage 3  Query rewriting     → add rewrite only if it shows positive delta
Stage 4  Critic + retry      → add loop only if it improves hard.csv
Stage 5  Vision path         → add vision only for figure/table pages
Stage 6  Synthesis           → multi-page answers if ≥ 3 such questions
```

---

## Keep As-Is

- `build_cache.py`, `corpus.json`, `search_tool.py`, `evals/easy.csv`, `evals/hard.csv`
- **Delete and rebuild:** `search_agent.py`

---

## Stage 0 — Eval Harness

Build `run_evals.py`. No LLM. Run after every stage.

- Load CSV. Strip `project2/input/` prefix from gold PDF paths to match corpus keys.
- Call `grep_corpus(question, max_results=10)`, parse `Top individual pages:` block.
- Check if `(gold_source, gold_page)` is in top-1 and top-3.
- Skip rows with blank `Page` (hard.csv), mark as `no_gold_page`.
- `--csv evals/easy.csv` flag; default runs both.

Output:
```
easy.csv  — top-1: 14/20 (70%)   top-3: 17/20 (85%)
hard.csv  — top-1: 8/20  (40%)   top-3: 12/20 (60%)   [7 no_gold_page skipped]
```

**Gate:** runs cleanly on both CSVs.

---

## Stage 1 — Retrieval Baseline

- Run Stage 0, record numbers.
- For each miss: inspect what `grep_corpus` returns and why the gold page ranked low.
- Fix only `_score_chunk` / `_extract_keywords` in `search_tool.py` if needed.
- Re-run after each fix.

```
Stage 1 result:  easy top-1 __/20  top-3 __/20  |  hard top-1 __/20  top-3 __/20
```

**Gate:** easy top-3 ≥ 80%.

---

## Stage 2 — Minimal Single-Page Answer

`search_agent.py` v1 — plain function, no LangGraph:
```
query → grep_corpus → top hit → read_pages → LLM extract → answer + citation
```

Add `answer_match` metric to `run_evals.py` (gold answer substring in response, case-insensitive).

```
Stage 2 result:  easy top-3 __/20  answer_match __/20
```

**Gate:** easy answer_match ≥ 70%.

---

## Stage 3 — Query Rewriting

Add one LLM rewrite step before `grep_corpus`. A/B against Stage 2.
Keep only if net positive delta on both CSVs.

```
Stage 3 result:  easy __/20  hard __/20
```

---

## Stage 4 — Critic + Retry Loop

Convert to LangGraph: `rewrite → model → search → critic → summarize`.
Critic: DONE if citation found, CONTINUE otherwise. Cap at `MAX_TOOL_CALLS = 6`.

```
Stage 4 result:  easy __/20  hard __/20  avg_tool_calls __
```

**Gate:** hard.csv improves over Stage 3.

---

## Stage 5 — Vision Path

Trigger `answer_from_page_vision` only when:
- page tagged `[LOW TEXT]`, OR
- page tagged `[HAS FIGURE/TABLE REFERENCE]` AND question asks for a plotted/tabulated value.

Add `type` column to `hard.csv` (`text | figure | multi_page`). Report metrics by type.

```
Stage 5 result:  hard __/20  — text __/__  figure __/__  multi_page __/__
```

**Gate:** figure questions improve over Stage 4.

---

## Stage 6 — Multi-Page Synthesis

Only build if `hard.csv` has ≥ 3 `multi_page` questions.
`model` reads multiple ranges per loop; `summarize` synthesizes across all snippets.

```
Stage 6 result:  easy __/20  hard __/20
```

---

## Score Tracker

| Stage | easy top-3 | easy ans | hard top-3 | hard ans | notes              |
|-------|-----------|----------|-----------|----------|--------------------|
| 1     |           |          |           |          | retrieval only     |
| 2     |           |          |           |          | +answer            |
| 3     |           |          |           |          | +rewrite           |
| 4     |           |          |           |          | +loop              |
| 5     |           |          |           |          | +vision            |
| 6     |           |          |           |          | +synthesis         |
