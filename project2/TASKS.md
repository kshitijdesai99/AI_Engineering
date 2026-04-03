# Project 2 — Task Tracker

## Architecture

```
ONE-TIME: build_cache.py → PDF pages → corpus.json

PER QUERY:
  query → rewrite → grep corpus.json → find hits (file, page, score)
    → reader sub-agents (1 per hit, read page + neighbors)
      → each reader decides: text enough? or need page image for vision?
      → returns snippet + metadata
    → final agent synthesizes all snippets into answer
    → critic: DONE or RETRY with new keywords?
    → summarize + cite (file, page, section)
```

## Tasks

### Phase 1: Text extraction cache
- [x] Task 1.1 — `build_cache.py`: recursively scan `input/` for PDFs
- [ ] Task 1.2 — Extract text per page using `pypdf`, skip empty pages
- [ ] Task 1.3 — Store as `corpus.json`: array of `{source, page, text}`
- [ ] Task 1.4 — Add a quality check: flag pages with <20 words as `low_text`
- [ ] Task 1.5 — Test on NCERT PDFs, verify output looks sane

### Phase 2: Grep tool
- [ ] Task 2.1 — `grep_corpus(keywords, max_results)` tool: loads corpus.json, scores chunks by keyword overlap
- [ ] Task 2.2 — Return hits as `[{source, page, score, snippet (first 200 chars)}]`
- [ ] Task 2.3 — Deduplicate neighboring pages (if page 12 and 13 both hit, merge into one read range)

### Phase 3: Reader sub-agent
- [ ] Task 3.1 — `read_pages(source, start_page, end_page)` tool: returns full text for a page range from corpus.json
- [ ] Task 3.2 — Reader agent prompt: given a query + page text, extract the relevant snippet
- [ ] Task 3.3 — Reader decides: is extracted text clear enough, or is this a table/garbled layout?
- [ ] Task 3.4 — If table/garbled: `render_page_image(source, page)` tool renders PDF page as image
- [ ] Task 3.5 — Send page image to vision model, get structured text back
- [ ] Task 3.6 — Reader returns: `{snippet, source, page, method: "text"|"vision"}`

### Phase 4: Agent orchestration (LangGraph)
- [ ] Task 4.1 — Rewrite node: rewrites vague query into search keywords
- [ ] Task 4.2 — Grep node: calls grep_corpus, gets hit locations
- [ ] Task 4.3 — Reader dispatch: spawn reader sub-agents (parallel, capped at 5)
- [ ] Task 4.4 — Synthesize node: final agent gets all snippets, produces answer
- [ ] Task 4.5 — Critic node: judges answer quality, decides DONE or RETRY
- [ ] Task 4.6 — RETRY path: critic suggests new keywords, loops back to grep
- [ ] Task 4.7 — Summarize node: final answer with citations (file, page, section)

### Phase 5: Eval harness
- [ ] Task 5.1 — Create `evals/` folder with train.csv and test.csv
- [ ] Task 5.2 — Eval type: factoid extraction (single page, direct answer)
- [ ] Task 5.3 — Eval type: needle in haystack (keyword appears many times, only one is correct)
- [ ] Task 5.4 — Eval type: multi-context (answer requires 2+ pages/documents)
- [ ] Task 5.5 — Eval type: unanswerable (answer not in corpus, agent should say so)
- [ ] Task 5.6 — Eval type: table comprehension (needs vision path)
- [ ] Task 5.7 — Port `run_evals.py` from project 1 (parallel, tool diagnostics, --async)
- [ ] Task 5.8 — Add citation accuracy as a metric (did it cite the right file + page?)

### Phase 6: Polish
- [ ] Task 6.1 — Update README.md with architecture, usage, eval results
- [ ] Task 6.2 — Add `--provider` support across gemini/openai/openrouter
- [ ] Task 6.3 — Handle edge cases: empty corpus, no hits found, all pages low_text
