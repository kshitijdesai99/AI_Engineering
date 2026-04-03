# Project 2 — Agentic Document Search Agent

An LLM agent that answers questions from PDF textbooks by searching a prebuilt page-level cache in `corpus.json`.

## Workflow

1. Drop PDFs into `input/`
2. Put your question in `query.txt`
3. Build or rebuild the cache
4. Run the agent with your preferred provider

```bash
cp .env_example .env
# Fill in your API keys

python build_cache.py
python search_agent.py openai      # or gemini / openrouter
```

If `corpus.json` already exists and you changed the PDFs, rebuild with:

```bash
python build_cache.py --force
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` | Google Gemini |
| `OPEN_ROUTER_API_KEY` | OpenRouter |
| `LANGSMITH_API_KEY` | LangSmith tracing |
| `MAX_TOOL_CALLS` | Max tool calls per query loop, default `6` |

## What The Agent Does

- Rewrites the user query into a search-friendly form
- Preserves exact numbers and literal wording for copied textbook problems
- Searches `corpus.json` with phrase-aware and number-aware scoring
- Forces a read of the top individual page hit for detailed worked problems
- Extracts direct worked answers from the retrieved page instead of recomputing them
- Falls back to broader reading or vision tools when text alone is insufficient

## Architecture

```text
User query
    ↓
[rewrite]
    - keep literal details for exact textbook problems
    - suggest scope when useful
    ↓
[grep_corpus]
    - page-level search over corpus.json
    - boosts exact phrases, rare terms, and numeric matches
    ↓
[critic]
    - for detailed copied problems, force-read the top individual page hit
    - otherwise continue searching or broaden scope if needed
    ↓
[read_pages / vision tools]
    - read the exact page or short range
    - use vision only when the answer depends on visual content
    ↓
[summarize]
    - prefer explicit document answers over fresh recomputation
    - always cite the source page
```

## Tools

- `grep_corpus(query, scope="", max_results=5)` — ranked page search over `corpus.json`
- `read_pages(source, start_page, end_page=0)` — reads one or more cached pages from a document
- `list_documents()` — lists available scopes and documents in the cache
- `read_page_vision(source, page)` — OCR-style visual extraction for a rendered PDF page
- `answer_from_page_vision(source, page, question)` — visual question answering for chart/table/image-heavy pages

## Retrieval Notes

- Exact worked examples are handled better when the original question text is passed in full.
- For copied textbook problems, the agent now prefers the page containing the full example statement and answer over generic theory pages.
- When a retrieved page already contains the solved result, the final answer is taken from that page directly and cited.

## Supported Documents

Current cache-building support is PDF-based:

- `.pdf`
