# Project 2 — Agentic Document Search Agent

An LLM agent that answers questions by searching documents in the `input/` directory using agentic grep-style tools.

## Usage

```bash
python search_agent.py openai      # or gemini / openrouter
```

Put your query in `query.txt`, drop documents into `input/`, then run.

## Setup

```bash
cp .env_example .env
# Fill in your API keys
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` | Google Gemini |
| `OPEN_ROUTER_API_KEY` | OpenRouter |
| `LANGSMITH_API_KEY` | LangSmith tracing |
| `MAX_TOOL_CALLS` | Max search calls per query (default: 3) |

## Architecture

```
User query
    ↓
 [rewrite] — Rewrites vague query into precise, document-grounded query
    ↓
 [model] — Decides which tool to use (search_documents, extract_code_patterns, list_documents)
    ↓ tool call                   ↓ direct answer
 [search]                      [summarize]
    ↓                               ↓
 [critic] — DONE or CONTINUE?      END
    ↓ CONTINUE     ↓ DONE
 [model]       [summarize] → END
```

## Tools

- **search_documents** — keyword/scored search across all documents
- **extract_code_patterns** — regex extraction from Python files (for code structure questions)
- **list_documents** — lists available files in `input/`

## Supported Document Types

`.txt`, `.md`, `.pdf`, `.py`
