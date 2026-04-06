# Preferences

## Language & Runtime

- Python 3.12+
- `uv` for dependency management (`pyproject.toml`, `uv.lock`, `uv sync`, `uv run`)
- `.python-version` file at repo root
- `.env` + `python-dotenv` for secrets; `.env_example` committed with placeholder values

## Code Style

### Naming

- `snake_case` for functions, variables, files
- `UPPER_SNAKE` for module-level constants
- Private helpers prefixed with `_` (e.g. `_load_prompt`, `_content_text`, `_get_client`)
- `TypedDict` for structured state (e.g. `AgentState`)

### Imports

- stdlib → third-party → local, separated by blank lines
- Explicit imports only — no wildcard `*`
- Type hints from `typing` / `typing_extensions` as needed

### Structure

- `_DIR = os.path.dirname(os.path.abspath(__file__))` at module top for relative paths
- Constants defined at module level, below imports
- `if __name__ == "__main__":` block at bottom for CLI entry
- `argparse` for CLI args with sensible defaults
- Auto-generated output paths when `--out` not specified (pattern: `results_{stem}_{provider}.json`)

### Functions

- Docstrings on public functions and module level — triple-quoted `"""`
- Inline comments on the same line or directly above for non-obvious logic
- Comments explain *why*, not *what* — marked with `#` and a space
- Emoji sparingly in CLI output only (`✅`, `❌`, `⛔`, `⚠️`)
- Helper functions stay small and single-purpose

### Error Handling

- `try/except/finally` with specific exception types
- Truncate long error output before feeding to LLMs (500 char cap)
- Graceful fallbacks: return partial results over crashing
- Timeouts via `ThreadPoolExecutor` + `future.result(timeout=...)` (thread-safe, no signals)

### Typing

- Type hints on function signatures: `def rerank(query: str, candidates: list[dict], topk: int = TOP_K) -> list[dict]`
- Use built-in generics (`list[str]`, `dict[str, int]`) over `typing.List`, `typing.Dict`
- `str | None` over `Optional[str]`

## Project Layout

```
projectN/
├── .env / .env_example
├── *.py              # flat — no src/ or nested packages
├── *_prompt.txt      # prompt files separate from code
├── query.txt         # editable input query
├── evals/            # CSV inputs + JSON results
├── input/            # raw data (PDFs, etc.)
├── output/           # generated artifacts (plots, CSVs)
├── TASKS.md          # task tracker
└── README.md         # project docs
```

- Flat file layout — all Python files at project root
- Prompts in `.txt` files, loaded at module level via `_load_prompt()`
- Evals as CSV in/JSON out under `evals/`
- No test files, no `src/` package, no `__init__.py`

## Documentation Style

### Module Docstrings

- Every `.py` file starts with a `"""one-liner description."""` as the very first line
- Library modules (no CLI): one-liner only — e.g. `"""ZeroEntropy reranker — scores pairs, returns top-k."""`
- Script modules (with CLI): one-liner → blank line → `Usage:` block with CLI example

### README.md

- Title: `# Project N — Short Description`
- Sections in order: intro → pipeline/architecture (ASCII diagram) → eval results table → files table → setup → usage (with CLI examples + arg tables) → config table
- Tables for: eval results, file descriptions, CLI args, env vars, provider models
- ASCII art for architecture diagrams (inline in README and in module docstrings)
- Code blocks with `bash` fence for CLI examples

### TASKS.md

- Grouped by phase with numbered tasks (`Task N.M — description`)
- `[x]` checkboxes for completion tracking
- `~~strikethrough~~` + ✅ for completed phase headers
- Results table at bottom

### Inline Comments

- `# ── section label ──────────` dividers for major code blocks
- End-of-line comments for config values: `MAX_TOOL_CALLS = 3  # Maximum number of tool calls allowed`
- Step-by-step comments in complex flows (e.g. agent graph wiring)

## Dependencies & Tooling

- LangChain + LangGraph for agent orchestration
- LangSmith for tracing (`LANGCHAIN_TRACING_V2=true`)
- Multi-provider LLM support via `PROVIDERS` dict pattern (Gemini, OpenAI, OpenRouter)
- `rank-bm25` for keyword retrieval
- `zeroentropy` for reranking
- Docker for sandboxed code execution
- `pypdf` for PDF text extraction

## Eval Pattern

- CSV input (`question,answer` or `Question, PDF, Page`)
- JSON output with per-item results + summary stats
- Accuracy as `correct/total (pct%)`
- Console output: `[i/total] ✅/❌  expected=X  extracted=Y  ...  | question_preview`
- Auto-generated output path: `evals/results_{stem}_{method}.json`
- Support for sequential and parallel (`--async --workers N`) execution
