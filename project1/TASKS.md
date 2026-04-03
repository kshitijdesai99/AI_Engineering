# Project 1 — Task Tracker

## Tasks

### Phase 1: Eval harness
- [x] Task 1.1 — Create `run_evals.py`: reads CSV, runs agent, extracts answer, compares to ground truth
- [x] Task 1.2 — `extract_number()`: regex-based answer extraction from free-text agent output
- [x] Task 1.3 — `answers_match()`: float tolerance comparison for decimals, exact match for ints
- [x] Task 1.4 — Per-question logging: expected, extracted, latency, tool calls
- [x] Task 1.5 — Aggregate summary: accuracy, avg latency, avg tool calls, total errors
- [x] Task 1.6 — JSON output with full per-question results

### Phase 2: Parallel execution
- [x] Task 2.1 — Add `--async` flag for parallel execution via `ThreadPoolExecutor`
- [x] Task 2.2 — Add `--workers` arg to control concurrency
- [x] Task 2.3 — Replace `signal.SIGALRM` with futures-based timeout (thread-safe everywhere)
- [x] Task 2.4 — Flatten nested thread pools: parallel workers call `_stream_agent` directly

### Phase 3: Bug fixes
- [x] Task 3.1 — Fix Gemini multipart content crash: `_content_text()` helper for list-style `.content`
- [x] Task 3.2 — Patch `after_critic`, `critic_node` error check to use `_content_text()`
- [x] Task 3.3 — Fix extraction bug: strip markdown `**bold**` wrapping numbers before regex
- [x] Task 3.4 — Fix extraction regex: add `≈` / `approximately` patterns, prioritize standalone number lines

### Phase 4: Prompt engineering
- [x] Task 4.1 — Rewrite `model_prompt.txt`: force tool usage for ALL math/computation queries
- [x] Task 4.2 — Add few-shot examples (prime, geometry, trig) — no dataset leakage
- [x] Task 4.3 — Instruct clean `print()` output: raw number only, no prose

### Phase 5: Router + sub-agents
- [x] Task 5.1 — Create `router_prompt.txt`: classifies queries as COMPUTE / VISUALIZE / CHAT
- [x] Task 5.2 — Create `viz_prompt.txt`: dedicated prompt for chart/plot generation
- [x] Task 5.3 — Add router node to LangGraph: single LLM call, dispatches to 3 paths
- [x] Task 5.4 — Add `viz_model` node: uses `viz_prompt.txt`, shares tools + critic with compute path
- [x] Task 5.5 — Add `chat_model` node: no tools, direct LLM answer, exits to END
- [x] Task 5.6 — Add `viz_summarize` node: extracts file paths from tool output, skips prose generation
- [x] Task 5.7 — `after_critic` routes CONTINUE back to correct model node based on `query_type`
- [x] Task 5.8 — Update `AgentState` with `query_type` field

### Phase 6: Tool diagnostics
- [x] Task 6.1 — `_build_meta()` checks if tool raw output contains correct answer (`tool_correct`)
- [x] Task 6.2 — Per-question output shows `tool=✅(N)` / `tool=❌(N)` / `tool=⛔ (none)`
- [x] Task 6.3 — Summary shows tool usage breakdown: used / correct / wrong
- [x] Task 6.4 — Failure diagnosis table in README: final × tool status → root cause

### Phase 7: Documentation
- [x] Task 7.1 — Update README.md: eval results (OpenAI 20/20, Gemini 20/20), architecture diagram
- [x] Task 7.2 — Document eval CLI args, per-question output legend, summary metrics
- [x] Task 7.3 — Add ASCII architecture diagram to `docker_agent.py` module docstring
- [x] Task 7.4 — Update Gemini model to `gemini-3.1-flash-lite-preview`
- [x] Task 7.5 — Document provider models table in README

## Results

| Provider | Model | Train (20) | Test (20) |
|---|---|---|---|
| OpenAI | gpt-4o-mini | 20/20 (100%) | 20/20 (100%) |
| Gemini | gemini-3.1-flash-lite-preview | 20/20 (100%) | 20/20 (100%) |
