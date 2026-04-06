# Project 1 — LLM Code Execution Agent

Generates and executes Python code inside a persistent Docker container. Routes queries through a classifier to specialized sub-agents (compute, visualize, chat). Traces all runs to LangSmith. Includes a parallel eval harness with tool-level diagnostics.

## Eval Results

| Provider | Model | Train (20) | Test (20) | Viz |
|---|---|---|---|---|
| **OpenAI** | `gpt-4o-mini` | **20/20 (100%)** | **20/20 (100%)** | ✅ |
| **Gemini** | `gemini-3.1-flash-lite-preview` | **20/20 (100%)** | **20/20 (100%)** | ✅ |

## Files

| File | Description |
|---|---|
| `docker_agent.py` | LangGraph agent with router, compute/viz sub-agents, critic, and summarize nodes |
| `docker_tool.py` | LangChain tool that runs code in a persistent Docker container |
| `run_evals.py` | Eval runner: sequential or parallel, multi-provider, with tool diagnostics |
| `Dockerfile` | Image with `uv`, `numpy`, `matplotlib`, `pandas`, `seaborn` pre-installed |
| `model_prompt.txt` | System prompt for the compute sub-agent (math/calculations) |
| `viz_prompt.txt` | System prompt for the visualization sub-agent (charts/plots) |
| `router_prompt.txt` | Classifier prompt — routes queries to COMPUTE, VISUALIZE, or CHAT |
| `critic_prompt.txt` | Judge prompt (DONE / CONTINUE) |
| `summarize_prompt.txt` | Final answer synthesis prompt (compute path only) |
| `evals/train.csv` | 20 math questions for prompt tuning |
| `evals/test.csv` | 20 held-out math questions for validation |

## Setup

```bash
cp .env_example .env   # fill in API keys
docker build -t code-executor:latest .
```

## Usage

### Run the agent interactively

Edit `query.txt` with your query, then run:

```bash
uv run python docker_agent.py
```

Or pass the query directly:

```bash
uv run python docker_agent.py --query "What is the 100th prime number?"
uv run python docker_agent.py --query "Plot a sine wave" --provider openai
```

| Argument | Default | Description |
|---|---|---|
| `--provider` | `gemini` | LLM provider: `gemini`, `openai`, or `openrouter` |
| `--query` | `query.txt` | Query to run (overrides `query.txt`) |

### Run evals

```bash
# sequential
python run_evals.py gemini --file evals/train.csv

# parallel (3 concurrent workers)
python run_evals.py openai --file evals/test.csv --async --workers 3

# custom output path
python run_evals.py gemini --file evals/train.csv --out evals/my_results.json
```

| Argument | Default | Description |
|---|---|---|
| `provider` | `gemini` | LLM provider: `gemini`, `openai`, or `openrouter` |
| `--file` | `evals/train.csv` | Path to eval CSV (columns: `question`, `answer`) |
| `--out` | auto-generated | Output JSON path (defaults to `evals/results_{file}_{provider}.json`) |
| `--async` | off | Run questions in parallel using a thread pool |
| `--workers` | `5` | Number of parallel workers (only applies with `--async`) |

### Interpreting eval output

**Per-question line:**

```
[1/20] ✅  expected=541  extracted=541  tool=✅(1)  latency=5.2s  | What is the 100th prime number?
[2/20] ❌  expected=3.0  extracted=1000  tool=❌(1)  latency=3.9s  | What is log base 10 of 1000?
[3/20] ❌  expected=42   extracted=None  tool=⛔ (none)  latency=1.2s  | What is 6 * 7?
```

| Symbol | Meaning |
|---|---|
| `✅` / `❌` | Final answer matched / did not match expected |
| `tool=✅(N)` | Tool called N times, raw output contained the correct answer |
| `tool=❌(N)` | Tool called N times, raw output did NOT contain the correct answer |
| `tool=⛔ (none)` | Agent answered directly without calling the tool |
| `⚠️ recursion limit hit` | Agent exceeded `MAX_TOOL_CALLS` loop iterations |
| `⚠️ timed out` | Agent exceeded `AGENT_TIMEOUT` seconds |

**Summary block:**

```
======================================================================
  RESULTS: 20/20 correct (100.0%)
  Tool usage: 20/20 used tool | 20 tool✅ | 0 tool❌
  Wall time: 34.2s (vs ~99.0s sequential est.)
  Avg latency: 4.95s | Avg tool calls: 1.0
  Total errors: 0
  Saved to: evals/results_test_gemini.json
======================================================================
```

| Metric | Description |
|---|---|
| **correct** | Questions where extracted answer matched ground truth |
| **tool✅ / tool❌** | Whether the Docker tool output itself was correct (distinguishes tool bugs from extraction bugs) |
| **wall time** | Total elapsed time; parallel mode shows estimated sequential time for comparison |
| **avg latency** | Average per-question wall-clock time |
| **avg tool calls** | Average `execute_docker` invocations per question (ideally 1.0) |
| **total errors** | Tool executions that produced Python errors/tracebacks |

**Diagnosing failures:**

| Final | Tool | Diagnosis |
|---|---|---|
| ✅ | ✅(1) | Working as intended |
| ❌ | ✅(1) | Tool was correct but **extraction** failed — fix `extract_number` in `run_evals.py` |
| ❌ | ❌(1) | LLM generated **wrong code** — inspect `agent_raw` and `tool_raw` in results JSON |
| ❌ | ⛔ (none) | Agent **skipped the tool** — strengthen `model_prompt.txt` |
| ❌ | ❌(3) | Repeated code errors, hit **consecutive error guard** — check missing packages |

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini |
| `OPENAI_API_KEY` | — | OpenAI |
| `OPEN_ROUTER_API_KEY` | — | OpenRouter |
| `LANGSMITH_API_KEY` | — | LangSmith tracing |
| `MAX_TOOL_CALLS` | 3 | Max docker executions per query |
| `AGENT_TIMEOUT` | 120 | Wall-clock timeout in seconds |

## Architecture

```
User query
    ↓
 [router] — LLM classifies: COMPUTE / VISUALIZE / CHAT
    ↓ compute            ↓ visualize            ↓ chat
 [model]              [viz_model]           [chat_model] → END
  (model_prompt.txt)   (viz_prompt.txt)       (no tools)
    ↓                    ↓
 [tools] ←────────── [tools]         (shared Docker executor)
    ↓                    ↓
 [critic] ←────────── [critic]       (shared judge)
    ↓ CONTINUE           ↓ CONTINUE
 [model]              [viz_model]
    ↓ DONE               ↓ DONE
 [summarize]          [viz_summarize]
  (LLM synthesis)      (extract file paths)
    ↓                    ↓
   END                  END
```

**Router** classifies every query into one of three paths with a single LLM call (adds ~0.5s overhead). This keeps the compute prompt optimized for math evals (100% accuracy) while the viz prompt is tuned for chart generation — neither interferes with the other.

**Key design decisions:**

- **Persistent container** reused across runs (ID stored in `.container_id`)
- **On-demand packages** installed via `uv pip install`
- **Critic node** stops the loop early when the task is satisfied, preventing unnecessary extra tool calls
- **Summarize node** (compute path) — LLM synthesizes tool results into a final answer
- **Viz summarize node** (viz path) — extracts saved file paths from tool output, skips prose generation
- **Error truncation** — errors capped at 500 chars before being fed back to the LLM
- **Consecutive error guard** — exits after 3 repeated failures to avoid infinite loops
- **Multipart content safety** — `_content_text()` helper handles Gemini's list-style `.content` fields
- **LangSmith tracing** — all runs traced to the `docker-agent` project

## Provider Models

| Provider | Model | Notes |
|---|---|---|
| `gemini` | `gemini-3.1-flash-lite-preview` | Returns multipart content (list), handled by `_content_text()` |
| `openai` | `gpt-4o-mini` | Reliable tool calling, string content |
| `openrouter` | `minimax/minimax-m2.5:free` | Free tier via OpenRouter |
