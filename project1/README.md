# Project 1 — LLM Code Execution Agent

Generates and executes Python code inside a persistent Docker container. Traces all runs to LangSmith.

## Files

- **`docker_agent.py`** — LangGraph agent with critic node, accepts a provider argument
- **`docker_tool.py`** — LangChain tool that runs code in a persistent Docker container
- **`Dockerfile`** — image with `uv`, `numpy`, `matplotlib`, `pandas`, `seaborn` pre-installed

## Setup

```bash
cp .env_example .env   # fill in API keys
docker build -t code-executor:latest .
```

## Usage

```bash
python docker_agent.py [gemini|openai|openrouter]
```

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
 [model] — LLM decides: answer directly or call execute_docker
    ↓ tool call                        ↓ direct answer
 [tools] — Docker runs code          → END
    ↓
 [critic] — Judge LLM: DONE or CONTINUE?
    ↓ CONTINUE          ↓ DONE
 [model]            [summarize] — LLM writes final answer from results
                        ↓
                       END
```

- **Persistent container** reused across runs (ID stored in `.container_id`)
- **On-demand packages** installed via `uv pip install`
- **Critic node** stops the loop early when the task is satisfied, preventing unnecessary extra tool calls
- **Summarize node** — dedicated LLM call to synthesize tool results into a clear final answer
- **Error truncation** — errors capped at 500 chars before being fed back to the LLM
- **Consecutive error guard** — exits after 3 repeated failures to avoid infinite loops
- **LangSmith tracing** — all runs traced to the `docker-agent` project
