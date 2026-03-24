# Project 1 — LLM Code Execution Agent

An agent that generates and executes Python code inside a persistent Docker container, supporting multiple LLM providers.

## Files

- **`docker_agent.py`** — main agent, accepts a provider argument
- **`docker_tool.py`** — LangChain tool that runs code in a persistent Docker container
- **`Dockerfile`** — custom image with `uv`, `seaborn`, `pandas`, `numpy`, `matplotlib` pre-installed

## Setup

1. Copy `.env_example` to `.env` and fill in your API keys:

```bash
cp .env_example .env
```

2. Build the Docker image:

```bash
docker build -t code-executor:latest .
```

## Usage

```bash
python docker_agent.py gemini
python docker_agent.py openai
python docker_agent.py openrouter
```

Defaults to `gemini` if no argument is passed.

## Configuration (`.env`)

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPEN_ROUTER_API_KEY` | OpenRouter API key |
| `MAX_TOOL_CALLS` | Max docker executions per query (default: 3) |

## How It Works

```
User query
    ↓
System prompt + query → LLM
    ↓
LLM generates Python code → calls execute_docker(code=..., packages=...)
    ↓
Docker runs the code → returns stdout + saved file paths
    ↓
Tool output → back to LLM
    ↓
LLM decides: call tool again or give final answer
```

- A **persistent container** is started once and reused across runs (container ID stored in `.container_id`)
- Packages not in the image are installed on-demand via `uv pip install`
- Plots and files saved to `/output/` inside the container appear in `project1/output/` on the host
- The agent loop is capped at `MAX_TOOL_CALLS` docker executions to prevent runaway loops

## Adding a Provider

Add an entry to `PROVIDERS` in `docker_agent.py`:

```python
"anthropic": lambda: ChatAnthropic(model="claude-haiku-4-5", api_key=os.getenv("ANTHROPIC_API_KEY")),
```
