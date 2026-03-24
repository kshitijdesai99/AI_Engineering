# Project 1 — LLM Code Execution Agent

An agent that generates and executes Python code inside an isolated Docker container, supporting multiple LLM providers.

## Files

- **`docker_agent.py`** — main agent, accepts a provider argument
- **`docker_tool.py`** — LangChain tool that runs code in a Docker container

## Setup

Copy `.env_example` to `.env` and fill in your API keys:

```bash
cp .env_example .env
```

## Usage

```bash
python docker_agent.py gemini
python docker_agent.py openai
```

Defaults to `gemini` if no argument is passed.

## How It Works

1. The agent sends your query to the LLM
2. The LLM generates Python code and calls `execute_docker`
3. A Docker container (`python:3.12-slim`) is spawned, runs the code, and is destroyed
4. The output is returned to the LLM which produces the final answer

## Adding a Provider

Add an entry to `PROVIDERS` in `docker_agent.py`:

```python
"anthropic": lambda: ChatAnthropic(model="claude-4-5-haiku-latest", api_key=os.getenv("ANTHROPIC_API_KEY")),
```
