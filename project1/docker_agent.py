"""
Agent that executes LLM-generated code inside a local Docker container.
Uses docker_tool.execute_docker as the execution backend instead of the
model's native sandbox. Swap the LLM to change providers.
"""
import sys
import os
import time
import signal
import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_openai import ChatOpenAI
from docker_tool import execute_docker
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

load_dotenv()

PROVIDERS = {
    "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=os.getenv("GEMINI_API_KEY")),
    "openai": lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter": lambda: ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
}

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", 3))  # max number of docker executions per query
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 120))  # total agent wall-clock timeout in seconds

SYSTEM_PROMPT = """Generate Python code for queries, then execute with docker tool.
1. Write complete code with imports/print()
2. Save any plots or files to /output/ (e.g. plt.savefig('/output/plot.png'))
3. Always pass packages=[...] with every third-party library your code imports (e.g. packages=["seaborn", "pandas"])
4. When making HTTP requests always set headers={"User-Agent": "Mozilla/5.0"}
5. execute_docker(code=your_code, packages=[...])
Return execution result and any saved file paths."""

def get_agent(provider: str, max_tool_calls: int = MAX_TOOL_CALLS):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")
    agent = create_agent(PROVIDERS[provider](), tools=[execute_docker], system_prompt=SYSTEM_PROMPT)
    recursion_limit = max_tool_calls * 2 + 1  # each tool call = 2 graph steps (LLM + tool)
    return agent, recursion_limit

if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    agent, recursion_limit = get_agent(provider)

    t_total = time.time()

    messages = []
    recursion_limit_hit = False
    timed_out = False

    def _on_timeout(signum, frame):
        raise TimeoutError(f"Agent timed out after {AGENT_TIMEOUT}s")

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(AGENT_TIMEOUT)
    try:
        for chunk in agent.stream(
            # {"messages": [HumanMessage("download the image at https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Walking_tiger_female.jpg/1280px-Walking_tiger_female.jpg")]},
            {"messages":[HumanMessage('''
            vizualize how sin(x) * cos(x) looks like
            ''')]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]
    except GraphRecursionError:
        recursion_limit_hit = True
    except (TimeoutError, httpx.ReadTimeout):
        timed_out = True
    except (ValueError, ChatGoogleGenerativeAIError) as e:
        print(f"\n[Provider Error] {e}")
    finally:
        signal.alarm(0)

    for i, msg in enumerate(messages):
        print(f"[msg {i}] {type(msg).__name__}: {str(msg.content)[:80]}")

    def _extract_text(content) -> str:
        if isinstance(content, list):
            return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content)

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if tool_messages:
        print("\n--- Sources (Docker tool outputs) ---")
        for i, tm in enumerate(tool_messages, 1):
            print(f"[source {i}] {_extract_text(tm.content)[:300]}")
        print("-------------------------------------")

    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None)
    answer = _extract_text(last_ai.content) if last_ai else (messages[-1].content if messages else "No answer generated.")

    print("\nResult:", answer)
    if recursion_limit_hit:
        print(f"\n[Note] Recursion limit ({recursion_limit}) was reached. For a more accurate answer, increase MAX_TOOL_CALLS (currently {MAX_TOOL_CALLS}) via the env variable or pass a higher value to get_agent().")
    if timed_out:
        print(f"\n[Note] Agent timed out after {AGENT_TIMEOUT}s. For a more accurate answer, increase AGENT_TIMEOUT (currently {AGENT_TIMEOUT}s) via the env variable.")
    print(f"Total time: {time.time() - t_total:.2f}s")
