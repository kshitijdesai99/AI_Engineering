"""
Agent that executes LLM-generated code inside a local Docker container.
Uses docker_tool.execute_docker as the execution backend instead of the
model's native sandbox. Swap the LLM to change providers.
"""
import sys
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from docker_tool import execute_docker
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

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

    result = agent.invoke(
        {"messages": [HumanMessage("download the image at https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Walking_tiger_female.jpg/1280px-Walking_tiger_female.jpg")]},
        config={"recursion_limit": recursion_limit},
    )

    for i, msg in enumerate(result["messages"]):
        print(f"[msg {i}] {type(msg).__name__}: {str(msg.content)[:80]}")

    print("\nResult:", result["messages"][-1].content)
    print(f"Total time: {time.time() - t_total:.2f}s")
