"""
Agent that executes LLM-generated code inside a local Docker container.
Uses docker_tool.execute_docker as the execution backend instead of the
model's native sandbox. Swap the LLM to change providers.
"""
import sys
import os
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
}

SYSTEM_PROMPT = """Generate Python code for queries, then execute with docker tool.
1. Write complete code with imports/print()
2. execute_docker(code=your_code)
Return execution result."""

def get_agent(provider: str):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")
    return create_agent(PROVIDERS[provider](), tools=[execute_docker], system_prompt=SYSTEM_PROMPT)

if __name__ == "__main__":
    import time
    provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    agent = get_agent(provider)
    start_time = time.time()
    result = agent.invoke({"messages": [HumanMessage("What is 7*8?")]})
    print("Result:", result["messages"][-1].content)
    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
