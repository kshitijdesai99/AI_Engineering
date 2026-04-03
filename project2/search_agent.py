"""
Agentic document search agent.
Uses corpus.json (pre-extracted PDF text) for fast grep-style search.

Architecture:
    query → rewrite → model → grep_corpus → critic
                                  ↓ CONTINUE
                              model → read_pages → critic
                                  ↓ DONE
                              summarize (with citations) → END
"""
import sys
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from typing import Annotated
from typing_extensions import TypedDict
import operator

from search_tool import grep_corpus, read_pages, list_documents

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "search-agent")

PROVIDERS = {
    "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")),
    "openai": lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter": lambda: ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=str(os.getenv("OPEN_ROUTER_API_KEY", "")),
        base_url="https://openrouter.ai/api/v1",
    ),
}

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", 6))
_DIR = os.path.dirname(os.path.abspath(__file__))

TOOLS = [grep_corpus, read_pages, list_documents]

def _load_prompt(filename: str) -> str:
    return open(os.path.join(_DIR, filename)).read().strip()

MODEL_PROMPT = _load_prompt("model_prompt.txt")
CRITIC_PROMPT = _load_prompt("critic_prompt.txt")
SUMMARIZE_PROMPT = _load_prompt("summarize_prompt.txt")
REWRITE_PROMPT = _load_prompt("rewrite_prompt.txt")


def _content_text(content) -> str:
    """Safely extract text from message content (handles str and list forms)."""
    if isinstance(content, list):
        return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def get_agent(provider: str = "openai", max_tool_calls: int = MAX_TOOL_CALLS):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")

    llm = PROVIDERS[provider]()
    llm_with_tools = llm.bind_tools(TOOLS)
    critic_llm = PROVIDERS[provider]()

    def rewrite_node(state: AgentState):
        # get document listing for context
        doc_list = list_documents.invoke({})
        original = state["messages"][-1]
        rewritten = llm.invoke([
            SystemMessage(content=doc_list + "\n\n" + REWRITE_PROMPT),
            original,
        ])
        rewritten_text = _content_text(rewritten.content)
        return {"messages": [HumanMessage(content=rewritten_text)]}

    def model_node(state: AgentState):
        messages = [SystemMessage(content=MODEL_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def critic_node(state: AgentState):
        last = state["messages"][-1]
        if not isinstance(last, ToolMessage):
            return {"messages": []}
        verdict = critic_llm.invoke([
            SystemMessage(content=CRITIC_PROMPT),
            *state["messages"],
        ])
        verdict.name = "critic"
        return {"messages": [verdict]}

    def summarize_node(state: AgentState):
        response = llm.invoke([
            SystemMessage(content=SUMMARIZE_PROMPT),
            *state["messages"],
        ])
        return {"messages": [response]}

    def should_search(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "search"
        return "summarize"

    def after_critic(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and "CONTINUE" in _content_text(last.content).upper():
            return "model"
        return "summarize"

    graph = StateGraph(AgentState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("model", model_node)
    graph.add_node("search", ToolNode(TOOLS))
    graph.add_node("critic", critic_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "model")
    graph.add_conditional_edges("model", should_search, {"search": "search", "summarize": "summarize"})
    graph.add_edge("search", "critic")
    graph.add_conditional_edges("critic", after_critic, {"model": "model", "summarize": "summarize"})
    graph.add_edge("summarize", END)

    agent = graph.compile()
    recursion_limit = max_tool_calls * 3 + 2
    return agent, recursion_limit


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    agent, recursion_limit = get_agent(provider)

    query = open(os.path.join(_DIR, "query.txt")).read().strip()
    print(f"\nQuery: {query}\n")

    messages = []
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(query)]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]
    except GraphRecursionError:
        print("[Note] Recursion limit reached.")
    except (ValueError, ChatGoogleGenerativeAIError) as e:
        print(f"[Provider Error] {e}")

    last_ai = next(
        (m for m in reversed(messages)
         if isinstance(m, AIMessage) and m.content and getattr(m, "name", None) != "critic"),
        None,
    )
    if last_ai:
        print("Result:", _content_text(last_ai.content))
    else:
        print("No answer generated.")
