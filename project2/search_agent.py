"""
Agentic search agent that routes queries to either document search or direct LLM answer.
Uses LangGraph with: router -> (search | direct) -> summarize
"""
import sys
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from typing import Annotated
from typing_extensions import TypedDict
import operator

from search_tool import search_documents, list_documents, extract_code_patterns

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

_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS = [search_documents, list_documents, extract_code_patterns]

MODEL_PROMPT = open(os.path.join(_DIR, "model_prompt.txt")).read().strip()
CRITIC_PROMPT = open(os.path.join(_DIR, "critic_prompt.txt")).read().strip()
SUMMARIZE_PROMPT = open(os.path.join(_DIR, "summarize_prompt.txt")).read().strip()
REWRITE_PROMPT = open(os.path.join(_DIR, "rewrite_prompt.txt")).read().strip()

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def get_agent(provider: str = "openai", max_tool_calls: int = int(os.getenv("MAX_TOOL_CALLS", 3))):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")

    llm = PROVIDERS[provider]()
    llm_with_tools = llm.bind_tools(TOOLS)
    critic_llm = PROVIDERS[provider]()

    def rewrite_node(state: AgentState):
        from search_tool import _list_documents, _read_file
        doc_summaries = []
        for p in _list_documents():
            fname = os.path.basename(p)
            lines = [l.strip() for l in _read_file(p).split("\n") if l.strip()]
            preview = " | ".join(lines[:3])[:200]
            doc_summaries.append(f"- {fname}: {preview}")
        context = "Available documents:\n" + "\n".join(doc_summaries) + "\n\n" if doc_summaries else ""
        original = state["messages"][-1]
        rewritten = llm.invoke([SystemMessage(content=context + REWRITE_PROMPT), original])
        return {"messages": [HumanMessage(content=rewritten.content)]}

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
        if isinstance(last, AIMessage) and "CONTINUE" in last.content.upper():
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
    recursion_limit = max_tool_calls * 3 + 1
    return agent, recursion_limit

if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    agent, recursion_limit = get_agent(provider)

    query = open(os.path.join(_DIR, "query.txt")).read().strip()
    messages = []
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(query)]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]
    except GraphRecursionError:
        pass

    llm_summarize = PROVIDERS[provider]()
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content and getattr(m, "name", None) != "critic"), None)
    if not last_ai:
        tool_ids = {tc["id"] for m in messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
        responded_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        clean_messages = [m for m in messages if not (isinstance(m, AIMessage) and any(tc["id"] not in responded_ids for tc in (m.tool_calls or [])))]
        summary = llm_summarize.invoke([
            SystemMessage(content=open(os.path.join(_DIR, "summarize_prompt.txt")).read().strip()),
            *clean_messages,
        ])
        print("\nResult:", summary.content)
    else:
        print("\nResult:", last_ai.content)
