"""
Agent that executes LLM-generated code inside a local Docker container.
Uses docker_tool.execute_docker as the execution backend instead of the
model's native sandbox. Swap the LLM to change providers.

Includes a router that classifies queries and dispatches to specialized
sub-agents (compute vs. visualize) with dedicated prompts.
"""
import sys
import os
import signal
import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_openai import ChatOpenAI
from docker_tool import execute_docker
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from typing import Annotated
from typing_extensions import TypedDict
import operator

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "docker-agent")

PROVIDERS = {
    "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=os.getenv("GEMINI_API_KEY")),
    "openai": lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter": lambda: ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
}

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", 3))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 120))

_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_prompt(filename: str) -> str:
    return open(os.path.join(_DIR, filename)).read().strip()

SYSTEM_PROMPT = _load_prompt("model_prompt.txt")
CRITIC_PROMPT = _load_prompt("critic_prompt.txt")
SUMMARIZE_PROMPT = _load_prompt("summarize_prompt.txt")
ROUTER_PROMPT = _load_prompt("router_prompt.txt")
VIZ_PROMPT = _load_prompt("viz_prompt.txt")


def _content_text(content) -> str:
    """Safely extract text from message content (handles str and list forms)."""
    if isinstance(content, list):
        return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query_type: str  # "compute", "visualize", or "chat"


def get_agent(provider: str, max_tool_calls: int = MAX_TOOL_CALLS):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")

    llm = PROVIDERS[provider]()
    llm_with_tools = llm.bind_tools([execute_docker])
    critic_llm = PROVIDERS[provider]()

    # ── router ──────────────────────────────────────────────
    def router_node(state: AgentState):
        user_msg = state["messages"][-1].content
        verdict = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = _content_text(verdict.content).strip().upper()
        if "VISUALIZE" in raw:
            query_type = "visualize"
        elif "COMPUTE" in raw:
            query_type = "compute"
        else:
            query_type = "chat"
        return {"query_type": query_type}

    def after_router(state: AgentState):
        qt = state.get("query_type", "compute")
        if qt == "visualize":
            return "viz_model"
        if qt == "chat":
            return "chat_model"
        return "model"

    # ── model nodes ─────────────────────────────────────────
    def model_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def viz_model_node(state: AgentState):
        messages = [SystemMessage(content=VIZ_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def chat_model_node(state: AgentState):
        """Direct answer — no tools, no critic."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # ── shared nodes ────────────────────────────────────────
    def critic_node(state: AgentState):
        last = state["messages"][-1]
        if not isinstance(last, ToolMessage):
            return {"messages": []}
        tool_results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if len(tool_results) >= 3 and all(
            any(kw in _content_text(m.content).lower() for kw in ("error", "traceback", "syntaxerror", "line 1"))
            for m in tool_results[-3:]
        ):
            done = AIMessage(content="DONE")
            done.name = "critic"
            return {"messages": [done]}
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

    def viz_summarize_node(state: AgentState):
        """For viz queries: extract saved file paths from tool output, keep it short."""
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        saved = []
        for m in tool_msgs:
            txt = _content_text(m.content)
            for line in txt.splitlines():
                if "/output/" in line:
                    saved.append(line.strip())
        if saved:
            summary = "Visualization saved:\n" + "\n".join(saved)
        else:
            summary = "Visualization code executed. Check the /output/ directory for saved files."
        return {"messages": [AIMessage(content=summary)]}

    # ── edge logic ──────────────────────────────────────────
    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_critic(state: AgentState):
        last = state["messages"][-1]
        qt = state.get("query_type", "compute")
        if isinstance(last, AIMessage) and "CONTINUE" in _content_text(last.content).upper():
            return "viz_model" if qt == "visualize" else "model"
        # viz queries skip summarize — tool output has the file path
        if qt == "visualize":
            return "viz_summarize"
        return "summarize"

    # ── graph assembly ──────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("model", model_node)
    graph.add_node("viz_model", viz_model_node)
    graph.add_node("chat_model", chat_model_node)
    graph.add_node("tools", ToolNode([execute_docker]))
    graph.add_node("critic", critic_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", after_router, {
        "model": "model",
        "viz_model": "viz_model",
        "chat_model": "chat_model",
    })

    # both model nodes share the same downstream
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_conditional_edges("viz_model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("chat_model", END)

    graph.add_edge("tools", "critic")
    graph.add_node("viz_summarize", viz_summarize_node)

    graph.add_conditional_edges("critic", after_critic, {
        "model": "model",
        "viz_model": "viz_model",
        "summarize": "summarize",
        "viz_summarize": "viz_summarize",
    })
    graph.add_edge("summarize", END)
    graph.add_edge("viz_summarize", END)

    agent = graph.compile()
    recursion_limit = max_tool_calls * 3 + 2  # +2 for router + summarize
    return agent, recursion_limit


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    agent, recursion_limit = get_agent(provider)

    messages = []
    recursion_limit_hit = False
    timed_out = False

    def _on_timeout(signum, frame):
        raise TimeoutError(f"Agent timed out after {AGENT_TIMEOUT}s")

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(AGENT_TIMEOUT)
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(open(os.path.join(_DIR, "query.txt")).read().strip())]},
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

    def _extract_text(content) -> str:
        if isinstance(content, list):
            return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content)

    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content and getattr(m, "name", None) != "critic"), None)
    if last_ai:
        answer = _extract_text(last_ai.content)
    else:
        last_tool = next((m for m in reversed(messages) if isinstance(m, ToolMessage) and m.content), None)
        answer = _extract_text(last_tool.content) if last_tool else "No answer generated."

    print("\nResult:", answer)
    if recursion_limit_hit:
        print(f"\n[Note] Recursion limit ({recursion_limit}) was reached. Increase MAX_TOOL_CALLS (currently {MAX_TOOL_CALLS}) via env variable.")
    if timed_out:
        print(f"\n[Note] Agent timed out after {AGENT_TIMEOUT}s. Increase AGENT_TIMEOUT (currently {AGENT_TIMEOUT}s) via env variable.")
