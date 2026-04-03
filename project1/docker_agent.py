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

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true") # Used for LangSmith tracing
os.environ.setdefault("LANGCHAIN_PROJECT", "docker-agent") # Used for LangSmith project name

# List of available LLM providers
PROVIDERS = {
    "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=os.getenv("GEMINI_API_KEY")),
    "openai": lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter": lambda: ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
}

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", 3)) # Maximum number of tool calls allowed
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 120)) # Max execution time for agent in seconds
# 👆 Uses signal.alarm(AGENT_TIMEOUT) to set a system alarm, if execution doesn't complete within the above period, it raises TimeoutError

_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this script

def _load_prompt(filename: str) -> str:
    return open(os.path.join(_DIR, filename)).read().strip() # Read the prompt from the file

# Load all prompts
SYSTEM_PROMPT = _load_prompt("model_prompt.txt")
CRITIC_PROMPT = _load_prompt("critic_prompt.txt")
SUMMARIZE_PROMPT = _load_prompt("summarize_prompt.txt")
ROUTER_PROMPT = _load_prompt("router_prompt.txt")
VIZ_PROMPT = _load_prompt("viz_prompt.txt")


def _content_text(content) -> str:
    """Safely extract text from message content (handles str and list forms)."""
    if isinstance(content, list):
        # Extract text from list of content parts 
        # Example: [{"type": "text", "text": "Hello"}, {"type": "image", "image_url": "..."}] -> "Hello"
        return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list, operator.add]
    query_type: str  # "compute", "visualize", or "chat"


def get_agent(provider: str, max_tool_calls: int = MAX_TOOL_CALLS):
    """
    Create and return a LangGraph agent with the specified provider.
    
    Args:
        provider: The provider to use (e.g., "openai", "openrouter")
        max_tool_calls: Maximum number of tool calls allowed
        
    Returns:
        A compiled LangGraph agent
    """
    if provider not in PROVIDERS:
        # Log the available providers
        print(f"Available providers: {list(PROVIDERS)}")
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")

    llm = PROVIDERS[provider]() # Create the LLM instance
    llm_with_tools = llm.bind_tools([execute_docker]) # Bind the tool to the LLM
    critic_llm = PROVIDERS[provider]() # Create the critic LLM instance

    # ── router ──────────────────────────────────────────────
    def router_node(state: AgentState):
        """Route the conversation based on the user's query type."""
        user_msg = state["messages"][-1].content # Get the last message from the conversation
        verdict = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = _content_text(verdict.content).strip().upper() # Extract the text from the response
        # Determine the query type based on the response
        if "VISUALIZE" in raw:
            query_type = "visualize"
        elif "COMPUTE" in raw:
            query_type = "compute"
        else:
            query_type = "chat"
        return {"query_type": query_type}

    def after_router(state: AgentState):
        """Determine which model node to call based on the query type."""
        qt = state.get("query_type", "compute")
        if qt == "visualize":
            return "viz_model"
        if qt == "chat":
            return "chat_model"
        return "model"

    # ── model nodes ─────────────────────────────────────────
    def model_node(state: AgentState):
        # Role 1
        # Route: compute → model (with tools + critic)
        """Handle compute queries with tool calls and critic validation."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"] # Add the system prompt to the messages
        # Example: [SystemMessage("You are a helpful assistant..."), HumanMessage("What is 2+2?")] 
        response = llm_with_tools.invoke(messages) # Invoke the LLM with tools
        return {"messages": [response]}

    def viz_model_node(state: AgentState):
        # Role 2
        # Route: visualize → viz_model (with tools)
        messages = [SystemMessage(content=VIZ_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def chat_model_node(state: AgentState):
        # Role 3
        # Route: chat → chat_model (no tools, no critic)
        """Direct answer — no tools, no critic."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # ── shared nodes ────────────────────────────────────────
    def critic_node(state: AgentState):
        # Trigger condition: Only runs after a tool execution (ToolMessage)
        # Two different stopping mechanisms:
        # 1. MAX_TOOL_CALLS (hard limit): Controls maximum loop iterations, prevents infinite loops
        # 2. Critic error detection (smart limit): Stops immediately on 3 consecutive errors, saves time
        # Example: MAX_TOOL_CALLS=100, but [success, error1, error2, error3] → STOP (critic)
        # Responsibility:
        # 1. Check if the last message is a ToolMessage
        # 2. If it is, check if the last 3 consecutive ToolMessages all contain error keywords
        # 3. If they do, return DONE (prevent infinite error loops)
        # 4. Otherwise, invoke the critic LLM to assess task completion
        last = state["messages"][-1] # Get the last message from the state
        if not isinstance(last, ToolMessage): # If the last message is not a ToolMessage, return an empty list
            return {"messages": []}
        tool_results = [m for m in state["messages"] if isinstance(m, ToolMessage)] # Get all ToolMessages from the state
        if len(tool_results) >= 3 and all(
            any(kw in _content_text(m.content).lower() for kw in ("error", "traceback", "syntaxerror", "line 1"))
            for m in tool_results[-3:]
        ):
            # If the last 3 ToolMessages all contain error keywords, return DONE
            done = AIMessage(content="DONE") # Create a DONE message
            done.name = "critic" # Set the name of the message
            return {"messages": [done]} # Return the DONE message
        
        # If the last 3 ToolMessages don't all contain error keywords, invoke the critic LLM
        verdict = critic_llm.invoke([
            SystemMessage(content=CRITIC_PROMPT),
            *state["messages"],
        ])
        verdict.name = "critic"
        return {"messages": [verdict]}

    def summarize_node(state: AgentState):
        """Summarize the conversation for non-visualization tasks."""
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
            # If we found saved files, we tell the user where they are
            summary = "Visualization saved:\n" + "\n".join(saved)
        else:
            # Tool might execute successfully but not explicitly print file paths
            summary = "Visualization code executed. Check the /output/ directory for saved files."
        return {"messages": [AIMessage(content=summary)]}

    # ── edge logic ──────────────────────────────────────────
    def should_continue(state: AgentState):
        """Determines whether the agent should make another tool call or finish"""
        # Example
        # model/viz_model → should_continue → tools (if tool_calls) → critic → ...
        #                           ↓
        #                   END (if no tool_calls)
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_critic(state: AgentState):
        """
        Check if critic wants to continue or finish
        
        Returns:
            - "viz_model" if query type is "visualize" and critic wants to continue
            - "model" if query type is "compute" and critic wants to continue
            - "viz_summarize" if query type is "visualize" and critic wants to finish
            - "summarize" if query type is "compute" and critic wants to finish
        """
        last = state["messages"][-1]
        qt = state.get("query_type", "compute")
        if isinstance(last, AIMessage) and "CONTINUE" in _content_text(last.content).upper():
            return "viz_model" if qt == "visualize" else "model"
        # viz queries skip summarize — tool output has the file path
        if qt == "visualize":
            return "viz_summarize"
        return "summarize"

    #  ─────────────────── graph assembly ────────────────────────
    #                     ┌──────────┐
    #                     │  router  │
    #                     └────┬─────┘
    #        ┌─────────────────┼─────────────────┐
    #        ▼                 ▼                 ▼
    # ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
    # │    model    │  │  viz_model   │  │  chat_model  │
    # │ (compute)   │  │ (visualize)  │  │  (no tools)  │
    # └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
    #        │                │                 │
    #        └───────┬────────┘                 ▼
    #                ▼                         END
    #        ┌──────────────┐
    #        │    tools     │
    #        │ (Docker exec)│
    #        └──────┬───────┘
    #               ▼
    #        ┌──────────────┐
    #        │    critic    │
    #        │ (DONE/CONT.) │
    #        └──────┬───────┘
    #       ┌───────┴────────┐
    #       ▼                ▼
    #  CONTINUE             DONE
    #  (loops back           │
    #   to model or   ┌──────┴─────┐
    #   viz_model)    ▼            ▼
    #            ┌──────────┐ ┌───────────────┐
    #            │summarize │ │ viz_summarize │
    #            │(LLM text)│ │(file paths)   │
    #            └────┬─────┘ └───────┬───────┘
    #                 ▼               ▼
    #                END             END
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("model", model_node)
    graph.add_node("viz_model", viz_model_node)
    graph.add_node("chat_model", chat_model_node)
    graph.add_node("tools", ToolNode([execute_docker]))
    graph.add_node("critic", critic_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("router") # Start with router node
    # Route to model, viz_model, or chat_model based on query type
    graph.add_conditional_edges("router", after_router, {
        "model": "model",
        "viz_model": "viz_model",
        "chat_model": "chat_model",
    })

    # Model and viz_model nodes share the same downstream
    #        model ──┐
    #                ├── [tools] → [critic] → [summarize/viz_summarize] → END
    #    viz_model ──┘
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_conditional_edges("viz_model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("chat_model", END) # Chat model doesn't need tools or critic

    graph.add_edge("tools", "critic")
    graph.add_node("viz_summarize", viz_summarize_node)  # Add viz_summarize after tools for visualization queries

    # Routing table: translates critic's verdict + query type into the next graph node
    graph.add_conditional_edges("critic", after_critic, {    
        "model": "model",                      # CONTINUE + compute   → model         (loop back)
        "viz_model": "viz_model",              # CONTINUE + visualize → viz_model     (loop back)
        "summarize": "summarize",              # DONE + compute       → summarize     (final summary)
        "viz_summarize": "viz_summarize",      # DONE + visualize     → viz_summarize (extract file paths)
    })
    graph.add_edge("summarize", END)
    graph.add_edge("viz_summarize", END)

    agent = graph.compile()
    # Recursion limit: max_tool_calls * 3 (model→tools→critic cycles) + 2 (router + final node)
    recursion_limit = max_tool_calls * 3 + 2
    return agent, recursion_limit


if __name__ == "__main__":
    # ── setup ───────────────────────────────────────────────
    provider = sys.argv[1] if len(sys.argv) > 1 else "gemini" # Default to gemini if no argument provided
    agent, recursion_limit = get_agent(provider)

    messages = []
    recursion_limit_hit = False
    timed_out = False

    # ── timeout handler ─────────────────────────────────────
    def _on_timeout(signum, frame):
        raise TimeoutError(f"Agent timed out after {AGENT_TIMEOUT}s")

    signal.signal(signal.SIGALRM, _on_timeout) # Register the timeout handler
    signal.alarm(AGENT_TIMEOUT)                # Start the countdown
    try:
        # ── stream agent ────────────────────────────────────
        for chunk in agent.stream(
            {"messages": [HumanMessage(open(os.path.join(_DIR, "query.txt")).read().strip())]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]        # Accumulate messages from each chunk
    except GraphRecursionError:
        recursion_limit_hit = True             # MAX_TOOL_CALLS exceeded
    except (TimeoutError, httpx.ReadTimeout):
        timed_out = True                       # AGENT_TIMEOUT exceeded
    except (ValueError, ChatGoogleGenerativeAIError) as e:
        print(f"\n[Provider Error] {e}")
    finally:
        signal.alarm(0)                        # Cancel the alarm regardless of outcome

    # ── extract final answer ─────────────────────────────────
    def _extract_text(content) -> str:
        if isinstance(content, list):
            return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text") # Extract text from the content
        return str(content)

    # Prefer last AIMessage (excluding critic) over raw tool output
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content and getattr(m, "name", None) != "critic"), None) # Get the last AIMessage that is not the critic
    if last_ai:
        answer = _extract_text(last_ai.content)
    else:
        # Fallback to last ToolMessage if no AI response
        last_tool = next((m for m in reversed(messages) if isinstance(m, ToolMessage) and m.content), None) # Get the last ToolMessage with content
        answer = _extract_text(last_tool.content) if last_tool else "No answer generated." # Extract text from the tool message

    print("\nResult:", answer)
    if recursion_limit_hit:
        print(f"\n[Note] Recursion limit ({recursion_limit}) was reached. Increase MAX_TOOL_CALLS (currently {MAX_TOOL_CALLS}) via env variable.")
    if timed_out:
        print(f"\n[Note] Agent timed out after {AGENT_TIMEOUT}s. Increase AGENT_TIMEOUT (currently {AGENT_TIMEOUT}s) via env variable.")
