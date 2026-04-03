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
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from typing import Annotated
from typing_extensions import TypedDict
import operator

try:
    from search_tool import grep_corpus, read_pages, list_documents, render_page_image, _load_corpus
except ImportError:
    from project2.search_tool import grep_corpus, read_pages, list_documents, render_page_image, _load_corpus

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
except ImportError:
    ChatGoogleGenerativeAI = None

    class ChatGoogleGenerativeAIError(Exception):
        """Fallback Gemini error type when the package is unavailable."""

        pass

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "search-agent")

PROVIDERS = {
    "openai": lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter": lambda: ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=str(os.getenv("OPEN_ROUTER_API_KEY", "")),
        base_url="https://openrouter.ai/api/v1",
    ),
}

VISION_PROVIDERS = {
    "openai": lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter": lambda: ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=str(os.getenv("OPEN_ROUTER_API_KEY", "")),
        base_url="https://openrouter.ai/api/v1",
    ),
}

if ChatGoogleGenerativeAI is not None:
    PROVIDERS["gemini"] = lambda: ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    VISION_PROVIDERS["gemini"] = lambda: ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", 6))
_DIR = os.path.dirname(os.path.abspath(__file__))

TOOLS = [grep_corpus, read_pages, list_documents]

VISION_PROMPT = """You are reading a PDF page rendered as an image. Extract ALL text and data from this page.
If the page contains tables, reproduce them in a clear text format with aligned columns.
If the page contains diagrams or figures, describe what they show.
Be thorough — capture every piece of text, number, label, and caption on the page."""

VISION_QA_PROMPT = """You are answering a user question from a single PDF page image.

Instructions:
- Read the page visually and answer the specific question using only what is visible on the page.
- If the answer depends on a chart, graph, or table, extract the relevant axes, labels, scale, and approximate values.
- If the user asks for a threshold or crossing point (for example, where a value drops below another value), estimate that crossing from the chart and say it is approximate.
- If the page does not contain enough visible information to answer, say so explicitly.
- Do not give generic commentary. Produce this exact format:

Answer: <best answer, or 'unavailable from this page image'>
Evidence: <short evidence summary from the page, including chart/table details if relevant>
Confidence: <high|medium|low>"""

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


def _repair_tool_args(tool_calls: list[dict]) -> list[dict]:
    """Normalize near-miss tool argument keys produced by the model."""
    repaired: list[dict] = []
    for call in tool_calls or []:
        args = dict(call.get("args", {}))
        if "query" not in args and "query=" in args:
            args["query"] = args.pop("query=")
        if "source" not in args and "source=" in args:
            args["source"] = args.pop("source=")
        if "page" not in args and "page=" in args:
            args["page"] = args.pop("page=")
        repaired.append({**call, "args": args})
    return repaired


def _query_needs_exact_value(query: str) -> bool:
    query_lower = query.lower()
    exact_markers = [
        "at what",
        "exact",
        "drop below",
        "less than",
        "greater than",
        "threshold",
        "count",
        "angle",
        "value",
    ]
    has_exact_marker = any(marker in query_lower for marker in exact_markers)
    has_numeric_target = bool(
        re.search(r"\b\d+(\.\d+)?\b", query_lower)
        or re.search(r"10\^\d+", query_lower)
        or re.search(r"10\s*\^\s*\{?\d+\}?", query_lower)
    )
    return has_exact_marker or has_numeric_target


def _has_citation(text: str) -> bool:
    return bool(re.search(r"\.pdf, page \d+", text, re.IGNORECASE))


def _tool_message_texts(state: dict) -> list[str]:
    return [_content_text(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]


def _exact_answer_present(query: str, content: str) -> bool:
    if not _query_needs_exact_value(query):
        return True
    text = content.lower()
    if "answer: unavailable" in text:
        return True
    has_number = bool(re.search(r"\b\d+(\.\d+)?\b", text))
    if "angle" in query.lower():
        return has_number and bool(re.search(r"(degree|degrees|°|theta|angle)", text))
    return has_number


def _relevant_sources_from_tools(state: dict) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for m in state["messages"]:
        if not isinstance(m, ToolMessage):
            continue
        text = _content_text(m.content)
        for source, page in re.findall(r"\[([^\]]+\.pdf)\] Page (\d+)", text):
            refs.append((source.split("/")[-1], page))
    return refs


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    original_query: str
    rewritten_query: str
    suggested_scope: str
    scope_relaxed: bool
    forced_tool_retry_count: int


def get_agent(provider: str = "openai", max_tool_calls: int = MAX_TOOL_CALLS):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDERS)}")

    llm = PROVIDERS[provider]()
    # build vision tool with access to the vision LLM via closure
    vision_llm = VISION_PROVIDERS[provider]() if provider in VISION_PROVIDERS else None

    @tool
    def read_page_vision(source: str, page: int) -> str:
        """
        Render a PDF page as an image and use a vision model to extract its content.
        Use this when read_pages returns [LOW TEXT] or garbled/table content that
        text extraction couldn't capture properly.

        Args:
            source: Document path (e.g. "ncert_physics_class_12_part_1/leph101.pdf")
            page: Page number to render and read visually

        Returns:
            Text and data extracted from the page image by the vision model.
        """
        if vision_llm is None:
            return f"Vision not available for provider '{provider}'."

        img_b64 = render_page_image(source, page)
        if img_b64 is None:
            return f"Failed to render page {page} of '{source}'. Check that pymupdf is installed and the file exists."

        response = vision_llm.invoke([
            SystemMessage(content=VISION_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": f"Extract all content from this page (page {page} of {source})."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]),
        ])
        extracted = _content_text(response.content)
        return f"--- [{source}] Page {page} (vision extraction) ---\n{extracted}"

    @tool
    def answer_from_page_vision(source: str, page: int, question: str) -> str:
        """
        Render a PDF page as an image and answer a specific question using visual content on that page.
        Use this when the answer depends on a chart, graph, table, or other visual information.

        Args:
            source: Document path (e.g. "ncert_physics_class_12_part_2/leph204.pdf")
            page: Page number to render and inspect visually
            question: The specific question to answer from the page

        Returns:
            Structured answer, evidence, and confidence based only on the page image.
        """
        if vision_llm is None:
            return f"--- [{source}] Page {page} (vision QA) ---\nAnswer: unavailable from this page image\nEvidence: Vision not available for provider '{provider}'.\nConfidence: low"

        img_b64 = render_page_image(source, page)
        if img_b64 is None:
            return (
                f"--- [{source}] Page {page} (vision QA) ---\n"
                "Answer: unavailable from this page image\n"
                "Evidence: Failed to render the page image. Check that pymupdf is installed and the file exists.\n"
                "Confidence: low"
            )

        response = vision_llm.invoke([
            SystemMessage(content=VISION_QA_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": f"Question: {question}\nAnswer it from page {page} of {source}."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]),
        ])
        extracted = _content_text(response.content)
        return f"--- [{source}] Page {page} (vision QA) ---\n{extracted}"

    all_tools = [grep_corpus, read_pages, list_documents, read_page_vision, answer_from_page_vision]
    llm_with_tools = llm.bind_tools(all_tools)

    def rewrite_node(state: AgentState):
        doc_list = list_documents.invoke({})
        original = state["messages"][-1]
        original_text = _content_text(original.content)
        rewritten = llm.invoke([
            SystemMessage(content=doc_list + "\n\n" + REWRITE_PROMPT),
            original,
        ])
        rewritten_text = _content_text(rewritten.content)
        scope, query = "", rewritten_text
        for line in rewritten_text.splitlines():
            if line.strip().upper().startswith("SCOPE:"):
                scope = line.split(":", 1)[1].strip()
            elif line.strip().upper().startswith("QUERY:"):
                query = line.split(":", 1)[1].strip()
        if scope:
            valid_scopes = {c["source"].split("/")[0] for c in _load_corpus()}
            if scope not in valid_scopes:
                scope = ""
        if scope:
            query = f"[Search scope: {scope}]\n{query}"
        return {
            "messages": [HumanMessage(content=query)],
            "original_query": original_text,
            "rewritten_query": query,
            "suggested_scope": scope,
            "scope_relaxed": False,
            "forced_tool_retry_count": 0,
        }

    def model_node(state: AgentState):
        extra_rules = []
        suggested_scope = state.get("suggested_scope", "")
        if suggested_scope and not state.get("scope_relaxed", False):
            extra_rules.append(
                f"Suggested scope from rewrite: {suggested_scope}. Treat this as a hint, not a hard constraint."
            )
        if state.get("scope_relaxed", False):
            extra_rules.append(
                "A previous scoped attempt was not sufficient. Broaden the search now: remove scope or search a different scope."
            )

        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage) and getattr(last, "name", None) == "critic" and "CONTINUE" in _content_text(last.content).upper():
            extra_rules.append(
                "The critic said CONTINUE. You must make at least one tool call now. Do not answer directly."
            )
            extra_rules.append(
                "If the last search used a scope and the results look irrelevant or generic, rerun grep_corpus with broader keywords and no scope."
            )
            extra_rules.append(
                "If read_pages mentions a figure, graph, or table and the answer depends on a plotted or tabulated value, use answer_from_page_vision on that page with the original user question."
            )

        system_prompt = MODEL_PROMPT
        if extra_rules:
            system_prompt += "\n\nAdditional instructions:\n- " + "\n- ".join(extra_rules)

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        response.tool_calls = _repair_tool_args(list(response.tool_calls or []))
        forced_retry_count = state.get("forced_tool_retry_count", 0)
        if (
            isinstance(last, AIMessage)
            and getattr(last, "name", None) == "critic"
            and "CONTINUE" in _content_text(last.content).upper()
            and not response.tool_calls
        ):
            forced_retry_count += 1
        else:
            forced_retry_count = 0
        return {
            "messages": [response],
            "forced_tool_retry_count": forced_retry_count,
        }

    def critic_node(state: AgentState):
        last = state["messages"][-1]
        if not isinstance(last, ToolMessage):
            return {"messages": []}
        original_query = state.get("original_query", "")
        content = _content_text(last.content)

        verdict_text = "CONTINUE"
        if getattr(last, "status", "") == "error":
            verdict_text = "CONTINUE"
        elif last.name == "grep_corpus":
            verdict_text = "CONTINUE"
        elif last.name == "read_page_vision":
            verdict_text = "CONTINUE" if _query_needs_exact_value(original_query) else "DONE"
        elif last.name == "answer_from_page_vision":
            verdict_text = "DONE" if _exact_answer_present(original_query, content) else "CONTINUE"
        elif last.name == "read_pages":
            needs_exact = _query_needs_exact_value(original_query)
            has_figure_reference = "[HAS FIGURE/TABLE REFERENCE" in content or bool(
                re.search(r"(figs?\.?|figure|graph|table)", content, re.IGNORECASE)
            )
            if needs_exact and has_figure_reference:
                verdict_text = "CONTINUE"
            else:
                verdict_text = "DONE"

        verdict = AIMessage(content=verdict_text)
        verdict.name = "critic"
        return {"messages": [verdict]}

    def summarize_node(state: AgentState):
        original_query = state.get("original_query", "")
        tool_texts = _tool_message_texts(state)
        if _query_needs_exact_value(original_query):
            if not any(_exact_answer_present(original_query, txt) for txt in tool_texts):
                refs = _relevant_sources_from_tools(state)
                if refs:
                    ref_text = "; ".join(f"{source}, page {page}" for source, page in refs[:3])
                    return {
                        "messages": [AIMessage(content=(
                            "I could not determine the exact requested value from the retrieved content. "
                            "The relevant material appears in figure- or table-dependent pages, but the extracted text/vision output did not provide a reliable exact value "
                            f"({ref_text})."
                        ))]
                    }
                return {
                    "messages": [AIMessage(content=(
                        "I could not determine the exact requested value from the retrieved content, and I do not have reliable evidence to provide a precise cited answer."
                    ))]
                }
        messages = [
            SystemMessage(content=SUMMARIZE_PROMPT),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        if not _has_citation(_content_text(response.content)):
            response = llm.invoke([
                SystemMessage(
                    content=SUMMARIZE_PROMPT
                    + "\n\nYour previous answer lacked required citations. Rewrite it and include document name and page number for every factual claim."
                ),
                *state["messages"],
            ])
        return {"messages": [response]}

    def should_search(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "search"
        prev = state["messages"][-2] if len(state["messages"]) > 1 else None
        if (
            isinstance(last, AIMessage)
            and not last.tool_calls
            and isinstance(prev, AIMessage)
            and getattr(prev, "name", None) == "critic"
            and "CONTINUE" in _content_text(prev.content).upper()
            and state.get("forced_tool_retry_count", 0) < 2
        ):
            return "retry_model"
        return "summarize"

    def after_critic(state: AgentState):
        tool_call_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        if tool_call_count >= max_tool_calls:
            return "summarize"
        last = state["messages"][-1]
        if (
            isinstance(last, AIMessage)
            and "CONTINUE" in _content_text(last.content).upper()
            and state.get("suggested_scope")
            and not state.get("scope_relaxed", False)
        ):
            return "broaden_scope"
        if isinstance(last, AIMessage) and "CONTINUE" in _content_text(last.content).upper():
            return "model"
        return "summarize"

    def broaden_scope_node(state: AgentState):
        original_query = state.get("original_query") or state.get("rewritten_query") or ""
        hint = (
            "The previous scoped search was not sufficient. Search again with broader coverage. "
            "Remove the scope restriction or choose a better scope based on the topic. "
            f"Original question: {original_query}"
        )
        return {
            "messages": [HumanMessage(content=hint)],
            "scope_relaxed": True,
            "forced_tool_retry_count": 0,
        }

    graph = StateGraph(AgentState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("model", model_node)
    graph.add_node("search", ToolNode(all_tools))
    graph.add_node("critic", critic_node)
    graph.add_node("broaden_scope", broaden_scope_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "model")
    graph.add_conditional_edges("model", should_search, {"search": "search", "retry_model": "model", "summarize": "summarize"})
    graph.add_edge("search", "critic")
    graph.add_conditional_edges("critic", after_critic, {"model": "model", "broaden_scope": "broaden_scope", "summarize": "summarize"})
    graph.add_edge("broaden_scope", "model")
    graph.add_edge("summarize", END)

    agent = graph.compile()
    recursion_limit = max(max_tool_calls * 5 + 6, 30)
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
