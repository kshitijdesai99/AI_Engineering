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
import operator
import os
import re
import sys
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

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

_CITATION_RE = re.compile(r"\.pdf, page \d+", re.IGNORECASE)
_EXACT_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")
_NUMBER_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TEXT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?")
_ASSIGNMENT_RE = re.compile(r"\b[a-z]\d?\s*=\s*[-+]?\d+(?:\.\d+)?", re.IGNORECASE)
_CONCLUSION_RE = re.compile(r"\b(thus|therefore|hence|answer)\b", re.IGNORECASE)
_POW10_RE = re.compile(r"10\^\d+")
_POW10_BRACED_RE = re.compile(r"10\s*\^\s*\{?\d+\}?")
_ANGLE_HINT_RE = re.compile(r"(degree|degrees|°|theta|angle)")
_TOOL_REF_RE = re.compile(r"\[([^\]]+\.pdf)\] Page (\d+)")
_TOP_PAGE_LINE_RE = re.compile(r"^\s*-\s*\[([^\]]+)\]\s+p\.(\d+)\s+\(score:\s*([^)]+)\)", re.MULTILINE)
_PAGE_SECTION_RE = re.compile(r"--- \[([^\]]+)\] Page (\d+).*?---\n(.*?)(?=\n\n--- \[|\Z)", re.S)
_CONCLUSION_SENTENCE_RE = re.compile(r"\b(?:thus|therefore|hence)\b.*?(?:[.!?]\s*|$)", re.IGNORECASE | re.S)
_ASSIGNMENT_CAPTURE_RE = re.compile(r"\b([A-Za-z]\d?)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z/°^-]*)", re.IGNORECASE)
_QUERY_STOPWORDS = {
    "a", "an", "and", "answer", "assume", "at", "be", "by", "cross", "each", "end",
    "find", "from", "given", "homogeneous", "in", "is", "it", "its", "mass", "of",
    "on", "question", "section", "supported", "text", "textbook", "the", "this", "to",
    "uniform", "what", "with", "your",
}
_ANSWER_HINT_TERMS = {
    "answer", "answers", "reaction", "reactions", "support", "supports", "result", "results",
    "therefore the", "thus the", "hence the",
}
_TOOL_ARG_ALIASES = {
    "query=": "query",
    "source=": "source",
    "page=": "page",
    "question=": "question",
    "scope=": "scope",
    "max_results=": "max_results",
}
_READ_PAGES_ARG_ALIASES = {
    "start_page=": "start_page",
    "end_page=": "end_page",
    "start": "start_page",
    "end": "end_page",
}


def _build_openai_model():
    return ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def _build_openrouter_model():
    return ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=str(os.getenv("OPEN_ROUTER_API_KEY", "")),
        base_url="https://openrouter.ai/api/v1",
    )


PROVIDER_FACTORIES = {
    "openai": _build_openai_model,
    "openrouter": _build_openrouter_model,
}

if ChatGoogleGenerativeAI is not None:
    def _build_gemini_model():
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    PROVIDER_FACTORIES["gemini"] = _build_gemini_model

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", 6))
_DIR = os.path.dirname(os.path.abspath(__file__))

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
    with open(os.path.join(_DIR, filename), "r", encoding="utf-8") as file:
        return file.read().strip()

MODEL_PROMPT = _load_prompt("model_prompt.txt")
SUMMARIZE_PROMPT = _load_prompt("summarize_prompt.txt")
REWRITE_PROMPT = _load_prompt("rewrite_prompt.txt")


def _content_text(content) -> str:
    """Safely extract text from message content (handles str and list forms)."""
    if isinstance(content, list):
        return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


def _make_tool_call_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": name,
            "args": args,
            "id": call_id,
            "type": "tool_call",
        }],
    )


def _repair_tool_args(tool_calls: list[dict]) -> list[dict]:
    """Normalize near-miss tool argument keys produced by the model."""
    repaired: list[dict] = []
    for call in tool_calls or []:
        args = dict(call.get("args", {}))
        for wrong_key, correct_key in _TOOL_ARG_ALIASES.items():
            if correct_key not in args and wrong_key in args:
                args[correct_key] = args.pop(wrong_key)
        if call.get("name") == "read_pages":
            for wrong_key, correct_key in _READ_PAGES_ARG_ALIASES.items():
                if correct_key not in args and wrong_key in args:
                    args[correct_key] = args.pop(wrong_key)
            page_value = args.pop("page", None)
            if "start_page" not in args and page_value is not None:
                args["start_page"] = page_value
        repaired.append({**call, "args": args})
    return repaired


def _query_needs_exact_value(query: str) -> bool:
    query_lower = query.casefold()
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
        _EXACT_NUMBER_RE.search(query_lower)
        or _POW10_RE.search(query_lower)
        or _POW10_BRACED_RE.search(query_lower)
    )
    return has_exact_marker or has_numeric_target


def _query_has_specific_details(query: str) -> bool:
    return len(_NUMBER_LITERAL_RE.findall(query)) >= 2 or len(query.split()) >= 18


def _compose_search_query(rewritten_query: str, original_query: str) -> str:
    rewritten_query = rewritten_query.strip()
    original_query = " ".join(original_query.split())
    if not original_query or not _query_has_specific_details(original_query):
        return rewritten_query
    if original_query.casefold() in rewritten_query.casefold():
        return rewritten_query
    return f"{rewritten_query}\nLiteral problem statement: {original_query}"


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def _tool_message_texts(state: dict) -> list[str]:
    return [_content_text(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]


def _latest_tool_message(state: dict, name: str) -> ToolMessage | None:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage) and message.name == name:
            return message
    return None


def _already_read_page(state: dict, source: str, page: int) -> bool:
    marker = f"--- [{source}] Page {page} "
    for message in state.get("messages", []):
        if isinstance(message, ToolMessage) and message.name == "read_pages":
            if marker in _content_text(message.content):
                return True
    return False


def _exact_answer_present(query: str, content: str) -> bool:
    if not _query_needs_exact_value(query):
        return True
    text = content.casefold()
    if "answer: unavailable" in text:
        return True
    has_number = bool(_EXACT_NUMBER_RE.search(text))
    if "angle" in query.casefold():
        return has_number and bool(_ANGLE_HINT_RE.search(text))
    return has_number


def _has_direct_worked_answer(content: str) -> bool:
    text = content.casefold()
    if not _CONCLUSION_RE.search(text):
        return False
    if len(_ASSIGNMENT_RE.findall(content)) >= 2:
        return True
    number_count = len(_NUMBER_LITERAL_RE.findall(content))
    return "about" in text and number_count >= 2


def _clean_sentence(text: str) -> str:
    return " ".join(text.split())


def _query_terms(query: str) -> tuple[set[str], set[str]]:
    words: set[str] = set()
    numbers: set[str] = set()
    for token in _TEXT_TOKEN_RE.findall(query.casefold()):
        if any(char.isdigit() for char in token):
            numbers.add(token)
        elif len(token) >= 3 and token not in _QUERY_STOPWORDS:
            words.add(token)
    return words, numbers


def _section_relevance_score(query: str, section: str) -> int:
    query_words, query_numbers = _query_terms(query)
    section_tokens = set(_TEXT_TOKEN_RE.findall(section.casefold()))
    word_overlap = len(query_words & section_tokens)
    number_overlap = len(query_numbers & section_tokens)
    return (number_overlap * 6) + word_overlap


def _top_individual_page(tool_text: str) -> tuple[str, int] | None:
    match = _TOP_PAGE_LINE_RE.search(tool_text)
    if not match:
        return None
    source, page, _score = match.groups()
    return source, int(page)


def _conclusion_sentence_score(query: str, sentence: str, sentence_index: int, sentence_count: int) -> float:
    text = sentence.casefold()
    score = float(_section_relevance_score(query, sentence))
    score += len(_NUMBER_LITERAL_RE.findall(sentence)) * 2.0

    if any(term in text for term in _ANSWER_HINT_TERMS):
        score += 18.0
    if "about" in text:
        score += 8.0
    if len(_ASSIGNMENT_RE.findall(sentence)) >= 1:
        score += 5.0
    if sentence_count > 1:
        score += (sentence_index / max(sentence_count - 1, 1)) * 4.0
    return score


def _direct_answer_from_tool_texts(state: dict) -> str | None:
    original_query = state.get("original_query", "")
    best_answer: str | None = None
    best_score = -1

    for message in state.get("messages", []):
        if not isinstance(message, ToolMessage) or message.name != "read_pages":
            continue
        text = _content_text(message.content)
        for source, page, section in _PAGE_SECTION_RE.findall(text):
            if not _has_direct_worked_answer(section):
                continue

            relevance = _section_relevance_score(original_query, section)
            if _query_has_specific_details(original_query):
                query_words, query_numbers = _query_terms(original_query)
                section_tokens = set(_TEXT_TOKEN_RE.findall(section.casefold()))
                if not (query_numbers & section_tokens or len(query_words & section_tokens) >= 4):
                    continue
            elif relevance < 3:
                continue

            basename = source.split("/")[-1]
            candidate: str | None = None

            conclusion_matches = list(_CONCLUSION_SENTENCE_RE.finditer(section))
            if conclusion_matches:
                best_conclusion = max(
                    enumerate(conclusion_matches),
                    key=lambda item: _conclusion_sentence_score(
                        original_query,
                        item[1].group(0),
                        item[0],
                        len(conclusion_matches),
                    ),
                )[1]
                conclusion = _clean_sentence(best_conclusion.group(0)).rstrip(".")
                conclusion = re.sub(r"^(thus|therefore|hence)\s+", "", conclusion, flags=re.IGNORECASE)
                candidate = f"According to {basename}, page {page}, {conclusion}."
            else:
                assignments = []
                for var, value, unit in _ASSIGNMENT_CAPTURE_RE.findall(section):
                    unit_text = f" {unit.strip()}" if unit.strip() else ""
                    assignments.append(f"{var} = {value}{unit_text}")
                if len(assignments) >= 2:
                    candidate = f"According to {basename}, page {page}, " + ", ".join(assignments[:3]) + "."

            if candidate and relevance > best_score:
                best_score = relevance
                best_answer = candidate

    return best_answer


def _has_relevant_direct_answer(query: str, content: str) -> bool:
    state = {
        "original_query": query,
        "messages": [ToolMessage(content=content, name="read_pages", tool_call_id="relevance_check")],
    }
    return _direct_answer_from_tool_texts(state) is not None


def _relevant_sources_from_tools(state: dict) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for m in state["messages"]:
        if not isinstance(m, ToolMessage):
            continue
        text = _content_text(m.content)
        for source, page in _TOOL_REF_RE.findall(text):
            refs.append((source.split("/")[-1], page))
    return refs


def _has_any_tool_messages(state: dict) -> bool:
    return any(isinstance(m, ToolMessage) for m in state.get("messages", []))


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    original_query: str
    rewritten_query: str
    suggested_scope: str
    scope_relaxed: bool
    forced_tool_retry_count: int


def get_agent(provider: str = "openai", max_tool_calls: int = MAX_TOOL_CALLS):
    if provider not in PROVIDER_FACTORIES:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(PROVIDER_FACTORIES)}")

    llm = PROVIDER_FACTORIES[provider]()
    # build vision tool with access to the vision LLM via closure
    vision_llm = PROVIDER_FACTORIES[provider]()

    def _invoke_vision(system_prompt: str, user_text: str, source: str, page: int) -> str | None:
        img_b64 = render_page_image(source, page)
        if img_b64 is None:
            return None

        response = vision_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]),
        ])
        return _content_text(response.content)

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
        extracted = _invoke_vision(
            VISION_PROMPT,
            f"Extract all content from this page (page {page} of {source}).",
            source,
            page,
        )
        if extracted is None:
            return f"Failed to render page {page} of '{source}'. Check that pymupdf is installed and the file exists."

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
        extracted = _invoke_vision(
            VISION_QA_PROMPT,
            f"Question: {question}\nAnswer it from page {page} of {source}.",
            source,
            page,
        )
        if extracted is None:
            return (
                f"--- [{source}] Page {page} (vision QA) ---\n"
                "Answer: unavailable from this page image\n"
                "Evidence: Failed to render the page image. Check that pymupdf is installed and the file exists.\n"
                "Confidence: low"
            )

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
        query = _compose_search_query(query, original_text)
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
        original_query = state.get("original_query", "")
        last = state["messages"][-1] if state["messages"] else None
        latest_grep = _latest_tool_message(state, "grep_corpus")

        if (
            _query_has_specific_details(original_query)
            and latest_grep is not None
        ):
            top_page = _top_individual_page(_content_text(latest_grep.content))
            if top_page is not None:
                source, page = top_page
                should_force_top_page = (
                    not _already_read_page(state, source, page)
                    and (
                        isinstance(last, ToolMessage)
                        or (
                            isinstance(last, AIMessage)
                            and getattr(last, "name", None) == "critic"
                            and "CONTINUE" in _content_text(last.content).upper()
                        )
                    )
                )
                if should_force_top_page:
                    return {
                        "messages": [_make_tool_call_message(
                            "read_pages",
                            {"source": source, "start_page": page, "end_page": page},
                            call_id=f"auto_read_top_page_{page}",
                        )],
                        "forced_tool_retry_count": 0,
                    }

        if suggested_scope and not state.get("scope_relaxed", False):
            extra_rules.append(
                f"Suggested scope from rewrite: {suggested_scope}. Treat this as a hint, not a hard constraint."
            )
        if state.get("scope_relaxed", False):
            extra_rules.append(
                "A previous scoped attempt was not sufficient. Broaden the search now: remove scope or search a different scope."
            )
        if _query_has_specific_details(original_query):
            extra_rules.append(
                "The original question contains a specific worked problem with concrete numbers. When you call grep_corpus, include the literal setup and exact numeric details from the original question, not just topic keywords."
            )
            extra_rules.append(
                "Prefer pages that contain the full example/problem statement and its answer over generic theory pages."
            )

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
            has_direct_answer = _has_relevant_direct_answer(original_query, content)
            has_figure_reference = "[HAS FIGURE/TABLE REFERENCE" in content or bool(
                re.search(r"(figs?\.?|figure|graph|table)", content, re.IGNORECASE)
            )
            if has_direct_answer:
                verdict_text = "DONE"
            elif needs_exact and has_figure_reference:
                verdict_text = "CONTINUE"
            else:
                verdict_text = "DONE"

        verdict = AIMessage(content=verdict_text)
        verdict.name = "critic"
        return {"messages": [verdict]}

    def summarize_node(state: AgentState):
        original_query = state.get("original_query", "")
        tool_texts = _tool_message_texts(state)
        direct_answer = _direct_answer_from_tool_texts(state)
        if direct_answer:
            return {"messages": [AIMessage(content=direct_answer)]}
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
        if isinstance(last, AIMessage) and not _has_any_tool_messages(state):
            return "force_search"
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

    def force_search_node(state: AgentState):
        original_query = state.get("original_query") or state.get("rewritten_query") or ""
        search_hint = (
            "You must search the corpus before answering. "
            "Call grep_corpus first using the user question and relevant keywords. "
            f"Original question: {original_query}"
        )
        return {
            "messages": [HumanMessage(content=search_hint)],
            "forced_tool_retry_count": 0,
        }

    def read_top_hit_node(state: AgentState):
        latest_grep = _latest_tool_message(state, "grep_corpus")
        if latest_grep is None:
            return {"messages": []}
        top_page = _top_individual_page(_content_text(latest_grep.content))
        if top_page is None:
            return {"messages": []}
        source, page = top_page
        return {
            "messages": [_make_tool_call_message(
                "read_pages",
                {"source": source, "start_page": page, "end_page": page},
                call_id=f"forced_top_page_{page}",
            )],
            "forced_tool_retry_count": 0,
        }

    def after_critic(state: AgentState):
        tool_call_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        if tool_call_count >= max_tool_calls:
            return "summarize"
        original_query = state.get("original_query", "")
        latest_grep = _latest_tool_message(state, "grep_corpus")
        if _query_has_specific_details(original_query) and latest_grep is not None:
            top_page = _top_individual_page(_content_text(latest_grep.content))
            if top_page is not None:
                source, page = top_page
                if not _already_read_page(state, source, page):
                    return "read_top_hit"
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
    graph.add_node("force_search", force_search_node)
    graph.add_node("read_top_hit", read_top_hit_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "model")
    graph.add_conditional_edges("model", should_search, {"search": "search", "force_search": "force_search", "retry_model": "model", "summarize": "summarize"})
    graph.add_edge("search", "critic")
    graph.add_conditional_edges("critic", after_critic, {"model": "model", "broaden_scope": "broaden_scope", "read_top_hit": "read_top_hit", "summarize": "summarize"})
    graph.add_edge("broaden_scope", "model")
    graph.add_edge("force_search", "model")
    graph.add_edge("read_top_hit", "search")
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
