"""
Bare minimum RAG: grep corpus → read top page → LLM extract.
No loop. No rewriting. No critic.
"""
import os
import re
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from search_tool import grep_corpus, read_pages

load_dotenv()

_DIR = os.path.dirname(os.path.abspath(__file__))
_TOP_HIT_RE = re.compile(r"-\s+\[([^\]]+)\]\s+p\.(\d+)\s+\(score:")

EXTRACT_PROMPT = """\
Extract the answer to the question from the page text below.
Reply with one short sentence or phrase — match the style of a textbook answer key.
End with: (Source: <filename>, page <N>)
Do not explain or restate the question."""


def _build_llm(provider: str):
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    if provider == "openrouter":
        return ChatOpenAI(
            model="minimax/minimax-m2.5:free",
            api_key=str(os.getenv("OPEN_ROUTER_API_KEY", "")),
            base_url="https://openrouter.ai/api/v1",
        )
    raise ValueError(f"Unknown provider: {provider}")


def answer(query: str, provider: str = "openai") -> str:
    grep_result = grep_corpus.invoke({"query": query, "max_results": 5})

    match = _TOP_HIT_RE.search(grep_result)
    if not match:
        return "No relevant pages found in corpus."

    source, page = match.group(1), int(match.group(2))
    page_text = read_pages.invoke({"source": source, "start_page": page, "end_page": page})

    llm = _build_llm(provider)
    response = llm.invoke([
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nPage text:\n{page_text}"),
    ])
    return response.content.strip()


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    with open(os.path.join(_DIR, "query.txt"), encoding="utf-8") as f:
        query = f.read().strip()
    print(f"Query: {query}\n")
    print(answer(query, provider))
