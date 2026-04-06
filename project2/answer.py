"""
Stage 3: LLM answer generation using top-k re-ranked pages as context.

Usage:
    python answer.py --query "What is Coulomb's law?" [--topk 5] [--retrieval_k 100]
"""
import os
import argparse
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval import load_corpus, build_bm25, tokenize, search
from rerank import rerank

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")    # Enable LangSmith tracing
os.environ.setdefault("LANGCHAIN_PROJECT", "search-agent") # LangSmith project name

_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
CORPUS_PATH = os.path.join(_DIR, "corpus.json")

# List of available LLM providers
PROVIDERS = {
    "gemini":      lambda: ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=os.getenv("GEMINI_API_KEY")),
    "openai":      lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "openrouter":  lambda: ChatOpenAI(
        model="minimax/minimax-m2.5:free",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
}


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_DIR, filename), encoding="utf-8") as f:
        return f.read().strip()  # Read the prompt from the file


SYSTEM_PROMPT = _load_prompt("model_prompt.txt")  # Load system prompt once at startup


def _content_text(content) -> str:
    if isinstance(content, list):
        # Extract text from list of content parts (Gemini returns structured list)
        return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


def _expand_context(hits: list[dict], corpus: list[dict]) -> list[dict]:
    index: dict[tuple, list[dict]] = defaultdict(list)
    for chunk in corpus:
        index[(chunk["source"], chunk["page"])].append(chunk)  # Build (source, page) → chunks lookup

    seen: set[tuple] = set()
    expanded: list[dict] = []
    for hit in hits:
        for page in (hit["page"] - 1, hit["page"], hit["page"] + 1):  # Include prev/next page to avoid boundary misses
            key = (hit["source"], page)
            if key not in seen and key in index:
                seen.add(key)
                expanded.extend(index[key])  # Add all chunks from this page
    return expanded


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {c['source']}, Page {c['page']}, Chunk {c.get('chunk_index', '-')}\n{c['text']}")
    return "\n\n---\n\n".join(parts)  # Separate chunks with a divider for LLM readability


def answer(query: str, topk: int = 10, retrieval_k: int = 100, provider: str = "gemini") -> str:
    corpus = load_corpus(CORPUS_PATH)  # Load all chunks from corpus.json
    bm25 = build_bm25(corpus)          # Build BM25 index

    keywords = tokenize(query)
    candidates = search(bm25, corpus, keywords, retrieval_k)  # Stage 1: BM25 → top-retrieval_k chunks
    hits = rerank(query, candidates, topk)                    # Stage 2: ZeroEntropy reranker → top-k
    pages = _expand_context(hits, corpus)                     # Stage 3: Expand to prev/next page chunks

    context = _build_context(pages)  # Format chunks into numbered context string

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    llm = PROVIDERS[provider]()  # Create the LLM instance for the chosen provider
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    return _content_text(response.content)  # Safely extract text from response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a query using two-stage retrieval + LLM")
    parser.add_argument("--query",       default=None,           help="Question to answer (overrides query.txt)")
    parser.add_argument("--topk",        type=int, default=10,   help="Final pages passed to LLM")
    parser.add_argument("--retrieval_k", type=int, default=100,  help="BM25 candidate pool size")
    parser.add_argument("--provider",    default=os.getenv("ANSWER_PROVIDER", "gemini"), choices=list(PROVIDERS))
    args = parser.parse_args()

    if args.query:
        query = args.query  # CLI arg takes priority
    else:
        with open(os.path.join(_DIR, "query.txt"), encoding="utf-8") as f:
            query = f.read().strip()  # Fall back to query.txt
    result = answer(query, topk=args.topk, retrieval_k=args.retrieval_k, provider=args.provider)
    print("\nAnswer:", result)
