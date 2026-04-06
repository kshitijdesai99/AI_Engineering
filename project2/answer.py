"""
Stage 3: LLM answer generation using top-k re-ranked pages as context.

Usage:
    python answer.py --query "What is Coulomb's law?" [--topk 5] [--retrieval_k 100]
"""
import os
import argparse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from keyword_search import load_corpus, build_bm25, tokenize, search
from rerank import rerank

load_dotenv()

_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_DIR, "corpus.json")

PROVIDERS = {
    "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")),
    "openai": lambda: ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
}


def _load_prompt(filename: str) -> str:
    return open(os.path.join(_DIR, filename)).read().strip()


def _build_context(pages: list[dict]) -> str:
    parts = []
    for i, p in enumerate(pages, 1):
        parts.append(f"[{i}] Source: {p['source']}, Page {p['page']}\n{p['text']}")
    return "\n\n---\n\n".join(parts)


def answer(query: str, topk: int = 5, retrieval_k: int = 100, provider: str = "gemini") -> str:
    corpus = load_corpus(CORPUS_PATH)
    bm25 = build_bm25(corpus)

    keywords = tokenize(query)
    candidates = search(bm25, corpus, keywords, retrieval_k)
    pages = rerank(query, candidates, topk)

    context = _build_context(pages)
    system_prompt = _load_prompt("answer_prompt.txt")

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    llm = PROVIDERS[provider]()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    return response.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a query using two-stage retrieval + LLM")
    parser.add_argument("--query",       required=True,          help="Question to answer")
    parser.add_argument("--topk",        type=int, default=5,    help="Final pages passed to LLM")
    parser.add_argument("--retrieval_k", type=int, default=100,  help="BM25 candidate pool size")
    parser.add_argument("--provider",    default=os.getenv("ANSWER_PROVIDER", "gemini"), choices=list(PROVIDERS))
    args = parser.parse_args()

    result = answer(args.query, topk=args.topk, retrieval_k=args.retrieval_k, provider=args.provider)
    print("\nAnswer:", result)
