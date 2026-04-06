"""Shared BM25 retrieval utilities used by eval_keyword.py, eval_rerank.py, and answer.py."""
import os
import json
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script

CORPUS_PATH  = os.path.join(_DIR, "corpus.json")
INPUT_PREFIX = "project2/input/"  # Prefix to strip from CSV source paths


def load_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # Load all chunks from corpus.json into memory


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())  # Lowercase + extract alphanumeric tokens
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]      # Remove stop words (e.g. "the", "is")


def build_bm25(corpus: list[dict]) -> BM25Okapi:
    tokenized = [tokenize(item["text"]) for item in corpus]  # Tokenize all chunks once at index time
    return BM25Okapi(tokenized)                               # Build BM25 index over tokenized corpus


def search(bm25: BM25Okapi, corpus: list[dict], keywords: list[str], topk: int) -> list[dict]:
    """Score each corpus item using BM25 and return top-k results."""
    scores = bm25.get_scores(keywords)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [{"score": round(scores[i], 4), **corpus[i]} for i in top_indices if scores[i] > 0]


def normalise_source(csv_source: str) -> str:
    """Strip 'project2/input/' prefix to match corpus.json source format."""
    s = csv_source.strip()
    if s.startswith(INPUT_PREFIX):
        return s[len(INPUT_PREFIX):]
    return s


def parse_pages(page_str: str) -> set[int]:
    """Parse '20-22' or '3' into a set of ints."""
    return {int(p.strip()) for p in page_str.strip().split("-") if p.strip().isdigit()}


def is_hit(results: list[dict], expected_source: str, expected_pages: set[int]) -> bool:
    """Return True if any top-k result matches the expected source and page."""
    for r in results:
        if r["source"] == expected_source and r["page"] in expected_pages:
            return True
    return False
