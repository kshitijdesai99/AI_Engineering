"""
BM25-based retrieval eval for corpus.json.
Reads question/answer/source pairs from a CSV, scores pages using BM25
against corpus.json (no LLM), and reports retrieval accuracy.

Usage:
    python keyword_search.py [--file evals/easy.csv] [--out evals/results_easy_keyword.json] [--topk 5]
"""
import os
import csv
import json
import re
from time import perf_counter
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_DIR = os.path.dirname(os.path.abspath(__file__))

CORPUS_PATH = os.path.join(_DIR, "corpus.json")
INPUT_PREFIX = "project2/input/"  # Prefix to strip from CSV source paths

def load_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


def build_bm25(corpus: list[dict]) -> BM25Okapi:
    tokenized = [tokenize(item["text"]) for item in corpus]
    return BM25Okapi(tokenized)


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


def run_evals(csv_path: str, out_path: str, topk: int = 5, retrieval_k: int = 100):
    corpus = load_corpus(CORPUS_PATH)
    bm25   = build_bm25(corpus)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Evaluating {total} questions from {csv_path} (retrieval_k={retrieval_k}, topk={topk})\n")

    t_start = perf_counter()
    results = []

    for i, row in enumerate(rows, 1):
        question     = row["Question"].strip()
        csv_source   = row[" PDF"].strip()
        page_str     = row[" Page"].strip()
        expected_source = normalise_source(csv_source)
        expected_pages  = parse_pages(page_str)
        keywords        = tokenize(question)
        hits            = search(bm25, corpus, keywords, retrieval_k)[:topk]
        match           = is_hit(hits, expected_source, expected_pages)

        status = "✅" if match else "❌"
        top_result = f"{hits[0]['source']}:p{hits[0]['page']}" if hits else "none"
        print(f"[{i}/{total}] {status}  expected={expected_source}:{page_str}  top={top_result}  | {question[:60]}")

        results.append({
            "question":        question,
            "expected_source": expected_source,
            "expected_pages":  list(expected_pages),
            "keywords":        keywords,
            "match":           match,
            "top_results": [
                {"source": r["source"], "page": r["page"], "score": r["score"]}
                for r in hits
            ],
        })

    elapsed = round(perf_counter() - t_start, 2)
    correct  = sum(1 for r in results if r["match"])
    accuracy = correct / total if total else 0

    summary = {
        "file":     csv_path,
        "topk":     topk,
        "total":    total,
        "correct":  correct,
        "accuracy": round(accuracy, 4),
        "results":  results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{correct}/{total} correct ({accuracy:.1%}) in {elapsed}s — {out_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Keyword-based retrieval eval against corpus.json")
    parser.add_argument("--file",  default=os.path.join(_DIR, "evals", "easy.csv"))
    parser.add_argument("--out",   default=None)
    parser.add_argument("--topk",        type=int, default=5)
    parser.add_argument("--retrieval_k", type=int, default=100)
    args = parser.parse_args()

    if args.out is None:
        stem     = os.path.splitext(os.path.basename(args.file))[0]
        args.out = os.path.join(_DIR, "evals", f"results_{stem}_keyword.json")

    run_evals(args.file, args.out, topk=args.topk, retrieval_k=args.retrieval_k)
