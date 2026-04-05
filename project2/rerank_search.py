"""
Two-stage retrieval eval: BM25 (Stage 1) → cross-encoder re-rank (Stage 2).

Usage:
    python rerank_search.py [--file evals/easy.csv] [--out evals/results_easy_rerank.json] [--retrieval_k 100] [--topk 5]
"""
import os
import argparse
import csv
import json
from time import perf_counter

from keyword_search import load_corpus, build_bm25, tokenize, search, normalise_source, parse_pages, is_hit
from rerank import rerank

_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_DIR, "corpus.json")


def run_evals(csv_path: str, out_path: str, topk: int = 5, retrieval_k: int = 100):
    corpus = load_corpus(CORPUS_PATH)
    bm25 = build_bm25(corpus)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Evaluating {total} questions from {csv_path} (retrieval_k={retrieval_k}, topk={topk})\n")

    t_start = perf_counter()
    results = []

    for i, row in enumerate(rows, 1):
        question        = row["Question"].strip()
        csv_source      = row[" PDF"].strip()
        page_str        = row[" Page"].strip()
        expected_source = normalise_source(csv_source)
        expected_pages  = parse_pages(page_str)

        keywords    = tokenize(question)
        candidates  = search(bm25, corpus, keywords, retrieval_k)
        hits        = rerank(question, candidates, topk)
        match       = is_hit(hits, expected_source, expected_pages)

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
                {"source": r["source"], "page": r["page"], "rerank_score": r["rerank_score"]}
                for r in hits
            ],
        })

    elapsed = round(perf_counter() - t_start, 2)
    correct  = sum(1 for r in results if r["match"])
    accuracy = correct / total if total else 0

    summary = {
        "file":        csv_path,
        "retrieval_k": retrieval_k,
        "topk":        topk,
        "total":       total,
        "correct":     correct,
        "accuracy":    round(accuracy, 4),
        "results":     results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{correct}/{total} correct ({accuracy:.1%}) in {elapsed}s — {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage BM25 + re-rank eval")
    parser.add_argument("--file",        default=os.path.join(_DIR, "evals", "easy.csv"))
    parser.add_argument("--out",         default=None)
    parser.add_argument("--topk",        type=int, default=5)
    parser.add_argument("--retrieval_k", type=int, default=100)
    args = parser.parse_args()

    if args.out is None:
        stem     = os.path.splitext(os.path.basename(args.file))[0]
        args.out = os.path.join(_DIR, "evals", f"results_{stem}_rerank.json")

    run_evals(args.file, args.out, topk=args.topk, retrieval_k=args.retrieval_k)
