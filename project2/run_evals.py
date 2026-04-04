"""
Eval harness: measures retrieval accuracy and answer match against gold CSVs.

Usage:
    python run_evals.py                    # runs both easy.csv and hard.csv
    python run_evals.py --csv evals/easy.csv
    python run_evals.py --csv evals/easy.csv --answers --provider openai
"""
import argparse
import csv
import os
import re

from search_tool import grep_corpus

_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLD_PREFIX = "project2/input/"
_TOP_HIT_RE = re.compile(r"-\s+\[([^\]]+)\]\s+p\.(\d+)\s+\(score:")
_PDF_RE = re.compile(r"project2/input/\S+\.pdf")


def _strip_gold_path(pdf: str) -> str:
    pdf = pdf.strip()
    return pdf[len(_GOLD_PREFIX):] if pdf.startswith(_GOLD_PREFIX) else pdf


def _parse_row(raw_row: list[str], header: list[str]) -> dict:
    pdf_index = next(i for i, value in enumerate(raw_row) if _PDF_RE.fullmatch(value.strip()))
    question = ", ".join(part.strip() for part in raw_row[:pdf_index]).strip()
    pdf = raw_row[pdf_index].strip()
    page = raw_row[pdf_index + 1].strip()
    answer = ", ".join(part.strip() for part in raw_row[pdf_index + 2:]).strip()
    return {
        header[0].strip(): question,
        header[1].strip(): pdf,
        header[2].strip(): page,
        header[3].strip(): answer,
    }


def _load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        rows = []
        for raw_row in reader:
            if not raw_row:
                continue
            rows.append(_parse_row(raw_row, header))
        return rows


def _parse_pages(page_text: str) -> set[int]:
    if not page_text:
        return set()
    if "-" in page_text:
        start_text, end_text = page_text.split("-", 1)
        start = int(start_text.strip())
        end = int(end_text.strip())
        if start > end:
            start, end = end, start
        return set(range(start, end + 1))
    return {int(page_text)}


def _retrieval_hits(question: str, gold_source: str, gold_pages: set[int]) -> tuple[bool, bool]:
    result = grep_corpus.invoke({"query": question, "max_results": 10})
    hits = [(m.group(1), int(m.group(2))) for m in _TOP_HIT_RE.finditer(result)]
    top1 = any(source == gold_source and page in gold_pages for source, page in hits[:1])
    top3 = any(source == gold_source and page in gold_pages for source, page in hits[:3])
    return top1, top3


def _answer_match(response: str, gold_answer: str) -> bool:
    return gold_answer.casefold() in response.casefold()


def run_csv(path: str, check_answers: bool = False, provider: str = "openai"):
    rows = _load_csv(path)
    csv_name = os.path.basename(path)

    if check_answers:
        from search_agent import answer as rag_answer

    col_ans = "  Ans" if check_answers else ""
    header = f"{'Q':<3} {'Question':<44} {'Gold':<24} {'T1':>3} {'T3':>3}{col_ans}"
    print(f"\n{csv_name}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    top1 = top3 = ans_match = total = 0

    for i, row in enumerate(rows, 1):
        question = row["Question"]
        gold_source = _strip_gold_path(row["PDF"])
        gold_page_raw = row["Page"]
        gold_answer = row.get("Answer", "").strip()

        gold_pages = _parse_pages(gold_page_raw)
        gold_label = f"{gold_source.split('/')[-1]} p.{gold_page_raw}"

        total += 1

        in_top1, in_top3 = _retrieval_hits(question, gold_source, gold_pages)
        top1 += in_top1
        top3 += in_top3

        t1 = "✓" if in_top1 else "✗"
        t3 = "✓" if in_top3 else "✗"

        ans_col = ""
        if check_answers and gold_answer:
            response = rag_answer(question, provider)
            matched = _answer_match(response, gold_answer)
            ans_match += matched
            ans_col = f"  {'✓' if matched else '✗'}"

        print(f"{i:<3} {question[:43]:<44} {gold_label:<24} {t1:>3} {t3:>3}{ans_col}")

    print("-" * len(header))
    ans_note = f"  answer: {ans_match}/{total}" if check_answers else ""
    print(f"top-1: {top1}/{total}  top-3: {top3}/{total}{ans_note}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to a specific eval CSV")
    parser.add_argument("--answers", action="store_true", help="Also run answer extraction and check match")
    parser.add_argument("--provider", default="openai", help="LLM provider (openai, gemini, openrouter)")
    args = parser.parse_args()

    targets = (
        [args.csv]
        if args.csv
        else [
            os.path.join(_DIR, "evals/easy.csv"),
            os.path.join(_DIR, "evals/hard.csv"),
        ]
    )
    for path in targets:
        run_csv(path, check_answers=args.answers, provider=args.provider)
