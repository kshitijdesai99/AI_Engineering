"""
Extracts text from every PDF in the input/ directory and saves it to corpus.json.
Run this once before using any search or eval scripts.

Usage:
    python build_cache.py [--input input/] [--output corpus.json] [--force]

Requires: pip install pypdf
"""
import argparse
import json
import os
import sys
from time import perf_counter
import pypdf

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_DIR, "input")
DEFAULT_OUTPUT = os.path.join(_DIR, "corpus.json")


def find_pdfs(input_dir: str) -> list[str]:
    """Walk input_dir and return all PDF paths, sorted alphabetically."""
    pdfs: list[str] = []
    for root, _, files in os.walk(input_dir):
        pdfs.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.lower().endswith(".pdf")
        )
    return pdfs


def extract_pages(pdf_path: str, input_dir: str) -> list[dict]:
    """Extract text page-by-page from a PDF. Skips blank pages."""
    chunks = []
    rel_path = os.path.relpath(pdf_path, input_dir)

    reader = pypdf.PdfReader(pdf_path)

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""

        text = text.strip()
        word_count = len(text.split())

        if not text:
            continue

        chunks.append({
            "source": rel_path,
            "page": page_num,
            "text": text,
            "word_count": word_count,
            "low_text": word_count < 20,
        })

    return chunks


def build_cache(input_dir: str, output_path: str, force: bool = False):
    if os.path.exists(output_path) and not force:
        print(f"{output_path} already exists — pass --force to rebuild.")
        return

    pdfs = find_pdfs(input_dir)
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        sys.exit(1)

    t0 = perf_counter()
    corpus: list[dict] = []
    total_pages = 0

    for i, pdf_path in enumerate(pdfs, 1):
        chunks = extract_pages(pdf_path, input_dir)
        total_pages += len(chunks)
        corpus.extend(chunks)
        print(f"[{i}/{len(pdfs)}] {os.path.relpath(pdf_path, input_dir)}  ({len(chunks)} pages)")

    elapsed = round(perf_counter() - t0, 2)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(corpus, fp, ensure_ascii=False, indent=2)

    print(f"\nDone — {total_pages} pages from {len(pdfs)} PDFs in {elapsed}s → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build text cache from PDFs")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input directory with PDFs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing corpus.json")
    args = parser.parse_args()

    build_cache(args.input, args.output, force=args.force)
