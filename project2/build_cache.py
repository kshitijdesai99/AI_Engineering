"""
One-time text extraction from PDFs in input/ directory.
Recursively scans all subdirectories, extracts text per page,
and stores the result in corpus.json for fast grep-style search.

Usage:
    python build_cache.py [--input input/] [--output corpus.json] [--force]

Requires: pip install pypdf
"""
import os
import sys
import json
import argparse
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_DIR, "input")
DEFAULT_OUTPUT = os.path.join(_DIR, "corpus.json")


def find_pdfs(input_dir: str) -> list[str]:
    """Recursively find all PDF files under input_dir."""
    pdfs = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return pdfs


def extract_pages(pdf_path: str, input_dir: str) -> list[dict]:
    """Extract text from each page of a PDF. Returns list of page chunks."""
    import pypdf

    chunks = []
    rel_path = os.path.relpath(pdf_path, input_dir)

    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as e:
        print(f"  ⚠️  failed to open: {e}")
        return chunks

    for page_num, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

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
        print(f"corpus.json already exists. Use --force to rebuild.")
        sys.exit(1)

    pdfs = find_pdfs(input_dir)
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Building corpus from {len(pdfs)} PDFs")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    corpus = []
    total_pages = 0
    low_text_pages = 0
    skipped_empty = 0

    for i, pdf_path in enumerate(pdfs, 1):
        rel = os.path.relpath(pdf_path, input_dir)
        print(f"[{i}/{len(pdfs)}] {rel}", end="")

        chunks = extract_pages(pdf_path, input_dir)
        page_count = len(chunks)
        low = sum(1 for c in chunks if c["low_text"])

        total_pages += page_count
        low_text_pages += low

        print(f"  → {page_count} pages" + (f" ({low} low-text)" if low else ""))
        corpus.extend(chunks)

    elapsed = round(time.time() - t0, 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    file_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed}s")
    print(f"  PDFs processed:  {len(pdfs)}")
    print(f"  Pages extracted: {total_pages}")
    print(f"  Low-text pages:  {low_text_pages} (flagged, may need vision)")
    print(f"  Corpus size:     {file_size_mb} MB")
    print(f"  Saved to:        {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build text cache from PDFs")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input directory with PDFs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing corpus.json")
    args = parser.parse_args()

    build_cache(args.input, args.output, force=args.force)
