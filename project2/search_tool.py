"""
Search tools that operate on corpus.json (pre-extracted PDF text).
Three tools for the agent:
  - grep_corpus: keyword search across all pages, returns scored hits
  - read_pages: reads full text for a page range from a specific document
  - list_documents: lists all documents in the corpus
"""
import os
import re
import json
import base64
from langchain_core.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.join(_BASE_DIR, "input")
_CORPUS_PATH = os.path.join(_BASE_DIR, "corpus.json")

_corpus_cache: list[dict] | None = None


def _load_corpus() -> list[dict]:
    """Load corpus.json into memory (cached after first call)."""
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache
    if not os.path.exists(_CORPUS_PATH):
        return []
    with open(_CORPUS_PATH, "r", encoding="utf-8") as f:
        _corpus_cache = json.load(f)
    return _corpus_cache


def _extract_keywords(query: str) -> list[str]:
    """Extract non-numeric keywords from a query, preserving order."""
    seen: set[str] = set()
    keywords: list[str] = []
    for kw in re.findall(r"[A-Za-z][A-Za-z0-9]*", query.lower()):
        if len(kw) <= 1 or kw in seen:
            continue
        seen.add(kw)
        keywords.append(kw)
    return keywords


def render_page_image(source: str, page: int, dpi: int = 200) -> str | None:
    """
    Render a single PDF page as a base64-encoded PNG image.
    Returns base64 string or None if rendering fails.
    Requires: pip install pymupdf
    """
    pdf_path = os.path.join(_INPUT_DIR, source)
    if not os.path.exists(pdf_path):
        return None
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        if page < 1 or page > len(doc):
            doc.close()
            return None
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")
    except ImportError:
        return None
    except Exception:
        return None


MAX_RANGE_PAGES = 5  # never merge into ranges larger than this


def _merge_hits(hits: list[dict], context: int = 1) -> list[dict]:
    """
    Merge neighboring page hits into read ranges.
    If page 12 and 13 both match in the same file, merge into one range
    with ±context pages for surrounding context.
    Ranges are capped at MAX_RANGE_PAGES to avoid reading entire chapters.

    Returns list of {source, start_page, end_page, total_score, best_snippet}
    """
    if not hits:
        return []

    by_source: dict[str, list[dict]] = {}
    for h in hits:
        by_source.setdefault(h["source"], []).append(h)

    merged = []
    for source, pages in by_source.items():
        pages.sort(key=lambda x: x["page"])

        corpus = _load_corpus()
        max_page = max(c["page"] for c in corpus if c["source"] == source)

        ranges = []
        for p in pages:
            start = max(1, p["page"] - context)
            end = min(max_page, p["page"] + context)

            can_merge = (
                ranges
                and ranges[-1]["end_page"] >= start - 1
                and (max(ranges[-1]["end_page"], end) - ranges[-1]["start_page"] + 1) <= MAX_RANGE_PAGES
            )

            if can_merge:
                ranges[-1]["end_page"] = max(ranges[-1]["end_page"], end)
                ranges[-1]["score"] += p["score"]
                if p["score"] > ranges[-1]["best_score"]:
                    ranges[-1]["best_snippet"] = p["snippet"]
                    ranges[-1]["best_score"] = p["score"]
            else:
                ranges.append({
                    "source": source,
                    "start_page": start,
                    "end_page": end,
                    "score": p["score"],
                    "best_score": p["score"],
                    "best_snippet": p["snippet"],
                })

        for r in ranges:
            merged.append({
                "source": r["source"],
                "start_page": r["start_page"],
                "end_page": r["end_page"],
                "total_score": r["score"],
                "best_snippet": r["best_snippet"],
            })

    merged.sort(key=lambda x: -x["total_score"])
    return merged


@tool
def grep_corpus(query: str, scope: str = "", max_results: int = 5) -> str:
    """
    Search pages in corpus.json for keywords. Returns the top matching
    page locations with scores and snippets. Use this to find WHERE relevant
    content is located before reading it in full.

    Args:
        query: Search keywords (e.g. "Ohm's law resistance current")
        scope: Optional prefix to narrow search. Can be a folder (e.g. "ncert_physics_class_12_part_1") or a specific file (e.g. "ncert_physics_class_12_part_1/leph101.pdf"). Leave empty to search all.
        max_results: Maximum number of hit ranges to return (default 5)

    Returns:
        Scored hit locations with source file, page range, and preview snippet.
        Use read_pages() to get the full text of any hit.
    """
    corpus = _load_corpus()
    if not corpus:
        return "No corpus found. Run build_cache.py first."

    # filter by scope if provided
    if scope:
        scope_lower = scope.lower().strip("/")
        pool = [c for c in corpus if c["source"].lower().startswith(scope_lower)]
        if not pool:
            available = sorted(set(c["source"].split("/")[0] for c in corpus))
            return f"No documents match scope '{scope}'. Available scopes: {available}"
    else:
        pool = corpus

    keywords = _extract_keywords(query)
    if not keywords:
        return "No valid search keywords provided."

    hits = []
    for chunk in pool:
        text_lower = chunk["text"].lower()
        score = sum(text_lower.count(kw) for kw in keywords)
        if score > 0:
            snippet = chunk["text"][:200].replace("\n", " ").strip()
            hits.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "score": score,
                "snippet": snippet,
            })

    scope_msg = f" in scope '{scope}'" if scope else ""
    if not hits:
        unique_docs = len(set(c["source"] for c in pool))
        return f"No matches for '{query}'{scope_msg} across {unique_docs} documents ({len(pool)} pages)."

    top_pages = sorted(hits, key=lambda x: -x["score"])[:8]
    merged = _merge_hits(hits)[:max_results]

    lines = [f"Found {len(hits)} page hits across {len(set(h['source'] for h in hits))} documents{scope_msg}.\n"]

    lines.append("Top individual pages:")
    for p in top_pages:
        lines.append(f"  - [{p['source']}] p.{p['page']} (score: {p['score']}) {p['snippet'][:100]}...")

    lines.append(f"\nSuggested read ranges (max {MAX_RANGE_PAGES} pages each):")
    for i, r in enumerate(merged, 1):
        page_str = f"p.{r['start_page']}" if r["start_page"] == r["end_page"] else f"p.{r['start_page']}-{r['end_page']}"
        lines.append(
            f"  {i}. [{r['source']}] {page_str} (score: {r['total_score']})"
        )

    lines.append(f"\nUse read_pages(source, start_page, end_page) to read the most relevant pages.")
    return "\n".join(lines)


@tool
def read_pages(source: str, start_page: int, end_page: int = 0) -> str:
    """
    Read the full text of specific pages from a document in the corpus.
    Use this after grep_corpus to deeply read the content at a hit location.

    Args:
        source: Document path as returned by grep_corpus (e.g. "ncert_physics_class_12_part_1/leph101.pdf")
        start_page: First page to read
        end_page: Last page to read (defaults to start_page if 0)

    Returns:
        Full text of the requested pages with page markers.
    """
    corpus = _load_corpus()
    if not corpus:
        return "No corpus found. Run build_cache.py first."

    if end_page == 0:
        end_page = start_page

    pages = [
        c for c in corpus
        if c["source"] == source and start_page <= c["page"] <= end_page
    ]

    if not pages:
        return f"No pages found for source='{source}' pages {start_page}-{end_page}."

    pages.sort(key=lambda x: x["page"])

    sections = []
    for p in pages:
        low_tag = " [LOW TEXT - may need vision]" if p.get("low_text") else ""
        figure_tag = ""
        if re.search(r"(figs?\.?|figure|table|graph)", p["text"], re.IGNORECASE):
            figure_tag = " [HAS FIGURE/TABLE REFERENCE - consider vision if answer depends on plotted or tabulated values]"
        sections.append(
            f"--- [{p['source']}] Page {p['page']} ({p['word_count']} words){low_tag}{figure_tag} ---\n"
            f"{p['text']}"
        )

    return "\n\n".join(sections)


@tool
def list_documents() -> str:
    """
    List all documents available in the corpus, grouped by folder (scope).
    Use this to understand what scopes and documents are available before searching.
    Pass a scope to grep_corpus to narrow your search to a specific folder.

    Returns:
        Documents grouped by scope (folder) with page and word counts.
    """
    corpus = _load_corpus()
    if not corpus:
        return "No corpus found. Run build_cache.py first."

    # aggregate by folder and source
    scopes: dict[str, dict[str, dict]] = {}
    for c in corpus:
        parts = c["source"].split("/")
        folder = parts[0] if len(parts) > 1 else "(root)"
        src = c["source"]
        if folder not in scopes:
            scopes[folder] = {}
        if src not in scopes[folder]:
            scopes[folder][src] = {"pages": 0, "words": 0, "low_text": 0}
        scopes[folder][src]["pages"] += 1
        scopes[folder][src]["words"] += c["word_count"]
        if c.get("low_text"):
            scopes[folder][src]["low_text"] += 1

    total_docs = sum(len(docs) for docs in scopes.values())
    lines = [f"Corpus: {total_docs} documents, {len(corpus)} pages, {len(scopes)} scopes\n"]
    lines.append("Available scopes (use with grep_corpus scope parameter):")

    for folder in sorted(scopes):
        docs = scopes[folder]
        total_pages = sum(d["pages"] for d in docs.values())
        total_words = sum(d["words"] for d in docs.values())
        lines.append(f"\n  [{folder}] — {len(docs)} files, {total_pages} pages, ~{total_words} words")
        for src in sorted(docs):
            d = docs[src]
            fname = src.split("/")[-1]
            low = f" ({d['low_text']} low-text)" if d["low_text"] else ""
            lines.append(f"    - {fname}: {d['pages']}p, ~{d['words']}w{low}")

    return "\n".join(lines)
