"""
Search tools that operate on corpus.json (pre-extracted PDF text).
Three tools for the agent:
  - grep_corpus: keyword search across all pages, returns scored hits
  - read_pages: reads full text for a page range from a specific document
  - list_documents: lists all documents in the corpus
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import re
import json
import base64
from langchain_core.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.join(_BASE_DIR, "input")
_CORPUS_PATH = os.path.join(_BASE_DIR, "corpus.json")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?")
_VISUAL_REF_RE = re.compile(r"(figs?\.?|figure|table|graph)", re.IGNORECASE)
_NO_CORPUS_MESSAGE = "No corpus found. Run build_cache.py first."
MAX_RANGE_PAGES = 5  # never merge into ranges larger than this
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "each", "end", "find",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or", "question",
    "support", "supported", "that", "the", "their", "to", "what", "which", "with",
    "your",
}


@dataclass
class CorpusIndex:
    pages: list[dict]
    by_source: dict[str, list[dict]]
    max_page_by_source: dict[str, int]
    scopes: dict[str, dict[str, dict[str, int]]]
    token_doc_freq: dict[str, int]
    document_listing: str


def _empty_index() -> CorpusIndex:
    return CorpusIndex(
        pages=[],
        by_source={},
        max_page_by_source={},
        scopes={},
        token_doc_freq={},
        document_listing=_NO_CORPUS_MESSAGE,
    )


def _document_scope(source: str) -> str:
    parts = source.split("/", 1)
    return parts[0] if len(parts) > 1 else "(root)"


def _build_document_listing(corpus: list[dict], scopes: dict[str, dict[str, dict[str, int]]]) -> str:
    total_docs = sum(len(docs) for docs in scopes.values())
    lines = [f"Corpus: {total_docs} documents, {len(corpus)} pages, {len(scopes)} scopes\n"]
    lines.append("Available scopes (use with grep_corpus scope parameter):")

    for folder in sorted(scopes):
        docs = scopes[folder]
        total_pages = sum(d["pages"] for d in docs.values())
        total_words = sum(d["words"] for d in docs.values())
        lines.append(f"\n  [{folder}] — {len(docs)} files, {total_pages} pages, ~{total_words} words")
        for src in sorted(docs):
            doc_stats = docs[src]
            fname = src.split("/")[-1]
            low = f" ({doc_stats['low_text']} low-text)" if doc_stats["low_text"] else ""
            lines.append(f"    - {fname}: {doc_stats['pages']}p, ~{doc_stats['words']}w{low}")

    return "\n".join(lines)


def _normalize_phrase_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9.\-\s]+", " ", text)
    text = text.replace("-", " ")
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize_phrase_text(text))


def _is_numeric_token(token: str) -> bool:
    return any(char.isdigit() for char in token)


def _load_corpus() -> list[dict]:
    """Load corpus.json into memory (cached after first call)."""
    return _load_corpus_index().pages


@lru_cache(maxsize=1)
def _load_corpus_index() -> CorpusIndex:
    if not os.path.exists(_CORPUS_PATH):
        return _empty_index()

    with open(_CORPUS_PATH, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    by_source: dict[str, list[dict]] = {}
    max_page_by_source: dict[str, int] = {}
    scopes: dict[str, dict[str, dict[str, int]]] = {}
    token_doc_freq: Counter[str] = Counter()

    for page in corpus:
        text = page.get("text", "")
        source = page["source"]
        page["_phrase_text"] = _normalize_phrase_text(text)
        page["_token_counts"] = Counter(_tokenize(text))
        page["_has_visual_reference"] = bool(_VISUAL_REF_RE.search(text))
        token_doc_freq.update(page["_token_counts"].keys())

        source_pages = by_source.setdefault(source, [])
        source_pages.append(page)
        max_page_by_source[source] = max(max_page_by_source.get(source, 0), page["page"])

        folder = _document_scope(source)
        doc_stats = scopes.setdefault(folder, {}).setdefault(source, {"pages": 0, "words": 0, "low_text": 0})
        doc_stats["pages"] += 1
        doc_stats["words"] += page.get("word_count", 0)
        if page.get("low_text"):
            doc_stats["low_text"] += 1

    for pages in by_source.values():
        pages.sort(key=lambda page: page["page"])

    return CorpusIndex(
        pages=corpus,
        by_source=by_source,
        max_page_by_source=max_page_by_source,
        scopes=scopes,
        token_doc_freq=dict(token_doc_freq),
        document_listing=_build_document_listing(corpus, scopes),
    )


def _available_scopes(index: CorpusIndex) -> list[str]:
    return sorted(index.scopes)


def _pages_for_scope(index: CorpusIndex, scope: str) -> list[dict]:
    if not scope:
        return index.pages

    scope_prefix = scope.casefold().strip("/")
    matching_pages: list[dict] = []
    for source, pages in index.by_source.items():
        if source.casefold().startswith(scope_prefix):
            matching_pages.extend(pages)
    return matching_pages


def _extract_keywords(query: str) -> list[str]:
    """Extract distinctive keywords from a query, preserving order."""
    seen: set[str] = set()
    keywords: list[str] = []
    for token in _tokenize(query):
        if token in seen:
            continue
        if _is_numeric_token(token):
            if len(token) >= 2 or "." in token:
                seen.add(token)
                keywords.append(token)
            continue
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _extract_query_phrases(query: str, max_phrases: int = 12) -> list[str]:
    tokens = _tokenize(query)
    if len(tokens) < 4:
        return []

    phrases: list[str] = []
    seen: set[str] = set()
    for window_size in (8, 7, 6, 5, 4):
        if len(tokens) < window_size:
            continue
        for start in range(len(tokens) - window_size + 1):
            window = tokens[start:start + window_size]
            if not any(_is_numeric_token(token) for token in window) and sum(len(token) >= 6 for token in window) < 2:
                continue
            phrase = " ".join(window)
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def _token_weight(token: str, doc_freq: int, total_pages: int) -> float:
    base_weight = 1.0 + math.log((total_pages + 1) / (doc_freq + 1))
    if _is_numeric_token(token):
        return base_weight * 4.0
    if len(token) >= 8:
        return base_weight * 1.6
    if len(token) >= 5:
        return base_weight * 1.2
    return base_weight


def _score_chunk(
    chunk: dict,
    keywords: list[str],
    phrases: list[str],
    token_doc_freq: dict[str, int],
    total_pages: int,
) -> float:
    token_counts: Counter[str] = chunk["_token_counts"]
    phrase_text = chunk["_phrase_text"]
    score = 0.0
    matched_keywords = 0
    matched_numeric_tokens = 0
    matched_phrases = 0

    for token in keywords:
        term_frequency = token_counts.get(token, 0)
        if not term_frequency:
            continue
        matched_keywords += 1
        if _is_numeric_token(token):
            matched_numeric_tokens += 1
        score += term_frequency * _token_weight(token, token_doc_freq.get(token, 0), total_pages)

    for phrase in phrases:
        if phrase in phrase_text:
            matched_phrases += 1
            score += 12.0 + (len(phrase.split()) * 2.5)

    if matched_numeric_tokens >= 2:
        score += matched_numeric_tokens * 6.0
    if matched_keywords >= 4:
        score += matched_keywords
    if matched_phrases and "answer" in phrase_text:
        score += 10.0
    if matched_phrases and "example" in phrase_text:
        score += 6.0

    return score


def _best_snippet(text: str, keywords: list[str], snippet_chars: int = 220) -> str:
    text_flat = text.replace("\n", " ").strip()
    lowered = text_flat.casefold()

    match_index = -1
    for token in keywords:
        if len(token) <= 2:
            continue
        match_index = lowered.find(token.casefold())
        if match_index >= 0:
            break

    if match_index < 0:
        return text_flat[:snippet_chars].strip()

    start = max(0, match_index - 80)
    end = min(len(text_flat), match_index + snippet_chars - 80)
    snippet = text_flat[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text_flat):
        snippet = snippet + "..."
    return snippet


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
        try:
            if page < 1 or page > len(doc):
                return None
            pix = doc[page - 1].get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
        finally:
            doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")
    except ImportError:
        return None
    except Exception:
        return None


def _merge_hits(hits: list[dict], max_page_by_source: dict[str, int], context: int = 1) -> list[dict]:
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
        max_page = max_page_by_source.get(source, 0)

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
    index = _load_corpus_index()
    corpus = index.pages
    if not corpus:
        return _NO_CORPUS_MESSAGE

    # filter by scope if provided
    if scope:
        pool = _pages_for_scope(index, scope)
        if not pool:
            return f"No documents match scope '{scope}'. Available scopes: {_available_scopes(index)}"
    else:
        pool = corpus

    keywords = _extract_keywords(query)
    if not keywords:
        return "No valid search keywords provided."
    phrases = _extract_query_phrases(query)
    total_pages = len(index.pages)

    hits = []
    for chunk in pool:
        score = _score_chunk(chunk, keywords, phrases, index.token_doc_freq, total_pages)
        if score > 0:
            snippet = _best_snippet(chunk["text"], keywords)
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

    top_pages = sorted(hits, key=lambda x: (-x["score"], x["source"], x["page"]))[:8]
    merged = _merge_hits(hits, index.max_page_by_source)[:max_results]

    lines = [f"Found {len(hits)} page hits across {len(set(h['source'] for h in hits))} documents{scope_msg}.\n"]

    lines.append("Top individual pages:")
    for p in top_pages:
        lines.append(f"  - [{p['source']}] p.{p['page']} (score: {round(p['score'], 2)}) {p['snippet'][:100]}...")

    lines.append(f"\nSuggested read ranges (max {MAX_RANGE_PAGES} pages each):")
    for i, r in enumerate(merged, 1):
        page_str = f"p.{r['start_page']}" if r["start_page"] == r["end_page"] else f"p.{r['start_page']}-{r['end_page']}"
        lines.append(
            f"  {i}. [{r['source']}] {page_str} (score: {round(r['total_score'], 2)})"
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
    index = _load_corpus_index()
    if not index.pages:
        return _NO_CORPUS_MESSAGE

    if end_page == 0:
        end_page = start_page

    source_pages = index.by_source.get(source, [])
    pages = [page for page in source_pages if start_page <= page["page"] <= end_page]

    if not pages:
        return f"No pages found for source='{source}' pages {start_page}-{end_page}."

    sections = []
    for p in pages:
        low_tag = " [LOW TEXT - may need vision]" if p.get("low_text") else ""
        figure_tag = ""
        if p.get("_has_visual_reference"):
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
    return _load_corpus_index().document_listing
