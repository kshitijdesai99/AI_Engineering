"""
LangChain tool that searches documents in the input/ directory.
Supports .txt, .md, and .pdf files. Uses keyword/grep-style search.
"""
import os
import re
from langchain_core.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.join(_BASE_DIR, "input")

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py"}

def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return "[PDF support requires pypdf: pip install pypdf]"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _list_documents() -> list[str]:
    docs = []
    for fname in os.listdir(_INPUT_DIR):
        if os.path.splitext(fname)[1].lower() in _SUPPORTED_EXTENSIONS:
            docs.append(os.path.join(_INPUT_DIR, fname))
    return docs

@tool
def search_documents(query: str, max_results: int = 5) -> str:
    """
    Search documents in the input/ directory for content relevant to the query.

    Args:
        query: The search query or keywords to look for
        max_results: Maximum number of matching excerpts to return

    Returns:
        Matching excerpts from documents with source filenames
    """
    docs = _list_documents()
    if not docs:
        return "No documents found in input/ directory."

    keywords = re.findall(r'\w+', query.lower())
    results = []

    scored = []
    for doc_path in docs:
        content = _read_file(doc_path)
        fname = os.path.basename(doc_path)
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = sum(1 for kw in keywords if kw in line_lower)
            if score > 0:
                start = max(0, i - 1)
                end = min(len(lines), i + 4)
                excerpt = "\n".join(lines[start:end]).strip()
                if excerpt:
                    scored.append((-score, fname, excerpt))

    scored.sort(key=lambda x: x[0])
    for _, fname, excerpt in scored[:max_results]:
        results.append(f"[{fname}]\n{excerpt}")

    if not results:
        return f"No matches found for '{query}' in {len(docs)} document(s): {[os.path.basename(d) for d in docs]}"

    return "\n\n---\n\n".join(results)

@tool
def extract_code_patterns(pattern: str) -> str:
    """
    Extract specific code patterns from Python files in input/ directory.
    Use this for questions about code structure, e.g. counting nodes, edges, functions, classes.

    Args:
        pattern: A regex or keyword pattern to extract (e.g. 'add_node', 'def ', 'class ')

    Returns:
        All matching lines with their filenames and line numbers
    """
    docs = [p for p in _list_documents() if p.endswith(".py")]
    if not docs:
        return "No Python files found in input/ directory."

    results = []
    for doc_path in docs:
        fname = os.path.basename(doc_path)
        content = _read_file(doc_path)
        lines = content.split("\n")
        matches = []
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                start = max(0, i - 2)
                end = min(len(lines), i + 8)
                block = "\n".join(f"  line {start+j+1}: {lines[start+j].rstrip()}" for j in range(end - start))
                matches.append(block)
        if matches:
            results.append(f"[{fname}] ({len(matches)} matches):\n" + "\n---\n".join(matches))

    if not results:
        return f"No matches for pattern '{pattern}' in {[os.path.basename(d) for d in docs]}"
    return "\n\n".join(results)

@tool
def list_documents() -> str:
    """
    List all documents available in the input/ directory.

    Returns:
        List of document filenames
    """
    docs = _list_documents()
    if not docs:
        return "No documents found in input/ directory."
    return "Available documents:\n" + "\n".join(f"- {os.path.basename(d)}" for d in docs)
