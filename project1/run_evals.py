"""
Eval runner for the LLM Code Execution Agent.
Reads question/answer pairs from a CSV, runs each through the agent,
extracts numeric answers, and reports accuracy.

Usage:
    python run_evals.py [gemini|openai|openrouter] [--file evals/train.csv] [--out evals/results.json]
    python run_evals.py gemini --file evals/train.csv --async --workers 5
"""
import sys
import os
import csv
import json
import re
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

load_dotenv()

_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
sys.path.insert(0, _DIR)                           # Ensure docker_agent is importable
from docker_agent import get_agent, AGENT_TIMEOUT, _content_text


def extract_number(text: str) -> str | None:
    """Extract the most likely final numeric answer from free-text agent output."""
    if not text:
        return None
    if isinstance(text, list):
        text = " ".join(p["text"] for p in text if isinstance(p, dict) and p.get("type") == "text")
    text = str(text)
    text = re.sub(r"\*{1,2}(-?[\d,]+\.?\d*)\*{1,2}", r"\1", text)  # Strip markdown bold around numbers

    # Strategy 1: last line that is a bare number
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if re.fullmatch(r"-?[\d,]+\.?\d*", line):
            return line.replace(",", "")

    # Strategy 2: number after keywords like "is", "=", "answer:"
    pattern = r"(?:is|=|equals|\u2248|approximately|result[:\s]|answer[:\s])\s*(-?[\d,]+\.?\d*)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].replace(",", "")

    # Strategy 3: fallback — last number anywhere in the text
    all_nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if all_nums:
        return all_nums[-1].replace(",", "")
    return None


def answers_match(extracted: str | None, expected: str, tolerance: float = 0.01) -> bool:
    """Compare extracted answer to expected, using 1% relative tolerance for numerics."""
    if extracted is None:
        return False
    try:
        ext = float(extracted)
        exp = float(expected)
        if exp == 0:
            return abs(ext - exp) < tolerance      # Absolute tolerance when expected is 0
        return abs(ext - exp) / max(abs(exp), 1e-9) < tolerance  # Relative tolerance otherwise
    except ValueError:
        return extracted.strip() == expected.strip()  # Fallback: exact string match


def _extract_answer(messages: list) -> str:
    """Extract the final answer text — prefer last AIMessage (excluding critic), fallback to last ToolMessage."""
    def _to_str(content) -> str:
        if isinstance(content, list):
            return " ".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content)

    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content and getattr(m, "name", None) != "critic"),
        None,
    )
    if last_ai:
        return _to_str(last_ai.content)
    last_tool = next((m for m in reversed(messages) if isinstance(m, ToolMessage) and m.content), None)
    return _to_str(last_tool.content) if last_tool else ""  # Empty string if no answer found


def _build_meta(messages: list, expected: str = None) -> dict:
    """Build metadata dict from the agent's message history for a single question."""
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    tool_count = len(tool_msgs)                          # Total tool calls made
    error_count = sum(
        1 for m in tool_msgs
        if any(kw in _content_text(m.content).lower() for kw in ("error", "traceback"))  # Count error tool results
    )
    tool_raw = _content_text(tool_msgs[-1].content).strip() if tool_msgs else ""  # Raw output of last tool call
    tool_correct = None
    if tool_count > 0 and expected is not None:
        tool_extracted = extract_number(tool_raw)
        tool_correct = answers_match(tool_extracted, expected)  # Did the tool output contain the right answer?

    return {
        "tool_calls": tool_count,
        "errors": error_count,
        "tool_raw": tool_raw[:200],
        "tool_correct": tool_correct,
        "recursion_hit": False,   # Overwritten by caller if recursion limit was hit
        "timed_out": False,       # Overwritten by caller if timeout occurred
    }


def _tool_tag(meta: dict) -> str:
    """Format tool usage status for console output."""
    tc = meta["tool_calls"]
    if tc == 0:
        return "tool=⛔ (none)"     # Agent answered without using tools
    ok = meta.get("tool_correct")
    if ok is True:
        return f"tool=✅({tc})"     # Tool output contained correct answer
    elif ok is False:
        return f"tool=❌({tc})"     # Tool output contained wrong answer
    return f"tool=?({tc})"          # No expected value to compare against


def _stream_agent(agent, recursion_limit: int, question: str):
    """Stream a single question through the agent and return messages + error flags."""
    messages = []
    recursion_hit = False
    error_count = 0

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]  # Accumulate latest message state
    except GraphRecursionError:
        recursion_hit = True   # MAX_TOOL_CALLS exceeded
    except (httpx.ReadTimeout,):
        pass                   # Network timeout — return whatever messages we have
    except (ValueError, ChatGoogleGenerativeAIError):
        error_count += 1       # Provider/API error

    return messages, recursion_hit, error_count


def run_single(agent, recursion_limit: int, question: str, expected: str = None, timeout: int = AGENT_TIMEOUT):
    """Run one question sequentially with a timeout enforced via ThreadPoolExecutor."""
    t0 = time.time()
    timed_out = False
    messages = []
    recursion_hit = False
    error_count = 0

    # Run agent in a thread so we can enforce wall-clock timeout
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_stream_agent, agent, recursion_limit, question)
        try:
            messages, recursion_hit, error_count = future.result(timeout=timeout)
        except Exception:
            timed_out = True   # future.result timed out — agent took too long
            future.cancel()

    meta = _build_meta(messages, expected)
    meta["latency_s"] = round(time.time() - t0, 2)
    meta["recursion_hit"] = recursion_hit
    meta["timed_out"] = timed_out
    meta["errors"] = max(meta["errors"], error_count)  # Take the higher error count

    return _extract_answer(messages), meta


def _run_one_question(agent, recursion_limit, idx, total, question, expected, timeout):
    t0 = time.time()
    try:
        messages, recursion_hit, error_count = _stream_agent(agent, recursion_limit, question)
    except Exception:
        messages, recursion_hit, error_count = [], False, 1

    meta = _build_meta(messages, expected)
    meta["latency_s"] = round(time.time() - t0, 2)
    meta["recursion_hit"] = recursion_hit
    meta["timed_out"] = False
    meta["errors"] = max(meta["errors"], error_count)

    answer_text = _extract_answer(messages)
    extracted = extract_number(answer_text)
    match = answers_match(extracted, expected)
    status = "✅" if match else "❌"
    tag = _tool_tag(meta)
    print(f"[{idx}/{total}] {status}  expected={expected}  extracted={extracted}  {tag}  latency={meta['latency_s']}s  | {question[:55]}")
    if meta["recursion_hit"]:
        print(f"  ⚠️  recursion limit hit")
    return {
        "question": question,
        "expected": expected,
        "extracted": extracted,
        "agent_raw": answer_text[:300],
        "match": match,
        **meta,
    }


def _print_line(i, total, question, expected, extracted, meta, match):
    status = "✅" if match else "❌"
    tag = _tool_tag(meta)
    print(f"[{i}/{total}] {status}  expected={expected}  extracted={extracted}  {tag}  latency={meta['latency_s']}s  | {question[:55]}")
    if meta["recursion_hit"]:
        print(f"  ⚠️  recursion limit hit")
    if meta["timed_out"]:
        print(f"  ⚠️  timed out")


def run_evals(provider: str, csv_path: str, out_path: str, parallel: bool = False, workers: int = 5):
    """Main eval loop — runs all questions and writes results to JSON."""
    agent, recursion_limit = get_agent(provider)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))  # Load all question/answer rows from CSV

    total = len(rows)
    mode = f"parallel ({workers} workers)" if parallel else "sequential"

    print(f"\n{'='*70}")
    print(f"  EVAL RUN — {total} questions | provider: {provider}")
    print(f"  file: {csv_path} | mode: {mode}")
    print(f"{'='*70}\n")

    t_start = time.time()

    if parallel:  # ── parallel mode ───────────────────────────────────────────
        results = [None] * total
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, row in enumerate(rows):
                f = executor.submit(
                    _run_one_question,
                    agent, recursion_limit,
                    i + 1, total,
                    row["question"], row["answer"],
                    AGENT_TIMEOUT,
                )
                futures[f] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=AGENT_TIMEOUT + 30)
                except Exception as e:
                    print(f"  ⚠️  [{idx+1}/{total}] worker exception: {e}")
                    results[idx] = {
                        "question": rows[idx]["question"],
                        "expected": rows[idx]["answer"],
                        "extracted": None,
                        "agent_raw": str(e)[:300],
                        "match": False,
                        "tool_calls": 0, "errors": 1, "tool_raw": "",
                        "tool_correct": None, "recursion_hit": False,
                        "timed_out": True, "latency_s": AGENT_TIMEOUT,
                    }
    else:  # ── sequential mode ──────────────────────────────────────────────
        results = []
        for i, row in enumerate(rows, 1):
            question = row["question"]
            expected = row["answer"]

            answer_text, meta = run_single(agent, recursion_limit, question, expected=expected)
            extracted = extract_number(answer_text)
            match = answers_match(extracted, expected)

            _print_line(i, total, question, expected, extracted, meta, match)

            results.append({
                "question": question,
                "expected": expected,
                "extracted": extracted,
                "agent_raw": answer_text[:300],
                "match": match,
                **meta,
            })

    # ── compute summary stats ────────────────────────────────────────────
    wall_time = round(time.time() - t_start, 2)
    correct = sum(1 for r in results if r["match"])
    accuracy = correct / total if total else 0
    tool_used = sum(1 for r in results if r["tool_calls"] > 0)          # Questions where tool was called
    tool_correct = sum(1 for r in results if r.get("tool_correct") is True)   # Tool output was correct
    tool_wrong = sum(1 for r in results if r.get("tool_correct") is False)     # Tool output was wrong

    summary = {
        "provider": provider,
        "file": csv_path,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "wall_time_s": wall_time,
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / total, 2) if total else 0,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in results) / total, 2) if total else 0,
        "tool_used": tool_used,
        "tool_correct": tool_correct,
        "tool_wrong": tool_wrong,
        "total_errors": sum(r["errors"] for r in results),
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)  # Write full results to JSON

    print(f"\n{'='*70}")
    print(f"  RESULTS: {correct}/{total} correct ({accuracy:.1%})")
    print(f"  Tool usage: {tool_used}/{total} used tool | {tool_correct} tool✅ | {tool_wrong} tool❌")
    print(f"  Wall time: {wall_time}s" + (f" (vs ~{round(summary['avg_latency_s'] * total, 1)}s sequential est.)" if parallel else ""))
    print(f"  Avg latency: {summary['avg_latency_s']}s | Avg tool calls: {summary['avg_tool_calls']}")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Saved to: {out_path}")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    # ── CLI argument parsing ─────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser(description="Run evals for the docker agent")
    parser.add_argument("provider", nargs="?", default="gemini", choices=["gemini", "openai", "openrouter"])
    parser.add_argument("--file", default=os.path.join(_DIR, "evals", "train.csv"))  # Default eval CSV
    parser.add_argument("--out", default=None)  # Auto-generated if not provided
    parser.add_argument("--async", dest="parallel", action="store_true", help="Run questions in parallel")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    args = parser.parse_args()

    # ── auto-generate output path if not specified ───────────────────────
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.file))[0]  # e.g. "train" from "train.csv"
        args.out = os.path.join(_DIR, "evals", f"results_{stem}_{args.provider}.json")

    run_evals(args.provider, args.file, args.out, parallel=args.parallel, workers=args.workers)
