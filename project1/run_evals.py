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

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
from docker_agent import get_agent, AGENT_TIMEOUT, _content_text


def extract_number(text: str) -> str | None:
    """Extract the most likely final numeric answer from free-text agent output."""
    if not text:
        return None
    if isinstance(text, list):
        text = " ".join(p["text"] for p in text if isinstance(p, dict) and p.get("type") == "text")
    text = str(text)
    text = re.sub(r"\*{1,2}(-?[\d,]+\.?\d*)\*{1,2}", r"\1", text)

    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if re.fullmatch(r"-?[\d,]+\.?\d*", line):
            return line.replace(",", "")

    pattern = r"(?:is|=|equals|\u2248|approximately|result[:\s]|answer[:\s])\s*(-?[\d,]+\.?\d*)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].replace(",", "")

    all_nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if all_nums:
        return all_nums[-1].replace(",", "")
    return None


def answers_match(extracted: str | None, expected: str, tolerance: float = 0.01) -> bool:
    if extracted is None:
        return False
    try:
        ext = float(extracted)
        exp = float(expected)
        if exp == 0:
            return abs(ext - exp) < tolerance
        return abs(ext - exp) / max(abs(exp), 1e-9) < tolerance
    except ValueError:
        return extracted.strip() == expected.strip()


def _extract_answer(messages: list) -> str:
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
    return _to_str(last_tool.content) if last_tool else ""


def _build_meta(messages: list, expected: str = None) -> dict:
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    tool_count = len(tool_msgs)
    error_count = sum(
        1 for m in tool_msgs
        if any(kw in _content_text(m.content).lower() for kw in ("error", "traceback"))
    )
    tool_raw = _content_text(tool_msgs[-1].content).strip() if tool_msgs else ""
    tool_correct = None
    if tool_count > 0 and expected is not None:
        tool_extracted = extract_number(tool_raw)
        tool_correct = answers_match(tool_extracted, expected)

    return {
        "tool_calls": tool_count,
        "errors": error_count,
        "tool_raw": tool_raw[:200],
        "tool_correct": tool_correct,
        "recursion_hit": False,
        "timed_out": False,
    }


def _tool_tag(meta: dict) -> str:
    tc = meta["tool_calls"]
    if tc == 0:
        return "tool=⛔ (none)"
    ok = meta.get("tool_correct")
    if ok is True:
        return f"tool=✅({tc})"
    elif ok is False:
        return f"tool=❌({tc})"
    return f"tool=?({tc})"


def _stream_agent(agent, recursion_limit: int, question: str):
    messages = []
    recursion_hit = False
    error_count = 0

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]
    except GraphRecursionError:
        recursion_hit = True
    except (httpx.ReadTimeout,):
        pass
    except (ValueError, ChatGoogleGenerativeAIError):
        error_count += 1

    return messages, recursion_hit, error_count


def run_single(agent, recursion_limit: int, question: str, expected: str = None, timeout: int = AGENT_TIMEOUT):
    t0 = time.time()
    timed_out = False
    messages = []
    recursion_hit = False
    error_count = 0

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_stream_agent, agent, recursion_limit, question)
        try:
            messages, recursion_hit, error_count = future.result(timeout=timeout)
        except Exception:
            timed_out = True
            future.cancel()

    meta = _build_meta(messages, expected)
    meta["latency_s"] = round(time.time() - t0, 2)
    meta["recursion_hit"] = recursion_hit
    meta["timed_out"] = timed_out
    meta["errors"] = max(meta["errors"], error_count)

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
    agent, recursion_limit = get_agent(provider)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    mode = f"parallel ({workers} workers)" if parallel else "sequential"

    print(f"\n{'='*70}")
    print(f"  EVAL RUN — {total} questions | provider: {provider}")
    print(f"  file: {csv_path} | mode: {mode}")
    print(f"{'='*70}\n")

    t_start = time.time()

    if parallel:
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
    else:
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

    wall_time = round(time.time() - t_start, 2)
    correct = sum(1 for r in results if r["match"])
    accuracy = correct / total if total else 0
    tool_used = sum(1 for r in results if r["tool_calls"] > 0)
    tool_correct = sum(1 for r in results if r.get("tool_correct") is True)
    tool_wrong = sum(1 for r in results if r.get("tool_correct") is False)

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
        json.dump(summary, f, indent=2)

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
    import argparse
    parser = argparse.ArgumentParser(description="Run evals for the docker agent")
    parser.add_argument("provider", nargs="?", default="gemini", choices=["gemini", "openai", "openrouter"])
    parser.add_argument("--file", default=os.path.join(_DIR, "evals", "train.csv"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--async", dest="parallel", action="store_true", help="Run questions in parallel")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    args = parser.parse_args()

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.file))[0]
        args.out = os.path.join(_DIR, "evals", f"results_{stem}_{args.provider}.json")

    run_evals(args.provider, args.file, args.out, parallel=args.parallel, workers=args.workers)
