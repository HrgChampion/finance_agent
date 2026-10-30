"""Evaluate the agents: run each case several times and measure grounding, consistency and routing.

All checks are deterministic (no LLM judge):
- grounding: share of numbers in an answer that can be traced to a tool output in the same run
- rating consistency: same BUY / HOLD / SELL across runs of the same ticker
- rating rule: stated rating matches the average-score rule
- workflow: briefs call all five specialists, step 1 in parallel, then valuation, then the writer
- routing: questions call the specialists they need and no others

Usage:
    python evaluate.py                    # every case in evals/cases.json, 3 runs each
    python evaluate.py --runs 2 --only qa
    python evaluate.py --match "Nvidia|weather"     # only matching cases
    python evaluate.py --rescore evals/results/<stamp>.json   # rebuild a report, no model calls
"""
import argparse
import json
import re
import statistics
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

from bank_agent import SPECIALISTS, orchestrator_agent, reset_metrics
from observability import setup_logging

EVAL_DIR = Path("evals")
SPECIALIST_NAMES = {a.name for a in SPECIALISTS}
STEP_1 = ("FinancialDataAgent", "NewsAndSentimentAgent", "RiskAgent")
DIMENSIONS = ("Financials", "Valuation", "Risk", "Sentiment")


class ToolRecorder(HookProvider):
    """Records every tool call (start, end, output) made by any agent during a run."""

    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.start)
        registry.add_callback(AfterToolCallEvent, self.end)

    def start(self, event: BeforeToolCallEvent) -> None:
        self._add(event.agent.name, "start", event.tool_use["name"], "")

    def end(self, event: AfterToolCallEvent) -> None:
        text = "".join(block.get("text", "") for block in event.result.get("content", []))
        self._add(event.agent.name, "end", event.tool_use["name"], text)

    def _add(self, agent, kind, tool, text):
        with self._lock:
            self.events.append({"t": time.perf_counter(), "agent": agent, "kind": kind, "tool": tool, "text": text})


# --- grounding -------------------------------------------------------------------------------

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def numbers(text, with_decimals=False):
    """All numbers in text, e.g. '$1,575.64 (35%)' -> [1575.64, 35.0]. Skips ones glued to words (FY2025, Q3).
    with_decimals=True returns (value, decimal places as written) pairs instead."""
    found = []
    for match in NUMBER.finditer(text):
        raw = match.group().replace(",", "").rstrip(".")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        found.append((value, decimals) if with_decimals else value)
    return found


def checkable(x):
    """Years, list numbers and 1-5 scores are not facts to trace."""
    return not (x.is_integer() and (abs(x) <= 5 or 1990 <= x <= 2100))


def grounded(x, facts, decimals=2):
    """True if x matches a fact, allowing rounding and a change of units (millions <-> billions etc.).
    decimals is how precisely x was written: '35%' may round 35.4, '0.04' may not match 0.0001."""
    x = abs(x)
    for scale in (1, 1e3, 1e-3, 1e6, 1e-6):
        y = x * scale
        for t in facts:
            if scale == 1:
                tolerance = max(0.005 * t, 0.5 * 10 ** -decimals)   # 0.5%, or rounding at the written precision
            elif t >= 1:
                tolerance = 0.005 * t
            else:
                continue
            if abs(y - t) <= tolerance:
                return True
    return False


LIST_MARKER = re.compile(r"^\s*\d+\.\s+")


def grounding(answer, tool_texts, derived=()):
    """(share of checkable numbers that are grounded, checked count, ungrounded numbers with context).
    derived: numbers the answer may legitimately compute itself, e.g. the rating rule's average."""
    facts = {abs(n) for text in tool_texts for n in numbers(text) if n} | {abs(d) for d in derived}
    checked, missing = 0, []
    for line in answer.splitlines():
        line = LIST_MARKER.sub("", line)                    # "6. Bottom line" is not the fact 6
        for x, decimals in numbers(line, with_decimals=True):
            if not checkable(x):
                continue
            checked += 1
            if not grounded(x, facts, decimals):
                missing.append(f"{x:g} in: {line.strip()[:120]}")
    return (None if not checked else 1 - len(missing) / checked), checked, missing


# --- briefs ----------------------------------------------------------------------------------

def parse_brief(text):
    """(stated rating, {dimension: score or None for N/A}) from a brief."""
    rating = re.search(r"\*\*(BUY|HOLD|SELL)\*\*", text) or re.search(r"\b(BUY|HOLD|SELL)\b", text)
    scores = {}
    for dim in DIMENSIONS:
        row = re.search(rf"^\|\s*\**{dim}\**\s*\|.*\|\s*\**(\d(?:\.\d+)?|N/A)\**\s*\|\s*$", text, re.M)
        if row:
            scores[dim] = None if row.group(1) == "N/A" else float(row.group(1))
    return (rating.group(1) if rating else None), scores


def average_score(scores):
    """Average of the scored dimensions, N/A excluded; None if nothing was scored."""
    values = [v for v in scores.values() if v is not None]
    return sum(values) / len(values) if values else None


def rule_rating(scores):
    """The rating the ReportWriter's rule implies: average >= 3.75 BUY, <= 2.25 SELL, else HOLD."""
    average = average_score(scores)
    if average is None:
        return None
    return "BUY" if average >= 3.75 else "SELL" if average <= 2.25 else "HOLD"


# --- workflow and routing --------------------------------------------------------------------

def spans(events):
    """{specialist: [[start, end], ...]} for each specialist the orchestrator called."""
    out = {}
    for e in events:
        if e["agent"] != "OrchestratorAgent" or e["tool"] not in SPECIALIST_NAMES:
            continue
        calls = out.setdefault(e["tool"], [])
        if e["kind"] == "start":
            calls.append([e["t"], float("inf")])
        else:
            next(c for c in calls if c[1] == float("inf"))[1] = e["t"]
    return out


def check_order(sp):
    problems = []
    if "FinancialDataAgent" in sp and "ValuationAgent" in sp:
        if sp["ValuationAgent"][0][0] < sp["FinancialDataAgent"][0][1]:
            problems.append("ValuationAgent started before FinancialDataAgent finished")
    return problems


def check_workflow(sp):
    """Problems with a full-brief run, or [] if it followed the workflow."""
    if set(sp) != SPECIALIST_NAMES:
        return [f"called {sorted(sp)} instead of all five specialists"]
    problems = check_order(sp)
    step_1 = [sp[name][0] for name in STEP_1]
    if max(start for start, _ in step_1) > min(end for _, end in step_1):
        problems.append("step 1 specialists did not run in parallel")
    if sp["ReportWriterAgent"][0][0] < sp["ValuationAgent"][0][1]:
        problems.append("ReportWriterAgent started before ValuationAgent finished")
    return problems


def check_routing(sp, case):
    """Problems with a question's routing, or [] if it called the right specialists."""
    called = set(sp)
    problems = []
    if missing := set(case.get("required", [])) - called:
        problems.append(f"missing {sorted(missing)}")
    if case.get("required_any") and not called & set(case["required_any"]):
        problems.append(f"called none of {case['required_any']}")
    if extra := called - set(case.get("allowed", [])):
        problems.append(f"unnecessary {sorted(extra)}")
    if repeated := [name for name, calls in sp.items() if len(calls) > 1]:
        problems.append(f"called more than once: {repeated}")
    return problems + check_order(sp)


# --- running ---------------------------------------------------------------------------------

def run_once(question, recorder):
    orchestrator = orchestrator_agent()
    orchestrator.hooks.add_hook(recorder)
    agents = [orchestrator] + SPECIALISTS
    reset_metrics(agents)
    recorder.events.clear()

    start = time.perf_counter()
    try:
        answer, error = str(orchestrator(question)), None
    except Exception as e:
        answer, error = "", f"{type(e).__name__}: {e}"[:300]
    wall = time.perf_counter() - start

    events = list(recorder.events)
    tool_texts = [question] + [e["text"] for e in events if e["kind"] == "end" and e["tool"] not in SPECIALIST_NAMES]
    result = {
        "question": question, "answer": answer, "error": error, "wall_s": round(wall, 1),
        "tokens": sum(a.event_loop_metrics.accumulated_usage["totalTokens"] for a in agents),
        "tool_texts": tool_texts,                           # saved so grounding can be recomputed offline
        "spans": spans(events),
    }
    score_grounding(result)
    return result


def score_grounding(result):
    """Grounding for one run; a brief's rating-rule average counts as a legitimately derived number."""
    average = average_score(parse_brief(result["answer"])[1])
    derived = [] if average is None else [round(average, 2)]
    result["grounding"], result["numbers_checked"], result["ungrounded"] = grounding(
        result["answer"], result["tool_texts"], derived)


def evaluate(cases, runs, only, pause, match=None):
    plan = []
    if only in (None, "briefs"):
        plan += [{"kind": "brief", "ticker": t, "question": f"Write an investment brief for {t}"} for t in cases["briefs"]]
    if only in (None, "qa"):
        plan += [{"kind": "qa", **case} for case in cases["questions"]]
    if match:
        plan = [case for case in plan if re.search(match, case["question"], re.I)]

    for agent in SPECIALISTS:
        agent.hooks.add_hook(RECORDER)
    results, total, n = [], len(plan) * runs, 0
    for case in plan:
        for run in range(1, runs + 1):
            n += 1
            result = {**case, "run": run, **run_once(case["question"], RECORDER)}
            if case["kind"] == "brief":
                result["rating"], result["scores"] = parse_brief(result["answer"])
                result["rule_rating"] = rule_rating(result["scores"])
                result["problems"] = check_workflow(result["spans"])
            else:
                result["problems"] = check_routing(result["spans"], case)
            if result["error"]:
                result["problems"].insert(0, result["error"])
            results.append(result)
            status = "FAILED" if result["error"] else ("ok" if not result["problems"] else "issues")
            print(f"[{n}/{total}] {case['question'][:55]:<55} run {run}: {status} ({result['wall_s']:.0f}s)", flush=True)
            if n < total:
                time.sleep(pause)                           # stay under the free-tier rate limits
    return results


# --- report ----------------------------------------------------------------------------------

def pct(part, whole):
    return "n/a" if not whole else f"{part / whole:.0%} ({part}/{whole})"


def mean(values, fmt="{:,.0f}"):
    values = [v for v in values if v is not None]
    return fmt.format(statistics.mean(values)) if values else "n/a"


def report(results, runs, cases_file):
    done = [r for r in results if not r["error"]]
    briefs = [r for r in done if r["kind"] == "brief"]
    qa = [r for r in done if r["kind"] == "qa"]
    checked = sum(r["numbers_checked"] for r in done)
    missing = sum(len(r["ungrounded"]) for r in done)

    lines = [
        "# Evaluation report",
        "",
        f"{datetime.now():%Y-%m-%d %H:%M} | {runs} runs per case | cases: `{cases_file.as_posix()}`",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Runs completed | {pct(len(done), len(results))} |",
        f"| Numbers traceable to a tool output | {pct(checked - missing, checked)} |",
    ]
    tickers = sorted({r["ticker"] for r in briefs})
    if briefs:
        unanimous = sum(len({r["rating"] for r in briefs if r["ticker"] == t}) == 1 for t in tickers)
        rule_ok = sum(r["rating"] == r["rule_rating"] for r in briefs)
        workflow_ok = sum(not r["problems"] for r in briefs)
        lines += [
            f"| Briefs: same rating in every run | {pct(unanimous, len(tickers))} of tickers |",
            f"| Briefs: rating follows the score rule | {pct(rule_ok, len(briefs))} |",
            f"| Briefs: workflow followed | {pct(workflow_ok, len(briefs))} |",
            f"| Briefs: mean time / tokens | {mean([r['wall_s'] for r in briefs], '{:.0f}s')} / {mean([r['tokens'] for r in briefs])} |",
        ]
    if qa:
        routing_ok = sum(not r["problems"] for r in qa)
        lines += [
            f"| Questions: routed correctly | {pct(routing_ok, len(qa))} |",
            f"| Questions: mean time / tokens | {mean([r['wall_s'] for r in qa], '{:.0f}s')} / {mean([r['tokens'] for r in qa])} |",
        ]

    if briefs:
        lines += ["", "## Briefs", "",
                  "| Ticker | Ratings | Score range " + " / ".join(DIMENSIONS) + " | Rule ok | Workflow ok | Grounded | Time | Tokens |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in tickers:
            rs = [r for r in briefs if r["ticker"] == t]
            ranges = []
            for dim in DIMENSIONS:
                values = [r["scores"].get(dim) for r in rs]
                nums = [v for v in values if v is not None]
                text = (f"{min(nums):g}" if min(nums) == max(nums) else f"{min(nums):g}-{max(nums):g}") if nums else "-"
                ranges.append(text + ("*" if None in values else ""))
            lines.append(
                f"| {t} | {', '.join(str(r['rating']) for r in rs)} | {' / '.join(ranges)} | "
                f"{sum(r['rating'] == r['rule_rating'] for r in rs)}/{len(rs)} | {sum(not r['problems'] for r in rs)}/{len(rs)} | "
                f"{mean([r['grounding'] for r in rs], '{:.0%}')} | {mean([r['wall_s'] for r in rs], '{:.0f}s')} | {mean([r['tokens'] for r in rs])} |")
        lines.append("\n`*` a run scored this dimension N/A (data unavailable).")

    if qa:
        lines += ["", "## Questions", "", "| Question | Routed ok | Specialists called (most common) | Grounded | Time | Tokens |",
                  "|---|---|---|---|---|---|"]
        for question in dict.fromkeys(r["question"] for r in qa):
            rs = [r for r in qa if r["question"] == question]
            called = Counter(", ".join(sorted(r["spans"])) or "none" for r in rs).most_common(1)[0][0]
            lines.append(f"| {question} | {sum(not r['problems'] for r in rs)}/{len(rs)} | {called} | "
                         f"{mean([r['grounding'] for r in rs], '{:.0%}')} | {mean([r['wall_s'] for r in rs], '{:.0f}s')} | "
                         f"{mean([r['tokens'] for r in rs])} |")

    issues = [r for r in results if r["problems"]]
    if issues:
        lines += ["", "## Problems", ""]
        lines += [f"- **{r['question']}** (run {r['run']}): {'; '.join(r['problems'])}" for r in issues]

    untraced = [r for r in done if r["ungrounded"]]
    if untraced:
        lines += ["", "## Numbers not traceable to a tool output", "",
                  "Usually values the model computed itself (e.g. an implied price) or rounded heavily; "
                  "occasionally a real error. Review them by hand.", ""]
        for r in untraced:
            lines.append(f"- **{r['question']}** (run {r['run']}):")
            lines += [f"  - {m}" for m in r["ungrounded"][:6]]
            if len(r["ungrounded"]) > 6:
                lines.append(f"  - ...and {len(r['ungrounded']) - 6} more")

    lines += ["", "## Method", "",
              "- **Grounding:** every number in the final answer (except years, list numbers and 1-5 scores) is matched "
              "against the numbers in that run's tool outputs, allowing 0.5% rounding and unit changes "
              "(millions/billions). Approximate: a number can match a fact by coincidence, so treat it as an upper bound.",
              "- **Workflow / routing:** from the orchestrator's tool-call timings, recorded with Strands hooks.",
              "- **Rating rule:** the average of the scored dimensions (N/A excluded): >= 3.75 BUY, <= 2.25 SELL, else HOLD."]
    return "\n".join(lines) + "\n"


RECORDER = ToolRecorder()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate the finance research agents.")
    parser.add_argument("--runs", type=int, default=3, help="runs per case (default 3)")
    parser.add_argument("--only", choices=["briefs", "qa"], help="run only briefs or only questions")
    parser.add_argument("--pause", type=float, default=20, help="seconds between runs, for rate limits (default 20)")
    parser.add_argument("--cases", type=Path, default=EVAL_DIR / "cases.json")
    parser.add_argument("--match", help="run only cases whose question matches this regex, e.g. 'Nvidia|weather'")
    parser.add_argument("--rescore", type=Path, metavar="RUNS_JSON",
                        help="rebuild the report from a saved results .json, without calling any model")
    args = parser.parse_args()

    if args.rescore:
        results = json.loads(args.rescore.read_text(encoding="utf-8"))
        for r in results:
            score_grounding(r)
        text = report(results, max(r["run"] for r in results), args.cases)
        args.rescore.with_suffix(".md").write_text(text, encoding="utf-8")
        (EVAL_DIR / "REPORT.md").write_text(text, encoding="utf-8")
        print(text)
        return

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y-%m-%d_%H%M%S}"
    setup_logging(out_dir / f"{stamp}.log", console=False)

    results = evaluate(json.loads(args.cases.read_text(encoding="utf-8")), args.runs, args.only, args.pause, args.match)
    for r in results:
        r["spans"] = {k: [[round(s, 2), round(e, 2)] for s, e in v] for k, v in r["spans"].items()}
    (out_dir / f"{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    text = report(results, args.runs, args.cases)
    (out_dir / f"{stamp}.md").write_text(text, encoding="utf-8")
    (EVAL_DIR / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"\n{text}\nSaved to {EVAL_DIR / 'REPORT.md'} (raw runs: {out_dir / f'{stamp}.json'})")


if __name__ == "__main__":
    main()
