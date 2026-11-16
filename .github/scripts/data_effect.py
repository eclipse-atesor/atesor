#!/usr/bin/env python3
r"""Report what the learned data (data/) gave a batch of ports.

Reads the ``agent-call_<package>.jsonl`` call logs of one logs folder.
Each LLM call record names the few-shot examples its prompt carried
(``metadata.examples``), and each run ends with one ``data_effect``
record (written by ``graph._log_data_effect``): the plan source, the
recalled recipe, the examples shown and copied by a fix that worked,
the fixer rounds, the cost and the duration.

Prints a Markdown report to stdout (and appends it to
``$GITHUB_STEP_SUMMARY`` when that variable is set):

    1. calls with an example, distinct examples, fixes that copied one;
    2. recipe seeds used, and seeds that passed without a fixer;
    3. the time of each package against its earlier run;
    4. prune candidates: examples shown 10 times or more and never
       copied (the netmaker example reached 9 prompts and helped none).

The script reads only; it changes no data file.

Example::

    python .github/scripts/data_effect.py --logs-dir workspace-native/logs
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

# An example shown this often and never copied is a prune candidate.
PRUNE_SHOWN = 10


def read_records(logs_dir: str) -> List[Dict[str, Any]]:
    """Return every JSON record of the call logs in ``logs_dir``.

    A line that is not valid JSON is skipped.
    """
    records: List[Dict[str, Any]] = []
    pattern = os.path.join(logs_dir, "agent-call_*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the report figures for the records of one batch."""
    calls = [r for r in records if r.get("record") != "data_effect"]
    runs = [r for r in records if r.get("record") == "data_effect"]
    shown: Counter = Counter()
    with_example = 0
    for call in calls:
        ids = (call.get("metadata") or {}).get("examples") or []
        if ids:
            with_example += 1
            shown.update(ids)
    copied: Counter = Counter()
    for run in runs:
        copied.update(run.get("examples_copied") or [])
    seeds = [
        r for r in runs if r.get("plan_source") in ("recipe", "cached_recipe")
    ]
    by_repo: Dict[str, List[Dict[str, Any]]] = {}
    for run in sorted(runs, key=lambda r: str(r.get("timestamp", ""))):
        by_repo.setdefault(str(run.get("repo", "?")), []).append(run)
    times = []
    for repo, history in sorted(by_repo.items()):
        last = history[-1]
        earlier = history[-2] if len(history) > 1 else None
        times.append(
            {
                "repo": repo,
                "outcome": last.get("outcome", "?"),
                "plan_source": last.get("plan_source", ""),
                "seconds": last.get("duration_seconds"),
                "earlier_seconds": (
                    earlier.get("duration_seconds") if earlier else None
                ),
            }
        )
    return {
        "llm_calls": len(calls),
        "calls_with_example": with_example,
        "distinct_examples": len(shown),
        "runs": len(runs),
        "fixes_that_copied": sum(1 for r in runs if r.get("examples_copied")),
        "seeds": len(seeds),
        "seeds_without_fixer": sum(
            1
            for r in seeds
            if r.get("outcome") == "verified" and not r.get("fixer_rounds")
        ),
        "shown": dict(shown),
        "copied": dict(copied),
        "prune": sorted(
            example
            for example, count in shown.items()
            if count >= PRUNE_SHOWN and not copied.get(example)
        ),
        "times": times,
    }


def render(summary: Dict[str, Any]) -> str:
    """Return the Markdown report of ``summarize`` output."""
    lines = [
        "## Effect of data/",
        "",
        "| Figure | Value |",
        "|---|---|",
        f"| LLM calls | {summary['llm_calls']} |",
        f"| Calls with a few-shot example | {summary['calls_with_example']} |",
        f"| Distinct examples shown | {summary['distinct_examples']} |",
        f"| Runs | {summary['runs']} |",
        f"| Runs whose fix copied an example | "
        f"{summary['fixes_that_copied']} |",
        f"| Recipe seeds used | {summary['seeds']} |",
        f"| Seeds verified without a fixer | "
        f"{summary['seeds_without_fixer']} |",
        "",
    ]
    if summary["shown"]:
        lines += ["| Example | Shown | Copied |", "|---|---|---|"]
        for example, count in sorted(
            summary["shown"].items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(
                f"| {example} | {count} | "
                f"{summary['copied'].get(example, 0)} |"
            )
        lines.append("")
    if summary["prune"]:
        lines.append(
            f"Prune candidates (shown {PRUNE_SHOWN} times or more, never "
            f"copied): {', '.join(summary['prune'])}"
        )
        lines.append("")
    if summary["times"]:
        lines += [
            "| Package | Outcome | Plan | Seconds | Earlier run |",
            "|---|---|---|---|---|",
        ]
        for row in summary["times"]:
            earlier = row["earlier_seconds"]
            lines.append(
                f"| {row['repo']} | {row['outcome']} | "
                f"{row['plan_source'] or '?'} | {row['seconds']} | "
                f"{earlier if earlier is not None else '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Print the report of one logs folder; exit 1 without call logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        default="workspace-native/logs",
        help="folder with the agent-call_<package>.jsonl files",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the figures as JSON"
    )
    args = parser.parse_args()
    records = read_records(args.logs_dir)
    if not records:
        print(f"No call logs in {args.logs_dir}", file=sys.stderr)
        return 1
    summary = summarize(records)
    text = json.dumps(summary, indent=2) if args.json else render(summary)
    print(text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary and not args.json:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
