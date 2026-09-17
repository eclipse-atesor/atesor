#!/usr/bin/env python3
r"""Aggregate per-category regression JUnit results into a report.

Reads ``results-<category>.xml`` JUnit files and the matching
``paths-<category>.txt`` file lists produced by the regression-tests
workflow matrix, then:

    1. Writes a Markdown results table to ``$GITHUB_STEP_SUMMARY``
       (falls back to stdout when unset).
    2. Verifies every ``tests/test_*.py`` file in the repository is
       claimed by exactly one category, so new test files cannot
       silently escape CI.

Exits non-zero when any category reports failures or errors, or when
a test file is uncategorized or double-categorized.

Example::

    python .github/scripts/report_regression.py \\
        --results-dir test-results --tests-dir tests
"""

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ElementTree
from typing import Dict, List, Tuple


def parse_junit(path: str) -> Dict[str, float]:
    """Parse a pytest JUnit XML file into aggregate counters.

    Args:
        path: Path to a ``results-<category>.xml`` file.

    Returns:
        Mapping with ``tests``, ``failures``, ``errors``, ``skipped``
        and ``time`` totals summed across all test suites in the file.
    """
    totals = {
        "tests": 0.0,
        "failures": 0.0,
        "errors": 0.0,
        "skipped": 0.0,
        "time": 0.0,
    }
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    for suite in suites:
        for key in totals:
            totals[key] += float(suite.get(key, 0) or 0)
    return totals


def collect_categories(
    results_dir: str,
) -> List[Tuple[str, Dict[str, float], List[str]]]:
    """Collect (category, totals, claimed test paths) triples.

    Args:
        results_dir: Directory holding the downloaded CI artifacts.

    Returns:
        One triple per ``results-*.xml`` file, sorted by category.
    """
    rows = []
    pattern = os.path.join(results_dir, "results-*.xml")
    for xml_path in sorted(glob.glob(pattern)):
        base = os.path.basename(xml_path)
        category = base[len("results-"):-len(".xml")]
        paths_file = os.path.join(results_dir, f"paths-{category}.txt")
        claimed: List[str] = []
        if os.path.exists(paths_file):
            with open(paths_file, encoding="utf-8") as handle:
                claimed = handle.read().split()
        rows.append((category, parse_junit(xml_path), claimed))
    return rows


def check_coverage(
    rows: List[Tuple[str, Dict[str, float], List[str]]],
    tests_dir: str,
) -> List[str]:
    """Verify every test file is claimed by exactly one category.

    Args:
        rows: Output of :func:`collect_categories`.
        tests_dir: Repository directory containing ``test_*.py`` files.

    Returns:
        Human-readable problem descriptions; empty when consistent.
    """
    problems: List[str] = []
    claims: Dict[str, List[str]] = {}
    for category, _, claimed in rows:
        for path in claimed:
            claims.setdefault(os.path.normpath(path), []).append(
                category
            )
    repo_files = sorted(
        os.path.normpath(p)
        for p in glob.glob(os.path.join(tests_dir, "test_*.py"))
    )
    for path in repo_files:
        owners = claims.get(path, [])
        if not owners:
            problems.append(
                f"`{path}` is not part of any workflow category"
            )
        elif len(owners) > 1:
            problems.append(
                f"`{path}` is claimed by multiple categories: "
                f"{', '.join(sorted(owners))}"
            )
    for path, owners in sorted(claims.items()):
        if path not in repo_files:
            problems.append(
                f"category {owners[0]} references missing file "
                f"`{path}`"
            )
    return problems


def build_summary(
    rows: List[Tuple[str, Dict[str, float], List[str]]],
    problems: List[str],
) -> str:
    """Render the Markdown report.

    Args:
        rows: Output of :func:`collect_categories`.
        problems: Output of :func:`check_coverage`.

    Returns:
        Markdown text for the workflow step summary.
    """
    lines = [
        "## Regression test results",
        "",
        "| Category | Tests | Passed | Failed | Errors | Skipped "
        "| Time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    grand = {
        "tests": 0.0,
        "failures": 0.0,
        "errors": 0.0,
        "skipped": 0.0,
        "time": 0.0,
    }
    for category, totals, _ in rows:
        for key in grand:
            grand[key] += totals[key]
        broken = totals["failures"] + totals["errors"]
        icon = "✅" if broken == 0 else "❌"
        passed = (
            totals["tests"]
            - totals["failures"]
            - totals["errors"]
            - totals["skipped"]
        )
        lines.append(
            f"| {icon} {category} | {int(totals['tests'])} "
            f"| {int(passed)} | {int(totals['failures'])} "
            f"| {int(totals['errors'])} | {int(totals['skipped'])} "
            f"| {totals['time']:.1f} |"
        )
    grand_passed = (
        grand["tests"]
        - grand["failures"]
        - grand["errors"]
        - grand["skipped"]
    )
    lines.append(
        f"| **Total** | **{int(grand['tests'])}** "
        f"| **{int(grand_passed)}** | **{int(grand['failures'])}** "
        f"| **{int(grand['errors'])}** | **{int(grand['skipped'])}** "
        f"| **{grand['time']:.1f}** |"
    )
    if problems:
        lines += ["", "### Category coverage problems", ""]
        lines += [f"- ❌ {problem}" for problem in problems]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Aggregate results, emit the summary, and gate the workflow.

    Returns:
        Process exit code: 0 on full success, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="test-results",
        help="Directory containing results-*.xml and paths-*.txt",
    )
    parser.add_argument(
        "--tests-dir",
        default="tests",
        help="Repository test directory to validate coverage against",
    )
    args = parser.parse_args()

    rows = collect_categories(args.results_dir)
    if not rows:
        print(f"No results-*.xml found in {args.results_dir}")
        return 1
    problems = check_coverage(rows, args.tests_dir)
    summary = build_summary(rows, problems)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary)
    print(summary)

    broken = sum(r[1]["failures"] + r[1]["errors"] for r in rows)
    if broken or problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
