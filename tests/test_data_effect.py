"""Tests for .github/scripts/data_effect.py (the data/ effect report)."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / ".github"
    / "scripts"
    / "data_effect.py"
)


def _load():
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location("data_effect", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDataEffectReport(unittest.TestCase):
    """The report counts examples, copies, seeds and run times."""

    def setUp(self) -> None:
        self.script = _load()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name: str, records) -> None:
        path = os.path.join(self.tmp.name, f"agent-call_{name}.jsonl")
        with open(path, "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

    def test_figures_of_a_batch(self) -> None:
        noisy = {"agent_role": "fixer", "metadata": {"examples": ["auto-10"]}}
        self._write(
            "jr",
            [noisy] * 10
            + [
                {"agent_role": "scout", "metadata": {"examples": []}},
                {
                    "record": "data_effect",
                    "timestamp": "2026-10-20T10:00:00",
                    "repo": "jr",
                    "outcome": "failed",
                    "duration_seconds": 900.0,
                },
                {
                    "record": "data_effect",
                    "timestamp": "2026-10-23T10:00:00",
                    "repo": "jr",
                    "outcome": "verified",
                    "plan_source": "recipe",
                    "fixer_rounds": 0,
                    "duration_seconds": 300.0,
                },
            ],
        )
        self._write(
            "pv",
            [
                {"agent_role": "fixer", "metadata": {"examples": ["auto-24"]}},
                {
                    "record": "data_effect",
                    "timestamp": "2026-10-23T11:00:00",
                    "repo": "pv",
                    "outcome": "verified",
                    "plan_source": "heuristic",
                    "fixer_rounds": 1,
                    "examples_copied": ["auto-24"],
                    "duration_seconds": 120.0,
                },
            ],
        )
        summary = self.script.summarize(
            self.script.read_records(self.tmp.name)
        )
        self.assertEqual(summary["llm_calls"], 12)
        self.assertEqual(summary["calls_with_example"], 11)
        self.assertEqual(summary["distinct_examples"], 2)
        self.assertEqual(summary["fixes_that_copied"], 1)
        self.assertEqual(summary["seeds"], 1)
        self.assertEqual(summary["seeds_without_fixer"], 1)
        self.assertEqual(summary["prune"], ["auto-10"])
        jr = next(row for row in summary["times"] if row["repo"] == "jr")
        self.assertEqual(
            (jr["seconds"], jr["earlier_seconds"]), (300.0, 900.0)
        )
        report = self.script.render(summary)
        self.assertIn("| Recipe seeds used | 1 |", report)
        self.assertIn("Prune candidates", report)

    def test_a_bad_line_is_skipped(self) -> None:
        path = os.path.join(self.tmp.name, "agent-call_x.jsonl")
        with open(path, "w") as fh:
            fh.write("not json\n" + json.dumps({"agent_role": "fixer"}) + "\n")
        self.assertEqual(len(self.script.read_records(self.tmp.name)), 1)


if __name__ == "__main__":
    unittest.main()
