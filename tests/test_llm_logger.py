"""Tests for src/llm_logger.py — audit log writer, per-repo file switching."""

import os
import shutil
import unittest
import uuid
from unittest import mock

from src.llm_logger import LLMCallLogger, log_llm_call, set_llm_log_repo


class TestLogCall(unittest.TestCase):
    """Tests for log_llm_call, set_llm_log_repo, and log-file setup."""

    def setUp(self) -> None:
        self.tmpdir = self._tmp()
        # Reset singleton state pointing at the temp dir
        inst = LLMCallLogger()
        inst.log_file = os.path.join(self.tmpdir, "agent-call.log")
        inst._logs_dir = self.tmpdir
        with open(inst.log_file, "w") as f:
            f.write("")

    def _tmp(self):
        """Create a project-local scratch directory for log files."""
        root = os.path.join(os.getcwd(), "workspace", "test-llm-logger")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"atesor-logtest-{uuid.uuid4().hex}")
        os.makedirs(path, exist_ok=False)
        return path

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_log(self, path=None):
        """Return the text of ``path``, or of the logger's current file."""
        path = path or LLMCallLogger().log_file
        with open(path) as f:
            return f.read()

    def test_log_call_writes_record(self) -> None:
        call_id = log_llm_call(
            agent_role="SCOUT",
            prompt="please analyze",
            response='{"foo": "bar"}',
            model="gemini-flash-lite",
            cost_usd=0.001,
            metadata={"repo": "zlib", "phase": "scout"},
        )
        self.assertTrue(call_id.startswith("call_"))

        content = self._read_log()
        self.assertIn("SCOUT", content)
        self.assertIn("gemini-flash-lite", content)
        self.assertIn("please analyze", content)
        self.assertIn('{"foo": "bar"}', content)
        self.assertIn("zlib", content)
        self.assertIn("$0.001000", content)
        self.assertIn(call_id, content)

    def test_long_prompt_is_truncated(self) -> None:
        big = "x" * 15000
        log_llm_call("FIXER", big, "ok", "m", 0.0)
        content = self._read_log()
        self.assertIn("truncated 5000 characters", content)

    def test_long_response_is_truncated(self) -> None:
        big = "y" * 12000
        log_llm_call("FIXER", "p", big, "m", 0.0)
        content = self._read_log()
        self.assertIn("truncated 2000 characters", content)

    def test_set_repo_name_switches_file(self) -> None:
        set_llm_log_repo("mypkg")
        inst = LLMCallLogger()
        self.assertTrue(inst.log_file.endswith("agent-call_mypkg.log"))
        log_llm_call("BUILDER", "p", "r", "m", 0.0)
        self.assertTrue(os.path.exists(inst.log_file))
        # The old default file should NOT have the new record
        default = os.path.join(self.tmpdir, "agent-call.log")
        if os.path.exists(default):
            with open(default) as f:
                self.assertNotIn("BUILDER", f.read())

    def test_empty_repo_name_does_not_switch(self) -> None:
        inst = LLMCallLogger()
        original = inst.log_file
        set_llm_log_repo("")
        self.assertEqual(inst.log_file, original)

    def test_metadata_serialized_as_json(self) -> None:
        log_llm_call(
            "PLANNER",
            "p",
            "r",
            "m",
            0.0,
            metadata={"a": 1, "nested": {"b": 2}},
        )
        content = self._read_log()
        self.assertIn('"nested"', content)
        self.assertIn('"b": 2', content)

    def test_call_ids_are_unique(self) -> None:
        ids = {log_llm_call("R", "p", "r", "m", 0.0) for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_ensure_log_file_writes_header_for_new_file(self) -> None:
        fresh_dir = os.path.join(self.tmpdir, "fresh")
        inst = LLMCallLogger()
        with mock.patch("src.config.LOGS_DIR", fresh_dir):
            inst._ensure_log_file()

        content = self._read_log(os.path.join(fresh_dir, "agent-call.log"))
        self.assertIn("ATESOR AI - LLM CALL LOG", content)
        self.assertIn("Created:", content)

    def test_ensure_log_file_failure_disables_file(self) -> None:
        inst = LLMCallLogger()
        with (
            mock.patch("src.llm_logger.os.makedirs", side_effect=OSError),
            self.assertLogs("src.llm_logger", level="ERROR"),
        ):
            inst._ensure_log_file()

        self.assertIsNone(inst.log_file)
        self.assertIsNone(inst._logs_dir)

    def test_log_call_write_failure_still_returns_id(self) -> None:
        """File-write failures are logged but do not break callers."""
        inst = LLMCallLogger()
        inst.log_file = os.path.join(self.tmpdir, "unwritable.log")
        before = len(inst.calls)
        with (
            mock.patch("builtins.open", side_effect=OSError("blocked")),
            self.assertLogs("src.llm_logger", level="ERROR") as logs,
        ):
            call_id = inst.log_call("SCOUT", "p", "r", "m")

        self.assertTrue(call_id.startswith("call_"))
        self.assertEqual(len(inst.calls), before + 1)
        self.assertIn("Failed to write LLM call log", "\n".join(logs.output))


class TestLoggerSingleton(unittest.TestCase):
    """Tests for the LLMCallLogger singleton and its call-record cap."""

    def test_returns_same_instance(self) -> None:
        self.assertIs(LLMCallLogger(), LLMCallLogger())

    def test_call_records_deque_capped(self) -> None:
        inst = LLMCallLogger()
        self.assertEqual(inst.calls.maxlen, 1000)


if __name__ == "__main__":
    unittest.main()
