"""Tests for the native target in .github/scripts/batch_test.py.

A native batch runs on the user's own riscv64 machine, for local runs.
These tests cover the worker count, the machine probe and the --target
flag. The QEMU calls to main.py must keep today's argv. No test opens
an ssh connection or starts a real process.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.sandbox import SandboxUnavailableError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BATCH_TEST = _REPO_ROOT / ".github" / "scripts" / "batch_test.py"
_URL = "https://github.com/madler/zlib"
_ALIAS = "builder@board-1"
_WORKER = "atesor-ai-sandbox-debian-w1"


def _load_batch_module():
    """Import .github/scripts/batch_test.py as a new module."""
    spec = importlib.util.spec_from_file_location(
        "batch_test_native", str(_BATCH_TEST)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed(returncode: int, stdout: str, stderr: str = ""):
    """Return a finished process with this result."""
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class _BatchTestCase(unittest.TestCase):
    """Load batch_test with no ATESOR_* key in the environment."""

    def setUp(self) -> None:
        """Clear the ATESOR_* keys, then load a new batch_test module."""
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in [key for key in os.environ if key.startswith("ATESOR_")]:
            del os.environ[key]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.bt = _load_batch_module()
        self.bt._shutdown_event.clear()
        self.bt.BATCH_LOGS_DIR = str(self.tmp / "logs")
        self.bt._USE_COLOR = False


class TestNativeDefaultWorkers(_BatchTestCase):
    """The native default gives each worker one core and 4 GiB."""

    def test_formula(self) -> None:
        """The default is max(1, min(cores, GiB // 4))."""
        cases = [
            ((4, 25), 4),  # The cores set the limit.
            ((8, 12), 3),  # The memory sets the limit.
            ((64, 3), 1),  # Less than 4 GiB still gives one worker.
            ((2, 100), 2),
        ]
        for (cores, mem_gib), expected in cases:
            with self.subTest(cores=cores, mem_gib=mem_gib):
                self.assertEqual(
                    expected, self.bt._native_default_workers(cores, mem_gib)
                )


class TestReadMachineResources(_BatchTestCase):
    """One ssh call reads the host name, the cores and the memory."""

    def test_one_call_reads_all_three_values(self) -> None:
        """The memory comes back in whole GiB."""
        output = "rv-box\n4\nMemTotal:       26358872 kB\n"
        with mock.patch.object(
            self.bt.sandbox, "run_raw", return_value=_completed(0, output)
        ) as run_raw:
            result = self.bt._read_machine_resources()

        self.assertEqual(("rv-box", 4, 25), result)
        run_raw.assert_called_once()
        argv = run_raw.call_args.args[0]
        self.assertEqual(["sh", "-c"], argv[:2])
        self.assertIn("nproc", argv[2])
        self.assertIn("MemTotal", argv[2])

    def test_a_failed_call_or_other_output_raises(self) -> None:
        """The batch must not guess the resources of the machine."""
        cases = [(0, "rv-box\nfour\n", ""), (1, "", "nproc: not found")]
        for returncode, stdout, stderr in cases:
            with self.subTest(returncode=returncode):
                with mock.patch.object(
                    self.bt.sandbox,
                    "run_raw",
                    return_value=_completed(returncode, stdout, stderr),
                ):
                    with self.assertRaises(RuntimeError):
                        self.bt._read_machine_resources()


class TestMainPyArgv(_BatchTestCase):
    """Each main.py call gets --target native on native only."""

    def _setup_argv(self, rebuild: bool = False) -> list:
        """Return the argv of one _run_setup() call."""
        with mock.patch.object(
            self.bt.subprocess, "run", return_value=_completed(0, "")
        ) as run:
            ok, _ = self.bt._run_setup(_WORKER, rebuild=rebuild)
        self.assertTrue(ok)
        return run.call_args.args[0]

    def _agent_argv(self) -> list:
        """Return the argv of one run_agent() call."""
        proc = mock.MagicMock(pid=4242, returncode=0)
        self.bt._populate_container_pool(1)
        with (
            mock.patch.object(
                self.bt.subprocess, "Popen", return_value=proc
            ) as popen,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            success, _, _ = self.bt.run_agent(_URL, "zlib")
        self.assertTrue(success)
        return popen.call_args.args[0]

    def test_child_gets_the_kill_deadline(self) -> None:
        """main.py learns when the batch kills it, to size its tests."""
        proc = mock.MagicMock(pid=4242, returncode=0)
        self.bt._populate_container_pool(1)
        with (
            mock.patch.object(
                self.bt.subprocess, "Popen", return_value=proc
            ) as popen,
            mock.patch.object(self.bt.time, "time", return_value=5000.0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.bt.run_agent(_URL, "zlib")

        env = popen.call_args.kwargs["env"]
        self.assertEqual(
            env["ATESOR_RUN_DEADLINE"],
            str(5000 + self.bt._AGENT_TIMEOUT_SECONDS),
        )
        # The rest of the environment is inherited unchanged.
        self.assertEqual(env.get("PATH"), os.environ.get("PATH"))

    def test_qemu_argv_is_unchanged(self) -> None:
        """QEMU calls keep today's argv, with no --target."""
        self.assertEqual(
            [
                "python3",
                "main.py",
                "--setup-only",
                "--platform",
                "debian",
                "--container",
                _WORKER,
            ],
            self._setup_argv(),
        )
        self.assertEqual(
            [
                "python3",
                "main.py",
                "--repo",
                _URL,
                "--max-attempts",
                "5",
                "--force",
                "--container",
                _WORKER,
                "--platform",
                "debian",
            ],
            self._agent_argv(),
        )

    def test_native_calls_get_target_native(self) -> None:
        """Both call sites pass --target native to main.py."""
        os.environ["ATESOR_TARGET"] = "native"
        self.assertEqual(
            [
                "python3",
                "main.py",
                "--setup-only",
                "--platform",
                "debian",
                "--container",
                _WORKER,
                "--target",
                "native",
                "--rebuild",
            ],
            self._setup_argv(rebuild=True),
        )
        self.assertEqual(
            [
                "python3",
                "main.py",
                "--repo",
                _URL,
                "--max-attempts",
                "5",
                "--force",
                "--container",
                _WORKER,
                "--platform",
                "debian",
                "--target",
                "native",
            ],
            self._agent_argv(),
        )


class TestBatchMain(_BatchTestCase):
    """main() picks the target, the worker count and the header rows."""

    def setUp(self) -> None:
        """Write a one-package list and the fake .env values."""
        super().setUp()
        self.list_path = self.tmp / "one.json"
        self.list_path.write_text(
            json.dumps(
                {
                    "$schema_version": 1,
                    "packages": [{"name": "zlib", "url": _URL}],
                }
            )
        )
        self.env_file = {
            "ATESOR_PLATFORM": "debian",
            "ATESOR_SSH_HOST": _ALIAS,
        }

    def _run(
        self,
        *args: str,
        resources=("rv-box", 4, 25),
        error=None,
        run_result=None,
        signal_handler=None,
    ):
        """Run main() with the machine, the workers and .env faked.

        Returns:
            The exit code, stdout, stderr and the mock of the probe.
        """
        bt = self.bt
        env_file = self.env_file

        def fake_load_dotenv(_path):
            # load_dotenv() never replaces a key that is already set.
            for key, value in env_file.items():
                os.environ.setdefault(key, value)
            return True

        if error is None:
            probe = mock.MagicMock(return_value=resources)
        else:
            probe = mock.MagicMock(side_effect=error)
        if run_result is None:
            run_result = self.bt.PackageRunResult("zlib", "PASS", "ok", 1.0)
        if signal_handler is None:
            signal_handler = mock.MagicMock()
        out, err = io.StringIO(), io.StringIO()
        argv = ["batch_test.py", "--list", str(self.list_path), *args]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(bt, "load_dotenv", fake_load_dotenv),
            mock.patch.object(bt, "_read_machine_resources", probe),
            mock.patch.object(
                bt, "_refresh_worker_pool_if_needed", return_value=True
            ),
            mock.patch.object(bt, "run_agent", return_value=run_result),
            mock.patch.object(
                bt, "_install_signal_handlers", signal_handler
            ),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = bt.main()
        return code, out.getvalue(), err.getvalue(), probe

    def test_native_default_comes_from_the_machine(self) -> None:
        """The machine sets the default, and the header names it."""
        code, out, _, probe = self._run(
            "--target", "native", resources=("rv-box", 8, 12)
        )

        self.assertEqual(0, code)
        probe.assert_called_once()
        self.assertEqual(3, self.bt.MAX_WORKERS)
        self.assertIn("Target     native", out)
        self.assertIn(
            f"Machine    {_ALIAS} (host rv-box), 8 cores, 12 GiB", out
        )

    def test_workers_flag_overrides_the_native_default(self) -> None:
        """--workers still wins over the native default."""
        code, _, _, _ = self._run("--target", "native", "--workers", "2")

        self.assertEqual(0, code)
        self.assertEqual(2, self.bt.MAX_WORKERS)

    def test_native_workers_stay_within_the_machine_cores(self) -> None:
        """On native, the limit is the core count of the machine."""
        code, _, err, _ = self._run("--target", "native", "--workers", "5")

        self.assertEqual(2, code)
        self.assertIn(
            f"--workers=5 exceeds the 4 cores of {_ALIAS} (host rv-box).",
            err,
        )

    def test_native_probe_failure_stops_the_batch(self) -> None:
        """A machine that cannot answer stops the batch before any run."""
        code, out, err, _ = self._run(
            "--target",
            "native",
            error=SandboxUnavailableError("The ssh connection did not open."),
        )

        self.assertEqual(1, code)
        self.assertIn("Cannot read the cores and memory", err)
        self.assertIn("python3 main.py --target native --preflight", err)
        self.assertNotIn("PORTING AGENT", out)

    def test_qemu_keeps_the_local_cpu_default(self) -> None:
        """QEMU uses the local CPU count and never probes a machine."""
        code, out, _, probe = self._run()

        self.assertEqual(0, code)
        probe.assert_not_called()
        self.assertEqual(self.bt._MAX_AVAILABLE_WORKERS, self.bt.MAX_WORKERS)
        self.assertNotIn("Machine", out)

    def test_qemu_workers_limit_message_is_unchanged(self) -> None:
        """QEMU keeps today's error for too many workers."""
        too_many = self.bt._MAX_AVAILABLE_WORKERS + 1
        code, _, err, _ = self._run("--workers", str(too_many))

        self.assertEqual(2, code)
        self.assertEqual(
            f"[ERROR] --workers={too_many} exceeds available CPU count "
            f"({self.bt._MAX_AVAILABLE_WORKERS}).\n",
            err,
        )

    def test_target_flag_wins_over_env_file(self) -> None:
        """--target native wins over ATESOR_TARGET=qemu in .env."""
        self.env_file["ATESOR_TARGET"] = "qemu"
        code, _, _, probe = self._run("--target", "native")

        self.assertEqual(0, code)
        probe.assert_called_once()
        self.assertEqual("native", os.environ["ATESOR_TARGET"])

    def test_qemu_flag_wins_over_native_env_file(self) -> None:
        """--target qemu wins over ATESOR_TARGET=native in .env."""
        self.env_file["ATESOR_TARGET"] = "native"
        code, _, _, probe = self._run("--target", "qemu")

        self.assertEqual(0, code)
        probe.assert_not_called()
        self.assertEqual("qemu", os.environ["ATESOR_TARGET"])

    def test_env_file_selects_native_without_the_flag(self) -> None:
        """With no flag, ATESOR_TARGET in .env selects the target."""
        self.env_file["ATESOR_TARGET"] = "native"
        code, _, _, probe = self._run()

        self.assertEqual(0, code)
        probe.assert_called_once()

    def test_same_basename_repos_share_a_batch_lock(self) -> None:
        """Repos cloned to the same directory cannot run concurrently."""
        # derive_repo_name gives both "zlib": one clone directory.
        lock_a = self.bt._repo_lock("https://github.com/madler/zlib")
        lock_b = self.bt._repo_lock("https://github.com/someone/zlib")
        # Generic basenames get an owner prefix, so these two clone to
        # hetznercloud-cli and urfave-cli and may run side by side.
        lock_c = self.bt._repo_lock("https://github.com/hetznercloud/cli")
        lock_d = self.bt._repo_lock("https://github.com/urfave/cli")

        self.assertIs(lock_a, lock_b)
        self.assertIsNot(lock_c, lock_d)
        self.assertIsNot(lock_a, lock_c)

    def test_platform_flag_wins_over_invalid_env(self) -> None:
        """An explicit --platform ignores a bad inherited env value."""
        os.environ["ATESOR_PLATFORM"] = "bogus"
        code, _, err, _ = self._run("--platform", "debian")

        self.assertEqual(0, code)
        self.assertEqual("", err)
        self.assertEqual("debian", os.environ["ATESOR_PLATFORM"])

    def test_invalid_env_platform_fails_without_flag(self) -> None:
        """A bad env platform is reported only when no flag overrides it."""
        self.env_file["ATESOR_PLATFORM"] = "bogus"
        code, _, err, _ = self._run()

        self.assertEqual(2, code)
        self.assertIn("Unknown ATESOR_PLATFORM='bogus'", err)

    def test_unverified_result_is_not_counted_as_pass(self) -> None:
        """main.py exit 3 maps to UNVERIFIED and fails the batch gate."""
        result = self.bt.PackageRunResult(
            "zlib",
            self.bt.STATUS_UNVERIFIED,
            "no ELF verified",
            1.0,
            3,
        )
        code, out, _, _ = self._run(run_result=result)

        self.assertEqual(1, code)
        self.assertIn("UNVERIFIED 1", out)
        self.assertIn("PASS 0", out)
        self.assertIn("PASS RATE        0.0%", out)

    def test_soft_stop_returns_130_and_skips_gate(self) -> None:
        """A requested shutdown returns 130 even with no failed packages."""
        result = self.bt.PackageRunResult(
            "zlib",
            self.bt.STATUS_SKIPPED,
            "Skipped: shutdown requested",
            0.0,
        )

        def mark_shutdown(_future_map):
            self.bt._shutdown_event.set()

        code, out, _, _ = self._run(
            run_result=result,
            signal_handler=mark_shutdown,
        )

        self.assertEqual(130, code)
        self.assertIn("SKIPPED 1", out)
        self.assertIn("TOTAL 0", out)


if __name__ == "__main__":
    unittest.main()
