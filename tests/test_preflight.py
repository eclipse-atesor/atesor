"""Tests for src/preflight.py and its wiring into main.py."""

import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import main
from src import preflight, target

_URL = "https://github.com/example/project"


def _tag_colors(text: str, color: str) -> str:
    """Mark the text with its color name, so a test can read the color."""
    return f"<{color}>{text}</{color}>"


class FakeMachine:
    """Answer the preflight's remote scripts like a riscv64 board.

    Every attribute describes one fact of the board. Set an attribute
    to an empty value to make that fact missing.
    """

    def __init__(self) -> None:
        self.login = None
        self.arch = "riscv64"
        self.hostname = "board-os"
        self.compatible = "milkv,megrez eswin,eic7700"
        self.mvendorid = "0x489"
        self.podman = "podman version 5.4.2"
        self.newuidmap = "/usr/bin/newuidmap"
        self.subuid = "1"
        self.subgid = "1"
        self.podman_info = (
            0,
            "true /home/tester/.local/share/containers/storage",
        )
        self.rsync = "/usr/bin/rsync"
        self.home = "/home/tester"
        self.workdir = "/home/tester/atesor-ai"
        self.free = ("graph=/dev/root 29360128", "work=/dev/root 29360128")
        self.calls = []

    def run(self, argv, **kwargs):
        """Return what the board prints for the script in argv[-1]."""
        script = argv[-1]
        self.calls.append(list(argv))

        def done(code=0, out="", err=""):
            """Build a completed process for this call."""
            return subprocess.CompletedProcess(argv, code, out, err)

        if script.startswith("printf %s "):
            if self.login is not None:
                return done(*self.login)
            return done(0, preflight.LOGIN_PROBE)
        if script == "uname -n -m":
            return done(0, f"{self.hostname} {self.arch}\n")
        if "/proc/device-tree/compatible" in script:
            out = self.compatible + "\n"
            if self.mvendorid:
                out += f"mvendorid\t: {self.mvendorid}\n"
            return done(0, out)
        if script == "podman --version":
            if not self.podman:
                return done(127, "", "sh: podman: not found")
            return done(0, self.podman + "\n")
        if "newuidmap" in script:
            return done(
                0,
                f"newuidmap={self.newuidmap}\n"
                f"subuid={self.subuid}\nsubgid={self.subgid}\n",
            )
        if script.startswith("podman info"):
            code, out = self.podman_info
            return done(code, out + "\n")
        if script == "command -v rsync":
            if not self.rsync:
                return done(1)
            return done(0, self.rsync + "\n")
        if script == 'printf "%s" "$HOME"':
            return done(0, self.home)
        if script.startswith("mkdir -p "):
            if not self.workdir:
                return done(1, "", "mkdir: Permission denied")
            return done(0, self.workdir + "\n")
        if "df -Pk" in script:
            return done(0, "\n".join(self.free) + "\n")
        raise AssertionError(f"unexpected remote script: {script!r}")


class _PreflightCase(unittest.TestCase):
    """Common setup: a valid alias-mode native config and a fake board."""

    def setUp(self) -> None:
        """Create the scratch cache folder, the board and the config."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # A short runtime dir keeps the ssh control-socket path under
        # the sun_path limit even when TMPDIR is long; only the
        # socket-length tests exercise the long-path rung.
        self.runtime_dir = tempfile.mkdtemp(prefix="atesor-rt-", dir="/tmp")
        self.addCleanup(shutil.rmtree, self.runtime_dir, True)
        self.machine = FakeMachine()
        self.env = {
            "ATESOR_TARGET": "native",
            "ATESOR_PLATFORM": "debian",
            "ATESOR_SSH_HOST": "tester@board-1",
            "XDG_CACHE_HOME": self._tmp.name,
            "XDG_RUNTIME_DIR": self.runtime_dir,
            "HOME": self._tmp.name,
            "PATH": os.environ.get("PATH", ""),
        }
        self.which = {"ssh": "/usr/bin/ssh", "rsync": "/usr/bin/rsync"}

    def run_ladder(
        self, first: int = 1, last: int = preflight.LAST_RUNG
    ) -> preflight.PreflightReport:
        """Run the ladder against the fake board."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(os.environ, self.env, clear=True)
            )
            stack.enter_context(
                mock.patch(
                    "src.preflight.subprocess.run",
                    side_effect=self.machine.run,
                )
            )
            stack.enter_context(
                mock.patch(
                    "src.preflight.shutil.which", side_effect=self.which.get
                )
            )
            target.reset_target_cache()
            return preflight.run_preflight(first, last)

    def scripts(self) -> list:
        """Return the remote scripts that the ladder sent."""
        return [argv[-1] for argv in self.machine.calls]


class TestPreflightLadder(_PreflightCase):
    """Tests for the stop rules and the pass path."""

    def test_report_names_the_machine(self) -> None:
        """Rung 1 names the ssh destination, and rung 5 the host name."""
        report = self.run_ladder()
        details = {result.number: result.detail for result in report.results}
        self.assertEqual(
            "native, alias mode (tester@board-1), platform debian", details[1]
        )
        self.assertEqual("riscv64, host board-os", details[5])
        self.assertEqual(
            "tester@board-1 (host board-os)", target.machine_label()
        )

    def test_ready_machine_passes_all_rungs(self) -> None:
        """A ready board passes rungs 1 to 11."""
        report = self.run_ladder()
        self.assertTrue(report.passed, report.render(verbose=True))
        self.assertEqual(
            list(range(1, 12)), [result.number for result in report.results]
        )
        self.assertEqual("/home/tester/atesor-ai", target.remote_workdir())
        self.assertIn("passed: rungs 1 to 11", report.render())

    def test_unsafe_physical_workdir_fails_rung_10(self) -> None:
        """A symlink-resolved workdir with spaces is rejected."""
        self.machine.workdir = "/home/tester/atesor ai"
        report = self.run_ladder()
        self.assertFalse(report.passed)
        self.assertIn(
            "physical work directory path without spaces",
            report.render(),
        )

    def test_every_call_uses_the_ssh_prefix_and_no_stdin(self) -> None:
        """Remote calls go through ssh_argv with stdin closed."""
        with mock.patch.dict(os.environ, self.env, clear=True):
            target.reset_target_cache()
            prefix = target.ssh_argv() + [target.ssh_alias()]
        runner = mock.Mock(side_effect=self.machine.run)
        with mock.patch.dict(os.environ, self.env, clear=True):
            with mock.patch("src.preflight.subprocess.run", runner):
                with mock.patch(
                    "src.preflight.shutil.which", side_effect=self.which.get
                ):
                    target.reset_target_cache()
                    preflight.run_preflight(4, 5)
        for call in runner.call_args_list:
            argv = call.args[0]
            self.assertEqual(prefix, argv[: len(prefix)])
            self.assertEqual(len(prefix) + 1, len(argv))
            self.assertIs(subprocess.DEVNULL, call.kwargs["stdin"])

    def test_config_error_stops_before_any_ssh_call(self) -> None:
        """Rung 1 stops the ladder, and nothing reaches the machine."""
        del self.env["ATESOR_PLATFORM"]
        report = self.run_ladder()
        self.assertEqual([1], [result.number for result in report.results])
        self.assertEqual(1, report.stopped_at)
        self.assertEqual([], self.machine.calls)
        self.assertIn("ATESOR_PLATFORM", report.render())

    def test_missing_local_rsync_stops_at_rung_2(self) -> None:
        """Rung 2 checks the local computer and stops the ladder."""
        self.which["rsync"] = None
        report = self.run_ladder()
        self.assertEqual(2, report.stopped_at)
        self.assertEqual([], self.machine.calls)
        self.assertIn("rsync on the local computer", report.render())

    def test_open_field_mode_key_fails_rung_3(self) -> None:
        """A key file that others can read stops the ladder."""
        key = os.path.join(self._tmp.name, "id_board")
        with open(key, "w", encoding="utf-8") as handle:
            handle.write("key\n")
        os.chmod(key, 0o644)
        del self.env["ATESOR_SSH_HOST"]
        self.env["ATESOR_SSH_HOSTNAME"] = "board.example"
        self.env["ATESOR_SSH_IDENTITY_FILE"] = key
        report = self.run_ladder()
        self.assertEqual(3, report.stopped_at)
        self.assertIn("chmod 600", report.render())
        self.assertEqual([], self.machine.calls)

    def test_private_field_mode_key_passes_rung_3(self) -> None:
        """A 0600 key passes, and rung 3 writes the SSH config."""
        key = os.path.join(self._tmp.name, "id_board")
        with open(key, "w", encoding="utf-8") as handle:
            handle.write("key\n")
        os.chmod(key, 0o600)
        del self.env["ATESOR_SSH_HOST"]
        self.env["ATESOR_SSH_HOSTNAME"] = "board.example"
        self.env["ATESOR_SSH_IDENTITY_FILE"] = key
        report = self.run_ladder(1, 3)
        self.assertTrue(report.passed, report.render(verbose=True))
        config_file = os.path.join(self._tmp.name, "atesor-ai", "ssh_config")
        self.assertTrue(os.path.isfile(config_file))

    def test_long_control_socket_path_fails_rung_3(self) -> None:
        """Rung 3 catches a socket path that ssh would refuse."""
        self.env["XDG_CACHE_HOME"] = os.path.join(self._tmp.name, "x" * 80)
        del self.env["XDG_RUNTIME_DIR"]
        report = self.run_ladder()
        self.assertEqual(3, report.stopped_at)
        self.assertIn("a short directory for the ssh", report.render())
        self.assertEqual([], self.machine.calls)

    def test_login_failures_name_the_fix(self) -> None:
        """Rung 4 turns each ssh error into the command that fixes it."""
        cases = [
            (
                (255, "", "tester@board-1: Permission denied (publickey)."),
                "ssh-copy-id tester@board-1",
            ),
            (
                (
                    255,
                    "",
                    "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!\n"
                    "Host key verification failed.",
                ),
                "ssh-keygen -R",
            ),
            ((255, "", "Host key verification failed."), "accept the key"),
            (
                (255, "", "ssh: Could not resolve hostname board-1: x"),
                "fix ATESOR_SSH_HOST",
            ),
            (
                (255, "", "ssh: connect to host 10.0.0.5 port 22: refused"),
                "check the address",
            ),
            ((0, "atesor probe changed", ""), "chsh -s /bin/bash"),
        ]
        for login, fix in cases:
            with self.subTest(fix=fix):
                self.machine = FakeMachine()
                self.machine.login = login
                report = self.run_ladder()
                self.assertEqual(4, report.stopped_at)
                self.assertIn(fix, report.render())
                self.assertEqual(1, len(self.machine.calls))

    def test_login_error_detail_is_redacted(self) -> None:
        """An address in an ssh error never reaches the output."""
        self.machine.login = (
            255,
            "",
            "ssh: connect to host 10.0.0.5 port 22: Connection refused",
        )
        report = self.run_ladder()
        self.assertNotIn("10.0.0.5", report.render(verbose=True))

    def test_login_timeout_counts_as_unreachable(self) -> None:
        """A hanging ssh call fails rung 4 instead of hanging Atesor."""
        timeout = subprocess.TimeoutExpired(cmd="ssh", timeout=60)
        with mock.patch.object(self.machine, "run", side_effect=timeout):
            report = self.run_ladder()
        self.assertEqual(4, report.stopped_at)
        self.assertIn("a reachable SSH server", report.render())

    def test_connection_loss_after_login_is_listed_once(self) -> None:
        """Later ssh errors give one item and no address."""
        real_run = self.machine.run

        def flaky(argv, **kwargs):
            """Answer the login, then drop every later call."""
            if argv[-1].startswith("printf %s "):
                return real_run(argv, **kwargs)
            return subprocess.CompletedProcess(
                argv, 255, "", "Connection closed by 10.0.0.5 port 22"
            )

        with mock.patch.object(self.machine, "run", side_effect=flaky):
            report = self.run_ladder()
        items = [entry.item for entry in report.missing_items()]
        self.assertEqual(["a stable SSH connection to the machine"], items)
        self.assertNotIn("10.0.0.5", report.render(verbose=True))


class TestMachineRungs(_PreflightCase):
    """Tests for rungs 5 to 11, which all run and build one list."""

    def test_wrong_architecture_is_reported(self) -> None:
        """Rung 5 fails, and the later rungs still run."""
        self.machine.arch = "x86_64"
        report = self.run_ladder()
        self.assertFalse(report.passed)
        self.assertIsNone(report.stopped_at)
        self.assertEqual(11, len(report.results))
        self.assertIn("a riscv64 machine", report.render())

    def test_qemu_virt_vm_is_rejected(self) -> None:
        """A virt board without a vendor ID is emulated."""
        self.machine.compatible = "riscv-virtio"
        self.machine.mvendorid = "0x0"
        report = self.run_ladder()
        self.assertIn("not a QEMU virt VM", report.render())

    def test_virt_board_with_a_vendor_id_passes(self) -> None:
        """A virt board with a real vendor ID is not rejected."""
        self.machine.compatible = "riscv-virtio"
        self.machine.mvendorid = "0x489"
        report = self.run_ladder()
        self.assertTrue(report.passed, report.render(verbose=True))

    def test_megrez_like_machine_lists_three_items(self) -> None:
        """No podman, uidmap or rsync gives exactly those three items."""
        self.machine.podman = ""
        self.machine.newuidmap = ""
        self.machine.rsync = ""
        report = self.run_ladder()
        items = [entry.item for entry in report.missing_items()]
        self.assertEqual(
            [
                "podman 4.0 or newer",
                "uidmap (newuidmap)",
                "rsync on the machine",
            ],
            items,
        )
        text = report.render()
        self.assertIn("Still required on the machine", text)
        self.assertIn("fix: sudo apt install podman", text)
        self.assertIn("Atesor does not install or change these", text)

    def test_no_fix_command_ever_runs(self) -> None:
        """The ladder only reads; it never installs or uses sudo."""
        self.machine.podman = ""
        self.machine.newuidmap = ""
        self.machine.rsync = ""
        self.machine.subuid = "0"
        self.run_ladder()
        for script in self.scripts():
            for word in ("sudo", "apt", "usermod", "loginctl", "chmod"):
                self.assertNotIn(word, script)

    def test_old_podman_is_reported(self) -> None:
        """Podman 3 is too old."""
        self.machine.podman = "podman version 3.4.4"
        report = self.run_ladder()
        self.assertIn("podman 4.0 or newer", report.render())
        self.assertIn("found podman version 3.4.4", report.render(True))

    def test_rootful_podman_is_reported(self) -> None:
        """Podman that runs as root is not accepted."""
        self.machine.podman_info = (0, "false /var/lib/containers/storage")
        report = self.run_ladder()
        self.assertIn("rootless podman for the login user", report.render())

    def test_missing_subordinate_ids_are_reported(self) -> None:
        """No /etc/subuid entry gives the usermod fix."""
        self.machine.subuid = "0"
        report = self.run_ladder()
        self.assertIn("subordinate UID and GID ranges", report.render())
        self.assertIn("usermod --add-subuids", report.render())

    def test_podman_info_failure_is_reported(self) -> None:
        """A failed podman info with valid IDs gives its own item."""
        self.machine.podman_info = (125, "")
        report = self.run_ladder()
        self.assertIn("a working rootless podman", report.render())

    def test_podman_graph_root_feeds_the_disk_check(self) -> None:
        """Rung 11 measures the storage path that podman reports."""
        self.run_ladder()
        disk = [script for script in self.scripts() if "df -Pk" in script]
        self.assertIn("/home/tester/.local/share/containers/storage", disk[0])

    def test_unsafe_workdirs_are_rejected_before_mkdir(self) -> None:
        """$HOME, its parents and ~/.ssh are never used or created."""
        cases = [
            ("~", "other than $HOME"),
            ("/home", "not a parent of $HOME"),
            ("/", "not a parent of $HOME"),
            ("~/.ssh/atesor", "outside ~/.ssh"),
        ]
        for workdir, problem in cases:
            with self.subTest(workdir=workdir):
                self.machine = FakeMachine()
                self.env["ATESOR_REMOTE_WORKDIR"] = workdir
                report = self.run_ladder()
                self.assertIn(problem, report.render())
                self.assertFalse(
                    any(s.startswith("mkdir") for s in self.scripts())
                )

    def test_workdir_symlink_into_ssh_is_rejected(self) -> None:
        """The physical path is checked again after mkdir."""
        self.machine.workdir = "/home/tester/.ssh/atesor-ai"
        report = self.run_ladder()
        self.assertIn("outside ~/.ssh", report.render())
        self.assertIsNone(target.remote_workdir())

    def test_unwritable_workdir_is_reported(self) -> None:
        """A failed mkdir or write test gives the workdir item."""
        self.machine.workdir = ""
        report = self.run_ladder()
        self.assertIn(
            "a writable work directory at ~/atesor-ai", report.render()
        )

    def test_one_filesystem_needs_20_gb(self) -> None:
        """Podman storage and the workdir on one disk need 20 GB."""
        self.machine.free = (
            "graph=/dev/root 15728640",
            "work=/dev/root 15728640",
        )
        report = self.run_ladder()
        self.assertIn("20 GB free on the filesystem", report.render())
        self.assertIn("(has 15.0 GB)", report.render())

    def test_two_filesystems_need_10_gb_each(self) -> None:
        """Separate disks need 10 GB each."""
        self.machine.free = ("graph=/dev/a 12582912", "work=/dev/b 12582912")
        self.assertTrue(self.run_ladder().passed)
        self.machine = FakeMachine()
        self.machine.free = ("graph=/dev/a 5242880", "work=/dev/b 12582912")
        report = self.run_ladder()
        self.assertIn(
            "10 GB free for podman storage (has 5.0 GB)", report.render()
        )

    def test_cleanup_subset_stops_at_rung_8(self) -> None:
        """Cleanup runs rungs 2 to 8, so a full disk cannot block it."""
        report = self.run_ladder(2, preflight.CLEANUP_LAST_RUNG)
        self.assertEqual(
            list(range(2, 9)), [result.number for result in report.results]
        )
        for script in self.scripts():
            self.assertNotIn("rsync", script)
            self.assertNotIn("mkdir", script)
            self.assertNotIn("df -Pk", script)


class TestReportColors(unittest.TestCase):
    """Tests for the console colors of the report."""

    def setUp(self) -> None:
        """Make a report with one passed rung and one failed rung."""
        podman = preflight.Missing(
            "podman 4.0 or newer", "sudo apt install podman"
        )
        self.report = preflight.PreflightReport(
            [
                preflight.RungResult(6, True, "mvendorid 0x489"),
                preflight.RungResult(7, False, "not installed", [podman]),
            ]
        )

    def test_default_render_has_no_color(self) -> None:
        """Without a paint function, the text has no color codes."""
        expected = [
            "  [  ok]  6 Real hardware: mvendorid 0x489",
            "  [FAIL]  7 Podman: not installed",
            "",
            "Native preflight failed. Still required on the machine:",
            "  - podman 4.0 or newer  fix: sudo apt install podman",
            "Atesor does not install or change these. The user or the "
            "machine provider",
            "must fix them, then run: atesor-ai --target native --preflight",
        ]
        self.assertEqual("\n".join(expected), self.report.render(True))

    def test_only_failures_are_red(self) -> None:
        """The failed rung and the verdict are red, and nothing else."""
        lines = self.report.render(True, _tag_colors).splitlines()
        self.assertEqual(
            "  <green>[  ok]</green>  6 Real hardware: mvendorid 0x489",
            lines[0],
        )
        self.assertEqual(
            "  <red>[FAIL]  7 Podman: not installed</red>", lines[1]
        )
        self.assertEqual(
            "<red>Native preflight failed. Still required on the "
            "machine:</red>",
            lines[3],
        )
        red = [line for line in lines if "<red>" in line]
        self.assertEqual([lines[1], lines[3]], red)

    def test_stopped_verdict_is_the_only_red_line(self) -> None:
        """A stop at rung 4 paints its verdict red and the rest plain."""
        login = preflight.Missing(
            "SSH key login to the machine", "ssh-copy-id board-1"
        )
        report = preflight.PreflightReport(
            [preflight.RungResult(4, False, missing=[login])]
        )
        lines = report.render(paint=_tag_colors).splitlines()
        self.assertTrue(
            lines[0].startswith("<red>Native preflight stopped at rung 4")
        )
        self.assertEqual([lines[0]], [line for line in lines if "</" in line])

    def test_passed_verdict_is_green(self) -> None:
        """A passed report has a green verdict and no red."""
        report = preflight.PreflightReport(
            [preflight.RungResult(6, True, "mvendorid 0x489")]
        )
        text = report.render(True, _tag_colors)
        self.assertNotIn("<red>", text)
        self.assertTrue(
            text.endswith(
                "<green>Native preflight passed: rungs 6 to 6.</green>"
            )
        )


class TestMainPreflightWiring(unittest.TestCase):
    """Tests for where main() runs the preflight."""

    def setUp(self) -> None:
        """Create a scratch cache folder and a valid native config."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.env = {
            "ATESOR_TARGET": "native",
            "ATESOR_PLATFORM": "debian",
            "ATESOR_SSH_HOST": "tester@board-1",
            "XDG_CACHE_HOME": self._tmp.name,
            "HOME": self._tmp.name,
        }

    def _run_main(self, args, env, ladder_passes: bool = True, values=None):
        """Run main.main() with the ladder and heavy calls mocked.

        Args:
            args: The command-line arguments.
            env: The environment for the run.
            ladder_passes: The result of the faked machine rungs.
            values: Return values that replace the defaults below. An
                exception value is raised instead. ``self.calls`` keeps
                the call order, also when main() raises.
        """
        calls = self.calls = []
        real_run = preflight.run_preflight

        def fake_run(first=1, last=preflight.LAST_RUNG):
            """Run rung 1 for real; fake the machine rungs."""
            calls.append(("preflight", first, last))
            if (first, last) == (1, 1):
                return real_run(1, 1)
            result = preflight.RungResult(7, ladder_passes)
            if not ladder_passes:
                result.missing.append(
                    preflight.Missing(
                        "podman 4.0 or newer", "sudo apt install podman"
                    )
                )
            return preflight.PreflightReport([result])

        def recorder(name, value):
            """Return a side effect that logs the call."""

            def side_effect(*args, **kwargs):
                """Log the call and return the canned value."""
                calls.append((name,))
                if isinstance(value, BaseException):
                    raise value
                return value

            return side_effect

        canned = {
            "configure_logging": None,
            "check_keys": True,
            "setup_docker_environment": True,
            "cleanup_container": None,
            "cleanup_workspace": [],
            "rebuild_all_sandboxes": True,
            "run_agent": 0,
            "setup_native_environment": True,
            "stop_native_container": None,
            "cleanup_native_container": None,
            "clean_native_repos": None,
            "remove_native_repo": None,
        }
        canned.update(values or {})
        mocks = {}
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
            stack.enter_context(mock.patch("sys.argv", ["atesor-ai"] + args))
            stack.enter_context(
                mock.patch.object(
                    preflight, "run_preflight", side_effect=fake_run
                )
            )
            for name, value in canned.items():
                mocks[name] = stack.enter_context(
                    mock.patch.object(
                        main, name, side_effect=recorder(name, value)
                    )
                )
            mocks["get_cached_recipe"] = stack.enter_context(
                mock.patch(
                    "src.memory.get_cached_recipe",
                    side_effect=recorder("get_cached_recipe", None),
                )
            )
            stack.enter_context(mock.patch("src.llm_logger.set_llm_log_repo"))
            stack.enter_context(contextlib.redirect_stdout(output))
            code = main.main()
        return code, calls, mocks, output.getvalue()

    def test_console_report_paints_each_line(self) -> None:
        """The main function paints the failures red, not the items."""
        with mock.patch.object(main, "colored", side_effect=_tag_colors):
            code, _, _, out = self._run_main(
                ["--preflight"], self.env, ladder_passes=False
            )
        self.assertEqual(1, code)
        self.assertIn("  <red>[FAIL]  7 Podman</red>\n", out)
        self.assertIn(
            "\n  - podman 4.0 or newer  fix: sudo apt install podman\n", out
        )
        self.assertEqual(2, out.count("<red>"))

    def test_qemu_rejects_the_preflight_flag(self) -> None:
        """--preflight on the qemu target exits with 1 after rung 1."""
        code, calls, _, out = self._run_main(["--preflight"], {})
        self.assertEqual(1, code)
        self.assertEqual([("preflight", 1, 1)], calls)
        self.assertIn("native machine only", out)

    def test_bad_native_config_stops_before_everything(self) -> None:
        """Rung 1 runs before logging and the recipe-cache fast path."""
        env = dict(self.env)
        del env["ATESOR_PLATFORM"]
        code, calls, _, out = self._run_main(["--repo", _URL], env)
        self.assertEqual(1, code)
        self.assertEqual([("preflight", 1, 1)], calls)
        self.assertIn("ATESOR_PLATFORM", out)

    def test_preflight_flag_runs_the_full_ladder(self) -> None:
        """--preflight runs rungs 1 to 11 and returns their result."""
        code, calls, mocks, _ = self._run_main(["--preflight"], self.env)
        self.assertEqual(0, code)
        self.assertEqual(("preflight", 1, preflight.LAST_RUNG), calls[-1])
        mocks["configure_logging"].assert_not_called()
        code, _, _, _ = self._run_main(
            ["--preflight"], self.env, ladder_passes=False
        )
        self.assertEqual(1, code)

    def test_repo_run_checks_the_cache_before_the_ladder(self) -> None:
        """Rungs 2 to 11 run after the fast path and before the keys.

        Then the native container starts, the agent runs, the tree on
        the machine goes, and the container stops. The local Docker
        setup never runs.
        """
        code, calls, mocks, _ = self._run_main(["--repo", _URL], self.env)
        self.assertEqual(0, code)
        order = [call for call in calls if call[0] != "configure_logging"]
        self.assertEqual(
            [
                ("preflight", 1, 1),
                ("get_cached_recipe",),
                ("preflight", 2, preflight.LAST_RUNG),
                ("check_keys",),
                ("setup_native_environment",),
                ("run_agent",),
                ("remove_native_repo",),
                ("stop_native_container",),
            ],
            order,
        )
        mocks["setup_docker_environment"].assert_not_called()

    def test_failed_ladder_lists_what_is_missing(self) -> None:
        """A failed ladder prints the list and skips the key check."""
        code, _, mocks, out = self._run_main(
            ["--repo", _URL], self.env, ladder_passes=False
        )
        self.assertEqual(1, code)
        mocks["check_keys"].assert_not_called()
        self.assertIn("Still required on the machine", out)
        self.assertIn("sudo apt install podman", out)

    def test_native_cleanup_never_touches_local_docker(self) -> None:
        """Cleanup runs rungs 2 to 8, then the podman cleanup only."""
        code, calls, mocks, _ = self._run_main(["--cleanup"], self.env)
        self.assertEqual(0, code)
        self.assertIn(("preflight", 2, preflight.CLEANUP_LAST_RUNG), calls)
        mocks["cleanup_native_container"].assert_called_once_with(
            remove_image=False
        )
        mocks["cleanup_container"].assert_not_called()
        mocks["setup_native_environment"].assert_not_called()

    def test_native_clean_image_removes_the_podman_image(self) -> None:
        """--clean-image on native removes the image on the machine."""
        code, _, mocks, _ = self._run_main(["--clean-image"], self.env)
        self.assertEqual(0, code)
        mocks["cleanup_native_container"].assert_called_once_with(
            remove_image=True
        )
        mocks["cleanup_container"].assert_not_called()

    def test_native_rebuild_never_rebuilds_local_images(self) -> None:
        """--rebuild on native rebuilds the podman image only."""
        code, calls, mocks, _ = self._run_main(["--rebuild"], self.env)
        self.assertEqual(0, code)
        mocks["rebuild_all_sandboxes"].assert_not_called()
        mocks["setup_docker_environment"].assert_not_called()
        self.assertIn(("preflight", 2, preflight.LAST_RUNG), calls)
        self.assertIn(("setup_native_environment",), calls)
        self.assertEqual(("stop_native_container",), calls[-1])

    def test_native_setup_only_stops_the_container(self) -> None:
        """--setup-only starts the container, then stops it again."""
        code, calls, mocks, out = self._run_main(["--setup-only"], self.env)
        self.assertEqual(0, code)
        mocks["check_keys"].assert_not_called()
        self.assertEqual(
            [("setup_native_environment",), ("stop_native_container",)],
            calls[-2:],
        )
        self.assertIn("Setup complete!", out)

    def test_native_container_stops_after_an_exception(self) -> None:
        """An agent crash still stops the container on the machine."""
        with self.assertRaises(RuntimeError):
            self._run_main(
                ["--repo", _URL],
                self.env,
                values={"run_agent": RuntimeError("agent crashed")},
            )
        self.assertEqual(("stop_native_container",), self.calls[-1])

    def test_native_port_removes_the_tree_after_a_crash(self) -> None:
        """The tree on the machine goes, then the container stops."""
        with self.assertRaises(RuntimeError):
            self._run_main(
                ["--repo", _URL],
                self.env,
                values={"run_agent": RuntimeError("agent crashed")},
            )
        self.assertEqual(
            [("remove_native_repo",), ("stop_native_container",)],
            self.calls[-2:],
        )

    def test_native_clean_workspace_cleans_the_machine_repos(self) -> None:
        """Choosing repos/ also cleans repos/ on the machine."""
        code, _, mocks, _ = self._run_main(
            ["--clean-workspace"],
            self.env,
            values={"cleanup_workspace": ["repos"]},
        )
        self.assertEqual(0, code)
        mocks["clean_native_repos"].assert_called_once_with()

    def test_native_clean_workspace_keeps_machine_repos(self) -> None:
        """Choosing logs/ only leaves the machine alone."""
        code, _, mocks, _ = self._run_main(
            ["--clean-workspace"],
            self.env,
            values={"cleanup_workspace": ["logs"]},
        )
        self.assertEqual(0, code)
        mocks["clean_native_repos"].assert_not_called()

    def test_short_target_flag_is_rejected(self) -> None:
        """A short --targ would give the wrong workspace, so it fails."""
        with mock.patch.object(main, "_TARGET_AT_IMPORT", "qemu"):
            code, calls, _, out = self._run_main(
                ["--targ", "native", "--preflight"], self.env
            )
        self.assertEqual(1, code)
        self.assertEqual([], calls)
        self.assertIn("write --target in full", out)

    def test_full_target_flag_is_accepted(self) -> None:
        """--target native works when the early read saw it."""
        with mock.patch.object(main, "_TARGET_AT_IMPORT", "native"):
            code, _, _, _ = self._run_main(
                ["--target", "native", "--preflight"], self.env
            )
        self.assertEqual(0, code)


if __name__ == "__main__":
    unittest.main()
