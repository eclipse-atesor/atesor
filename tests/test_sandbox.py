"""Tests for src/sandbox.py: the exec argv, secrets and transport errors."""

import os
import shlex
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from src import sandbox, target

_ALIAS = "tester@board-1"
_SECRET = "Authorization: bearer ghp_TESTSECRET"
_NO_HOST = "ssh: Could not resolve hostname x: Name or service not known"
_REFUSED = "ssh: connect to host 10.0.0.5 port 22: Connection refused"
_DROPPED = "Connection to board-1 closed by remote host."
_NO_CONTAINER = (
    'Error: no container with name or ID "box" found: no such container'
)
_STOPPED = (
    "Error: can only create exec sessions on running containers: "
    "container state improper"
)

# Commands that a remote shell must pass through unchanged.
_HARD_COMMANDS = [
    "echo 'single' \"double\"",
    "echo $HOME ${PATH} $(id -u) `uname`",
    "printf '%s\\n' back\\slash 'a\\b'",
    "echo one\necho two",
    "grep -E '^(a|b)*$' file; true && false || echo *",
]


def _native_env(**extra: str) -> dict:
    """Return the environment of a valid alias-mode native setup."""
    env = {
        "ATESOR_TARGET": "native",
        "ATESOR_PLATFORM": "debian",
        "ATESOR_SSH_HOST": _ALIAS,
    }
    env.update(extra)
    return env


def _done(argv, code=0, out="", err=""):
    """Return a completed process for a faked call."""
    return subprocess.CompletedProcess(argv, code, out, err)


class _NativeCase(unittest.TestCase):
    """Run each test with a valid native config and a scratch cache."""

    def setUp(self) -> None:
        """Select the native target with a scratch cache folder."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = _native_env(XDG_CACHE_HOME=self._tmp.name, HOME=self._tmp.name)
        env["PATH"] = os.environ.get("PATH", "")
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        target.reset_target_cache()

    def remote_script(self, argv) -> str:
        """Check the ssh prefix of argv and return its remote script."""
        self.assertEqual(target.ssh_argv(), argv[:-2])
        self.assertEqual(_ALIAS, argv[-2])
        return argv[-1]

    def fake_podman(self, body: str) -> str:
        """Write a fake podman script and return the folder that holds it."""
        folder = os.path.join(self._tmp.name, "bin")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "podman")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)
        return folder

    def run_on_fake_machine(self, script: str, folder: str, stdin=""):
        """Run a remote script in a local POSIX sh, as the machine does."""
        env = {"PATH": folder + os.pathsep + os.environ["PATH"]}
        return subprocess.run(
            ["sh", "-c", script],
            input=stdin,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )


class TestQemuArgv(unittest.TestCase):
    """The qemu argv stays exactly what Atesor used before native."""

    def test_plain_command(self) -> None:
        """A command with no env and no workdir is plain docker exec."""
        call = sandbox.build_exec_argv("box", ["bash", "-c", "ls"])
        self.assertEqual(
            ["docker", "exec", "box", "bash", "-c", "ls"], call.argv
        )
        self.assertFalse(call.native)
        self.assertEqual({"env": None}, call.run_kwargs())

    def test_env_workdir_and_secret(self) -> None:
        """Secrets go by name, with the value in the local environment."""
        call = sandbox.build_exec_argv(
            "box",
            ["bash", "-c", "git status"],
            env={"LANG": "C", "GIT_CONFIG_VALUE_0": _SECRET},
            workdir="/workspace/repos/x",
        )
        self.assertEqual(
            [
                "docker",
                "exec",
                "--env",
                "LANG=C",
                "--env",
                "GIT_CONFIG_VALUE_0",
                "-w",
                "/workspace/repos/x",
                "box",
                "bash",
                "-c",
                "git status",
            ],
            call.argv,
        )
        self.assertNotIn("ghp_TESTSECRET", " ".join(call.argv))
        self.assertEqual(_SECRET, call.env["GIT_CONFIG_VALUE_0"])

    def test_remove_container_runs_docker_rm_locally(self) -> None:
        """The batch refresh keeps its docker rm -f argv on qemu."""
        with mock.patch("src.sandbox.subprocess.run") as run:
            run.return_value = _done([], 0)
            sandbox.remove_container("box-w1")
        run.assert_called_once_with(
            ["docker", "rm", "-f", "box-w1"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )


class TestNativeArgv(_NativeCase):
    """The native call is one ssh call with one quoted podman script."""

    def test_script_gives_back_the_podman_argv(self) -> None:
        """shlex.split() of the script gives exec and the podman argv."""
        for command in _HARD_COMMANDS:
            with self.subTest(command=command):
                call = sandbox.build_exec_argv(
                    "box", ["bash", "-c", command], workdir="/workspace/r"
                )
                script = self.remote_script(call.argv)
                self.assertEqual(
                    [
                        "exec",
                        "podman",
                        "exec",
                        "-w",
                        "/workspace/r",
                        "box",
                        "bash",
                        "-c",
                        command,
                    ],
                    shlex.split(script),
                )
                self.assertEqual(
                    {"stdin": subprocess.DEVNULL}, call.run_kwargs()
                )

    def test_a_real_shell_passes_each_word_unchanged(self) -> None:
        """A POSIX sh on the machine gets the exact podman argv."""
        folder = self.fake_podman("printf '%s\\0' \"$@\"\n")
        for command in _HARD_COMMANDS:
            with self.subTest(command=command):
                call = sandbox.build_exec_argv("box", ["bash", "-c", command])
                result = self.run_on_fake_machine(call.argv[-1], folder)
                words = result.stdout.split("\0")[:-1]
                self.assertEqual(["exec", "box", "bash", "-c", command], words)

    def test_secret_travels_on_stdin_only(self) -> None:
        """The prelude reads the secret, and podman gets its name only."""
        call = sandbox.build_exec_argv(
            "box",
            ["bash", "-c", "git status"],
            env={"LANG": "C", "GIT_CONFIG_VALUE_0": _SECRET},
        )
        self.assertNotIn("ghp_TESTSECRET", " ".join(call.argv))
        self.assertEqual(_SECRET + "\n", call.stdin_text)
        self.assertEqual({"input": _SECRET + "\n"}, call.run_kwargs())
        script = self.remote_script(call.argv)
        prelude = (
            "IFS= read -r GIT_CONFIG_VALUE_0 && "
            "export GIT_CONFIG_VALUE_0 && "
        )
        self.assertTrue(script.startswith(prelude))
        self.assertEqual(
            [
                "exec",
                "podman",
                "exec",
                "--env",
                "LANG=C",
                "--env",
                "GIT_CONFIG_VALUE_0",
                "box",
                "bash",
                "-c",
                "git status",
            ],
            shlex.split(script[len(prelude) :]),
        )

    def test_a_real_shell_exports_the_secret(self) -> None:
        """The podman process sees the secret in its environment only."""
        folder = self.fake_podman(
            'printf "%s\\n" "$*"\nprintf "value=%s\\n" "$GITHUB_TOKEN"\n'
        )
        value = "  tab\there $not 'expanded' \\n  "
        call = sandbox.build_exec_argv(
            "box", ["true"], env={"GITHUB_TOKEN": value}
        )
        result = self.run_on_fake_machine(
            call.argv[-1], folder, stdin=call.stdin_text
        )
        lines = result.stdout.split("\n")
        self.assertEqual("exec --env GITHUB_TOKEN box true", lines[0])
        self.assertEqual(f"value={value}", lines[1])

    def test_secret_with_a_line_break_is_refused(self) -> None:
        """One line per key: a line break in a secret is an error."""
        with self.assertRaises(ValueError) as ctx:
            sandbox.build_exec_argv(
                "box", ["true"], env={"GITHUB_TOKEN": "a\nb"}
            )
        self.assertNotIn("a\nb", str(ctx.exception))

    def test_secret_key_must_be_a_shell_name(self) -> None:
        """The prelude cannot read into a name such as MY-TOKEN."""
        with self.assertRaises(ValueError):
            sandbox.build_exec_argv("box", ["true"], env={"MY-TOKEN": "x"})

    def test_run_raw_quotes_each_word(self) -> None:
        """run_raw() sends one quoted script over ssh."""
        with mock.patch("src.sandbox.subprocess.run") as run:
            run.return_value = _done([], 0)
            sandbox.run_raw(["podman", "rm", "-f", "a b"])
        script = self.remote_script(run.call_args.args[0])
        self.assertEqual(["podman", "rm", "-f", "a b"], shlex.split(script))
        self.assertIs(subprocess.DEVNULL, run.call_args.kwargs["stdin"])


class TestTransportErrors(_NativeCase):
    """Tests for the ssh transport classification and the retry rule."""

    def test_classification(self) -> None:
        """Only exit 255 with an ssh message is a transport error."""
        never = sandbox.NEVER_OPENED
        dropped = sandbox.DROPPED
        cases = [
            (_NO_HOST, never),
            (_REFUSED, never),
            ("ssh: connect to host h port 22: Connection timed out", never),
            ("kex_exchange_identification: read: Connection reset", never),
            ("Connection closed by 10.0.0.5 port 22", never),
            ("tester@board-1: Permission denied (publickey).", never),
            ("Host key verification failed.", never),
            (_DROPPED, dropped),
            ("client_loop: send disconnect: Broken pipe", dropped),
            ("Timeout, server board-1 not responding.", dropped),
            ("mux_client_read_packet: read header failed", dropped),
            ("make: *** [all] Error 255", ""),
        ]
        for stderr, kind in cases:
            with self.subTest(stderr=stderr):
                self.assertEqual(kind, sandbox.transport_error(255, stderr))
        self.assertEqual(
            "", sandbox.transport_error(1, "ssh: Could not resolve hostname")
        )

    def _run_call(self, results, login=True):
        """Run run_call() with faked ssh results and a faked login."""
        call = sandbox.build_exec_argv("box", ["true"])
        with (
            mock.patch(
                "src.sandbox.subprocess.run", side_effect=results
            ) as run,
            mock.patch(
                "src.sandbox.login_works", return_value=login
            ) as login_check,
        ):
            try:
                return sandbox.run_call(call, 60), run, login_check
            except sandbox.SandboxUnavailableError as exc:
                return exc, run, login_check

    def test_connect_failure_retries_once_after_a_good_login(self) -> None:
        """A connection that never opened gets one more try."""
        refused = _done([], 255, err=_REFUSED)
        result, run, login = self._run_call([refused, _done([], 0, "ok")])
        self.assertEqual("ok", result.stdout)
        self.assertEqual(2, run.call_count)
        login.assert_called_once_with()

    def test_connect_failure_with_a_bad_login_raises(self) -> None:
        """No retry when the login check also fails."""
        refused = _done([], 255, err=_REFUSED)
        error, run, _ = self._run_call([refused], login=False)
        self.assertIsInstance(error, sandbox.SandboxUnavailableError)
        self.assertEqual(1, run.call_count)
        self.assertNotIn("10.0.0.5", str(error))

    def test_second_connect_failure_raises(self) -> None:
        """A retry that fails again raises."""
        refused = _done([], 255, err=_REFUSED)
        error, run, _ = self._run_call([refused, refused])
        self.assertIsInstance(error, sandbox.SandboxUnavailableError)
        self.assertEqual(2, run.call_count)

    def test_dropped_connection_never_retries(self) -> None:
        """A half-done command is never run twice."""
        dropped = _done([], 255, err=_DROPPED)
        error, run, login = self._run_call([dropped])
        self.assertIsInstance(error, sandbox.SandboxUnavailableError)
        self.assertIn("dropped", str(error))
        self.assertEqual(1, run.call_count)
        login.assert_not_called()

    def test_command_failure_is_a_normal_result(self) -> None:
        """A failing remote command is returned, not raised."""
        result, run, login = self._run_call(
            [_done([], 2, err="make: Error 2")]
        )
        self.assertEqual(2, result.returncode)
        login.assert_not_called()

    def test_not_running_mapping(self) -> None:
        """Exit 125 (missing) or 255 (stopped), with the podman text."""
        self.assertTrue(sandbox.not_running(125, _NO_CONTAINER))
        self.assertTrue(sandbox.not_running(125, _STOPPED))
        # Seen live with podman 5.2.3: exec on a stopped container.
        self.assertTrue(sandbox.not_running(255, _STOPPED))
        self.assertFalse(sandbox.not_running(125, "Error: OCI runtime error"))
        self.assertFalse(sandbox.not_running(255, _DROPPED))
        self.assertFalse(sandbox.not_running(1, _NO_CONTAINER))


class TestPodmanHelpers(_NativeCase):
    """Tests for the argv of each podman call on the machine."""

    def _podman(self, func, *args, result=None):
        """Call a helper with faked ssh and return (value, podman argv)."""
        with mock.patch("src.sandbox.subprocess.run") as run:
            run.return_value = result or _done([], 0)
            value = func(*args)
        return value, shlex.split(self.remote_script(run.call_args.args[0]))

    def test_each_podman_argv(self) -> None:
        """The helpers send the podman commands that the plan names."""
        mount = "/home/u/atesor-ai:/workspace:Z"
        probe = ["sh", "-lc", "go version"]
        cases = [
            (
                sandbox.create_container,
                ("box", "img:1", "/home/u/atesor-ai"),
                ["podman", "run", "-d", "--init", "--name", "box"]
                + ["-v", mount, "img:1"],
            ),
            (sandbox.start_container, ("box",), ["podman", "start", "box"]),
            (sandbox.stop_container, ("box",), ["podman", "stop", "box"]),
            (
                sandbox.remove_container,
                ("box",),
                ["podman", "rm", "-f", "--volumes", "box"],
            ),
            (sandbox.remove_image, ("img:1",), ["podman", "rmi", "img:1"]),
            (
                sandbox.run_in_throwaway,
                ("img:1", probe),
                ["podman", "run", "--rm", "img:1"] + probe,
            ),
        ]
        for func, args, expected in cases:
            with self.subTest(func=func.__name__):
                _, argv = self._podman(func, *args)
                self.assertEqual(expected, argv)

    def test_image_exists(self) -> None:
        """Exit 0 is yes, 1 is no, and anything else is an error."""
        found, argv = self._podman(sandbox.image_exists, "img:1")
        self.assertTrue(found)
        self.assertEqual(["podman", "image", "exists", "img:1"], argv)
        found, _ = self._podman(
            sandbox.image_exists, "img:1", result=_done([], 1)
        )
        self.assertFalse(found)
        with self.assertRaises(sandbox.PodmanError):
            self._podman(
                sandbox.image_exists, "img:1", result=_done([], 125, err="x")
            )

    def test_inspect_container(self) -> None:
        """Inspect gives the state and the /workspace source, or None."""
        state, argv = self._podman(
            sandbox.inspect_container,
            "box",
            result=_done([], 0, "running\t/home/u/atesor-ai\n"),
        )
        self.assertEqual(
            sandbox.ContainerState("running", "/home/u/atesor-ai"), state
        )
        self.assertEqual(
            ["podman", "container", "inspect", "--format"], argv[:4]
        )
        self.assertIn('eq .Destination "/workspace"', argv[4])
        missing = _done([], 125, err="Error: no such container box")
        state, _ = self._podman(
            sandbox.inspect_container, "box", result=missing
        )
        self.assertIsNone(state)
        with self.assertRaises(sandbox.PodmanError):
            self._podman(
                sandbox.inspect_container,
                "box",
                result=_done([], 125, err="x"),
            )

    def test_create_failure_raises(self) -> None:
        """A failed podman run raises PodmanError with the podman text."""
        with self.assertRaises(sandbox.PodmanError) as ctx:
            self._podman(
                sandbox.create_container,
                "box",
                "img:1",
                "/w",
                result=_done([], 125, err="Error: name in use"),
            )
        self.assertIn("name in use", str(ctx.exception))


class _FakeStdin:
    """Collect what build_image() writes to the ssh stdin."""

    def __init__(self) -> None:
        self.parts = []

    def write(self, text: str) -> int:
        """Keep the text."""
        self.parts.append(text)
        return len(text)

    def close(self) -> None:
        """Accept the close."""

    def getvalue(self) -> str:
        """Return all written text."""
        return "".join(self.parts)


class _FakeBuild:
    """A fake Popen for podman build over ssh."""

    def __init__(self, lines, code=0, hang=False) -> None:
        self.stdin = _FakeStdin()
        self._lines = lines
        self._code = code
        self._hang = hang
        self._killed = threading.Event()

    @property
    def stdout(self):
        """Yield the output lines, or wait until the process is killed."""
        if self._hang:
            self._killed.wait(5)
            return iter([])
        return iter(line + "\n" for line in self._lines)

    def kill(self) -> None:
        """Record the kill."""
        self._killed.set()

    def wait(self) -> int:
        """Return the exit status."""
        return -9 if self._killed.is_set() else self._code

    def poll(self):
        """Return the exit status once the process ended."""
        return self.wait()


class TestBuildImage(_NativeCase):
    """Tests for the image build on the machine."""

    def test_build_streams_lines_and_sends_the_dockerfile(self) -> None:
        """The Dockerfile goes on stdin, and each line reaches on_line."""
        fake = _FakeBuild(["STEP 1/9: FROM debian", "COMMIT img:1"])
        seen = []
        with mock.patch(
            "src.sandbox.subprocess.Popen", return_value=fake
        ) as popen:
            code = sandbox.build_image("img:1", "FROM x\n", seen.append, 60)
        self.assertEqual(0, code)
        self.assertEqual(["STEP 1/9: FROM debian", "COMMIT img:1"], seen)
        self.assertEqual("FROM x\n", fake.stdin.getvalue())
        script = self.remote_script(popen.call_args.args[0])
        self.assertIn(
            "timeout --kill-after=30s --signal=TERM 60s "
            "podman build --pull=always -t img:1 "
            '-f "$ctx/Containerfile" "$ctx"',
            script,
        )

    def test_remote_timeout_exit_is_reported(self) -> None:
        """A remote timeout exit maps to BUILD_TIMED_OUT."""
        fake = _FakeBuild([], code=124)
        with mock.patch("src.sandbox.subprocess.Popen", return_value=fake):
            code = sandbox.build_image("img:1", "FROM x\n", print, 60)
        self.assertEqual(sandbox.BUILD_TIMED_OUT, code)

    def test_local_margin_timer_kills_stuck_ssh(self) -> None:
        """If ssh outlives the remote timeout margin, it is killed."""
        fake = _FakeBuild([], hang=True)
        timers = []

        def fake_timer(delay, callback):
            """Run the timer callback when start() is called."""
            timer = mock.Mock()
            timer.delay = delay
            timer.start.side_effect = callback
            timers.append(timer)
            return timer

        with mock.patch(
            "src.sandbox.subprocess.Popen", return_value=fake
        ), mock.patch(
            "src.sandbox.threading.Timer", side_effect=fake_timer
        ) as timer_class:
            code = sandbox.build_image("img:1", "FROM x\n", print, 60)
        self.assertEqual(sandbox.BUILD_TIMED_OUT, code)
        timer_class.assert_called_once()
        self.assertEqual(150, timers[0].delay)

    def test_build_transport_error_raises(self) -> None:
        """A dropped connection during the build raises."""
        fake = _FakeBuild([_DROPPED], code=255)
        with mock.patch("src.sandbox.subprocess.Popen", return_value=fake):
            with self.assertRaises(sandbox.SandboxUnavailableError):
                sandbox.build_image("img:1", "FROM x\n", lambda line: None, 60)

    def test_build_script_in_a_real_shell(self) -> None:
        """The script puts stdin into a temporary context, then cleans up."""
        folder = self.fake_podman(
            'while [ "$1" != "-f" ]; do shift; done\n'
            'cat "$2"\n'
            'printf "context=%s\\n" "$3"\n'
        )
        podman = shlex.join(["podman", "build", "--pull=always", "-t", "img"])
        script = sandbox._BUILD_PRELUDE + podman
        script += ' -f "$ctx/Containerfile" "$ctx"'
        result = self.run_on_fake_machine(script, folder, stdin="FROM x\n")
        self.assertEqual(0, result.returncode, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual("FROM x", lines[0])
        context = lines[1].split("=", 1)[1]
        self.assertFalse(os.path.exists(context))


if __name__ == "__main__":
    unittest.main()
