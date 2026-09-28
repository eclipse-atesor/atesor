"""Tests for src/mirror.py: the local copy of the native repositories."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from src import config, mirror, target
from src.sandbox import SandboxUnavailableError

_ALIAS = "tester@board-1"
_WORKDIR = "/home/tester/atesor-ai"
_MISSING = (
    'rsync: [sender] change_dir "/home/tester/atesor-ai/repos/zlib" '
    "failed: No such file or directory (2)"
)


def _done(code: int = 0, err: str = ""):
    """Return a completed rsync call."""
    return subprocess.CompletedProcess([], code, "", err)


class _MirrorCase(unittest.TestCase):
    """A scratch workspace, a known work directory and a target."""

    native = True

    def setUp(self) -> None:
        """Select the target and point the workspace at a scratch folder."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = {
            "XDG_CACHE_HOME": self._tmp.name,
            "HOME": self._tmp.name,
            "PATH": os.environ.get("PATH", ""),
        }
        if self.native:
            env.update(
                {
                    "ATESOR_TARGET": "native",
                    "ATESOR_PLATFORM": "debian",
                    "ATESOR_SSH_HOST": _ALIAS,
                }
            )
        workspace = os.path.join(self._tmp.name, "ws")
        for patcher in (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(config, "WORKSPACE_ROOT", workspace),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        target.reset_target_cache()
        target.set_remote_workdir(_WORKDIR)

    def rsync(self, results):
        """Replace rsync with canned results and return the mock."""
        patcher = mock.patch("src.mirror.subprocess.run", side_effect=results)
        run = patcher.start()
        self.addCleanup(patcher.stop)
        return run


class TestRepoOf(unittest.TestCase):
    """Tests for the repository name of a container path."""

    def test_repo_paths(self) -> None:
        """Only /workspace/repos/<repo> paths have a repository."""
        cases = {
            "/workspace/repos/zlib": "zlib",
            "/workspace/repos/zlib/src/x.c": "zlib",
            "/workspace/repos/zlib/../cJSON/x.c": "cJSON",
            "/workspace/repos/": None,
            "/workspace/repos/..": None,
            "/workspace/repos/../etc/passwd": None,
            "/workspace/repos/.hidden": None,
            "/workspace/output/zlib": None,
            "/tmp/zlib": None,
        }
        for path, repo in cases.items():
            with self.subTest(path=path):
                self.assertEqual(repo, mirror.repo_of(path))


class TestPull(_MirrorCase):
    """Tests for the rsync copy from the machine."""

    def test_rsync_argv(self) -> None:
        """One rsync call over the open ssh connection, with no stdin."""
        run = self.rsync([_done()])
        mirror.pull("zlib")
        local = os.path.join(config.WORKSPACE_ROOT, "repos", "zlib")
        self.assertEqual(
            [
                "rsync",
                "-a",
                "--delete",
                "--modify-window=-1",
                "--exclude=.git",
                "-e",
                shlex.join(target.ssh_argv()),
                f"{_ALIAS}:{_WORKDIR}/repos/zlib/",
                local + "/",
            ],
            run.call_args.args[0],
        )
        self.assertIs(subprocess.DEVNULL, run.call_args.kwargs["stdin"])
        self.assertTrue(os.path.isdir(local))

    def test_copies_only_when_stale(self) -> None:
        """A copy runs after a command, or when forced."""
        run = self.rsync([_done(), _done(), _done()])
        mirror.pull("zlib")
        mirror.pull("zlib")
        self.assertEqual(1, run.call_count)
        mirror.mark_stale()
        self.assertTrue(mirror.is_stale("zlib"))
        mirror.pull("zlib")
        self.assertEqual(2, run.call_count)
        mirror.pull("zlib", force=True)
        self.assertEqual(3, run.call_count)

    def test_warning_exits_do_not_raise(self) -> None:
        """The rsync exits 23 and 24 are warnings; the copy is fresh."""
        run = self.rsync([_done(23, "rsync: permission denied"), _done(24)])
        mirror.pull("zlib")
        mirror.mark_stale()
        mirror.pull("zlib")
        self.assertEqual(2, run.call_count)
        self.assertFalse(mirror.is_stale("zlib"))

    def test_other_exits_raise_without_the_address(self) -> None:
        """Any other failure stops the run, and hides the address."""
        self.rsync([_done(12, "rsync: connection closed (10.0.0.5)")])
        with self.assertRaises(SandboxUnavailableError) as ctx:
            mirror.pull("zlib")
        self.assertIn("rsync exit 12", str(ctx.exception))
        self.assertNotIn("10.0.0.5", str(ctx.exception))
        self.assertTrue(mirror.is_stale("zlib"))

    def test_missing_repo_removes_the_local_copy(self) -> None:
        """A tree that is gone on the machine is also gone locally."""
        local = mirror.mirror_dir("zlib")
        os.makedirs(local)
        open(os.path.join(local, "old.c"), "w").close()
        self.rsync([_done(23, _MISSING)])
        mirror.pull("zlib")
        self.assertFalse(os.path.exists(local))

    def test_timeout_raises(self) -> None:
        """A copy that runs too long stops the run."""
        self.rsync([subprocess.TimeoutExpired("rsync", 1)])
        with self.assertRaises(SandboxUnavailableError):
            mirror.pull("zlib")

    def test_unknown_work_directory_raises(self) -> None:
        """Without preflight rung 10, there is nothing to copy from."""
        target.reset_target_cache()
        run = self.rsync([_done()])
        with self.assertRaises(SandboxUnavailableError):
            mirror.pull("zlib")
        run.assert_not_called()

    def test_unsafe_names_are_refused(self) -> None:
        """A name can never point outside the repos folder."""
        run = self.rsync([_done()])
        for name in ("..", ".git", "a/b", ""):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    mirror.pull(name)
        run.assert_not_called()


class TestToHostPath(_MirrorCase):
    """Tests for the copy before a host-side read."""

    def test_repo_path_copies_first(self) -> None:
        """A read in a repository copies it, then maps the path."""
        run = self.rsync([_done()])
        host = config.to_host_path("/workspace/repos/zlib/CMakeLists.txt")
        expected = os.path.join(
            config.WORKSPACE_ROOT, "repos", "zlib", "CMakeLists.txt"
        )
        self.assertEqual(expected, host)
        self.assertEqual(1, run.call_count)

    def test_other_paths_do_not_copy(self) -> None:
        """Paths outside a repository never start rsync."""
        run = self.rsync([])
        config.to_host_path("/workspace/output/zlib_recipe.md")
        config.to_host_path("/workspace/repos")
        run.assert_not_called()


class TestQemuHasNoMirror(_MirrorCase):
    """On qemu, the container mounts the local workspace."""

    native = False

    def test_nothing_is_copied(self) -> None:
        """pull() and to_host_path() never start rsync on qemu."""
        run = self.rsync([])
        mirror.pull("zlib")
        host = config.to_host_path("/workspace/repos/zlib/x.c")
        self.assertEqual(config.WORKSPACE_ROOT + "/repos/zlib/x.c", host)
        run.assert_not_called()


@unittest.skipUnless(shutil.which("rsync"), "needs the rsync program")
class TestRealRsync(_MirrorCase):
    """Run the real rsync, with a fake ssh that runs the command here."""

    def setUp(self) -> None:
        """Make a fake machine folder and a fake ssh program."""
        super().setUp()
        self.machine = os.path.join(self._tmp.name, "machine")
        target.set_remote_workdir(self.machine)
        fake_ssh = os.path.join(self._tmp.name, "fake-ssh")
        with open(fake_ssh, "w", encoding="utf-8") as handle:
            handle.write(
                "#!/bin/sh\n"
                "# Drop the ssh options and the host; run the command here.\n"
                'while [ $# -gt 0 ]; do case "$1" in\n'
                "  -o|-F) shift 2 ;;\n"
                "  *) break ;;\n"
                "esac; done\n"
                "shift\n"
                'exec "$@"\n'
            )
        os.chmod(fake_ssh, 0o755)
        for patcher in (
            mock.patch.object(target, "ssh_argv", return_value=[fake_ssh]),
            mock.patch.object(target, "ssh_alias", return_value="board"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write(self, rel: str, text: str) -> None:
        """Write a file in the repository on the fake machine."""
        path = os.path.join(self.machine, "repos", "zlib", rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def _read(self, rel: str) -> str:
        """Read a file from the local mirror."""
        with open(os.path.join(mirror.mirror_dir("zlib"), rel)) as handle:
            return handle.read()

    def test_copy_follows_the_machine(self) -> None:
        """New, changed and removed files reach the mirror; .git never."""
        self._write("zlib.h", "v1")
        self._write("old.c", "old")
        self._write(".git/HEAD", "ref: refs/heads/main")
        mirror.pull("zlib")
        local = mirror.mirror_dir("zlib")
        self.assertEqual("v1", self._read("zlib.h"))
        self.assertFalse(os.path.exists(os.path.join(local, ".git")))

        self._write("zlib.h", "v2")
        self._write("build/libz.a", "object code")
        os.remove(os.path.join(self.machine, "repos", "zlib", "old.c"))
        mirror.mark_stale()
        mirror.pull("zlib")
        self.assertEqual("v2", self._read("zlib.h"))
        self.assertEqual("object code", self._read("build/libz.a"))
        self.assertFalse(os.path.exists(os.path.join(local, "old.c")))

    def test_missing_repo_on_the_machine_removes_the_copy(self) -> None:
        """The real rsync message for a missing tree is recognized."""
        self._write("zlib.h", "v1")
        mirror.pull("zlib")
        shutil.rmtree(os.path.join(self.machine, "repos", "zlib"))
        mirror.mark_stale()
        mirror.pull("zlib")
        self.assertFalse(os.path.exists(mirror.mirror_dir("zlib")))


if __name__ == "__main__":
    unittest.main()
