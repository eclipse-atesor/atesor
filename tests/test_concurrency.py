"""Multi-worker / multi-thread concurrency regression tests.

batch_test.py drives up to ``--workers N`` parallel ``main.py``
processes that share one host workspace, one logs directory, one
recipe cache, and one Docker daemon. These tests lock in the
concurrency contracts that setup relies on:

  * the provisioning flock serializes the expensive critical section —
    a cold start with N workers must build the sandbox image ONCE, not
    N times;
  * the batch container pool never double-leases a container, restores
    itself after every outcome, and never returns a container while
    the agent process is still alive in it;
  * the per-clone GitHub auth overlay is thread-local — concurrent
    clones through the shared ScriptedOperations instance cannot see
    each other's headers;
  * concurrent recipe-cache writers (separate real processes, like
    batch workers) serialize via filelock and never corrupt the cache.

No test touches the network, a real Docker daemon, or an LLM.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.state import CommandResult

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BATCH_TEST = _REPO_ROOT / ".github" / "scripts" / "batch_test.py"

_real_sleep = time.sleep


def _load_batch_module():
    """Import .github/scripts/batch_test.py as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "batch_test_under_test", str(_BATCH_TEST)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# Provisioning flock — one image build across N concurrent workers
# ===========================================================================


class _FakeContainer:
    """Running container with the expected /workspace mount."""

    def __init__(self, workspace_root: str) -> None:
        self.status = "running"
        self.short_id = "cafe1234"
        self.attrs = {
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": os.path.realpath(workspace_root),
                }
            ]
        }

    def reload(self) -> None:
        """Reload."""

    def exec_run(self, cmd: str):
        """Answer the provisioning self-checks."""
        if "os-release" in cmd:
            return SimpleNamespace(exit_code=0, output=b"ID=alpine\n")
        if "uname" in cmd:
            return SimpleNamespace(exit_code=0, output=b"riscv64\n")
        return SimpleNamespace(exit_code=0, output=b"ok\n")


class _FakeDockerClient:
    """Thread-safe fake: image is missing until api.build runs."""

    def __init__(self, workspace_root: str) -> None:
        import docker as docker_pkg

        self._docker_errors = docker_pkg.errors
        self._lock = threading.Lock()
        self.image_built = False
        self.build_calls = 0
        self._container = _FakeContainer(workspace_root)
        self.images = SimpleNamespace(get=self._images_get)
        self.api = SimpleNamespace(build=self._api_build)
        self.containers = SimpleNamespace(get=self._containers_get)

    def _images_get(self, name: str):
        with self._lock:
            if not self.image_built:
                raise self._docker_errors.ImageNotFound(name)
        return SimpleNamespace(id="sha256:fake")

    def _api_build(self, **_kwargs):
        with self._lock:
            self.build_calls += 1
        _real_sleep(0.25)  # emulate a slow build inside the lock
        with self._lock:
            self.image_built = True
        return iter([{"stream": "Step 1/1 : FROM fake"}])

    def _containers_get(self, _name: str):
        return self._container

    def close(self) -> None:
        """Close."""


class TestProvisionLockSerializesImageBuild:
    """Concurrent provisioning must build the sandbox image once."""

    def test_image_built_once_across_workers(
        self, tmp_path, monkeypatch
    ) -> None:
        """Two concurrent _provision_sandbox calls -> one api.build."""
        import main as main_mod
        from src import platforms

        monkeypatch.setattr(main_mod, "LOGS_DIR", str(tmp_path))
        monkeypatch.setattr(main_mod, "_ensure_riscv64_binfmt", lambda: True)
        # Neutralize the real waits inside the provisioning body; the
        # fake build keeps its own (pre-captured) real sleep.
        monkeypatch.setattr(main_mod.time, "sleep", lambda _s: None)

        client = _FakeDockerClient(main_mod.WORKSPACE_ROOT)
        profile = platforms.ALPINE_RISCV

        def provision() -> bool:
            return main_mod._provision_sandbox(
                client,
                profile,
                profile.image_name,
                "atesor-test-lock",
                profile.dockerfile,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda _i: provision(), range(2))
            )

        assert results == [True, True]
        assert client.build_calls == 1, (
            "the flock must serialize provisioning so only the first "
            "worker builds; the second must see the image and skip"
        )


# ===========================================================================
# Batch container pool — lease/return invariants under N workers
# ===========================================================================


class _TrackingQueue(queue.Queue):
    """Queue that flags a container leased twice at the same time."""

    def __init__(self) -> None:
        super().__init__()
        self.leased: set = set()
        self.double_leases: list = []
        self._tlock = threading.Lock()

    def get(self, *args, **kwargs):
        """Get."""
        name = super().get(*args, **kwargs)
        with self._tlock:
            if name in self.leased:
                self.double_leases.append(name)
            self.leased.add(name)
        return name

    def put(self, item, *args, **kwargs):
        """Put."""
        with self._tlock:
            self.leased.discard(item)
        return super().put(item, *args, **kwargs)


class TestBatchContainerPool:
    """run_agent must lease exclusively and always restore the pool."""

    def _drain(self, q: queue.Queue) -> list:
        names = []
        while True:
            try:
                names.append(q.get_nowait())
            except queue.Empty:
                return names

    def test_concurrent_runs_never_double_lease(
        self, tmp_path, monkeypatch
    ) -> None:
        """6 packages on 3 workers: unique leases, pool restored."""
        bt = _load_batch_module()
        monkeypatch.setattr(bt, "BATCH_LOGS_DIR", str(tmp_path / "logs"))
        bt._shutdown_event.clear()

        pool_q = _TrackingQueue()
        for i in range(3):
            pool_q.put(f"{bt._BASE_CONTAINER}-w{i + 1}")
        monkeypatch.setattr(bt, "_container_pool", pool_q)

        real_popen = subprocess.Popen

        def fake_popen(cmd, **kwargs):
            # Replace the real agent with a short-lived process; keep
            # the process-group plumbing (preexec_fn) intact.
            assert "--container" in cmd
            return real_popen(["sleep", "0.1"], **kwargs)

        monkeypatch.setattr(bt.subprocess, "Popen", fake_popen)

        packages = [(f"pkg{i}", f"https://example.org/pkg{i}") for i in
                    range(6)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = [
                ex.submit(bt.run_agent, url, name)
                for name, url in packages
            ]
            results = [f.result() for f in futures]

        assert all(success for success, _out, _dur in results)
        assert pool_q.double_leases == [], (
            "a container name must never be leased by two runs at once"
        )
        remaining = self._drain(pool_q)
        assert sorted(remaining) == sorted(
            f"{bt._BASE_CONTAINER}-w{i + 1}" for i in range(3)
        ), "every lease must be returned exactly once"
        assert not bt._live_pids, "all child PIDs must be unregistered"

    def test_exception_path_kills_child_and_returns_container(
        self, tmp_path, monkeypatch
    ) -> None:
        """A failure after spawn must not leak a live child."""
        bt = _load_batch_module()
        monkeypatch.setattr(bt, "BATCH_LOGS_DIR", str(tmp_path / "logs"))
        bt._shutdown_event.clear()
        # Cap the SIGTERM->SIGKILL grace period so the test stays fast.
        monkeypatch.setattr(
            bt.time, "sleep", lambda s: _real_sleep(min(s, 0.05))
        )

        pool_q: queue.Queue = queue.Queue()
        pool_q.put(f"{bt._BASE_CONTAINER}-w1")
        monkeypatch.setattr(bt, "_container_pool", pool_q)

        spawned: list = []
        real_popen = subprocess.Popen

        def fake_popen(cmd, **kwargs):
            proc = real_popen(["sleep", "30"], **kwargs)
            spawned.append(proc)
            return proc

        monkeypatch.setattr(bt.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            bt,
            "_register_pid",
            mock.MagicMock(side_effect=RuntimeError("injected")),
        )

        success, output, _dur = bt.run_agent(
            "https://example.org/pkg", "pkg-exc"
        )

        assert success is False
        assert "Exception" in output
        assert pool_q.qsize() == 1, "container must return to the pool"
        assert spawned, "the child must have been spawned"
        deadline = time.time() + 5
        while spawned[0].poll() is None and time.time() < deadline:
            _real_sleep(0.05)
        assert spawned[0].poll() is not None, (
            "the child process must be dead before the container is "
            "handed to the next package"
        )


# ===========================================================================
# Thread-local auth overlay — concurrent clones on one instance
# ===========================================================================


class TestAuthThreadIsolation:
    """Concurrent clones must not observe each other's auth overlay."""

    def test_parallel_clones_keep_auth_separate(
        self, stub_execute_command, monkeypatch
    ) -> None:
        """The GitHub clone keeps its header while another clone runs."""
        from src.scripted_ops import ScriptedOperations

        monkeypatch.setenv("GIT_TOKEN", "sekret-token")
        monkeypatch.setattr("os.path.exists", lambda _p: False)
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations."
            "_ensure_container_healthy",
            lambda _self: None,
        )
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations."
            "_init_submodules_if_present",
            lambda _self, _p: None,
        )

        barrier = threading.Barrier(2, timeout=10)
        clone_envs: dict = {}
        lock = threading.Lock()

        def fake(cmd, **kwargs):
            text = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
            if "git clone" in text:
                # Hold both threads inside their clone at the same
                # moment so both auth overlays are active concurrently.
                barrier.wait()
                with lock:
                    clone_envs[threading.current_thread().name] = dict(
                        kwargs.get("extra_env") or {}
                    )
            return CommandResult(text, 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ops = ScriptedOperations()

        def clone(url: str, name: str) -> None:
            ops.clone_or_update_repository(url, name)

        t_github = threading.Thread(
            target=clone,
            args=("https://github.com/foo/bar.git", "bar"),
            name="github-clone",
        )
        t_other = threading.Thread(
            target=clone,
            args=("https://git.example.org/baz/qux.git", "qux"),
            name="other-clone",
        )
        t_github.start()
        t_other.start()
        t_github.join(timeout=15)
        t_other.join(timeout=15)

        github_env = clone_envs.get("github-clone", {})
        other_env = clone_envs.get("other-clone", {})
        assert github_env.get("GIT_CONFIG_VALUE_0", "").endswith(
            "sekret-token"
        ), "the GitHub clone must keep its header while another runs"
        assert "GIT_CONFIG_KEY_0" not in other_env, (
            "the non-GitHub clone must never see the sibling's header"
        )


# ===========================================================================
# Recipe cache — real multi-process writers (like batch workers)
# ===========================================================================


class TestRecipeCacheMultiProcessWriters:
    """N processes writing the cache concurrently must not corrupt it."""

    def test_six_process_writers_all_land(self, tmp_path) -> None:
        """All writers succeed and the cache stays valid JSON."""
        n = 6
        code_template = (
            "import sys; sys.path.insert(0, {root!r})\n"
            "from src.memory import save_to_recipe_cache\n"
            "ok = save_to_recipe_cache(\n"
            "    repo_name='pkg{i}', repo_url='https://x/pkg{i}',\n"
            "    build_system='go',\n"
            "    build_plan={{'phases': [{{'name': 'build',\n"
            "                'commands': ['go build .']}}]}},\n"
            "    dependencies=[], patches=[], artifacts=[],\n"
            "    build_duration_seconds=1.0, sandbox='alpine-riscv64')\n"
            "sys.exit(0 if ok else 1)\n"
        )
        env = dict(os.environ)
        env["ATESOR_HOME"] = str(tmp_path)

        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code_template.format(root=str(_REPO_ROOT), i=i),
                ],
                env=env,
                cwd=str(_REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for i in range(n)
        ]
        exit_codes = [p.wait(timeout=120) for p in procs]

        assert exit_codes == [0] * n, [
            p.stderr.read().decode()[-300:] for p in procs
        ]

        cache_path = tmp_path / "data" / "recipe_cache.json"
        cache = json.loads(cache_path.read_text())  # must stay valid
        packages = cache.get("packages", {})
        for i in range(n):
            assert f"pkg{i}" in packages, (
                f"pkg{i} lost in concurrent write — filelock broken"
            )
            assert "alpine-riscv64" in packages[f"pkg{i}"]
