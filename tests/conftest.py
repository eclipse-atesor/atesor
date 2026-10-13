"""Shared pytest fixtures for the Atesor AI test suite.

Goals:
  * Isolate every test from shared global state (MEMORY_INSTANCES,
    the active platform profile, the cached execution target, and the
    native mirror state).
  * Never touch the network, never launch a real Docker exec, never
    call an LLM.
  * Make file-system side effects opt-in via the `tmp_path` builtin.
    The session fails if a test changes a git-tracked file under data/.
"""

from __future__ import annotations

import hashlib
import subprocess
from typing import Callable, Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _guard_tracked_data_files() -> Iterator[None]:
    """Fail if tests mutate tracked files under data/.

    The recipe cache and the few-shot examples under data/ are committed
    files, so a test must write to a tmp_path copy. The guard compares
    the SHA-256 digest of each ``git ls-files data`` path before and
    after the whole session. The paths are relative to the working
    directory, so run pytest from the repository root.
    """
    result = subprocess.run(
        ["git", "ls-files", "data"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in result.stdout.splitlines() if line]

    def digest(path: str) -> str:
        """Return the SHA-256 digest for a tracked data file."""
        with open(path, "rb") as file_obj:
            return hashlib.sha256(file_obj.read()).hexdigest()

    before = {path: digest(path) for path in paths}
    yield
    changed = [
        path
        for path, old_digest in before.items()
        if digest(path) != old_digest
    ]
    assert not changed, "Tests mutated git-tracked data files: " + ", ".join(
        changed
    )


@pytest.fixture(autouse=True)
def _no_package_test_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep finish_node's package test phase off in unit tests.

    The phase looks for a test suite in the live workspace and runs it
    in the sandbox. Tests of the phase itself set the variable back.
    """
    monkeypatch.setenv("ATESOR_PACKAGE_TEST_TIMEOUT", "0")


@pytest.fixture(autouse=True)
def _clear_memory_instances() -> Iterator[None]:
    """Wipe the MEMORY_INSTANCES singleton so tests don't bleed."""
    from src import memory

    saved = dict(memory.MEMORY_INSTANCES)
    memory.MEMORY_INSTANCES.clear()
    try:
        yield
    finally:
        memory.MEMORY_INSTANCES.clear()
        memory.MEMORY_INSTANCES.update(saved)


@pytest.fixture(autouse=True)
def _restore_active_profile() -> Iterator[None]:
    """Reset the platforms cache before/after each test.

    We pre-seed the cache with ALPINE_RISCV so that any code path that calls
    get_active_profile() during a test does NOT trigger a real `docker exec`
    against a (likely non-running) container — which would block for ~10s
    per call and balloon the suite runtime. A test can swap the profile
    freely: the fixture restores the saved profile after the test.
    """
    from src import platforms

    saved = platforms._cached_profile
    try:
        platforms._cached_profile = platforms.ALPINE_RISCV
        yield
    finally:
        platforms._cached_profile = saved


@pytest.fixture(autouse=True)
def _reset_target_cache() -> Iterator[None]:
    """Forget the cached target before and after each test.

    ``get_target()`` caches the qemu or native config from the
    environment. Without the reset, a test that selects the native
    target leaks it, and the preflight results, into later tests.
    """
    from src import target

    target.reset_target_cache()
    try:
        yield
    finally:
        target.reset_target_cache()


@pytest.fixture(autouse=True)
def _reset_mirror_state() -> Iterator[None]:
    """Forget which repositories the mirror copied, between tests.

    The mirror module keeps the pull state (which repositories are
    fresh) in module globals, so one test must not see another test's
    pulls.
    """
    from src import mirror

    mirror.reset()
    try:
        yield
    finally:
        mirror.reset()


@pytest.fixture
def stub_execute_command(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, Callable[..., object]], Callable[..., object]]:
    """Return a helper that stubs execute_command in any module.

    The returned helper replaces execute_command with a deterministic
    callable in one module, so the test does not run docker. The
    monkeypatch undo restores the real function after the test.
    Example::

        def fake(cmd, **kwargs):
            return CommandResult(cmd, 0, "ok", "", 0.0)
        stub_execute_command("src.scripted_ops", fake)
    """

    def _install(module_path: str, fake):
        """Install fake as execute_command in the named module."""
        import importlib

        mod = importlib.import_module(module_path)
        monkeypatch.setattr(mod, "execute_command", fake)
        return fake

    return _install
