"""Shared pytest fixtures for the Atesor AI test suite.

Goals:
  * Isolate every test from shared global state (MEMORY_INSTANCES,
    active platform profile, llm_logger file handle).
  * Never touch the network, never launch a real Docker exec, never
    call an LLM.
  * Make file-system side effects opt-in via the `tmp_path` builtin.
"""

from __future__ import annotations

from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Isolate the global agent-memory singleton between tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Isolate the cached active PlatformProfile so tests can swap it freely
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_active_profile() -> Iterator[None]:
    """Reset the platforms cache before/after each test.

    We pre-seed the cache with ALPINE_RISCV so that any code path that calls
    get_active_profile() during a test does NOT trigger a real `docker exec`
    against a (likely non-running) container — which would block for ~10s
    per call and balloon the suite runtime.
    """
    from src import platforms

    saved = platforms._cached_profile
    try:
        platforms._cached_profile = platforms.ALPINE_RISCV
        yield
    finally:
        platforms._cached_profile = saved


# ---------------------------------------------------------------------------
# Safe helper: stub execute_command in a single module so tests
# don't fork docker
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_execute_command(monkeypatch):
    """Return a helper that stubs execute_command in any module.

    The returned helper replaces execute_command with a deterministic
    callable. Example::

        def fake(cmd, **kwargs):
            return CommandResult(cmd, 0, "ok", "", 0.0)
        stub_execute_command("src.scripted_ops", fake)
    """

    def _install(module_path: str, fake):
        """Install."""
        import importlib

        mod = importlib.import_module(module_path)
        monkeypatch.setattr(mod, "execute_command", fake)
        return fake

    return _install
