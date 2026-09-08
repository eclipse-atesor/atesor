#!/usr/bin/env python3
"""Batch test script to run the RISC-V Porting Agent on repos.

Packages are defined in JSON lists under ``.github/packages/`` (see the
README there for the schema). The default list is ``smoke.json`` — a
small, fast Go + C mix used as a sanity check before the full sweep.

Usage:
  python batch_test.py                            # run smoke list
  python batch_test.py --list full                # run full catalog
  python batch_test.py --list full anew gron      # filter by name
  python batch_test.py --list path/to/custom.json # absolute / relative path
"""

import argparse
import collections
import concurrent.futures
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any

# Default worker count auto-detected from the host CPU. Override with
# --workers <N> (1 .. _MAX_AVAILABLE_WORKERS). os.cpu_count() can return
# None on exotic platforms, so fall back to a conservative 2.
_MAX_AVAILABLE_WORKERS = os.cpu_count() or 2
MAX_WORKERS = _MAX_AVAILABLE_WORKERS
# RLock so a thread that already holds the lock (e.g. a future _emit caller
# inside a critical section) cannot self-deadlock.
_print_lock = threading.RLock()

# ----------------------------------------------------------------------------
# Pretty-printing helpers (presentation only — no impact on agent flow).
# ----------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    """Wrap ``text`` in an ANSI color code when stdout is a TTY."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return _c("1", text)


def _dim(text: str) -> str:
    return _c("2", text)


def _green(text: str) -> str:
    return _c("32", text)


def _red(text: str) -> str:
    return _c("31", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def _blue(text: str) -> str:
    return _c("34", text)


def _fmt_duration(seconds: float) -> str:
    """Render seconds as ``1h 23m 45s`` / ``12m 34s`` / ``42.1s``."""
    if seconds < 60:
        return f"{seconds:5.1f}s"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _short_container(name: str) -> str:
    """Strip the verbose base prefix so worker IDs read as ``w3``."""
    if name.startswith(_BASE_CONTAINER + "-"):
        return name[len(_BASE_CONTAINER) + 1 :]
    return name


_progress_counter = 0
_progress_total = 0
_progress_pass = 0
_progress_fail = 0
_progress_timeout = 0
_progress_active = 0
_progress_start = 0.0
_bar_installed = False
_TERM_ROWS = 24
_TERM_COLS = 80


def _progress_tag() -> str:
    """Return ``[ 12/172]`` style progress tag for the next completion."""
    width = len(str(_progress_total))
    return f"[{_progress_counter:>{width}}/{_progress_total}]"


def _term_size() -> tuple[int, int]:
    """Best-effort current terminal size as ``(rows, cols)``."""
    try:
        size = os.get_terminal_size()
        return size.lines, size.columns
    except OSError:
        return 24, 80


def _install_sticky_bar() -> None:
    """Reserve the bottom row for a sticky progress bar.

    Sets the DEC scroll region to rows ``1..H-1`` so subsequent prints
    scroll above the bar, then parks the cursor inside the region.
    """
    global _bar_installed, _TERM_ROWS, _TERM_COLS
    if not _USE_COLOR:
        return
    _TERM_ROWS, _TERM_COLS = _term_size()
    # Make room for the bar: emit a newline so the cursor isn't on the last
    # row when we shrink the scroll region (otherwise terminals scroll once).
    sys.stdout.write("\n")
    # Set scroll region [1, H-1], move cursor inside it, save state.
    sys.stdout.write(f"\033[1;{_TERM_ROWS - 1}r")
    sys.stdout.write(f"\033[{_TERM_ROWS - 1};1H")
    sys.stdout.flush()
    _bar_installed = True


def _remove_sticky_bar() -> None:
    """Restore the scroll region and clear the sticky bar row."""
    global _bar_installed
    if not _bar_installed:
        return
    rows, _ = _term_size()
    # Reset scroll region, clear bar line, move cursor to bottom.
    sys.stdout.write("\033[r")
    sys.stdout.write(f"\033[{rows};1H\033[2K")
    sys.stdout.flush()
    _bar_installed = False


def _redraw_bar() -> None:
    """Draw the sticky progress bar on the reserved bottom row."""
    if not _bar_installed:
        return
    rows, cols = _term_size()
    total = max(_progress_total, 1)
    done = _progress_counter
    pct = done / total

    # Stats prefix and ETA suffix sized first so the bar fills the middle.
    elapsed = time.time() - _progress_start
    eta_txt = ""
    if done > 0 and done < total:
        eta = elapsed / done * (total - done)
        eta_txt = f"  ETA {_fmt_duration(eta)}"

    prefix_plain = (
        f" {done}/{total}  "
        f"PASS {_progress_pass}  FAIL {_progress_fail}  "
        f"TIMEOUT {_progress_timeout}  RUN {_progress_active} "
    )
    prefix = (
        f" {_bold(f'{done}/{total}')}  "
        f"{_green('PASS')} {_progress_pass}  "
        f"{_red('FAIL')} {_progress_fail}  "
        f"{_yellow('TIMEOUT')} {_progress_timeout}  "
        f"{_cyan('RUN')} {_progress_active} "
    )
    suffix_plain = f" {pct * 100:5.1f}%{eta_txt} "
    suffix = f" {_bold(f'{pct * 100:5.1f}%')}{_dim(eta_txt)} "

    bar_w = max(cols - len(prefix_plain) - len(suffix_plain) - 2, 10)
    filled = int(bar_w * pct)
    bar_body = "\u2588" * filled + "\u2591" * (bar_w - filled)
    if pct >= 1.0:
        bar_render = _green(bar_body)
    elif pct >= 0.5:
        bar_render = _cyan(bar_body)
    else:
        bar_render = _blue(bar_body)

    line = f"{prefix}[{bar_render}]{suffix}"

    sys.stdout.write("\0337")  # save cursor
    sys.stdout.write(f"\033[{rows};1H")  # move to bar row
    sys.stdout.write("\033[2K")  # clear line
    sys.stdout.write(line)
    sys.stdout.write("\0338")  # restore cursor
    sys.stdout.flush()


def _emit(line: str) -> None:
    """Print ``line`` above the sticky bar and refresh the bar."""
    with _print_lock:
        print(line)
        _redraw_bar()


# ----------------------------------------------------------------------------
# Per-worker container pool
# ----------------------------------------------------------------------------
# Each batch worker reserves its own Docker sandbox container by passing
# --container atesor-ai-sandbox-w<N> to main.py. This eliminates contention on
# dpkg / apk locks that previously deadlocked parallel runs when every worker
# shared a single container. Containers are created on demand by
# setup_docker_environment() (same image, just different names) and persist
# between packages so the pool only pays the ~5 s startup cost once.
#
# To switch distro for the whole batch, set ATESOR_PLATFORM=debian (or pass
# --platform <name> below); the suffix is appended to the active profile's
# default container name.
# ----------------------------------------------------------------------------
PLATFORM = os.environ.get("ATESOR_PLATFORM", "debian").strip().lower()
_PLATFORM_CONTAINERS = {
    "alpine": "atesor-ai-sandbox",
    "debian": "atesor-ai-sandbox-debian",
    "ubuntu": "atesor-ai-sandbox-debian",
}
if PLATFORM not in _PLATFORM_CONTAINERS:
    # SystemExit instead of ValueError: an env-var typo should print a
    # one-line error, not a traceback.
    raise SystemExit(
        f"[ERROR] Unknown ATESOR_PLATFORM={PLATFORM!r}; "
        f"expected one of {sorted(_PLATFORM_CONTAINERS)}"
    )
_BASE_CONTAINER = _PLATFORM_CONTAINERS[PLATFORM]

# Container pool is sized in main() after --workers is parsed, so the
# default-vs-CLI value is honoured. _populate_container_pool() is
# idempotent and safe to call once at startup.
_container_pool: "queue.Queue[str]" = queue.Queue()


def _populate_container_pool(n: int) -> None:
    """Fill the worker container pool with ``n`` slots (w1..wN)."""
    while not _container_pool.empty():
        try:
            _container_pool.get_nowait()
        except queue.Empty:
            break
    for i in range(n):
        _container_pool.put(f"{_BASE_CONTAINER}-w{i + 1}")


# ----------------------------------------------------------------------------
# Package-list loading
# ----------------------------------------------------------------------------
# Packages live in JSON files under ../packages/ (i.e. .github/packages/,
# a sibling of this scripts/ dir). The CLI accepts either a bare name
# (resolved against <packages_dir>/<name>.json) or an explicit path.
# Per-list defaults (max_attempts, timeout_seconds) flow into the globals
# below so run_agent() doesn't need to thread them through.
# ----------------------------------------------------------------------------
_PACKAGES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages")
)
_DEFAULT_LIST = "smoke"
_SCHEMA_VERSION = 1

_MAX_ATTEMPTS = 5
_AGENT_TIMEOUT_SECONDS = 3600
_LIST_DESCRIPTION = ""
_PACKAGE_BUILDS = False


def _resolve_list_path(value: str) -> str:
    """Resolve a ``--list`` argument to an absolute JSON file path."""
    if os.path.sep in value or value.endswith(".json"):
        return os.path.abspath(value)
    return os.path.join(_PACKAGES_DIR, f"{value}.json")


def _load_package_list(
    path: str,
) -> tuple[list[tuple[str, str]], dict[str, Any], str]:
    """Load and validate a package list JSON file.

    Returns ``(packages, defaults, description)`` where ``packages`` is a
    list of ``(name, url)`` tuples in declaration order.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"[ERROR] Package list not found: {path}")

    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[ERROR] Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] {path}: top-level must be an object")

    version = data.get("$schema_version")
    if version != _SCHEMA_VERSION:
        raise SystemExit(
            f"[ERROR] {path}: unsupported $schema_version={version!r} "
            f"(expected {_SCHEMA_VERSION})"
        )

    raw_packages = data.get("packages")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise SystemExit(
            f"[ERROR] {path}: 'packages' must be a non-empty list"
        )

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for idx, entry in enumerate(raw_packages):
        if not isinstance(entry, dict):
            raise SystemExit(
                f"[ERROR] {path}: packages[{idx}] is not an object"
            )
        name = entry.get("name")
        url = entry.get("url")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"[ERROR] {path}: packages[{idx}] missing 'name'")
        if not isinstance(url, str) or not url:
            raise SystemExit(
                f"[ERROR] {path}: packages[{idx}] ({name}) missing 'url'"
            )
        if name in seen:
            raise SystemExit(
                f"[ERROR] {path}: duplicate package name {name!r}"
            )
        seen.add(name)
        out.append((name, url))

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SystemExit(
            f"[ERROR] {path}: 'defaults' must be an object if present"
        )

    description = data.get("description", "")
    if not isinstance(description, str):
        description = ""

    return out, defaults, description


BATCH_LOGS_DIR = "output/batch_logs"
_MIN_TOOLCHAIN = {
    "debian": {"go": "1.26.5", "cargo": "1.90.0"},
    "ubuntu": {"go": "1.26.5", "cargo": "1.90.0"},
    "alpine": {"go": "1.26.5", "cargo": "1.90.0"},
}
_STREAM_SETUP = os.environ.get("ATESOR_SETUP_STREAM", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _parse_semver(value: str) -> tuple[int, int, int] | None:
    """Parse semantic versions like ``1.26`` or ``1.26.3``."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _version_lt(current: str, minimum: str) -> bool:
    """Return True when ``current`` is lower than ``minimum``.

    Returns ``False`` (i.e. do NOT force a rebuild) when either value
    cannot be parsed — a parse failure is logged to stderr so the
    operator sees it instead of silently triggering an hours-long
    container rebuild on every run.
    """
    current_parsed = _parse_semver(current)
    minimum_parsed = _parse_semver(minimum)
    if current_parsed is None or minimum_parsed is None:
        print(
            f"[WARN] _version_lt: unparseable version "
            f"(current={current!r}, minimum={minimum!r}); skipping check.",
            file=sys.stderr,
        )
        return False
    return current_parsed < minimum_parsed


def _run_setup(container_name: str, rebuild: bool = False) -> tuple[bool, str]:
    """Run ``main.py --setup-only`` for a specific worker container."""
    cmd = [
        "python3",
        "main.py",
        "--setup-only",
        "--platform",
        PLATFORM,
        "--container",
        container_name,
    ]
    if rebuild:
        cmd.append("--rebuild")
    if _STREAM_SETUP:
        # Stream setup output live to help operators see progress during
        # long preflight setup/rebuild steps.
        result = subprocess.run(
            cmd,
            text=True,
            timeout=5400,
        )
        return result.returncode == 0, ""

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=5400,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, combined


def _probe_toolchains(container_name: str) -> dict[str, str]:
    """Read toolchain versions from an existing container."""
    out: dict[str, str] = {}
    commands = {
        "go": "go version",
        "cargo": "cargo --version",
    }
    for tool, probe in commands.items():
        result = subprocess.run(
            ["docker", "exec", container_name, "sh", "-lc", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = ((result.stdout or "") + " " + (result.stderr or "")).strip()
        if tool == "go":
            match = re.search(r"\bgo(\d+\.\d+(?:\.\d+)?)\b", text)
        else:
            match = re.search(r"\bcargo\s+(\d+\.\d+(?:\.\d+)?)\b", text)
        if match:
            out[tool] = match.group(1)
    return out


def _stale_reasons(
    versions: dict[str, str], minimums: dict[str, str]
) -> list[str]:
    """Return human-readable reasons for a required toolchain refresh."""
    reasons: list[str] = []
    for tool, required in minimums.items():
        current = versions.get(tool)
        if not current:
            reasons.append(
                f"{tool} is missing in worker container "
                f"(requires >= {required})"
            )
            continue
        if _version_lt(current, required):
            reasons.append(f"{tool} {current} < required {required}")
    return reasons


def _refresh_worker_pool_if_needed() -> bool:
    """Refresh worker containers when toolchains are stale."""
    minimums = _MIN_TOOLCHAIN.get(PLATFORM, {})
    if not minimums:
        return True

    sample_container = f"{_BASE_CONTAINER}-w1"
    print(
        f"[PREFLIGHT] Running setup check for sample container: "
        f"{sample_container}"
    )
    ok, output = _run_setup(sample_container, rebuild=False)
    if not ok:
        print("[ERROR] Batch preflight setup failed for sample container.")
        if output.strip():
            print(output[-1200:])
        return False

    versions = _probe_toolchains(sample_container)
    reasons = _stale_reasons(versions, minimums)
    if not reasons:
        return True

    print("[PREFLIGHT] Stale sandbox toolchain detected:")
    for reason in reasons:
        print(f"  - {reason}")
    print("[PREFLIGHT] Rebuilding image and refreshing worker containers...")

    ok, output = _run_setup(sample_container, rebuild=True)
    if not ok:
        print("[ERROR] Failed to rebuild sandbox image in preflight.")
        if output.strip():
            print(output[-1200:])
        return False

    worker_names = [f"{_BASE_CONTAINER}-w{i + 1}" for i in range(MAX_WORKERS)]
    for container_name in worker_names:
        print(f"[PREFLIGHT] Refreshing worker container: {container_name}")
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ok, output = _run_setup(container_name, rebuild=False)
        if not ok:
            print(
                f"[ERROR] Failed to recreate worker container: "
                f"{container_name}"
            )
            if output.strip():
                print(output[-1200:])
            return False

    print("[PREFLIGHT] Worker pool refreshed with current toolchains.")
    return True


def _kill_proc_group(pid: int) -> None:
    """Send SIGTERM to the process group; escalate to SIGKILL after 10 s."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    time.sleep(10)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _tail(path: str, n: int = 40) -> str:
    """Return the last *n* lines of a file (best-effort, bounded memory)."""
    try:
        with open(path, errors="replace") as fh:
            return "".join(collections.deque(fh, maxlen=n))
    except OSError:
        return ""


# ----------------------------------------------------------------------------
# Two-stage Ctrl-C handling
# ----------------------------------------------------------------------------
# First Ctrl-C  → soft shutdown: cancel queued futures, SIGTERM every live
#                 child process-group so running agents exit promptly, then
#                 let the executor drain so we can still print the summary.
# Second Ctrl-C → hard shutdown: SIGKILL every tracked PGID, drop the sticky
#                 bar, and exit immediately with code 130.
# ----------------------------------------------------------------------------
_live_pids: set[int] = set()
_live_lock = threading.Lock()
_shutdown_event = threading.Event()


def _register_pid(pid: int) -> None:
    with _live_lock:
        _live_pids.add(pid)


def _unregister_pid(pid: int) -> None:
    with _live_lock:
        _live_pids.discard(pid)


def _signal_all(sig: int) -> int:
    """Send ``sig`` to every tracked process-group; return count signalled."""
    with _live_lock:
        pids = list(_live_pids)
    sent = 0
    for pid in pids:
        try:
            pgid = os.getpgid(pid)
        except OSError:
            continue
        try:
            os.killpg(pgid, sig)
            sent += 1
        except OSError:
            pass
    return sent


def _install_signal_handlers(future_map: dict) -> None:
    """Install a two-stage SIGINT handler bound to ``future_map``.

    The handler runs on the main thread; a worker may hold ``_print_lock``
    at any moment, so status messages go straight to stdout instead of
    through ``_emit()`` (which would block on the lock).
    """

    def _write(msg: str) -> None:
        try:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def _handler(_signum, _frame):
        if not _shutdown_event.is_set():
            _shutdown_event.set()
            cancelled = sum(1 for f in future_map if f.cancel())
            sent = _signal_all(signal.SIGTERM)
            _write(
                f"\n  {_yellow('SOFT-STOP')} "
                f"{_dim('Ctrl-C received — cancelling queued jobs and ')}"
                f"{_dim('asking running agents to exit. ')}"
                f"{_dim('Press Ctrl-C again to force kill.')}"
            )
            _write(
                f"  {_dim(f'cancelled {cancelled} queued, ')}"
                f"{_dim(f'SIGTERM → {sent} running')}"
            )
            return
        # Second Ctrl-C → escalate to SIGKILL and bail out immediately.
        sent = _signal_all(signal.SIGKILL)
        _remove_sticky_bar()
        _write(
            f"\n  {_red('FORCE-KILL')} "
            f"SIGKILL → {sent} process group(s). Exiting now."
        )
        os._exit(130)

    signal.signal(signal.SIGINT, _handler)


def run_agent(repo_url: str, repo_name: str) -> tuple[bool, str, float]:
    """Run the agent on a single repository, streaming to a log file.

    Thread-safe: leases one container name from the per-worker pool, runs
    the agent against it, then returns the name so a later package can
    reuse the same container (warm apt/apk cache). Uses ``--container`` so
    ``main.py`` knows which sandbox to spawn/attach to.
    """
    start_time = time.time()
    log_path = os.path.join(BATCH_LOGS_DIR, f"{repo_name}.log")

    # If a soft shutdown was requested while this future was queued, bail
    # out cheaply before claiming a container slot.
    if _shutdown_event.is_set():
        return False, "Skipped: shutdown requested", 0.0

    container_name = _container_pool.get()  # blocks until a slot is free

    global _progress_active
    with _print_lock:
        _progress_active += 1
    _emit(
        f"  {_dim('START  ')} "
        f"{_cyan(repo_name):<40} "
        f"{_dim('on')} {_blue(_short_container(container_name)):<6} "
        f"{_dim('→ ' + log_path)}"
    )

    proc: subprocess.Popen | None = None
    try:
        os.makedirs(BATCH_LOGS_DIR, exist_ok=True)
        with open(log_path, "w") as log_fh:
            log_fh.write(
                f"=== {repo_name} | {repo_url} | container={container_name} | "
                f"{datetime.now().isoformat()} ===\n\n"
            )
            log_fh.flush()

            cmd = [
                "python3",
                "main.py",
                "--repo",
                repo_url,
                "--max-attempts",
                str(_MAX_ATTEMPTS),
                "--force",
                "--container",
                container_name,
            ]
            if PLATFORM in {"alpine", "debian", "ubuntu"}:
                cmd.extend(["--platform", PLATFORM])
            if _PACKAGE_BUILDS:
                cmd.append("--package")

            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=log_fh,
                text=True,
                preexec_fn=os.setsid,  # New group → kill children.
            )
            _register_pid(proc.pid)

            try:
                proc.wait(timeout=_AGENT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                mins = max(_AGENT_TIMEOUT_SECONDS // 60, 1)
                _emit(
                    f"  {_yellow('TIMEOUT')} {_cyan(repo_name):<40} "
                    f"{_dim('killing process group …')}"
                )
                _kill_proc_group(proc.pid)
                proc.wait()
                log_fh.write(f"\n\n=== TIMED OUT after {mins} minutes ===\n")
                duration = time.time() - start_time
                return (
                    False,
                    f"Timeout ({mins} min) — see {log_path}",
                    duration,
                )

        duration = time.time() - start_time
        success = proc.returncode == 0
        tail = _tail(log_path)
        return success, tail, duration

    except Exception as exc:
        duration = time.time() - start_time
        msg = f"Exception: {exc}"
        # Never hand the container back to the pool with the agent
        # still running in it — the next package would race a live
        # sibling for apt/apk locks and repo state.
        if proc is not None and proc.poll() is None:
            _kill_proc_group(proc.pid)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
        try:
            with open(log_path, "a") as lf:
                lf.write(f"\n\n=== UNHANDLED EXCEPTION: {exc} ===\n")
        except OSError:
            pass
        _emit(
            f"  {_red('ERROR  ')} {_cyan(repo_name):<40} " f"{_dim(str(exc))}"
        )
        return False, msg, duration
    finally:
        if proc is not None:
            _unregister_pid(proc.pid)
        _container_pool.put(container_name)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI: ``--list <name|path>`` + optional positional name filters."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the RISC-V porting agent across a list of packages "
            "defined in JSON (see .github/packages/)."
        ),
    )
    parser.add_argument(
        "--list",
        dest="list_name",
        default=_DEFAULT_LIST,
        help=(
            f"Package list to run. Bare name (resolved against "
            f".github/packages/<name>.json) or a path. "
            f"Default: {_DEFAULT_LIST}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_MAX_AVAILABLE_WORKERS,
        metavar="N",
        help=(
            f"Number of parallel worker containers. Must be in "
            f"1..{_MAX_AVAILABLE_WORKERS} (host CPU count). "
            f"Default: {_MAX_AVAILABLE_WORKERS}."
        ),
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help=(
            "Forward --package to each main.py invocation, producing a "
            "zip artifact (recipe + source tree) under workspace/packages/ "
            "for every successful build. Used by CI to upload downloadable "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        metavar="I",
        help=(
            "Zero-based index of this shard in the workload (requires "
            "--shard-total). The package list is contiguously sliced "
            "into --shard-total chunks of ceil(N/total) packages each, "
            "and this run processes shard I only. Used by the CI "
            "workflow to keep each job under the 6h limit."
        ),
    )
    parser.add_argument(
        "--shard-total",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total number of shards the package list is split into "
            "(requires --shard-index). Pass 1 to disable sharding."
        ),
    )
    parser.add_argument(
        "--platform",
        dest="platform",
        default=None,
        choices=sorted(_PLATFORM_CONTAINERS),
        help=(
            "Sandbox distro to run against. Overrides ATESOR_PLATFORM "
            "for this invocation and picks the matching container "
            "image + per-worker container names. "
            f"Default: value of ATESOR_PLATFORM (currently {PLATFORM!r})."
        ),
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="Optional package names to filter from the chosen list.",
    )
    return parser.parse_args(argv)


def _apply_shard(
    packages: list[tuple[str, str]],
    shard_index: int,
    shard_total: int,
) -> list[tuple[str, str]]:
    """Return the contiguous slice of ``packages`` for one shard.

    Splits ``packages`` into ``shard_total`` chunks of ``ceil(N/total)``
    items and returns chunk ``shard_index`` (0-based). With ``total=1``
    returns the full list unchanged.

    Caller is responsible for validating that 0 <= index < total.
    """
    if shard_total <= 1:
        return list(packages)
    n = len(packages)
    # ceil division so every shard except possibly the last is full-sized
    chunk = -(-n // shard_total)
    start = shard_index * chunk
    end = start + chunk
    return list(packages[start:end])


def main() -> int:
    """Run the configured packages through the agent in parallel."""
    args = _parse_args(sys.argv[1:])

    global PLATFORM, _BASE_CONTAINER
    if args.platform and args.platform != PLATFORM:
        PLATFORM = args.platform
        _BASE_CONTAINER = _PLATFORM_CONTAINERS[PLATFORM]
        # Downstream helpers read ATESOR_PLATFORM (e.g. main.py); keep
        # env and module state in sync so child processes agree with us.
        os.environ["ATESOR_PLATFORM"] = PLATFORM

    global MAX_WORKERS
    if args.workers < 1:
        print(
            f"[ERROR] --workers must be >= 1 (got {args.workers}).",
            file=sys.stderr,
        )
        return 2
    if args.workers > _MAX_AVAILABLE_WORKERS:
        print(
            f"[ERROR] --workers={args.workers} exceeds available CPU "
            f"count ({_MAX_AVAILABLE_WORKERS}).",
            file=sys.stderr,
        )
        return 2
    MAX_WORKERS = args.workers
    _populate_container_pool(MAX_WORKERS)

    global _PACKAGE_BUILDS
    _PACKAGE_BUILDS = bool(args.package)

    list_path = _resolve_list_path(args.list_name)
    all_packages, defaults, description = _load_package_list(list_path)

    global _MAX_ATTEMPTS, _AGENT_TIMEOUT_SECONDS, _LIST_DESCRIPTION
    _MAX_ATTEMPTS = int(defaults.get("max_attempts", _MAX_ATTEMPTS))
    _AGENT_TIMEOUT_SECONDS = int(
        defaults.get("timeout_seconds", _AGENT_TIMEOUT_SECONDS)
    )
    _LIST_DESCRIPTION = description

    # Validate sharding flags before anything else that depends on the
    # final package list. --shard-index and --shard-total must come as a
    # pair; combining them with positional name filters is rejected
    # because the resulting semantics (filter within a shard? across?)
    # would confuse users.
    shard_index = args.shard_index
    shard_total = args.shard_total
    if (shard_index is None) != (shard_total is None):
        print(
            "[ERROR] --shard-index and --shard-total must be provided "
            "together.",
            file=sys.stderr,
        )
        return 2
    if shard_total is not None:
        if shard_total < 1:
            print(
                f"[ERROR] --shard-total must be >= 1 (got {shard_total}).",
                file=sys.stderr,
            )
            return 2
        if shard_index < 0 or shard_index >= shard_total:
            print(
                f"[ERROR] --shard-index={shard_index} out of range "
                f"[0, {shard_total}).",
                file=sys.stderr,
            )
            return 2
        if args.names:
            print(
                "[ERROR] Positional name filters cannot be combined "
                "with --shard-index/--shard-total.",
                file=sys.stderr,
            )
            return 2
        sharded = _apply_shard(all_packages, shard_index, shard_total)
        print(
            f"[shard] {shard_index + 1}/{shard_total} of {list_path}: "
            f"{len(sharded)} of {len(all_packages)} packages "
            f"(group size = {-(-len(all_packages) // shard_total)})",
            flush=True,
        )
        all_packages = sharded
        if not all_packages:
            print(
                "[shard] empty shard, nothing to run; exiting 0.",
                file=sys.stderr,
            )
            return 0

    filter_names = set(args.names)
    if filter_names:
        packages_to_run = [
            (n, u) for n, u in all_packages if n in filter_names
        ]
        missing = filter_names - {n for n, _ in packages_to_run}
        if missing:
            print(
                f"[ERROR] Names not in {list_path}: "
                f"{', '.join(sorted(missing))}",
                file=sys.stderr,
            )
            return 2
    else:
        packages_to_run = all_packages

    if not packages_to_run:
        print("[ERROR] No packages to run.", file=sys.stderr)
        return 2

    os.makedirs(BATCH_LOGS_DIR, exist_ok=True)
    if not _refresh_worker_pool_if_needed():
        return 1

    global _progress_total, _progress_counter, _progress_start
    global _progress_pass, _progress_fail, _progress_timeout, _progress_active
    _progress_total = len(packages_to_run)
    _progress_counter = 0
    _progress_start = time.time()

    started_at = time.time()
    started_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs_abs = os.path.abspath(BATCH_LOGS_DIR)
    home = os.path.expanduser("~")
    logs_disp = (
        "~" + logs_abs[len(home) :]
        if logs_abs.startswith(home + "/")
        else logs_abs
    )

    rows: list[tuple[str, str]] = [
        ("Started", started_human),
        (
            "List",
            f"{os.path.basename(list_path)}  ({len(packages_to_run)} pkgs)",
        ),
        ("Packages", str(len(packages_to_run))),
        ("Workers", str(MAX_WORKERS)),
        ("Platform", f"{PLATFORM}  (image: {_BASE_CONTAINER})"),
        (
            "Pool",
            f"{_BASE_CONTAINER}-w1 .. {_BASE_CONTAINER}-w{MAX_WORKERS}",
        ),
        ("Logs", logs_disp),
    ]
    if _LIST_DESCRIPTION:
        rows.insert(2, ("Desc", _LIST_DESCRIPTION))
    title = "ATESOR AI - RISC-V PORTING AGENT "
    # Auto-size to longest line; cap so we don't run past narrow terminals.
    _, term_cols = _term_size()
    inner = max(
        len(title) + 4,
        max(len(f" {lbl:<10} {val}") for lbl, val in rows) + 2,
    )
    inner = min(inner, max(term_cols - 2, 40))

    print(_bold("╔" + "═" * inner + "╗"))
    print(_bold("║") + _bold(_cyan(title.center(inner))) + _bold("║"))
    print(_bold("╠" + "═" * inner + "╣"))

    def _row(label: str, value: str) -> None:
        line = f" {label:<10} {value}"
        if len(line) > inner:
            line = line[: inner - 1] + "…"
        pad = inner - len(line)
        print(_bold("║") + line + " " * max(pad, 0) + _bold("║"))

    for lbl, val in rows:
        _row(lbl, val)
    print(_bold("╚" + "═" * inner + "╝"))
    print()

    _install_sticky_bar()
    _redraw_bar()

    results: list[tuple[str, bool, float, str]] = []

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            future_to_pkg = {
                executor.submit(run_agent, url, name): (name, url)
                for name, url in packages_to_run
            }
            _install_signal_handlers(future_to_pkg)
            for future in concurrent.futures.as_completed(future_to_pkg):
                name, _url = future_to_pkg[future]
                if future.cancelled():
                    continue
                try:
                    success, output, duration = future.result()
                except concurrent.futures.CancelledError:
                    continue
                except Exception as exc:
                    success, output, duration = (
                        False,
                        f"Unhandled exception: {exc}",
                        0.0,
                    )
                if output == "Skipped: shutdown requested":
                    continue

                results.append((name, success, duration, output))
                log_path = os.path.join(BATCH_LOGS_DIR, f"{name}.log")
                is_timeout = not success and output.startswith("Timeout")
                with _print_lock:
                    _progress_counter += 1
                    _progress_active = max(_progress_active - 1, 0)
                    if success:
                        _progress_pass += 1
                    elif is_timeout:
                        _progress_timeout += 1
                    else:
                        _progress_fail += 1
                if success:
                    badge = _green("✔ PASS   ")
                elif is_timeout:
                    badge = _yellow("⏱ TIMEOUT")
                else:
                    badge = _red("✘ FAIL   ")
                _emit(
                    f"{_dim(_progress_tag())} {badge} "
                    f"{_cyan(name):<40} "
                    f"{_fmt_duration(duration):>10}  "
                    f"{_dim(log_path)}"
                )
    finally:
        _remove_sticky_bar()

    if _shutdown_event.is_set():
        print(
            f"\n  {_yellow('SOFT-STOP')} "
            f"{_dim('all running agents drained; printing summary so far.')}"
        )

    passed = sum(1 for _, s, _, _ in results if s)
    failed = len(results) - passed
    timeouts = sum(
        1 for _, s, _, o in results if (not s and o.startswith("Timeout"))
    )
    errors = failed - timeouts
    elapsed = time.time() - started_at

    # Sort: failures first, then by duration desc, keeps eyes on what matters.
    sorted_results = sorted(results, key=lambda r: (r[1], -r[2]))

    # ---- Results table ----
    col_idx, col_status, col_pkg, col_dur = 4, 9, 36, 12
    table_w = 2 + col_idx + 3 + col_status + 3 + col_pkg + 3 + col_dur + 2

    print()
    print(_bold("┌" + "─" * (table_w - 2) + "┐"))
    header = (
        f" {'#':>{col_idx}} │ {'STATUS':<{col_status}} │ "
        f"{'PACKAGE':<{col_pkg}} │ {'DURATION':>{col_dur}} "
    )
    print(_bold("│") + _bold(header) + _bold("│"))
    print(
        _bold("├")
        + "─" * (col_idx + 2)
        + "┼"
        + "─" * (col_status + 2)
        + "┼"
        + "─" * (col_pkg + 2)
        + "┼"
        + "─" * (col_dur + 2)
        + _bold("┤")
    )

    for i, (name, success, duration, output) in enumerate(sorted_results, 1):
        is_timeout = not success and output.startswith("Timeout")
        if success:
            status_txt = _green("✔ PASS".ljust(col_status))
        elif is_timeout:
            status_txt = _yellow("⏱ TIMEOUT".ljust(col_status))
        else:
            status_txt = _red("✘ FAIL".ljust(col_status))
        name_disp = name if len(name) <= col_pkg else name[: col_pkg - 1] + "…"
        print(
            _bold("│")
            + f" {i:>{col_idx}} "
            + _bold("│")
            + f" {status_txt} "
            + _bold("│")
            + f" {name_disp:<{col_pkg}} "
            + _bold("│")
            + f" {_fmt_duration(duration):>{col_dur}} "
            + _bold("│")
        )

    print(_bold("└" + "─" * (table_w - 2) + "┘"))

    # ---- Stats panel ----
    rate = (passed / len(results) * 100) if results else 0.0
    rate_color = _green if rate >= 80 else (_yellow if rate >= 50 else _red)

    print()
    print(_bold("╔" + "═" * (table_w - 2) + "╗"))

    def _stat_row(text: str) -> None:
        pad = table_w - 2 - len(text)
        print(_bold("║") + " " + text + " " * max(pad - 1, 0) + _bold("║"))

    def _stat_row_colored(plain: str, colored: str) -> None:
        pad = table_w - 2 - len(plain)
        print(_bold("║") + " " + colored + " " * max(pad - 1, 0) + _bold("║"))

    summary_plain = (
        f"TOTAL {len(results)}   PASS {passed}   "
        f"FAIL {errors}   TIMEOUT {timeouts}"
    )
    summary_colored = (
        f"{_bold('TOTAL')} {len(results)}   "
        f"{_green('PASS')} {passed}   "
        f"{_red('FAIL')} {errors}   "
        f"{_yellow('TIMEOUT')} {timeouts}"
    )
    _stat_row_colored(summary_plain, summary_colored)

    rate_plain = f"SUCCESS RATE   {rate:5.1f}%"
    rate_colored = f"{_bold('SUCCESS RATE')}   {rate_color(f'{rate:5.1f}%')}"
    _stat_row_colored(rate_plain, rate_colored)

    _stat_row(f"ELAPSED        {_fmt_duration(elapsed)}")
    _stat_row(f"FINISHED       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _stat_row(f"LOGS           {logs_abs}")
    print(_bold("╚" + "═" * (table_w - 2) + "╝"))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
