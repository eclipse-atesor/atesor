#!/usr/bin/env python3
"""Main entry point for Atesor AI.

Handles CLI, Docker setup, and starts the multi-agent porting
workflow.
"""

import argparse
import fcntl
import logging
import os
import sys
import time
from datetime import datetime

import docker
from dotenv import load_dotenv
from termcolor import colored

# Load environment variables before importing project modules so that
# provider configuration is available at import time. The current
# directory is searched first, then standard per-user locations, so an
# installed CLI finds keys regardless of the working directory.
load_dotenv()
_home = os.path.expanduser("~")
_xdg_config = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
    _home, ".config"
)
_xdg_data = os.environ.get("XDG_DATA_HOME") or os.path.join(
    _home, ".local", "share"
)
_env_candidates = [
    os.path.join(_xdg_config, "atesor-ai", ".env"),
    os.path.join(_xdg_data, "atesor-ai", ".env"),
]
if os.environ.get("ATESOR_HOME"):
    _env_candidates.insert(0, os.path.join(os.environ["ATESOR_HOME"], ".env"))
for _env_path in _env_candidates:
    load_dotenv(_env_path, override=False)

from src import __version__  # noqa: E402
from src.config import (  # noqa: E402
    LOGS_DIR,
    OUTPUT_DIR,
    PACKAGES_DIR,
    REPOS_DIR,
    WORKSPACE_ROOT,
)
from src.models import check_api_keys, print_model_info  # noqa: E402
from src.state import AgentState  # noqa: E402

# Directory holding bundled resources (Dockerfiles, seed data), resolved
# from the install location so image builds work regardless of the CWD.
APP_DIR = os.path.dirname(os.path.abspath(__file__))


# Configure logging
def configure_logging(verbose: bool, repo_name: str = "") -> None:
    """Configure root logging handlers for console and file output.

    Args:
        verbose: If True, raise the console handler level to DEBUG.
        repo_name: Optional repository name used to derive a per-repo
            log file path.
    """
    # Capture everything at root level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Create a custom formatter for better readability
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    # Remove existing handlers to avoid duplicates. Close each one as
    # well: FileHandler keeps its log file open, so removing without
    # closing leaks one file descriptor per reconfiguration.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # 1. File Handler (Always Capture Everything)
    # Use per-repo log files to avoid cross-process corruption in batch mode
    log_filename = f"agent_{repo_name}.log" if repo_name else "agent.log"
    log_file = os.path.join(LOGS_DIR, log_filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # 2. Console Handler (Respect Verbose Flag)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        # User requested no in-depth logs in terminal unless verbose is set
        console_handler.setLevel(logging.ERROR)

    root_logger.addHandler(console_handler)

    # Reduce noise from sensitive/chatty packages
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langsmith").setLevel(logging.ERROR)
    logging.getLogger("google.ai").setLevel(logging.WARNING)

    # Ensure stdout is flushed
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)


logger = logging.getLogger(__name__)


def check_keys() -> bool:
    """Verify required API keys are set."""
    is_valid, msg, provider = check_api_keys()
    if not is_valid:
        print(colored(f"ERROR: {msg}", "red"))
        return False

    print(colored(f"{msg}", "green"))
    print_model_info()
    return True


def _ensure_riscv64_binfmt() -> bool:
    """Verify the host's binfmt_misc handler for riscv64.

    Auto-registers the handler if possible. Without it, every riscv64
    container exits immediately with
    ``exec /bin/<cmd>: exec format error`` and the agent silently loops
    recreating containers — observed in the 2026-05-23 batch run, where
    all 172 packages failed within seconds for this single reason.

    Returns:
        True if the handler is present after this call, False
        otherwise. Auto-registration is attempted exactly once via the
        standard ``tonistiigi/binfmt`` installer image (the canonical
        fix documented in Docker's multi-arch guides). If that fails,
        the caller prints the manual command.
    """
    import subprocess

    binfmt_path = "/proc/sys/fs/binfmt_misc/qemu-riscv64"
    if os.path.exists(binfmt_path):
        try:
            with open(binfmt_path) as f:
                if "enabled" in f.read():
                    return True
        except OSError:
            pass

    print(
        colored(
            "qemu-riscv64 binfmt handler not detected; attempting to register "
            "(one-time, requires --privileged docker access)...",
            "yellow",
        )
    )
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--privileged",
                "--rm",
                "tonistiigi/binfmt",
                "--install",
                "riscv64",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                f"binfmt install exit={result.returncode}: "
                f"{result.stderr[:300]}"
            )
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"binfmt install failed: {e}")
        return False

    # Re-check
    if os.path.exists(binfmt_path):
        print(colored("qemu-riscv64 binfmt handler registered", "green"))
        return True
    return False


def setup_docker_environment() -> bool:
    """Set up the Docker environment for RISC-V development.

    Image / container / Dockerfile are selected based on the active platform
    profile (Alpine or Debian). A self-check confirms the running container's
    `/etc/os-release` matches the selected profile, refusing to proceed on
    mismatch (the most common cause: user passes --platform debian but is
    still pointing at the legacy Alpine container).
    """
    from src.platforms import get_active_profile, get_container_name

    profile = get_active_profile()
    image_name = profile.image_name
    container_name = get_container_name()
    dockerfile = profile.dockerfile

    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        print(colored("ERROR: Docker is not running or not accessible", "red"))
        print(colored(f"Details: {e}", "yellow"))
        return False

    # The docker-py client owns a urllib3 connection pool on the Docker
    # socket; close it on every exit path so repeated setups (e.g.
    # rebuild_all_sandboxes, batch preflight) do not accumulate open FDs.
    try:
        return _provision_sandbox(
            client, profile, image_name, container_name, dockerfile
        )
    finally:
        client.close()


def _provision_sandbox(
    client: "docker.DockerClient",
    profile,
    image_name: str,
    container_name: str,
    dockerfile: str,
) -> bool:
    """Serialize sandbox provisioning across concurrent workers.

    Takes an exclusive host-side flock around the WHOLE provisioning
    sequence (binfmt registration, image existence check + build,
    container create/start/health-check). batch_test.py runs up to
    --workers N ``main.py`` processes at once; without the lock
    covering the image check, a cold start races N identical 15-45 min
    QEMU image builds (one per worker) plus N concurrent binfmt
    installer containers. The previous lock only covered container
    creation. Whoever wins the lock does the expensive work once; the
    waiters then see the image/container already present and continue
    in seconds.
    """
    lock_path = os.path.join(LOGS_DIR, ".container_setup.lock")
    os.makedirs(LOGS_DIR, exist_ok=True)
    lock_fd = open(lock_path, "w")
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(
                colored(
                    "   Another worker is provisioning the sandbox "
                    "(image build / container setup); waiting...",
                    "yellow",
                )
            )
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return _provision_sandbox_locked(
            client, profile, image_name, container_name, dockerfile
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _provision_sandbox_locked(
    client: "docker.DockerClient",
    profile,
    image_name: str,
    container_name: str,
    dockerfile: str,
) -> bool:
    """Build the sandbox image if needed and start the container.

    The caller (`_provision_sandbox`) holds the setup flock for the
    whole duration and closes the Docker client afterwards.
    """
    # Preflight: the host must have qemu-riscv64 registered in
    # binfmt_misc, otherwise every riscv64 container exits immediately
    # with "exec format error" before the agent can run anything. The
    # Debian image is riscv64-native so this is mandatory; the Alpine
    # image is too, just historically less likely to hit this because
    # docker-desktop ships binfmt by default. Auto-register if we can,
    # otherwise emit a clear instruction and exit.
    if not _ensure_riscv64_binfmt():
        print(
            colored(
                "ERROR: qemu-riscv64 is not registered in "
                "/proc/sys/fs/binfmt_misc. "
                "Without it, every riscv64 container exits with "
                "'exec format error'.",
                "red",
            )
        )
        print(
            colored(
                "Fix: docker run --privileged --rm tonistiigi/binfmt "
                "--install riscv64",
                "yellow",
            )
        )
        return False

    # Check/Build Docker Image
    print(
        colored(
            f"\nSetting up RISC-V development environment "
            f"({profile.display_name})...",
            "cyan",
        )
    )

    force_rebuild = os.environ.get("REBUILD_IMAGE") == "true"

    try:
        if force_rebuild:
            print(
                colored(
                    f"Forcing rebuild of image '{image_name}'...", "yellow"
                )
            )
            raise docker.errors.ImageNotFound("Triggering rebuild")

        client.images.get(image_name)
        print(colored(f"Image '{image_name}' found", "green"))
    except docker.errors.ImageNotFound:
        print(
            colored(
                f"   Building image '{image_name}' from {dockerfile}...",
                "yellow",
            )
        )
        print(
            colored(
                "   Note: RISC-V images build under QEMU emulation. "
                "Expect 15-45 minutes "
                "on first build (subsequent runs reuse the cached image).",
                "yellow",
            )
        )
        try:
            # Use low-level API to stream build progress so the user
            # sees activity
            build_stream = client.api.build(
                path=APP_DIR,
                dockerfile=dockerfile,
                tag=image_name,
                rm=True,
                forcerm=True,
                pull=True,
                platform="linux/riscv64",
                decode=True,
            )
            try:
                for chunk in build_stream:
                    if "stream" in chunk:
                        line = chunk["stream"].rstrip()
                        if not line:
                            continue
                        # Show step headers prominently; condense noisy
                        # lines
                        if line.startswith("Step "):
                            print(colored(f"   {line}", "cyan"))
                        else:
                            # Keep the user feeling alive: print a dot
                            # per log line, full text every 50
                            print(f"   {line}")
                    elif "errorDetail" in chunk:
                        raise docker.errors.BuildError(
                            chunk["errorDetail"].get(
                                "message", "build failed"
                            ),
                            build_stream,
                        )
                    elif "error" in chunk:
                        raise docker.errors.BuildError(
                            chunk["error"], build_stream
                        )
            finally:
                # The generator wraps the build HTTP response; close it
                # so an error raised mid-stream does not leak the
                # socket. Guarded: docker-py returns a generator, but
                # plain iterables (fakes, future API changes) have no
                # close().
                close_stream = getattr(build_stream, "close", None)
                if close_stream is not None:
                    close_stream()
            print(colored("Image built successfully", "green"))
        except docker.errors.BuildError as e:
            print(colored("ERROR: Failed to build Docker image", "red"))
            print(colored(f"{e}", "yellow"))
            return False
        except Exception as e:
            print(colored(f"ERROR: Build stream error: {e}", "red"))
            return False

    # Create/Start Container (the caller holds the setup flock, so
    # concurrent workers cannot race the create/recreate sequence).
    container = None
    for attempt in range(2):
        try:
            try:
                container = client.containers.get(container_name)
                # A container created before the workspace path
                # changed (e.g. after install/upgrade) keeps its old
                # bind mount and would operate on stale files while the
                # host-side checks look at the new location. Recreate it
                # when the /workspace source no longer matches.
                _expected_ws = os.path.realpath(WORKSPACE_ROOT)
                _ws_src = next(
                    (
                        m.get("Source")
                        for m in container.attrs.get("Mounts", [])
                        if m.get("Destination") == "/workspace"
                    ),
                    None,
                )
                if _ws_src and os.path.realpath(_ws_src) != _expected_ws:
                    print(
                        colored(
                            f"   Container '{container_name}' has a "
                            f"stale workspace mount ({_ws_src}); "
                            f"recreating for {_expected_ws}...",
                            "yellow",
                        )
                    )
                    container.remove(force=True)
                    raise docker.errors.NotFound("stale workspace mount")
                if container.status != "running":
                    print(
                        colored(
                            f"   Starting existing container "
                            f"'{container_name}'...",
                            "yellow",
                        )
                    )
                    container.start()
                    time.sleep(2)
                print(
                    colored(
                        f"Container '{container_name}' is running", "green"
                    )
                )
            except docker.errors.NotFound:
                print(
                    colored(
                        f"   Creating container '{container_name}'...",
                        "yellow",
                    )
                )

                os.makedirs(OUTPUT_DIR, exist_ok=True)

                container = client.containers.run(
                    image_name,
                    name=container_name,
                    detach=True,
                    tty=True,
                    platform="linux/riscv64",
                    network_mode="bridge",
                    dns=["8.8.8.8", "8.8.4.4"],
                    volumes={
                        os.path.abspath(WORKSPACE_ROOT): {
                            "bind": "/workspace",
                            "mode": "rw",
                        }
                    },
                    mem_limit="8g",
                    cpu_quota=-1,
                )
                print(colored("Container created and started", "green"))
                time.sleep(5)

            container.reload()
            time.sleep(1)
            if container.status != "running":
                print(
                    colored(
                        f"   Container status: {container.status}, "
                        f"recreating...",
                        "yellow",
                    )
                )
                try:
                    container.stop()
                    container.remove()
                except Exception as e:
                    logger.warning(f"Cleanup failed: {e}")
                continue

            break
        except docker.errors.NotFound:
            continue

    if container is None or container.status != "running":
        print(
            colored("ERROR: Failed to start container after 2 attempts", "red")
        )
        return False

    print(colored(f"   Container ID: {container.short_id}", "cyan"))

    logger.info(
        f"Container '{container_name}' is running with ID: "
        f"{container.short_id}"
    )
    logger.info("Workspace mounted at /workspace in container")

    # ---- Self-check: container's actual distro must match the
    # selected profile ----
    try:
        osr = container.exec_run("cat /etc/os-release")
        if osr.exit_code == 0:
            text = osr.output.decode("utf-8", errors="ignore")
            container_id = ""
            for line in text.splitlines():
                if line.startswith("ID="):
                    container_id = (
                        line.split("=", 1)[1].strip().strip('"').lower()
                    )
                    break
            # ubuntu profile is satisfied by either "ubuntu" or
            # "debian" (same package map)
            expected = {
                "alpine": {"alpine"},
                "debian": {"debian", "ubuntu"},
                "ubuntu": {"debian", "ubuntu"},
            }
            allowed = expected.get(profile.name, {profile.name})
            if container_id and container_id not in allowed:
                print(
                    colored(
                        f"ERROR: Platform mismatch — container "
                        f"'{container_name}' is "
                        f"'{container_id}' but --platform "
                        f"'{profile.name}' was selected.",
                        "red",
                    )
                )
                print(
                    colored(
                        f"  Fix: stop & remove the stale container, "
                        f"then re-run.\n"
                        f"    docker rm -f {container_name}\n"
                        f"  Or pick a profile that matches the "
                        f"container's distro.",
                        "yellow",
                    )
                )
                return False
            logger.info(
                f"Container distro check OK: {container_id} ∈ "
                f"{sorted(allowed)}"
            )
        else:
            logger.warning(
                f"Could not read /etc/os-release from container "
                f"(exit {osr.exit_code})"
            )
    except Exception as e:
        logger.warning(f"Distro self-check skipped: {e}")

    # Verify container is working and architecture is correct
    try:
        # Check architecture
        arch_result = container.exec_run("uname -m")
        if arch_result.exit_code == 0:
            arch = arch_result.output.decode("utf-8").strip()
            if "riscv64" in arch:
                print(
                    colored(
                        f"Container architecture verified: {arch} "
                        f"(Correctly Emulated)",
                        "green",
                    )
                )
            else:
                print(
                    colored(
                        f"WARNING: Container architecture is {arch} "
                        f"(Expected riscv64!)",
                        "yellow",
                    )
                )
                print(
                    colored(
                        "Try running: docker run --privileged --rm "
                        "tonistiigi/binfmt --install all",
                        "yellow",
                    )
                )
        else:
            print(
                colored("ERROR: Container is not responding correctly", "red")
            )
            return False

        # Simple health check
        exec_result = container.exec_run("echo 'Container ready!'")
        if exec_result.exit_code == 0:
            print(colored("Container is responsive", "green"))
        else:
            print(
                colored("ERROR: Container is not responding correctly", "red")
            )
            return False
    except Exception as e:
        print(colored(f"ERROR: Container health check failed: {e}", "red"))

        # Try to recreate container on health check failure
        print(colored("   Attempting to recreate container...", "yellow"))
        try:
            container.stop()
            container.remove()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        # Retry setup
        container = None
        for attempt in range(2):
            try:
                print(
                    colored(
                        f"   Creating container '{container_name}' "
                        f"(attempt {attempt + 1})...",
                        "yellow",
                    )
                )
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                container = client.containers.run(
                    image_name,
                    name=container_name,
                    detach=True,
                    tty=True,
                    platform="linux/riscv64",
                    network_mode="bridge",
                    dns=["8.8.8.8", "8.8.4.4"],
                    volumes={
                        os.path.abspath(WORKSPACE_ROOT): {
                            "bind": "/workspace",
                            "mode": "rw",
                        }
                    },
                    mem_limit="8g",
                    cpu_quota=-1,
                )
                print(colored("Container created and started", "green"))
                time.sleep(5)

                container.reload()
                time.sleep(1)
                if container.status != "running":
                    continue

                exec_result = container.exec_run("echo 'Container ready!'")
                if exec_result.exit_code == 0:
                    print(
                        colored(
                            "Container is responsive after recreation", "green"
                        )
                    )
                    return True
            except Exception as recreate_error:
                print(
                    colored(
                        f"   Recreation attempt failed: {recreate_error}",
                        "yellow",
                    )
                )
                if container:
                    try:
                        container.stop()
                        container.remove()
                    except Exception as e:
                        logger.warning(f"Cleanup failed: {e}")

        return False

    return True


def save_porting_outputs(state: AgentState, output_dir: str) -> None:
    """Save all porting outputs to files."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    repo_name = state.repo_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 0. Save complete state (JSON)
    state_path = os.path.join(
        output_dir, f"{repo_name}_state_{timestamp}.json"
    )
    try:
        state.save_to_json(state_path)
        print(colored(f"Full state saved: {state_path}", "green"))
    except Exception as e:
        logger.error(f"Failed to save state JSON: {e}")

    # 1. Save porting recipe
    if state.porting_recipe:
        recipe_path = os.path.join(output_dir, f"{repo_name}_recipe.md")
        with open(recipe_path, "w") as f:
            f.write(state.porting_recipe)
        print(colored(f"Porting recipe saved: {recipe_path}", "green"))

    # 2. Save detailed build report
    report = generate_detailed_report(state.to_dict())
    report_path = os.path.join(
        output_dir, f"{repo_name}_report_{timestamp}.md"
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(colored(f"Detailed report saved: {report_path}", "green"))

    # 3. Save patches
    if state.patches_generated:
        patches_dir = os.path.join(
            output_dir, f"{repo_name}_patches_{timestamp}"
        )
        os.makedirs(patches_dir, exist_ok=True)
        for i, patch in enumerate(state.patches_generated):
            patch_path = os.path.join(patches_dir, f"patch_{i + 1}.patch")
            with open(patch_path, "w") as f:
                f.write(patch)
        print(
            colored(
                f"{len(state.patches_generated)} patch(es) saved: "
                f"{patches_dir}/",
                "green",
            )
        )


def generate_detailed_report(state: dict) -> str:
    """Generate a comprehensive detailed report."""
    tokens = (
        f"{state.get('api_tokens_in', 0)}/{state.get('api_tokens_out', 0)}"
    )
    report = f"""# RISC-V Porting Report: {state.get("repo_name", "Unknown")}

## Executive Summary

**Status**: {state.get("build_status", "UNKNOWN")}\u0020\u0020
**Repository**: {state.get("repo_url", "N/A")}\u0020\u0020
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Metrics

| Metric | Value |
|--------|-------|
| Build Status | {state.get("build_status", "N/A")} |
| Total Duration | {state.get("execution_duration", 0):.1f}s |
| Attempts Made | {state.get("attempt_count", 0)} |
| API Calls | {state.get("api_calls_made", 0)} |
| Scripted Operations | {state.get("scripted_ops_count", 0)} |
| LLM Tokens (in/out) | {tokens} |
| Cost (from token usage) | ${state.get("api_cost_usd", 0):.4f} |

---

## Build System Analysis

"""

    build_plan = state.get("build_plan")
    if build_plan:
        report += (
            f"**Build System**: "
            f"{build_plan.get('build_system', 'Unknown')}\n\n"
        )

        phases = build_plan.get("phases", [])
        if phases:
            report += "### Build Phases\n\n"
            for i, phase in enumerate(phases, 1):
                phase_name = phase.get("name", f"Phase {i}")
                report += f"#### {i}. {phase_name}\n\n"

                commands = phase.get("commands", [])
                if commands:
                    report += "```bash\n"
                    for cmd in commands:
                        report += f"{cmd}\n"
                    report += "```\n\n"
    else:
        report += "*No build plan available*\n\n"

    # Dependencies
    dependencies = state.get("dependencies")
    if dependencies:
        report += "### Dependencies\n\n"

        build_tools = dependencies.get("build_tools", [])
        if build_tools:
            report += f"**Build Tools**: {', '.join(build_tools)}\n\n"

        system_packages = dependencies.get("system_packages", [])
        if system_packages:
            report += f"**System Packages**: {', '.join(system_packages)}\n\n"

    # Architecture-specific code
    arch_code = state.get("arch_specific_code", [])
    if arch_code:
        report += "---\n\n## Architecture-Specific Code\n\n"
        report += (
            f"Found {len(arch_code)} instances of "
            f"architecture-specific code:\n\n"
        )

        for i, code in enumerate(arch_code[:5], 1):  # Show first 5
            filepath = code.get("file", "Unknown")
            line_num = code.get("line", "N/A")
            snippet = code.get("code_snippet", "")

            report += f"### {i}. {filepath}:{line_num}\n\n"
            report += f"```c\n{snippet[:200]}\n```\n\n"

        if len(arch_code) > 5:
            report += f"*...and {len(arch_code) - 5} more instances*\n\n"

    # Patches
    patches = state.get("patches_generated", [])
    if patches:
        report += "---\n\n## Patches Applied\n\n"
        report += (
            f"{len(patches)} patch(es) were needed for RISC-V "
            f"compatibility:\n\n"
        )

        for i, patch in enumerate(patches, 1):
            report += f"### Patch {i}\n\n"
            report += f"```diff\n{patch[:500]}\n```\n\n"
            if len(patch) > 500:
                report += (
                    "*Truncated (full patch available in separate file)*\n\n"
                )

    # Error history
    error_log = state.get("error_log", [])
    if error_log:
        report += "---\n\n## Error Resolution History\n\n"

        for i, error in enumerate(error_log, 1):
            category = getattr(error, "category", "Unknown")
            message = getattr(error, "message", "N/A")

            report += f"### Error {i}: {category}\n\n"
            report += f"```\n{message[:300]}\n```\n\n"

    # Recommendations
    report += "---\n\n## Recommendations\n\n"

    status = state.get("build_status")
    if status == "SUCCESS":
        report += """
- Build completed successfully for RISC-V
- Test the binary on actual RISC-V hardware or QEMU
- Consider submitting patches upstream if modifications were needed
- Add RISC-V to your CI/CD pipeline
"""
    elif status == "ESCALATED":
        escalation_reason = state.get("escalation_reason", "Unknown")
        report += f"""
- Manual intervention required
- Reason: {escalation_reason}
- Review the error logs above
- Consider consulting RISC-V porting documentation
"""
    else:
        report += """
- Build did not complete successfully
- Review error logs above
- Check dependencies and build system compatibility
- Consider filing an issue with the upstream project
"""

    report += "\n---\n\n"
    report += "*Report generated by RISC-V Porting Foundry v1.0*\n"

    return report


def run_agent(
    repo_url: str,
    max_attempts: int = 5,
    verbose: bool = False,
    package: bool = False,
) -> int:
    """Run the RISC-V porting agent on a repository.

    Args:
        repo_url: The GitHub/GitLab repository URL
        max_attempts: Maximum fix attempts before escalation
        verbose: Enable verbose output
        package: If True, on a successful build, write a zip artifact to
            ``PACKAGES_DIR`` containing the recipe + source tree. Failure
            to package is treated as a fatal error (exit 2) when this is
            requested explicitly.

    Returns:
        Exit code (0 for success, 1 for build failure, 2 for packaging failure)
    """
    from langchain_core.messages import HumanMessage

    from src.graph import app
    from src.platforms import get_active_profile
    from src.state import BuildStatus, create_initial_state

    profile = get_active_profile()

    logger.info("Starting RISC-V Porting Agent")
    logger.info(f"Repository: {repo_url}")
    logger.info(f"Max attempts: {max_attempts}")
    logger.info(f"Platform profile: {profile.display_name}")
    logger.info("-" * 60)

    print(colored("\nStarting RISC-V Porting Agent", "cyan", attrs=["bold"]))
    print(colored(f"   Repository: {repo_url}", "white"))
    print(colored(f"   Max attempts: {max_attempts}", "white"))
    print(colored(f"   Platform:    {profile.display_name}", "white"))
    print(colored("-" * 60, "cyan"))

    initial_state = create_initial_state(repo_url, max_attempts=max_attempts)
    initial_state.messages = [
        HumanMessage(
            content=(
                f"Port this repository to RISC-V: {repo_url}. "
                f"The repository is cloned at "
                f"{initial_state.repo_path}"
            )
        )
    ]

    # Run the workflow
    try:
        final_state = initial_state
        step_count = 0

        # LangGraph's default recursion_limit (25 super-steps) is too
        # tight for a full replan + multi-fix run and aborts with
        # GraphRecursionError; the real loop bound is max_attempts +
        # the cost cap, so give the graph generous headroom.
        for output in app.stream(
            initial_state, config={"recursion_limit": 120}
        ):
            for node_name, state_update in output.items():
                step_count += 1

                # In LangGraph, state_update can be a dict of updates
                if isinstance(state_update, dict):
                    # Update final_state fields from dict
                    for k, v in state_update.items():
                        if hasattr(final_state, k):
                            setattr(final_state, k, v)
                elif isinstance(state_update, AgentState):
                    final_state = state_update

                # Print progress
                status = getattr(
                    final_state.build_status,
                    "value",
                    str(final_state.build_status),
                )
                status_color = {
                    "PENDING": "white",
                    "PLANNING": "cyan",
                    "SCOUTING": "blue",
                    "BUILDING": "yellow",
                    "FIXING": "magenta",
                    "SUCCESS": "green",
                    "FAILED": "red",
                    "ESCALATED": "red",
                }.get(status, "white")

                logger.info(
                    f"[Step {step_count}] {node_name} - Status: {status}"
                )

                print(
                    colored(
                        f"\n[Step {step_count}] {node_name}",
                        "cyan",
                        attrs=["bold"],
                    ),
                    flush=True,
                )
                print(
                    colored(f"   Status: {status}", status_color), flush=True
                )

                # Print messages if verbose
                if (
                    verbose
                    and hasattr(state_update, "messages")
                    and state_update.messages
                ):
                    for msg in state_update.messages:
                        role = getattr(msg, "name", "Assistant")
                        content = getattr(msg, "content", str(msg))

                        logger.debug(f"[{role}] {content}")

                        # Style the roles
                        role_color = (
                            "magenta" if role == "Supervisor" else "cyan"
                        )
                        if role == "Scout":
                            role_color = "blue"
                        if role == "Builder":
                            role_color = "yellow"
                        if role == "Fixer":
                            role_color = "green"

                        print(
                            colored(
                                f"   [{role}]", role_color, attrs=["bold"]
                            ),
                            end=" ",
                        )

                        # Handle long content
                        display_content = content
                        if not verbose:
                            if len(display_content) > 500:
                                display_content = display_content[:500] + "..."

                        print(colored(display_content, "white"), flush=True)
                elif not verbose and node_name != "Supervisor":
                    # Minimal feedback for non-verbose mode
                    status = (
                        final_state.build_status.value
                        if hasattr(final_state, "build_status")
                        else "ACTIVE"
                    )
                    print(
                        colored(f"   Working... ({status})", "white"),
                        end="\r",
                        flush=True,
                    )

        # Save and print results
        logger.info("=" * 60)

        print(colored("\n" + "=" * 60, "cyan"))

        save_porting_outputs(final_state, OUTPUT_DIR)

        final_status = final_state.build_status

        if final_status == BuildStatus.SUCCESS:
            logger.info("PORTING SUCCESSFUL!")
            logger.info(
                f"Recipe generated at: "
                f"{OUTPUT_DIR}/{final_state.repo_name}_recipe.md"
            )

            print(colored("\nPORTING SUCCESSFUL!", "green", attrs=["bold"]))
            print(
                colored(
                    f"   Recipe generated at: "
                    f"{OUTPUT_DIR}/{final_state.repo_name}_recipe.md",
                    "white",
                )
            )

            if package:
                from src.packager import package_build

                recipe_path = os.path.join(
                    OUTPUT_DIR, f"{final_state.repo_name}_recipe.md"
                )
                repo_path = os.path.join(REPOS_DIR, final_state.repo_name)
                agent_log_path = os.path.join(
                    LOGS_DIR, f"agent_{final_state.repo_name}.log"
                )
                # batch_test.py writes per-package logs to
                # ``output/batch_logs/<repo>.log`` relative to its CWD,
                # which is the project root. Resolve from CWD so this
                # works whether main.py is invoked directly or as a
                # batch_test child process. Falls back gracefully (warn
                # + omit) if absent.
                batch_log_path = os.path.join(
                    os.getcwd(),
                    "output",
                    "batch_logs",
                    f"{final_state.repo_name}.log",
                )
                try:
                    zip_path = package_build(
                        repo_name=final_state.repo_name,
                        repo_path=repo_path,
                        recipe_path=recipe_path,
                        platform_name=profile.name,
                        packages_dir=PACKAGES_DIR,
                        repo_url=repo_url,
                        agent_log_path=agent_log_path,
                        batch_log_path=batch_log_path,
                    )
                except Exception as exc:
                    logger.error("Packaging failed: %s", exc, exc_info=True)
                    print(
                        colored(
                            f"\nPackaging FAILED: {exc}",
                            "red",
                            attrs=["bold"],
                        )
                    )
                    return 2
                logger.info(f"Package generated at: {zip_path}")
                print(colored(f"   Package generated at: {zip_path}", "white"))

            return 0
        else:
            logger.info("PORTING STOPPED / FAILED")
            logger.info(f"Final Status: {final_status.value}")

            print(colored("\nPORTING STOPPED / FAILED", "red", attrs=["bold"]))
            print(colored(f"   Final Status: {final_status.value}", "white"))

            if final_state.last_error:
                logger.info("Last Error Capture:")
                err_cat = final_state.last_error_category
                err_cat_val = err_cat.value if err_cat else "Unknown"
                err_sev = final_state.last_error_severity
                err_sev_val = err_sev.value if err_sev else "unknown"
                logger.info(f"Category: {err_cat_val}")
                logger.info(f"Severity: {err_sev_val}")
                logger.info(f"{final_state.last_error[:500]}")

                print(colored("\nLast Error Capture:", "red", attrs=["bold"]))
                print(
                    colored(
                        f"   Category: {err_cat_val}",
                        "yellow",
                    )
                )
                print(
                    colored(
                        f"   Severity: {err_sev_val}",
                        "yellow",
                    )
                )
                print(colored(f"   {final_state.last_error[:500]}", "white"))

            # Print audit trail for transparency
            if final_state.audit_trail:
                logger.info("Agent Decision Audit:")
                print(
                    colored("\nAgent Decision Audit:", "cyan", attrs=["bold"])
                )
                for event in final_state.get_last_audit_events(10):
                    ts = event["timestamp"].split("T")[-1][:8]
                    agent = event.get("agent", "system")
                    etype = event.get("event")
                    data = event.get("data", {})

                    msg = ""
                    if etype == "decision":
                        msg = (
                            f"Decided to {data.get('action')} - "
                            f"{data.get('reason')}"
                        )
                    elif etype == "scripted_op":
                        msg = f"Ran scripted op: {data.get('operation')}"
                    elif etype == "error":
                        err_txt = str(data.get("message") or "")[:100]
                        msg = f"ERROR: {err_txt}..."

                    logger.info(f"[{ts}] {agent:12} | {etype:10} | {msg}")

                    print(
                        colored(
                            f"   [{ts}] {agent:12} | {etype:10} | {msg}",
                            "white",
                        )
                    )

            return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print(colored("\n\nInterrupted by user", "yellow"))
        return 130
    except Exception as e:
        logger.error(f"Unexpected error during agent execution: {e}")
        print(colored(f"\nERROR: {e}", "red"))
        logger.exception("Unexpected error during agent execution")
        return 1

    finally:
        # Ensure logs are flushed. Handlers live on the ROOT logger
        # (configure_logging attaches them there); the module logger has
        # none, so flushing `logger.handlers` was a silent no-op.
        for handler in logging.getLogger().handlers:
            handler.flush()


def cleanup_workspace(dry_run: bool = False) -> None:
    """Clean up workspace directory to manage size.

    Args:
        dry_run: If True, only show what would be deleted without
            actually deleting.
    """
    import shutil
    from pathlib import Path

    workspace_path = Path(WORKSPACE_ROOT)
    if not workspace_path.exists():
        print(colored("Workspace directory does not exist", "yellow"))
        return

    # Calculate current size
    total_size = 0
    file_count = 0

    for item in workspace_path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1

    size_mb = total_size / (1024 * 1024)
    size_gb = size_mb / 1024

    print(colored("\nWorkspace Statistics:", "cyan", attrs=["bold"]))
    print(colored(f"   Location: {workspace_path.absolute()}", "white"))
    print(
        colored(f"   Total Size: {size_gb:.2f} GB ({size_mb:.2f} MB)", "white")
    )
    print(colored(f"   Total Files: {file_count}", "white"))

    # List subdirectories with sizes
    print(colored("\nDirectory Breakdown:", "cyan"))
    subdirs = {}
    for subdir in ["repos", "output", "logs", ".cache", "patches"]:
        subdir_path = workspace_path / subdir
        if subdir_path.exists():
            subdir_size = sum(
                f.stat().st_size for f in subdir_path.rglob("*") if f.is_file()
            )
            subdirs[subdir] = subdir_size
            print(
                colored(
                    f"   {subdir}: {subdir_size / (1024 * 1024):.2f} MB",
                    "white",
                )
            )

    # Offer cleanup options
    print(colored("\nCleanup Options:", "cyan"))
    print(colored("   1. Clean repos/ (cloned repositories)", "white"))
    print(colored("   2. Clean output/ (generated reports)", "white"))
    print(colored("   3. Clean logs/ (execution logs)", "white"))
    print(colored("   4. Clean .cache/ (cached data)", "white"))
    print(colored("   5. Clean ALL (complete workspace reset)", "white"))

    if dry_run:
        print(colored("\n[DRY RUN] No files will be deleted", "yellow"))
        return

    try:
        choice = input(
            colored("\nSelect option (1-5, or 'q' to quit): ", "cyan")
        )
    except (EOFError, KeyboardInterrupt):
        # Non-interactive stdin (CI, piped input) or Ctrl-C: exit
        # cleanly instead of dumping a traceback.
        print(colored("\nCleanup cancelled (no interactive input)", "yellow"))
        return

    dirs_to_clean = []
    if choice == "1":
        dirs_to_clean = ["repos"]
    elif choice == "2":
        dirs_to_clean = ["output"]
    elif choice == "3":
        dirs_to_clean = ["logs"]
    elif choice == "4":
        dirs_to_clean = [".cache"]
    elif choice == "5":
        dirs_to_clean = ["repos", "output", "logs", ".cache", "patches"]
    elif choice.lower() == "q":
        print(colored("Cleanup cancelled", "yellow"))
        return
    else:
        print(colored("Invalid option", "red"))
        return

    # Perform cleanup
    for dirname in dirs_to_clean:
        dir_path = workspace_path / dirname
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                dir_path.mkdir(exist_ok=True)
                print(colored(f"   Cleaned {dirname}/", "green"))
            except Exception as e:
                print(colored(f"   Error cleaning {dirname}/: {e}", "red"))

    # Recalculate size
    new_total_size = sum(
        f.stat().st_size for f in workspace_path.rglob("*") if f.is_file()
    )
    new_size_mb = new_total_size / (1024 * 1024)
    freed_mb = size_mb - new_size_mb

    print(colored("\nCleanup Complete!", "green", attrs=["bold"]))
    print(colored(f"   Freed: {freed_mb:.2f} MB", "green"))
    print(colored(f"   New Size: {new_size_mb:.2f} MB", "white"))


def cleanup_container(remove_image: bool = False) -> None:
    """Stop and remove the active profile's sandbox container.

    Also removes the image when requested. Respects the
    ATESOR_CONTAINER override so per-worker containers are cleaned up
    correctly during batch runs.

    Args:
        remove_image: If True, also remove the sandbox image.
    """
    from src.platforms import get_active_profile, get_container_name

    profile = get_active_profile()
    container_name = get_container_name()
    image_name = profile.image_name
    client = None
    try:
        client = docker.from_env()

        # Remove container
        try:
            container = client.containers.get(container_name)
            if container.status == "running":
                print(
                    colored(
                        f"Stopping container '{container_name}'...", "yellow"
                    )
                )
                container.stop(timeout=10)
            print(
                colored(f"Removing container '{container_name}'...", "yellow")
            )
            container.remove()
            print(colored("Container cleaned up", "green"))
        except docker.errors.NotFound:
            print(colored(f"Container '{container_name}' not found", "yellow"))

        # Remove image if requested
        if remove_image:
            try:
                print(colored(f"Removing image '{image_name}'...", "yellow"))
                client.images.remove(image_name)
                print(colored(f"Image '{image_name}' removed", "green"))
            except docker.errors.ImageNotFound:
                print(colored(f"Image '{image_name}' not found", "yellow"))
            except Exception as e:
                print(colored(f"Error removing image: {e}", "red"))

    except Exception as e:
        print(colored(f"Error during cleanup: {e}", "red"))
    finally:
        if client is not None:
            client.close()


def rebuild_all_sandboxes() -> bool:
    """Rebuild Alpine and Debian sandbox images/containers.

    Returns:
        True if both sandbox environments are rebuilt successfully.
    """
    from src.platforms import set_active_profile

    original_platform = os.environ.get("ATESOR_PLATFORM")
    original_container = os.environ.get("ATESOR_CONTAINER")
    os.environ["REBUILD_IMAGE"] = "true"

    try:
        for platform_name in ("alpine", "debian"):
            os.environ["ATESOR_PLATFORM"] = platform_name
            os.environ.pop("ATESOR_CONTAINER", None)
            set_active_profile(platform_name)
            print(
                colored(
                    f"\nRebuilding sandbox: {platform_name}",
                    "cyan",
                    attrs=["bold"],
                )
            )
            if not setup_docker_environment():
                return False
    finally:
        if original_platform is None:
            os.environ.pop("ATESOR_PLATFORM", None)
        else:
            os.environ["ATESOR_PLATFORM"] = original_platform

        if original_container is None:
            os.environ.pop("ATESOR_CONTAINER", None)
        else:
            os.environ["ATESOR_CONTAINER"] = original_container

    return True


class _CleanHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Compact help layout.

    Two deviations from argparse's default rendering:

    * short and long options share one metavar, so an option reads
      ``-r, --repo URL`` instead of ``-r URL, --repo URL``;
    * help text starts in a fixed column so descriptions line up.

    The epilog is still emitted verbatim (RawDescription behaviour).
    """

    def __init__(self, prog: str, **kwargs) -> None:
        kwargs.setdefault("max_help_position", 26)
        super().__init__(prog, **kwargs)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            return super()._format_action_invocation(action)
        joined = ", ".join(action.option_strings)
        if action.nargs == 0:
            return joined
        metavar = self._format_args(
            action, self._get_default_metavar_for_optional(action)
        )
        return f"{joined} {metavar}"


def main() -> int:
    """Run the CLI entry point and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="atesor-ai",
        add_help=False,  # -h is declared in the "other" group below
        usage=(
            "atesor-ai --repo URL [options]\n"
            "       atesor-ai --setup-only | --cleanup | --clean-workspace"
        ),
        formatter_class=_CleanHelpFormatter,
        epilog=(
            "examples:\n"
            "  atesor-ai --setup-only"
            "                       # first-time sandbox setup\n"
            "  atesor-ai --repo https://github.com/madler/zlib\n"
            "  atesor-ai --repo https://github.com/madler/zlib --verbose\n"
            "  atesor-ai --repo https://github.com/sqlite/sqlite "
            "--platform debian --package\n"
            "\n"
            "environment:\n"
            "  LLM_PROVIDER   gemini | openai | openrouter, plus the "
            "matching *_API_KEY\n"
            "  GIT_TOKEN      token for cloning private GitHub repos\n"
            "  ATESOR_HOME    runtime state dir "
            "(default: ~/.local/share/atesor-ai)\n"
            "  Full list in .env-example; a .env in the current dir or\n"
            "  ~/.config/atesor-ai/ is loaded automatically.\n"
        ),
    )

    porting = parser.add_argument_group("porting")
    porting.add_argument(
        "-r", "--repo", metavar="URL", help="repository to port"
    )
    porting.add_argument(
        "-m",
        "--max-attempts",
        metavar="N",
        type=int,
        default=5,
        help="fix attempts before escalating (default: 5)",
    )
    porting.add_argument(
        "--platform",
        metavar="NAME",
        choices=["alpine", "debian", "ubuntu", "auto"],
        default="auto",
        help="sandbox distro: alpine|debian|ubuntu|auto",
    )
    porting.add_argument(
        "--package",
        action="store_true",
        help="zip recipe + sources on success",
    )
    porting.add_argument(
        "--force",
        action="store_true",
        help="ignore the recipe cache and re-run the full pipeline",
    )
    porting.add_argument(
        "-v", "--verbose", action="store_true", help="show agent activity"
    )

    sandbox = parser.add_argument_group("sandbox")
    sandbox.add_argument(
        "--setup-only",
        action="store_true",
        help="provision the build sandbox and exit",
    )
    sandbox.add_argument(
        "--rebuild",
        action="store_true",
        help="force a rebuild of the sandbox image before running",
    )
    sandbox.add_argument(
        "--cleanup",
        action="store_true",
        help="stop and remove the sandbox container, then exit",
    )
    sandbox.add_argument(
        "--clean-image",
        action="store_true",
        help="also remove the sandbox image (implies --cleanup)",
    )
    sandbox.add_argument(
        "--clean-workspace",
        action="store_true",
        help="reclaim disk space used by the workspace",
    )

    other = parser.add_argument_group("other")
    other.add_argument(
        "-h", "--help", action="help", help="show this help and exit"
    )
    other.add_argument(
        "--version",
        action="version",
        version=f"atesor-ai {__version__}",
        help="show the version and exit",
    )
    other.add_argument(
        "--container",
        metavar="NAME",
        default=None,
        help="override the sandbox container name (batch runs)",
    )

    args = parser.parse_args()

    # Apply platform override BEFORE any module reads get_active_profile()
    if args.platform != "auto":
        os.environ["ATESOR_PLATFORM"] = args.platform

    # Apply container override BEFORE any module reads get_container_name()
    if args.container:
        os.environ["ATESOR_CONTAINER"] = args.container

    # Extract repo name early for per-repo logging. Must match the
    # derivation in create_initial_state exactly — the recipe cache is
    # keyed by the state's repo_name, so any divergence here makes
    # cache lookups miss forever.
    from src.state import is_valid_repo_url, sanitize_repo_name

    repo_name = ""
    if args.repo:
        if not is_valid_repo_url(args.repo):
            print(
                colored(
                    f"ERROR: unsupported or unsafe repository URL: "
                    f"{args.repo!r}",
                    "red",
                )
            )
            print(
                colored(
                    "  Expected a plain http(s) URL without shell "
                    "metacharacters, e.g.\n"
                    "    atesor-ai --repo https://github.com/madler/zlib",
                    "yellow",
                )
            )
            return 1
        repo_name = sanitize_repo_name(
            args.repo.strip().rstrip("/").split("/")[-1].removesuffix(".git")
        )

    # Configure logging (per-repo log files avoid corruption in batch mode)
    configure_logging(args.verbose, repo_name=repo_name)

    # Switch LLM call log to per-repo file for batch safety
    if repo_name:
        from src.llm_logger import set_llm_log_repo

        set_llm_log_repo(repo_name)

    if (args.cleanup or args.clean_image) and args.repo and not args.rebuild:
        print(
            colored(
                "NOTE: --cleanup/--clean-image exit after teardown; "
                f"the repository {args.repo!r} will NOT be ported. "
                "Re-run without the cleanup flag to port it.",
                "yellow",
            )
        )

    if args.cleanup:
        cleanup_container(remove_image=args.clean_image)
        if not args.rebuild:
            return 0

    if args.clean_image and not args.cleanup:
        cleanup_container(remove_image=True)
        if not args.rebuild:
            return 0

    # Handle workspace cleanup
    if args.clean_workspace:
        cleanup_workspace()
        return 0

    infra_only = args.setup_only or (args.rebuild and not args.repo)
    dual_rebuild = (
        infra_only
        and args.rebuild
        and args.platform == "auto"
        and not args.container
    )

    # Require repo URL before starting long-running setup or API checks
    if not args.repo and not infra_only:
        print(colored("ERROR: --repo argument is required", "red"))
        parser.print_help()
        return 1

    if dual_rebuild:
        if not rebuild_all_sandboxes():
            return 1
        print(colored("\nRebuild complete!", "green"))
        return 0

    # Recipe-cache fast path (skip with --force): a cache hit needs
    # neither API keys nor a running sandbox — it only rewrites the
    # recipe file from the cache — so check it BEFORE key validation
    # and Docker provisioning instead of paying for both first.
    if args.repo and not args.setup_only and not args.force:
        from src.memory import get_cached_recipe, materialize_cached_recipe

        cached = get_cached_recipe(repo_name)
        if cached:
            print(
                colored(
                    f"\n✓ Found cached recipe for '{repo_name}'",
                    "green",
                    attrs=["bold"],
                )
            )
            print(
                colored(
                    f"  Build system: {cached.get('build_system', '?')}",
                    "white",
                )
            )
            print(
                colored(
                    f"  Last built: {cached.get('last_built', '?')}", "white"
                )
            )
            # Materialize the recipe into the output dir so the user has a
            # real file to open, regenerated from the cache on every hit.
            recipe_path = materialize_cached_recipe(
                repo_name, OUTPUT_DIR, cached
            )
            if recipe_path:
                print(colored(f"  Recipe written to: {recipe_path}", "white"))
            else:
                print(
                    colored(
                        "  Recipe file: (failed to write — check "
                        "permissions)",
                        "yellow",
                    )
                )
            print(
                colored("  Use --force to re-run the full pipeline.", "yellow")
            )
            if args.package:
                print(
                    colored(
                        "  Note: --package is skipped on cache hits "
                        "(no fresh source tree to package). "
                        "Re-run with --force to build and package.",
                        "yellow",
                    )
                )
            return 0

    # API keys are only required when we are actually running the agent.
    if args.repo and not args.setup_only:
        if not check_keys():
            return 1

    # Set up Docker environment
    if args.rebuild:
        os.environ["REBUILD_IMAGE"] = "true"

    if not setup_docker_environment():
        return 1

    # Setup/Rebuild only mode (no repository run)
    if infra_only:
        message = "\nSetup complete!"
        if args.rebuild and not args.setup_only:
            message = "\nRebuild complete!"
        print(colored(message, "green"))
        return 0

    # Run the agent
    exit_code = run_agent(
        repo_url=args.repo,
        max_attempts=args.max_attempts,
        verbose=args.verbose,
        package=args.package,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
