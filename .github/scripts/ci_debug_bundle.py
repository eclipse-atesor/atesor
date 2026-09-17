#!/usr/bin/env python3
"""Atesor CI debug bundle.

Given a GitHub PR number OR an Actions workflow run ID, this script
collects everything you need to investigate failures/timeouts in a
batch-port run and writes it to ``./debug-<id>/`` at the repo root.

What it collects (and where each piece comes from):

* **Run metadata** — ``gh run view --json …`` for the workflow run.
* **Per-shard CI logs** —
  ``gh api repos/<owner>/<repo>/actions/jobs/<id>/logs``
  for every ``atesor-{alpine,debian}-full`` job. Each line for a built
  package looks like::

      [12/47] ⏱ TIMEOUT cloudlist  1h 00m 10s  output/batch_logs/cloudlist.log
      [37/47] ✘ FAIL    nginx      3m 38s      output/batch_logs/nginx.log
      [ 1/47] ✔ PASS    anew       4m 37s      output/batch_logs/anew.log

  We parse those lines to derive per-shard PASS/FAIL/TIMEOUT lists.
* **Per-package log files** — the workflow's ``collect-full-shard-logs``
  job aggregates every ``output/batch_logs/<pkg>.log`` from every
  shard into two tiny (~8MB each) artifacts:

      full-logs-alpine
      full-logs-debian

  Those artifacts contain the ``agent_<pkg>.log`` / ``agent-call_<pkg>.log``
  / ``<pkg>.log`` files we want. We download both, extract, and copy
  every log file matching a failing/timeout package into
  ``debug-<id>/issues/<platform>/<pkg>/``.

* **Reports** — a human-readable ``REPORT.md`` and a machine-readable
  ``failures.json`` for downstream tooling.

Inputs (one of --pr / --run-id is required)::

    python3 .github/scripts/ci_debug_bundle.py --run-id 28020958388
    python3 .github/scripts/ci_debug_bundle.py --pr 11
    python3 .github/scripts/ci_debug_bundle.py --pr 11 -R akifejaz/atesor

Defaults: repo = current ``gh repo view`` repo; output dir = CWD.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Parsing patterns (see module docstring for source examples).
# ---------------------------------------------------------------------------

# Per-package status line printed by .github/scripts/batch_test.py.
# Anchored on the bracketed "[idx/total]" counter so we never match the
# duplicate one-shot " TIMEOUT pkg killing process group …" lines.
PKG_LINE_RE = re.compile(
    r"\[\s*(?P<idx>\d+)\s*/\s*(?P<total>\d+)\s*\]\s+"
    r"(?P<icon>[\u2714\u2718\u23f1])\s+"
    r"(?P<status>PASS|FAIL|TIMEOUT)\s+"
    r"(?P<pkg>\S+)\s+"
    r"(?P<duration>.+?)\s+"
    r"(?P<log>output/batch_logs/\S+\.log)\s*$"
)

# Final summary line printed at the end of every shard's batch run::
#     ║ TOTAL 47   PASS 42   FAIL 3   TIMEOUT 2  ║
BATCH_TOTALS_RE = re.compile(
    r"TOTAL\s+(?P<total>\d+)\s+"
    r"PASS\s+(?P<pass_>\d+)\s+"
    r"FAIL\s+(?P<fail>\d+)\s+"
    r"TIMEOUT\s+(?P<timeout>\d+)"
)

# "atesor-debian-full (shard 0/15)"  → platform=debian, shard=0, total=15
SHARD_NAME_RE = re.compile(
    r"atesor-(?P<platform>alpine|debian)-full"
    r"\s+\(shard\s+(?P<shard>\d+)/(?P<total>\d+)\)"
)

# CI error markers we want to surface in the report.
CI_ERROR_RE = re.compile(r"##\[error\](.+)$")

# Misc CI noise worth flagging (each tuple: regex, friendly label).
CI_FLAG_RES = [
    (re.compile(r"Unable to reserve cache key"), "Cache reserve conflict"),
    (re.compile(r"Cache restore failed"), "Cache restore failed"),
    (
        re.compile(r"Node\.js 20.*deprecat", re.I),
        "Node.js 20 deprecation warning",
    ),
    (
        re.compile(r"The runner has received a shutdown signal"),
        "Runner shutdown signal",
    ),
    (
        re.compile(r"Process completed with exit code (\d+)"),
        "Batch step exit code",
    ),
    (re.compile(r"The operation was canceled"), "Job cancelled"),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PackageResult:
    """One package's outcome parsed from a shard's CI log."""

    idx: int
    total: int
    status: str  # PASS / FAIL / TIMEOUT
    package: str
    duration: str
    log_path: str  # output/batch_logs/<pkg>.log (relative path in CI)


@dataclass
class ShardSummary:
    """Aggregated results for one CI shard job."""

    job_id: int
    job_name: str
    platform: str
    shard: int
    shard_total: int
    status: str  # queued/in_progress/completed
    conclusion: str | None  # success/failure/cancelled/...
    started_at: str | None
    completed_at: str | None
    html_url: str | None
    batch_totals: dict | None = None
    packages: list[PackageResult] = field(default_factory=list)
    ci_errors: list[str] = field(default_factory=list)
    ci_flags: list[str] = field(default_factory=list)
    log_file: str = ""  # relative path to saved CI log


# ---------------------------------------------------------------------------
# gh helpers
# ---------------------------------------------------------------------------


def _run(
    cmd: list[str], *, stdin: bytes | None = None, capture: bool = True
) -> bytes:
    """Run a subprocess command and return its stdout bytes.

    Raises CalledProcessError on non-zero exit. Stdout is captured;
    stderr is left attached to the parent process so failures are visible.
    """
    return subprocess.run(
        cmd,
        check=True,
        input=stdin,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def gh_json(args: list[str]) -> object:
    """Run a ``gh`` command and parse its stdout as JSON."""
    out = _run(["gh", *args])
    return json.loads(out.decode("utf-8", errors="replace"))


def gh_text(args: list[str]) -> str:
    """Run a ``gh`` command and return its raw stdout."""
    out = _run(["gh", *args])
    return out.decode("utf-8", errors="replace")


def gh_repo_from_env(explicit: str | None) -> str:
    """Resolve OWNER/REPO. Use explicit if given, else `gh repo view`."""
    if explicit:
        return explicit
    data = gh_json(["repo", "view", "--json", "nameWithOwner"])
    return data["nameWithOwner"]


def resolve_pr_to_run_id(repo: str, pr: int, workflow_name: str | None) -> int:
    """Find the most recent workflow run associated with a PR's head SHA."""
    pr_info = gh_json(
        [
            "pr",
            "view",
            str(pr),
            "-R",
            repo,
            "--json",
            "headRefOid,headRefName,number",
        ]
    )
    head_sha = pr_info["headRefOid"]
    head_ref = pr_info["headRefName"]

    # Filter runs by head SHA (most reliable), then optionally workflow name.
    runs = gh_json(
        [
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs?head_sha={head_sha}&per_page=100",
        ]
    )
    candidates = (
        runs.get("workflow_runs", []) if isinstance(runs, dict) else []
    )
    if workflow_name:
        candidates = [r for r in candidates if r.get("name") == workflow_name]
    if not candidates:
        # Fall back to head branch lookup.
        runs2 = gh_json(
            [
                "api",
                "--paginate",
                f"repos/{repo}/actions/runs?branch={head_ref}&per_page=100",
            ]
        )
        candidates = (
            runs2.get("workflow_runs", []) if isinstance(runs2, dict) else []
        )
        if workflow_name:
            candidates = [
                r for r in candidates if r.get("name") == workflow_name
            ]
    if not candidates:
        sys.exit(
            f"error: no workflow runs found for PR #{pr} "
            f"(head_sha={head_sha[:7]}, head_ref={head_ref})."
        )
    # Most recent first
    candidates.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return int(candidates[0]["id"])


def fetch_run_metadata(repo: str, run_id: int) -> dict:
    """Fetch workflow-run metadata via ``gh run view``."""
    return gh_json(
        [
            "run",
            "view",
            str(run_id),
            "-R",
            repo,
            "--json",
            "status,conclusion,headBranch,headSha,event,name,"
            "createdAt,updatedAt,url,databaseId",
        ]
    )


def fetch_jobs(repo: str, run_id: int) -> list[dict]:
    """Return all jobs for a run, following pagination."""
    data = gh_json(
        [
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
        ]
    )
    return data.get("jobs", []) if isinstance(data, dict) else []


def fetch_job_logs(repo: str, job_id: int) -> str:
    """Return the raw CI log text for a single job."""
    try:
        out = gh_text(["api", f"repos/{repo}/actions/jobs/{job_id}/logs"])
    except subprocess.CalledProcessError as exc:
        print(
            f"  warn: failed to fetch logs for job {job_id}: {exc}",
            file=sys.stderr,
        )
        return ""
    return out


def fetch_artifacts(repo: str, run_id: int) -> list[dict]:
    """List the artifacts attached to a workflow run."""
    data = gh_json(
        [
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
        ]
    )
    return data.get("artifacts", []) if isinstance(data, dict) else []


def download_artifact_zip(repo: str, artifact_id: int, dest_zip: Path) -> None:
    """Download an artifact zip via gh api into dest_zip."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_zip, "wb") as fh:
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/artifacts/{artifact_id}/zip",
                "-H",
                "Accept: application/vnd.github+json",
            ],
            check=True,
            stdout=fh,
        )


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_shard_log(
    log_text: str,
) -> tuple[list[PackageResult], dict | None, list[str], list[str]]:
    """Parse a single shard's CI log.

    Returns (packages, batch_totals, ci_errors, ci_flag_labels).
    """
    pkgs: list[PackageResult] = []
    seen_pkgs: set[str] = set()
    totals: dict | None = None
    errors: list[str] = []
    flags: set[str] = set()

    for line in log_text.splitlines():
        # GitHub prepends ISO timestamps to every log line; strip them
        # before regex matching so anchored patterns still work.
        stripped = re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s?", "", line)

        m = PKG_LINE_RE.search(stripped)
        if m:
            pkg = m.group("pkg")
            if pkg in seen_pkgs:
                continue
            seen_pkgs.add(pkg)
            pkgs.append(
                PackageResult(
                    idx=int(m.group("idx")),
                    total=int(m.group("total")),
                    status=m.group("status"),
                    package=pkg,
                    duration=m.group("duration").strip(),
                    log_path=m.group("log"),
                )
            )
            continue

        m = BATCH_TOTALS_RE.search(stripped)
        if m:
            totals = {
                "total": int(m.group("total")),
                "pass": int(m.group("pass_")),
                "fail": int(m.group("fail")),
                "timeout": int(m.group("timeout")),
            }
            continue

        m = CI_ERROR_RE.search(stripped)
        if m:
            text = m.group(1).strip()
            if text and text not in errors:
                errors.append(text)
            continue

        for pat, label in CI_FLAG_RES:
            if pat.search(stripped):
                flags.add(label)
                break

    return pkgs, totals, errors, sorted(flags)


# ---------------------------------------------------------------------------
# Artifact handling
# ---------------------------------------------------------------------------


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a downloaded artifact zip into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def index_log_files(roots: Iterable[Path]) -> dict[str, list[Path]]:
    """Index every ``*.log`` file by its stem (filename without extension).

    Multiple shards may have rebuilt the same package, so values are
    lists. Also indexes the common ``agent_<pkg>.log`` and
    ``agent-call_<pkg>.log`` companions used by the workflow.
    """
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            by_stem[path.stem].append(path)
    return by_stem


def collect_package_logs(
    pkg: str,
    indexes: dict[str, dict[str, list[Path]]],
    platform: str,
) -> list[Path]:
    """Return all log files belonging to ``pkg`` for ``platform``.

    Matches the package's primary log (``<pkg>.log``) plus any companion
    ``agent_<pkg>.log`` / ``agent-call_<pkg>.log`` files.
    """
    candidates: list[Path] = []
    stems = [pkg, f"agent_{pkg}", f"agent-call_{pkg}"]
    plat_idx = indexes.get(platform, {})
    for stem in stems:
        candidates.extend(plat_idx.get(stem, []))
    return candidates


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_dur(start: str | None, end: str | None) -> str:
    if not (start and end):
        return "?"
    from datetime import datetime

    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return "?"
    secs = int((b - a).total_seconds())
    if secs < 0:
        return "?"
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def build_report(
    run_meta: dict,
    repo: str,
    run_id: int,
    shards: list[ShardSummary],
    debug_root: Path,
) -> None:
    """Write REPORT.md and failures.json into the bundle directory."""
    by_platform: dict[str, list[ShardSummary]] = defaultdict(list)
    for s in shards:
        by_platform[s.platform].append(s)
    for s_list in by_platform.values():
        s_list.sort(key=lambda s: s.shard)

    md: list[str] = []
    md.append(f"# CI debug report — run {run_id}")
    md.append("")
    md.append(f"- **Workflow**: {run_meta.get('name')}")
    md.append(f"- **URL**: {run_meta.get('url')}")
    md.append(f"- **Repo**: `{repo}`")
    md.append(
        f"- **Status / Conclusion**: {run_meta.get('status')} / "
        f"{run_meta.get('conclusion')}"
    )
    md.append(f"- **Event**: {run_meta.get('event')}")
    md.append(
        f"- **Branch / SHA**: {run_meta.get('headBranch')} / "
        f"`{run_meta.get('headSha', '')[:12]}`"
    )
    md.append(f"- **Created**: {run_meta.get('createdAt')}")
    md.append(f"- **Updated**: {run_meta.get('updatedAt')}")
    md.append("")

    # Aggregate failure index across platforms
    total_fail = total_timeout = total_pass = total_total = 0
    fail_pkgs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    timeout_pkgs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for s in shards:
        if s.batch_totals:
            total_total += s.batch_totals["total"]
            total_pass += s.batch_totals["pass"]
            total_fail += s.batch_totals["fail"]
            total_timeout += s.batch_totals["timeout"]
        for p in s.packages:
            if p.status == "FAIL":
                fail_pkgs[s.platform].append((p.package, s.shard))
            elif p.status == "TIMEOUT":
                timeout_pkgs[s.platform].append((p.package, s.shard))

    md.append("## Aggregate totals (across all shards)")
    md.append("")
    md.append(
        f"- TOTAL={total_total}  PASS={total_pass}  "
        f"FAIL={total_fail}  TIMEOUT={total_timeout}"
    )
    md.append("")
    for plat in sorted(by_platform):
        f = sorted({p for p, _ in fail_pkgs[plat]})
        t = sorted({p for p, _ in timeout_pkgs[plat]})
        md.append(f"### {plat}")
        md.append(f"- Failing ({len(f)}): {', '.join(f) if f else '—'}")
        md.append(f"- Timeout ({len(t)}): {', '.join(t) if t else '—'}")
        md.append("")

    # Per-shard sections
    for platform in sorted(by_platform):
        md.append(f"## {platform.upper()} shards")
        md.append("")
        for s in by_platform[platform]:
            f_pkgs = [p.package for p in s.packages if p.status == "FAIL"]
            t_pkgs = [p.package for p in s.packages if p.status == "TIMEOUT"]
            md.append(
                f"### shard {s.shard}/{s.shard_total} — "
                f"{s.conclusion or s.status}"
            )
            md.append("")
            md.append(f"- **Job**: `{s.job_name}` (id={s.job_id})")
            md.append(f"- **Job URL**: {s.html_url}")
            md.append(
                f"- **Duration**: {_fmt_dur(s.started_at, s.completed_at)} "
                f"(start={s.started_at}, end={s.completed_at})"
            )
            if s.batch_totals:
                bt = s.batch_totals
                md.append(
                    f"- **Batch totals**: TOTAL={bt['total']}  "
                    f"PASS={bt['pass']}  FAIL={bt['fail']}  "
                    f"TIMEOUT={bt['timeout']}"
                )
            md.append(f"- **CI log**: `{s.log_file}`")
            md.append(
                f"- **Failing packages ({len(f_pkgs)})**: "
                f"{', '.join(f_pkgs) if f_pkgs else '—'}"
            )
            md.append(
                f"- **Timeout packages ({len(t_pkgs)})**: "
                f"{', '.join(t_pkgs) if t_pkgs else '—'}"
            )
            if s.ci_errors:
                md.append("- **CI errors**:")
                for e in s.ci_errors[:8]:
                    md.append(f"    - {e}")
                if len(s.ci_errors) > 8:
                    md.append(f"    - … ({len(s.ci_errors)-8} more)")
            if s.ci_flags:
                md.append(f"- **CI flags**: {', '.join(s.ci_flags)}")
            if f_pkgs or t_pkgs:
                md.append("")
                md.append("  | pkg | status | duration | local logs |")
                md.append("  |---|---|---|---|")
                for p in s.packages:
                    if p.status == "PASS":
                        continue
                    local = debug_root / "issues" / platform / p.package
                    if local.exists():
                        rel = local.relative_to(debug_root)
                        log_files = sorted(local.glob("*.log"))
                        files_str = (
                            ", ".join(
                                f"`{f.relative_to(debug_root)}`"
                                for f in log_files
                            )
                            or f"`{rel}/`"
                        )
                    else:
                        files_str = "(no log found)"
                    md.append(
                        f"  | `{p.package}` | {p.status} | "
                        f"{p.duration} | {files_str} |"
                    )
            md.append("")

    (debug_root / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Bundle CI debug data for an atesor batch-port run."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--run-id", type=int, help="GitHub Actions workflow run ID."
    )
    g.add_argument(
        "--pr", type=int, help="Pull request number (resolves latest run)."
    )
    ap.add_argument(
        "-R", "--repo", help="OWNER/REPO (default: current `gh repo view`)."
    )
    ap.add_argument(
        "--workflow",
        default="ATESOR AI Batch Porting Tests",
        help=(
            "Filter PR runs by workflow name "
            "(default: 'ATESOR AI Batch Porting Tests')."
        ),
    )
    ap.add_argument(
        "--output-root",
        default=".",
        help="Parent dir for the debug-<id> bundle (default: CWD).",
    )
    ap.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Don't download full-logs-* artifacts (CI-log parsing only).",
    )
    args = ap.parse_args()

    repo = gh_repo_from_env(args.repo)
    if args.pr:
        print(f"[1/5] Resolving PR #{args.pr} on {repo} → workflow run id …")
        run_id = resolve_pr_to_run_id(repo, args.pr, args.workflow)
        bundle_label = f"pr-{args.pr}-run-{run_id}"
    else:
        run_id = args.run_id
        bundle_label = f"run-{run_id}"

    debug_root = Path(args.output_root).resolve() / f"debug-{bundle_label}"
    if debug_root.exists():
        print(f"  (refreshing existing bundle: {debug_root})")
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True)
    (debug_root / "ci-logs").mkdir()
    (debug_root / "artifacts").mkdir()
    (debug_root / "issues").mkdir()

    print(f"[2/5] Fetching run metadata for {run_id} on {repo} …")
    run_meta = fetch_run_metadata(repo, run_id)
    (debug_root / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"      conclusion={run_meta.get('conclusion')} "
        f"branch={run_meta.get('headBranch')} "
        f"sha={run_meta.get('headSha', '')[:12]}"
    )

    print("[3/5] Fetching jobs and per-shard CI logs …")
    jobs = fetch_jobs(repo, run_id)
    (debug_root / "jobs.json").write_text(
        json.dumps(jobs, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"      {len(jobs)} job(s) total")

    shards: list[ShardSummary] = []
    for job in jobs:
        m = SHARD_NAME_RE.search(job.get("name", ""))
        if not m:
            continue
        platform = m.group("platform")
        shard_num = int(m.group("shard"))
        shard_total = int(m.group("total"))
        log_text = fetch_job_logs(repo, job["id"])
        log_file = (
            debug_root / "ci-logs" / f"{platform}-shard-{shard_num:02d}.log"
        )
        log_file.write_text(log_text, encoding="utf-8")
        pkgs, totals, errors, flags = parse_shard_log(log_text)
        shards.append(
            ShardSummary(
                job_id=job["id"],
                job_name=job.get("name", ""),
                platform=platform,
                shard=shard_num,
                shard_total=shard_total,
                status=job.get("status", ""),
                conclusion=job.get("conclusion"),
                started_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
                html_url=job.get("html_url"),
                batch_totals=totals,
                packages=pkgs,
                ci_errors=errors,
                ci_flags=flags,
                log_file=str(log_file.relative_to(debug_root)),
            )
        )
        print(
            f"      - {job['name']}: parsed {len(pkgs)} pkg result(s), "
            f"totals={totals}"
        )

    shards.sort(key=lambda s: (s.platform, s.shard))

    print("[4/5] Downloading full-logs-* artifacts …")
    artifacts = fetch_artifacts(repo, run_id)
    art_by_name = {a["name"]: a for a in artifacts}
    platform_log_roots: dict[str, Path] = {}
    if not args.skip_artifacts:
        for platform in ("alpine", "debian"):
            art_name = f"full-logs-{platform}"
            art = art_by_name.get(art_name)
            if not art:
                print(f"      ! artifact '{art_name}' not found (skipping)")
                continue
            if art.get("expired"):
                print(f"      ! artifact '{art_name}' has expired (skipping)")
                continue
            zip_path = debug_root / "artifacts" / f"{art_name}.zip"
            extract_dir = debug_root / "artifacts" / art_name
            print(
                f"      - downloading {art_name} "
                f"({art.get('size_in_bytes', 0)/1024/1024:.1f} MB) …"
            )
            try:
                download_artifact_zip(repo, art["id"], zip_path)
                extract_zip(zip_path, extract_dir)
                platform_log_roots[platform] = extract_dir
                zip_path.unlink()
            except subprocess.CalledProcessError as exc:
                print(f"        warn: download failed: {exc}")

    # Index log files per platform; copy logs for each failing/timeout pkg.
    indexes = {
        plat: index_log_files([root])
        for plat, root in platform_log_roots.items()
    }

    failures_index: list[dict] = []
    for s in shards:
        for p in s.packages:
            if p.status == "PASS":
                continue
            dest_dir = debug_root / "issues" / s.platform / p.package
            dest_dir.mkdir(parents=True, exist_ok=True)
            log_files = collect_package_logs(p.package, indexes, s.platform)
            copied: list[str] = []
            for src in log_files:
                dst = dest_dir / src.name
                # Disambiguate same-name files coming from different shards
                # by suffixing with their parent dir name.
                if dst.exists():
                    dst = dest_dir / f"{src.parent.name}-{src.name}"
                try:
                    shutil.copy2(src, dst)
                    copied.append(str(dst.relative_to(debug_root)))
                except OSError as exc:
                    print(f"        warn: copy {src} → {dst} failed: {exc}")
            failures_index.append(
                {
                    "platform": s.platform,
                    "shard": s.shard,
                    "package": p.package,
                    "status": p.status,
                    "duration": p.duration,
                    "ci_log": s.log_file,
                    "package_logs": copied,
                    "job_url": s.html_url,
                }
            )

    (debug_root / "failures.json").write_text(
        json.dumps(failures_index, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Per-shard summary for machine consumption
    shards_dump = [asdict(s) for s in shards]
    (debug_root / "shards.json").write_text(
        json.dumps(shards_dump, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    print("[5/5] Building REPORT.md …")
    build_report(run_meta, repo, run_id, shards, debug_root)

    n_fail = sum(1 for f in failures_index if f["status"] == "FAIL")
    n_timeout = sum(1 for f in failures_index if f["status"] == "TIMEOUT")
    print()
    print(f"Done. Bundle: {debug_root}")
    print(f"  shards parsed : {len(shards)}")
    print(f"  failing pkgs  : {n_fail}")
    print(f"  timeout pkgs  : {n_timeout}")
    print(f"  open          : {debug_root / 'REPORT.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
