#############################################################################
# Copyright (c) 2026 10xEngineers
#
# Author: Akif Ejaz <akif.ejaz@10xengineers.ai>
# This program and the accompanying materials are made available under the
# terms of the MIT License which is available at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#############################################################################

"""Agent memory system for few-shot and self-learning.

Provides few-shot learning and self-learning for the multi-agent
RISC-V porting pipeline.

Features:
    1. Compact example format per agent (scout, fixer, builder).
    2. Keyword + regex-based relevance matching.
    3. Auto-learning: agents save novel successful patterns back to
       examples.
    4. Recipe cache: incremental registry of successfully-built
       packages.
    5. Deduplication and pruning (max 100 per agent, oldest
       auto-learned first).
"""

import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import filelock

from .config import DATA_DIR
from .target import is_native

logger = logging.getLogger(__name__)

# Seed examples and the recipe cache ship with the package; the runtime
# copies live in the writable data directory resolved by src.config.
_BUNDLED_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXAMPLES_DIR = Path(DATA_DIR) / "examples"
RECIPE_CACHE_PATH = Path(DATA_DIR) / "recipe_cache.json"
MAX_EXAMPLES_PER_AGENT = 100


def _seed_bundled_data() -> None:
    """Seed the writable data dir from bundled defaults on first run.

    Copies packaged example files and the recipe cache into the
    writable data directory when absent, so a read-only install (for
    example under ``/opt``) still has a working, user-writable few-shot
    store and cache.
    """
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    bundled_examples = _BUNDLED_DATA_DIR / "examples"
    if bundled_examples.is_dir():
        for seed_file in bundled_examples.glob("*.json"):
            target = EXAMPLES_DIR / seed_file.name
            if not target.exists():
                shutil.copy2(seed_file, target)
    if not RECIPE_CACHE_PATH.exists():
        bundled_cache = _BUNDLED_DATA_DIR / "recipe_cache.json"
        if bundled_cache.exists():
            shutil.copy2(bundled_cache, RECIPE_CACHE_PATH)
        else:
            RECIPE_CACHE_PATH.write_text(
                json.dumps({"version": "2.0", "packages": {}})
            )


_seed_bundled_data()


@dataclass
class AgentExample:
    """A single few-shot example for an agent."""

    id: str
    name: str
    tags: List[str]
    build_system: str = ""
    source: str = "manual"
    repo_name: str = ""
    sandbox: str = ""
    timestamp: str = ""

    # Scout-specific
    trigger: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None

    # Fixer-specific
    error_pattern: str = ""
    fix: Optional[Dict[str, Any]] = None

    # Builder-specific
    phases: Optional[List[Dict[str, Any]]] = None
    timeout_recommendation: str = ""

    # Legacy compat
    context: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None
    solution: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    reasoning: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self, example_type: str) -> str:
        """Convert example to compact text for prompt inclusion."""
        if example_type == "scout":
            return self._to_scout_prompt()
        elif example_type == "fixer":
            return self._to_fixer_prompt()
        elif example_type == "builder":
            return self._to_builder_prompt()
        return ""

    def _to_scout_prompt(self) -> str:
        plan_data = self.plan or self.expected_output or {}
        phases = plan_data.get("phases", [])
        trigger = self.trigger or self.context or {}

        phases_json = []
        for p in phases:
            phases_json.append(
                {
                    "name": p.get("name", "unknown"),
                    "commands": p.get("commands", []),
                }
            )

        build = self.build_system or trigger.get("build_system", "?")
        main = trigger.get("main_path", trigger.get("has_main", "?"))
        module_dir = trigger.get("module_dir", "root")
        phases_blob = json.dumps({"phases": phases_json}, indent=2)
        return (
            f"## {self.name}\n"
            f"Build: {build} | "
            f"Main: {main} | "
            f"ModuleDir: {module_dir}\n"
            f"```json\n{phases_blob}\n```\n"
            f"Why: {self.reasoning}\n"
        )

    def _to_fixer_prompt(self) -> str:
        fix_data = self.fix or self.solution or {}
        error_ctx = self.raw.get("error_context", {})
        error_msg = (
            error_ctx.get("error_message") or self.error_pattern or ""
        )[:120]

        actions_text = ""
        for action in fix_data.get("actions", []):
            atype = action.get("type", "")
            if atype == "command":
                actions_text += f"\n  - `{action.get('command', '')}`"
            elif atype == "create_file":
                actions_text += f"\n  - create: {action.get('path', '')}"
            elif atype == "patch":
                actions_text += f"\n  - patch: {action.get('file', 'repo')}"

        strategy = fix_data.get("strategy", fix_data.get("analysis", "N/A"))
        return (
            f"## {self.name}\n"
            f"Error: {error_msg}\n"
            f"Strategy: {strategy}\n"
            f"Actions:{actions_text}\n"
            f"Why: {self.reasoning}\n"
        )

    def _to_builder_prompt(self) -> str:
        ctx = self.context or {}
        phases = self.phases or ctx.get("phases", [])

        cmds = []
        for p in phases:
            for c in p.get("commands", []):
                cmds.append(c)

        return (
            f"## {self.name}\n"
            f"Build: {self.build_system or ctx.get('build_system', '?')} | "
            f"Timeout: {self.timeout_recommendation or 'default'}\n"
            f"Commands: {', '.join(cmds[:4])}\n"
            f"Notes: {self.reasoning or self.raw.get('notes', '')}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "tags": self.tags,
            "build_system": self.build_system,
            "source": self.source,
            "repo_name": self.repo_name,
            "timestamp": self.timestamp or datetime.now().strftime("%Y-%m-%d"),
        }
        if self.trigger:
            d["trigger"] = self.trigger
        if self.plan:
            d["plan"] = self.plan
        if self.error_pattern:
            d["error_pattern"] = self.error_pattern
        if self.fix:
            d["fix"] = self.fix
        if self.phases:
            d["phases"] = self.phases
        if self.timeout_recommendation:
            d["timeout_recommendation"] = self.timeout_recommendation
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.context:
            d["context"] = self.context
        if self.expected_output:
            d["expected_output"] = self.expected_output
        if self.solution:
            d["solution"] = self.solution
        if self.execution:
            d["execution"] = self.execution
        return d


class AgentMemory:
    """Lightweight few-shot memory with auto-learning support.

    Attributes:
        agent_type: The agent role this memory serves.
        examples_dir: Directory holding the example JSON files.
        examples: The loaded ``AgentExample`` instances.
    """

    def __init__(
        self,
        agent_type: str,
        examples_dir: Path = EXAMPLES_DIR,
    ) -> None:
        self.agent_type = agent_type
        self.examples_dir = examples_dir
        self.examples: List[AgentExample] = []
        self._filepath = self.examples_dir / f"{self.agent_type}_examples.json"
        self._load_examples()

    def reload(self) -> None:
        """Reload examples from disk (call after auto-learning writes)."""
        self.examples = []
        self._load_examples()

    def _load_examples(self):
        """Load examples, supporting both new and legacy formats."""
        if not self._filepath.exists():
            logger.warning(f"No examples file found: {self._filepath}")
            return

        try:
            with open(self._filepath, "r") as f:
                data = json.load(f)

            for ex_data in data.get("examples", []):
                example = self._parse_example(ex_data)
                self.examples.append(example)

            logger.info(
                f"Loaded {len(self.examples)} examples for "
                f"{self.agent_type}"
            )
        except Exception as e:
            logger.error(f"Failed to load examples for {self.agent_type}: {e}")

    def _parse_example(self, ex_data: Dict[str, Any]) -> AgentExample:
        """Parse one example dict, handling both formats."""
        return AgentExample(
            id=ex_data.get("id", ""),
            name=ex_data.get("name", ""),
            tags=ex_data.get("tags", []),
            build_system=ex_data.get(
                "build_system",
                ex_data.get("context", {}).get("build_system", ""),
            ),
            source=ex_data.get("source", "manual"),
            repo_name=ex_data.get(
                "repo_name", ex_data.get("context", {}).get("repo_name", "")
            ),
            timestamp=ex_data.get("timestamp", ""),
            sandbox=ex_data.get("sandbox", ""),
            trigger=ex_data.get("trigger"),
            plan=ex_data.get("plan"),
            error_pattern=ex_data.get("error_pattern", ""),
            fix=ex_data.get("fix"),
            phases=ex_data.get("phases"),
            timeout_recommendation=ex_data.get("timeout_recommendation", ""),
            context=ex_data.get("context", {}),
            expected_output=ex_data.get("expected_output"),
            solution=ex_data.get("solution"),
            execution=ex_data.get("execution"),
            reasoning=ex_data.get("reasoning", ""),
            raw=ex_data,
        )

    def get_relevant_examples(
        self,
        context: Dict[str, Any],
        max_examples: int = 3,
    ) -> List[AgentExample]:
        """Get top-K relevant examples based on context matching."""
        if not self.examples:
            return []

        # Hard-filter wrong-sandbox examples BEFORE scoring. Otherwise a stock
        # cmake/go example using apk could still surface under Debian because
        # the build_system match boost (+0.5) outweighs the sandbox penalty.
        ctx_sandbox = context.get("sandbox")
        if not ctx_sandbox:
            try:
                from .platforms import get_active_profile

                ctx_sandbox = f"{get_active_profile().name}-riscv64"
            except Exception:
                ctx_sandbox = ""

        def _sandbox_ok(ex: AgentExample) -> bool:
            if not ctx_sandbox:
                return True
            ex_sandbox = (
                ex.sandbox or ex.raw.get("sandbox", "") or "alpine-riscv64"
            )
            return ex_sandbox == ctx_sandbox

        candidates = [ex for ex in self.examples if _sandbox_ok(ex)]
        if not candidates:
            # Nothing matches the active distro — better to inject zero
            # examples (forces the LLM to follow Platform knowledge) than
            # to bias it with wrong-distro commands. The first build on a
            # new distro will be example-free; subsequent successful
            # builds will populate the cache.
            logger.info(
                f"No {self.agent_type} examples for sandbox '{ctx_sandbox}'; "
                "returning empty (will rely on Platform knowledge in prompt)."
            )
            return []

        scored = [
            (self._calculate_relevance(ex, context), ex) for ex in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for score, ex in scored[:max_examples] if score > 0]

    def _calculate_relevance(
        self, example: AgentExample, context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score. Supports keyword + regex matching."""
        score = 0.0
        ex_bs = example.build_system or example.context.get("build_system", "")
        ex_trigger = example.trigger or example.context or {}
        ex_tags = set(t.lower() for t in example.tags)

        # Build system match (strongest signal)
        if context.get("build_system") and context["build_system"] == ex_bs:
            score += 0.5

        # Error pattern regex match (fixer-specific, very strong)
        if context.get("error_message") and example.error_pattern:
            try:
                if re.search(
                    example.error_pattern,
                    context["error_message"],
                    re.IGNORECASE,
                ):
                    score += 0.4
            except re.error:
                pass

        # Tag-in-error matching
        if context.get("error_message"):
            error_lower = context["error_message"].lower()
            for tag in ex_tags:
                if tag in error_lower:
                    score += 0.15

        # has_main match
        if context.get("has_main") is not None:
            ex_has_main = ex_trigger.get(
                "has_main", example.context.get("has_main")
            )
            if context["has_main"] == ex_has_main:
                score += 0.1

        # module_dir match
        ctx_md = context.get("module_dir", "")
        ex_md = ex_trigger.get(
            "module_dir", example.context.get("module_dir", "")
        )
        if ctx_md and ex_md and ctx_md == ex_md:
            score += 0.15
        elif ctx_md and ex_md:
            score += 0.05

        # CGO tag
        if context.get("has_cgo") and "cgo" in ex_tags:
            score += 0.2

        # Sandbox / distro match (apk commands won't run on Debian and
        # vice-versa). Same-sandbox is a strong positive; cross-sandbox
        # is a hard penalty so a generic build-system match alone doesn't
        # surface the wrong-distro recipe.
        ctx_sandbox = context.get("sandbox")
        if not ctx_sandbox:
            try:
                from .platforms import get_active_profile

                ctx_sandbox = f"{get_active_profile().name}-riscv64"
            except Exception:
                ctx_sandbox = ""
        ex_sandbox = (
            example.sandbox
            or example.raw.get("sandbox", "")
            or "alpine-riscv64"
        )
        if ctx_sandbox and ex_sandbox:
            if ctx_sandbox == ex_sandbox:
                score += 0.25
            else:
                score -= 0.35

        return min(score, 1.0)

    def format_examples_for_prompt(
        self, examples: List[AgentExample], max_chars: int = 3000
    ) -> str:
        """Format examples for inclusion in prompt."""
        if not examples:
            return ""

        sections = [f"# Few-Shot Examples ({self.agent_type})\n"]
        total_chars = 0

        for example in examples:
            text = example.to_prompt_text(self.agent_type)
            if total_chars + len(text) > max_chars:
                break
            sections.append(text)
            total_chars += len(text)

        return "\n".join(sections)

    def save_learned_example(self, example_data: Dict[str, Any]) -> bool:
        """Save an auto-learned example to the JSON file.

        Args:
            example_data: The example payload to persist.

        Returns:
            True if saved, False if a duplicate or invalid.
        """
        if not example_data.get("name") or not example_data.get(
            "build_system"
        ):
            logger.warning("Cannot save example: missing name or build_system")
            return False

        if self._is_duplicate(example_data):
            logger.info(
                f"Skipping duplicate example: {example_data.get('name')}"
            )
            return False

        # Use max existing suffix + 1, not count + 1: after pruning
        # removes old auto examples, a count-based id would collide
        # with a surviving one.
        prefix = f"{self.agent_type}-auto-"
        max_num = 0
        for ex in self.examples:
            if ex.id.startswith(prefix):
                try:
                    max_num = max(max_num, int(ex.id[len(prefix) :]))
                except ValueError:
                    continue
        example_data["id"] = f"{prefix}{max_num + 1:03d}"
        example_data["source"] = "auto"
        example_data["timestamp"] = datetime.now().strftime("%Y-%m-%d")
        if not example_data.get("sandbox"):
            from .platforms import get_active_profile

            example_data["sandbox"] = f"{get_active_profile().name}-riscv64"

        lock_path = str(self._filepath) + ".lock"
        try:
            lock = filelock.FileLock(lock_path, timeout=5)
            with lock:
                data = self._read_json_file()
                examples_list = data.get("examples", [])
                examples_list.append(example_data)

                examples_list = self._prune_examples_list(examples_list)
                data["examples"] = examples_list
                self._write_json_file(data)

            self.reload()
            logger.info(
                f"Saved learned example: {example_data['id']} - "
                f"{example_data['name']}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save learned example: {e}")
            return False

    def _is_duplicate(self, new_data: Dict[str, Any]) -> bool:
        """Check if an example duplicates an existing one (sandbox-aware)."""
        new_bs = new_data.get("build_system", "")
        new_repo = new_data.get("repo_name", "")
        # Auto-stamp sandbox here too so callers that haven't set it
        # still get accurate dedup.
        new_sandbox = new_data.get("sandbox", "")
        if not new_sandbox:
            try:
                from .platforms import get_active_profile

                new_sandbox = f"{get_active_profile().name}-riscv64"
            except Exception:
                new_sandbox = ""

        for ex in self.examples:
            ex_bs = ex.build_system or ex.context.get("build_system", "")
            ex_repo = ex.repo_name or ex.context.get("repo_name", "")
            # Treat an unset legacy sandbox as alpine-riscv64 (the
            # original default).
            ex_sandbox = (
                ex.sandbox or ex.raw.get("sandbox", "") or "alpine-riscv64"
            )

            # Different sandbox → never a duplicate (apk vs apt differ).
            if new_sandbox and ex_sandbox and new_sandbox != ex_sandbox:
                continue

            if (
                ex_repo
                and new_repo
                and ex_repo == new_repo
                and ex_bs == new_bs
            ):
                return True

            if (
                self.agent_type == "fixer"
                and new_data.get("error_pattern")
                and ex.error_pattern
                and new_data["error_pattern"] == ex.error_pattern
                and ex_bs == new_bs
            ):
                return True

            if self.agent_type in ("scout", "builder"):
                new_cmds = self._extract_commands(new_data)
                ex_cmds = self._extract_commands(
                    ex.raw if ex.raw else ex.to_dict()
                )
                if new_cmds and ex_cmds and new_cmds == ex_cmds:
                    return True

        return False

    def _extract_commands(self, data: Dict[str, Any]) -> str:
        """Extract a fingerprint of commands from an example for dedup."""
        cmds = []
        for phase in data.get("phases", []):
            cmds.extend(phase.get("commands", []))
        plan = data.get("plan", {})
        if plan:
            for phase in plan.get("phases", []):
                cmds.extend(phase.get("commands", []))
        output = data.get("expected_output", {})
        if output:
            for phase in output.get("phases", []):
                cmds.extend(phase.get("commands", []))
        if not cmds:
            return ""
        normalized = "|".join(sorted(cmds))
        return hashlib.md5(normalized.encode()).hexdigest()

    def _prune_examples_list(self, examples: List[Dict]) -> List[Dict]:
        """Prune to MAX_EXAMPLES_PER_AGENT, oldest auto-learned first."""
        if len(examples) <= MAX_EXAMPLES_PER_AGENT:
            return examples

        manual = [e for e in examples if e.get("source", "manual") == "manual"]
        auto = [e for e in examples if e.get("source", "manual") == "auto"]

        auto.sort(key=lambda e: e.get("timestamp", ""))
        overflow = len(examples) - MAX_EXAMPLES_PER_AGENT
        if overflow > 0 and len(auto) >= overflow:
            auto = auto[overflow:]
        elif overflow > 0:
            auto = []

        return manual + auto

    def _read_json_file(self) -> Dict[str, Any]:
        if not self._filepath.exists():
            return {
                "version": "2.0",
                "description": f"Examples for {self.agent_type} agent",
                "examples": [],
            }
        with open(self._filepath, "r") as f:
            return json.load(f)

    def _write_json_file(self, data: Dict[str, Any]):
        os.makedirs(self._filepath.parent, exist_ok=True)
        with open(self._filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# RECIPE CACHE
# ============================================================================


def _migrate_legacy_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a legacy flat recipe cache to the nested format.

    The old recipe cache stored one recipe per repo as a flat dict::

        packages[repo_name] = {build_plan: ..., sandbox: "...", ...}

    The new format nests by sandbox so multiple distros can coexist::

        packages[repo_name][sandbox] = {build_plan: ..., ...}

    Migration happens transparently on load.

    Args:
        cache: The loaded cache object.

    Returns:
        The same cache object, possibly mutated in place.
    """
    packages = cache.get("packages", {})
    for repo_name, entry in list(packages.items()):
        if not isinstance(entry, dict):
            continue
        # A nested entry has dict values keyed by sandbox; a flat
        # (legacy) entry has build_plan at the top level.
        if "build_plan" in entry:
            legacy_sandbox = entry.get("sandbox") or "alpine-riscv64"
            packages[repo_name] = {legacy_sandbox: entry}
    cache["packages"] = packages
    return cache


def load_recipe_cache() -> Dict[str, Any]:
    """Load the recipe cache from disk (auto-migrating legacy flat entries)."""
    if not RECIPE_CACHE_PATH.exists():
        return {"version": "2.0", "packages": {}}
    try:
        with open(RECIPE_CACHE_PATH, "r") as f:
            cache = json.load(f)
        return _migrate_legacy_cache(cache)
    except Exception as e:
        logger.error(f"Failed to load recipe cache: {e}")
        return {"version": "2.0", "packages": {}}


def _default_sandbox() -> str:
    """Return the active platform's sandbox key.

    The native target adds ``-native``, so native and QEMU recipes
    never share a cache entry.

    Returns:
        The sandbox key, e.g. ``"alpine-riscv64"`` or
        ``"debian-riscv64-native"``.
    """
    try:
        from .platforms import get_active_profile

        sandbox = f"{get_active_profile().name}-riscv64"
    except Exception:
        sandbox = "alpine-riscv64"
    if is_native():
        sandbox += "-native"
    return sandbox


def get_cached_recipe(
    repo_name: str,
    architecture: str = "riscv64",
    sandbox: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Look up a cached recipe for a package within a sandbox.

    A sandbox is a distro+arch combination. Falls back to the active
    platform's sandbox when `sandbox` is None. Returns None if no recipe
    exists for that combination — even if one exists for a different
    sandbox (apk vs apt commands are not interchangeable).
    """
    if sandbox is None:
        sandbox = _default_sandbox()
    cache = load_recipe_cache()
    pkg_entry = cache.get("packages", {}).get(repo_name)
    if not isinstance(pkg_entry, dict):
        return None
    recipe = pkg_entry.get(sandbox)
    if recipe and recipe.get("architecture", "riscv64") == architecture:
        return recipe
    return None


def save_to_recipe_cache(
    repo_name: str,
    repo_url: str,
    build_system: str,
    build_plan: Dict[str, Any],
    dependencies: List[str],
    patches: List[str],
    artifacts: List[Dict[str, Any]],
    build_duration_seconds: float,
    architecture: str = "riscv64",
    sandbox: Optional[str] = None,
    recipe_markdown: Optional[str] = None,
) -> bool:
    """Upsert a package recipe in the cache, scoped per sandbox.

    Recipes for different sandboxes (e.g. ``alpine-riscv64`` vs
    ``debian-riscv64``) coexist under the same ``repo_name`` so an
    Alpine zlib build does not overwrite a Debian zlib build.

    Args:
        repo_name: The repository name key.
        repo_url: The source repository URL.
        build_system: The detected build system.
        build_plan: The successful build plan to persist.
        dependencies: Resolved dependency names.
        patches: Patch identifiers applied during the build.
        artifacts: Build artifacts produced.
        build_duration_seconds: Wall-clock build duration.
        architecture: Target architecture. Defaults to ``"riscv64"``.
        sandbox: Sandbox key; defaults to the active platform.
        recipe_markdown: The rendered recipe guide. Stored verbatim so a
            later cache hit can reproduce the exact ``<repo>_recipe.md``.

    Returns:
        True if the cache was written successfully, else False.
    """
    if sandbox is None:
        sandbox = _default_sandbox()
    lock_path = str(RECIPE_CACHE_PATH) + ".lock"
    try:
        lock = filelock.FileLock(lock_path, timeout=5)
        with lock:
            cache = load_recipe_cache()
            packages = cache.setdefault("packages", {})

            compact_phases = []
            for phase in build_plan.get("phases", []):
                compact_phases.append(
                    {
                        "name": phase.get("name", "unknown"),
                        "commands": phase.get("commands", []),
                    }
                )

            recipe = {
                "repo_url": repo_url,
                "build_system": build_system,
                "architecture": architecture,
                "sandbox": sandbox,
                "last_built": datetime.now().isoformat(),
                "build_plan": {"phases": compact_phases},
                "dependencies": dependencies,
                "patches": patches[:10],
                "artifacts": [
                    {
                        "type": a.get("type", "binary"),
                        "path": a.get("filepath") or a.get("path", ""),
                        **({"role": a["role"]} if a.get("role") else {}),
                    }
                    for a in artifacts[:20]
                ],
                "build_duration_seconds": round(build_duration_seconds, 1),
                "recipe_file": f"output/{repo_name}_recipe.md",
            }
            if recipe_markdown:
                recipe["recipe_markdown"] = recipe_markdown

            # Ensure nested layout (entry is a dict keyed by sandbox)
            pkg_entry = packages.get(repo_name)
            if not isinstance(pkg_entry, dict) or "build_plan" in (
                pkg_entry or {}
            ):
                # Either missing or legacy-flat; (re-)init as nested
                pkg_entry = {}
                if (
                    isinstance(packages.get(repo_name), dict)
                    and "build_plan" in packages[repo_name]
                ):
                    # Preserve the legacy flat entry under its own sandbox key
                    legacy = packages[repo_name]
                    legacy_sb = legacy.get("sandbox") or "alpine-riscv64"
                    pkg_entry[legacy_sb] = legacy
            pkg_entry[sandbox] = recipe
            packages[repo_name] = pkg_entry

            cache.setdefault("version", "2.0")
            os.makedirs(RECIPE_CACHE_PATH.parent, exist_ok=True)
            with open(RECIPE_CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

        logger.info(f"Recipe cache updated for {repo_name} [{sandbox}]")
        return True
    except Exception as e:
        logger.error(f"Failed to save recipe cache for {repo_name}: {e}")
        return False


def render_recipe_markdown(repo_name: str, recipe: Dict[str, Any]) -> str:
    """Render a Markdown porting recipe from a cached recipe entry.

    Returns the stored ``recipe_markdown`` verbatim when present (the
    exact guide produced by the original build). Otherwise it
    reconstructs a readable recipe from the structured cache fields, so a
    cache hit can still materialize a usable ``<repo>_recipe.md``.

    Args:
        repo_name: The package/repository name.
        recipe: The cached recipe entry for a single sandbox.

    Returns:
        The recipe rendered as a Markdown document.
    """
    stored = recipe.get("recipe_markdown")
    if stored:
        return stored if stored.endswith("\n") else stored + "\n"

    lines = [f"# RISC-V Porting Recipe: {repo_name}", ""]
    if recipe.get("repo_url"):
        lines.append(f"- **Repository:** {recipe['repo_url']}")
    lines.append(
        f"- **Build system:** {recipe.get('build_system', 'unknown')}"
    )
    lines.append(
        f"- **Architecture:** {recipe.get('architecture', 'riscv64')}"
    )
    if recipe.get("sandbox"):
        lines.append(f"- **Sandbox:** {recipe['sandbox']}")
    if recipe.get("last_built"):
        lines.append(f"- **Last built:** {recipe['last_built']}")
    lines += [
        "",
        "> Reconstructed from the recipe cache. Re-run with `--force` to "
        "regenerate the full guide from a fresh build.",
        "",
    ]

    deps = recipe.get("dependencies") or []
    if deps:
        lines += ["## Dependencies", ""]
        lines += [f"- {d}" for d in deps]
        lines.append("")

    phases = (recipe.get("build_plan") or {}).get("phases") or []
    if phases:
        lines += ["## Build Steps", ""]
        for phase in phases:
            lines += [f"### {phase.get('name', 'step')}", "", "```bash"]
            lines += list(phase.get("commands", []))
            lines += ["```", ""]

    patches = recipe.get("patches") or []
    if patches:
        lines += ["## Patches / Fixes Applied", ""]
        lines += [f"- {p}" for p in patches]
        lines.append("")

    artifacts = recipe.get("artifacts") or []
    if artifacts:
        lines += ["## Build Artifacts", ""]
        for art in artifacts:
            role = f" ({art['role']})" if art.get("role") else ""
            path = art.get("path", "")
            lines.append(f"- **{art.get('type', 'binary')}**{role}: `{path}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def materialize_cached_recipe(
    repo_name: str,
    output_dir: str,
    recipe: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Write a cached recipe to ``<output_dir>/<repo_name>_recipe.md``.

    Looks up the active-sandbox recipe when ``recipe`` is not supplied.
    This lets a cache hit leave a tangible recipe file the user can open,
    instead of only printing cached metadata.

    Args:
        repo_name: The package/repository name.
        output_dir: Directory to write the recipe file into.
        recipe: A pre-fetched recipe entry. Falls back to
            ``get_cached_recipe(repo_name)`` when ``None``.

    Returns:
        The absolute path to the written file, or None if no recipe
        exists or the write failed.
    """
    if recipe is None:
        recipe = get_cached_recipe(repo_name)
    if not recipe:
        return None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{repo_name}_recipe.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_recipe_markdown(repo_name, recipe))
        return os.path.abspath(path)
    except OSError as e:
        logger.error(f"Failed to write recipe file for {repo_name}: {e}")
        return None


# ============================================================================
# SINGLETON + CONVENIENCE FUNCTIONS
# ============================================================================

MEMORY_INSTANCES: Dict[str, AgentMemory] = {}


def get_agent_memory(agent_type: str) -> AgentMemory:
    """Get or create memory instance for an agent type."""
    if agent_type not in MEMORY_INSTANCES:
        MEMORY_INSTANCES[agent_type] = AgentMemory(agent_type)
    return MEMORY_INSTANCES[agent_type]


def reload_agent_memory(agent_type: str) -> None:
    """Force reload an agent's memory (after auto-learning writes)."""
    if agent_type in MEMORY_INSTANCES:
        MEMORY_INSTANCES[agent_type].reload()
    else:
        MEMORY_INSTANCES[agent_type] = AgentMemory(agent_type)


def format_few_shot_examples(
    agent_type: str,
    context: Dict[str, Any],
    max_examples: int = 2,
    max_chars: int = 2500,
) -> str:
    """Build formatted few-shot examples for prompt inclusion.

    Args:
        agent_type: 'scout', 'fixer', 'builder', or 'supervisor'.
        context: Current context for relevance matching.
        max_examples: Maximum examples to include.
        max_chars: Maximum characters for the examples section.

    Returns:
        A formatted string for prompt inclusion.
    """
    memory = get_agent_memory(agent_type)
    examples = memory.get_relevant_examples(context, max_examples)
    return memory.format_examples_for_prompt(examples, max_chars)


def save_learned_example(
    agent_type: str, example_data: Dict[str, Any]
) -> bool:
    """Save a learned example for the given agent type."""
    memory = get_agent_memory(agent_type)
    return memory.save_learned_example(example_data)
