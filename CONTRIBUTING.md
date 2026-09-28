# Contributing to Atesor AI

Thanks for your interest in improving Atesor AI. This guide covers how to set up a dev environment, the coding standards we enforce, how to add new functionality, and what we look for in pull requests.

## Table of Contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Coding standards](#coding-standards)
- [Architecture invariants](#architecture-invariants)
- [Common extension points](#common-extension-points)
- [Submitting changes](#submitting-changes)
- [Reporting bugs](#reporting-bugs)

---

## Ways to contribute

- **Port a package and submit the recipe.** Successful runs auto-learn into `data/examples/*.json`. Sharing these makes future runs faster and cheaper for everyone.
- **Add a new platform profile** (e.g. Fedora, openSUSE) — usually a single `PROFILES` entry in `src/platforms.py` plus a `Dockerfile.<distro>`.
- **Improve error classification** in `src/state.py::classify_error` so the supervisor can react to new failure patterns.
- **File bug reports** for packages that fail in interesting or repeatable ways.
- **Improve prompts, tests, or docs.**

---

## Development setup

```bash
git clone https://github.com/akifejaz/atesor-ai
cd atesor-ai
pip install -r requirements.txt
cp .env-example .env   # add your API key
sh .github/scripts/setup_git_merge_driver.sh
```

The last command sets up a merge driver in your clone only. With it, git
merges `data/recipe_cache.json` by meaning (the newest recipe wins) instead
of line by line. Without it, git stops at a conflict for that file; see
`.gitattributes` for the manual steps.

You need Docker with RISC-V emulation enabled on non-RISC-V hosts:

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
python3 main.py --setup-only
```

---

## Running tests

The project is **not** pip-installed — every test command must set `PYTHONPATH=.`:

```bash
PYTHONPATH=. pytest                                          # full suite
PYTHONPATH=. pytest tests/test_graph_routing.py              # one file
PYTHONPATH=. pytest tests/test_state.py::TestState::test_add_error
PYTHONPATH=. pytest -k "memory and not slow"                 # by name
```

Tests live in `tests/`, one file per `src/` module, written in `unittest.TestCase` style. Add a test for every bug fix and every new public function.

---

## Coding standards

Non-negotiables for all Python in this repo:

- **PEP 8**, hard **79-char** line limit
- **4-space** indentation, no tabs
- **Double-quoted** strings
- **Type hints** on every public function/method
- **Google-style docstrings** on every public module, class, function, method
- **isort** import order: stdlib → third-party → local, blank-line separated
- `snake_case` for variables/functions/modules, `PascalCase` for classes, `UPPER_SNAKE` for constants

Comments: keep them minimal. Add one only when the *why* is non-obvious (a hidden constraint, an invariant, a workaround). Never restate what the code already says.

Linters available (not enforced in CI yet): `black`, `flake8`. Both are listed in `requirements.txt`.

---

## Architecture invariants

These are the rules that keep the graph correct. Break one and tests will (or should) fail.

1. **`AgentState` is the single source of truth.** Mutate it in place inside every node, never construct a fresh one.
2. **Every LangGraph node must be decorated** with `@agent_node(AgentRole.X)` from `src/graph.py`. This gives uniform error handling, rate-limit retry, and audit logging. Undecorated nodes will crash the graph on transient errors.
3. **Every LLM call goes through `invoke_llm(llm, messages, timeout=120)`** followed by `log_llm_call(...)`. Never call `llm.invoke()` directly — it bypasses the timeout and audit log.
4. **Every shell command goes through `execute_command()`** in `src/tools.py`. It enforces the `CommandValidator` regex whitelist and the Docker sandbox boundary. Don't shell out via `subprocess` directly.
5. **Path constants come from `src/config.py`** (`WORKSPACE_ROOT`, `REPOS_DIR`, `OUTPUT_DIR`, …). Never hardcode `/workspace` or host paths.
6. **Distro-specific logic goes through `get_active_profile()`** in `src/platforms.py`. Never hardcode `atesor-ai-sandbox`, `apk`, or `apt-get`.
7. **Recipe-cache and few-shot writes use `filelock`.** Batch runs will corrupt these files otherwise.
8. **Go builds always pass `-buildvcs=false`** (the sandbox runs as root and trips Go's VCS ownership check).

---

## Common extension points

### Adding a new agent / routing phase

1. Add the node function in `src/graph.py` and decorate it with `@agent_node(AgentRole.X)`.
2. Add the node in `create_workflow()`.
3. Write a per-node routing function (e.g. `route_my_node_to_next`) that inspects `AgentState` fields and returns the next node name.
4. Wire it with `add_conditional_edges("my_node", route_my_node_to_next, {dest1: dest1, dest2: dest2})`.

### Adding a new platform (distro)

1. Add a `PROFILES` entry in `src/platforms.py` with package manager commands and the `package_map`.
2. Add a `Dockerfile.<distro>` that uses `FROM --platform=linux/riscv64 …`.
3. Add the choice to `--platform` in `main.py`.
4. Update `src/knowledge.py` if the distro has unique gotchas worth surfacing to the LLM.

### Adding a new LLM provider

1. Extend `MODEL_CONFIG` in `src/models.py` with per-role model names and temperatures.
2. Add the `elif provider == "..."` branch in `create_llm()`.
3. Add the API-key check in `check_api_keys()`.

### Adding few-shot examples

Edit `data/examples/{scout,fixer,builder,supervisor}_examples.json`. Schema is documented in `data/examples/README.md`. Manual entries (no `source: "auto"`) are never pruned.

### Adding a new error pattern

Add a branch to `classify_error()` in `src/state.py`. **Order matters** — more specific patterns must come before generic ones. Add a corresponding test in `tests/test_state.py`.

---

## Submitting changes

1. Branch from `main` (or `dev` if the maintainer asks).
2. Make focused commits with clear messages (Conventional Commits style preferred: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
3. Run `PYTHONPATH=. pytest` and make sure everything passes.
4. Open a pull request with:
   - What changed and why
   - Any new env vars, CLI flags, or breaking changes
   - Test output for new functionality
5. Keep PRs reasonably small. If you're touching multiple concerns, split them.

---

## Reporting bugs

For build failures on specific packages, include:

- The exact `main.py` command used
- The package URL
- Sandbox profile (`alpine` or `debian`)
- The relevant section of `workspace/logs/agent_{repo}.log`
- The relevant section of `workspace/logs/agent-call_{repo}.log` (LLM transcript)
- The `AgentState` snapshot from `workspace/output/{repo}_state_*.json` if available

For everything else, open an issue with steps to reproduce.

---

Thanks for contributing to the RISC-V ecosystem.
