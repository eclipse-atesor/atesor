# Atesor AI

**Agentic System that autonomously ports x86/ARM Packages to RISC-V (riscv64).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-green.svg)](https://github.com/langchain-ai/langgraph)
[![Regression tests](https://img.shields.io/github/actions/workflow/status/eclipse-atesor/atesor/regression-tests.yml?branch=main&label=regression%20tests)](https://github.com/eclipse-atesor/atesor/actions/workflows/regression-tests.yml)
[![Batch ports](https://img.shields.io/github/actions/workflow/status/eclipse-atesor/atesor/batch-port.yml?branch=main&label=batch%20ports)](https://github.com/eclipse-atesor/atesor/actions/workflows/batch-port.yml)
[![LLM health](https://img.shields.io/github/actions/workflow/status/eclipse-atesor/atesor/llm-health.yml?branch=main&label=LLM%20health)](https://github.com/eclipse-atesor/atesor/actions/workflows/llm-health.yml)
[![.deb package](https://img.shields.io/github/actions/workflow/status/eclipse-atesor/atesor/package-deb.yml?branch=main&label=.deb%20package)](https://github.com/eclipse-atesor/atesor/actions/workflows/package-deb.yml)

Atesor AI takes a source code url (i.e GitHub repo link), builds the package natively inside a RISC-V Docker sandbox, fixes whatever breaks, and emits a reproducible porting recipe. Supported build systems cover the **C, C++, and Go** ecosystems (Make, CMake, Meson, autotools, Cargo, Go modules), more are coming.. It runs across both **Alpine (musl)** and **Debian/Ubuntu (glibc)** sandboxes (and on **native/real RISC-V hardware** too, see [Native RISC-V machine](#native-risc-v-machine)), so a package is verified on the two libc families that matter in practice.

**NOTE**: Builds use QEMU/binfmt emulation by default. To build on a real RISC-V machine, see [Native RISC-V machine](#native-risc-v-machine).

---

## Table of Contents

- [Why](#why)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Native RISC-V machine](#native-risc-v-machine)
- [Performance mode](#performance-mode---optimize)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Key features](#key-features)
- [Development](#development)
- [Contributing](#contributing)

---

## Why

The RISC-V software ecosystem still has gaps that x86 and ARM solved years ago. Porting work is repetitive: clone, detect the build system, install the right packages, hit the same handful of issues (stale `config.guess`, x86-only SIMD, Go's `-buildvcs` trap, musl vs glibc headers and many more), patch, retry. Atesor AI automates that loop with a small team of specialized LLM agents and a deterministic scripted-ops layer that handles the boring up to 70% for free.

Output is a reproducible Markdown recipe a human or CI can replay, plus the ready-to-use RISC-V build artifacts.

---

## How it works

1. **Scripted analysis** clones the repo inside the sandbox and detects the build system, dependencies, and architecture-specific code - zero LLM cost.
2. **Planner** drafts a high-level `TaskPlan` from that analysis.
3. **Supervisor** routes work between Scout, Builder, and Fixer, watches for error loops, and decides when to escalate.
4. **Builder** runs the build natively on RISC-V via QEMU/binfmt. **Fixer** patches whatever breaks. **Scout** answers targeted questions about the source tree.
5. **Artifact scanner** reads the ELF header of every build output and of every member of every static archive. One x86-64, aarch64 or riscv32 output fails the port, even next to riscv64 ones. A build whose commands pass but that proves no riscv64 output is reported as **not verified** (exit code 3).
6. **Package tests** (`make check`, `ctest`, `meson test`, `go test`, `cargo test`) run once after a successful build, with a time limit. The result goes in the report and the zip manifest; it never fails the port.
7. **Recipe** is written to disk and cached, keyed by `(package, sandbox)`. Only verified ports are cached or learned. A later cache hit re-renders `{repo}_recipe.md` from that entry and skips the pipeline entirely.

The supervisor → executor loop is built on [LangGraph](https://github.com/langchain-ai/langgraph), with a single `AgentState` carried between nodes.

---

## Quick start

### Prerequisites

- Docker, with RISC-V emulation enabled on x86/ARM hosts:
  ```bash
  docker run --privileged --rm tonistiigi/binfmt --install all
  ```
- Python 3.10+
- An API key for one of: Gemini, OpenAI, OpenRouter, or Anthropic.

### Install

```bash
git clone https://github.com/akifejaz/atesor-ai
cd atesor-ai
pip install -r requirements.txt
cp .env-example .env   # then add your API key
```

### Build the sandbox

```bash
python3 main.py --setup-only                       # Alpine (default)
python3 main.py --setup-only --platform debian     # Debian/Ubuntu
```

### Port a package

```bash
python3 main.py --repo https://github.com/madler/zlib --verbose
```

### Installation (from pre-built package)

A self-contained Debian package can be built for Ubuntu/Debian hosts - it
bundles a virtualenv and installs an `atesor-ai` launcher. See
[`packaging/deb/README.md`](packaging/deb/README.md):

```bash
packaging/deb/build_deb.sh                          # → dist/atesor-ai_*.deb
sudo apt-get install ./dist/atesor-ai_*.deb         # pulls docker/qemu deps
atesor-ai --help
atesor-ai --repo https://github.com/madler/zlib 
```

---

## Native RISC-V machine

Atesor can also build on a real riscv64 machine instead of QEMU. With
`--target native`, each build runs in a rootless podman container on your
own machine, and Atesor reaches the machine over SSH. QEMU stays the
default target. Use native for local builds and tests. CI does not use
native machines.

### What the machine needs

Atesor installs nothing on the machine and never runs `sudo`. The owner of
the machine installs these items:

| Requirement | Why |
|---|---|
| riscv64 Linux on real hardware, not a QEMU virtual machine | Native results are the purpose of this target. |
| SSH key login, and a POSIX login shell (bash, dash, zsh or ksh) | Atesor runs with no prompts and sends shell scripts. |
| podman 4.0 or newer, `uidmap`, and subordinate UID and GID ranges for the login user | The builds run in a rootless container. |
| `rsync` on the machine, and `ssh` and `rsync` on your computer | Atesor drives the machine with `ssh`, and copies the build tree back with `rsync` for local reads. |
| A work directory that is not `$HOME`, not a parent of `$HOME`, and not in `~/.ssh` | Atesor mounts the work directory into the container with SELinux relabeling (`:Z`). |
| 10 GB free for podman storage and 10 GB free in the work directory, or 20 GB when one filesystem holds both | Images, build trees and build caches use this space. |
| Outbound HTTPS | Image builds and package builds download toolchains and sources. |

On a Debian or Ubuntu machine, the owner can install them with:

```bash
sudo apt install podman uidmap rsync
sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$USER"
```

Skip the `usermod` line when `/etc/subuid` already lists the user. Then
install your SSH key from your computer, one time:

```bash
ssh-copy-id -i ~/.ssh/<key>.pub -p <port> <user>@<address>
```

### Configure `.env`

Set the target, the platform, and one of the two SSH modes:

| Key | Mode | Value |
|---|---|---|
| `ATESOR_TARGET` | both | `native`. The default is `qemu`. The `--target` flag overrides it. |
| `ATESOR_PLATFORM` | both | `alpine`, `debian` or `ubuntu`. Required on native. |
| `ATESOR_REMOTE_WORKDIR` | both | Work directory on the machine. Default `~/atesor-ai`. |
| `ATESOR_SSH_HOST` | alias | What you type after `ssh`: a `Host` alias from `~/.ssh/config`, a host name, or `user@host` |
| `ATESOR_SSH_HOSTNAME` | field | Address of the machine |
| `ATESOR_SSH_PORT` | field | SSH port. Default 22. |
| `ATESOR_SSH_USER` | field | Login user |
| `ATESOR_SSH_IDENTITY_FILE` | field | Path to your private key, never the key itself |

- **Alias mode**: use it when a plain `ssh <machine>` already logs in.
  Atesor gives the value to `ssh` unchanged.
- **Field mode**: use it when the machine provider gives you an address, a
  port, a user and a key. Atesor writes these values to
  `~/.cache/atesor-ai/ssh_config` with mode 0600, so the address stays out
  of the process list and the logs.

Set one mode, not both. Login uses a key only. Any other `ATESOR_SSH_*`
key, for example a password, stops the run. A key with a passphrase needs a
running `ssh-agent`.

The `debian` and `ubuntu` platforms build `Dockerfile.native` (Debian
trixie) on the machine. The `alpine` platform builds `Dockerfile`.

### Check the machine

```bash
python3 main.py --target native --preflight
```

The preflight runs 11 checks: the settings, the local tools, the SSH login
and shell, the architecture, real hardware, podman, the rootless IDs, rsync
on the machine, the work directory and the free disk space. It exits with 0
when all checks pass. When a check fails, it lists all items that the
machine still needs, with the command that provides each item. Atesor never
runs these commands. The owner of the machine runs them, then you run
`--preflight` again. Every native command runs the preflight first.

### Build on the machine

```bash
python3 main.py --target native --setup-only       # build the image one time
python3 main.py --target native --repo https://github.com/madler/zlib --package
```

The first `--setup-only` builds the image on the machine. On a 4-core
board, this takes about 10 minutes. Later commands start the container in
seconds. The container stops when each command ends, and each port removes
its build tree from the machine.

- `--cleanup` removes the container, and `--clean-image` also removes the
  image.
- `--clean-workspace` also removes the cloned repositories from the work
  directory on the machine.

A batch run on native gets its default worker count from the machine. The
default is the core count or the memory in GiB divided by 4, whichever is
smaller, and at least 1. `--workers` overrides the default, up to the core
count of the machine.

```bash
python3 .github/scripts/batch_test.py --target native --platform debian --list smoke
```

### Native results

Native results and QEMU results never mix:

| Result | QEMU | Native |
|---|---|---|
| State directory | `workspace/` | `workspace-native/` |
| Recipe cache key | `<platform>-riscv64` | `<platform>-riscv64-native` |
| Zip name | `<owner>-<repo>-<time>-<platform>.zip` | `<owner>-<repo>-<time>-<platform>-native.zip` |
| Report line | `**Target**: qemu` | `**Target**: native` |

---

## Performance mode (`--optimize`)

After a verified port, `--optimize` tries faster builds ("tunes") for
the machine that runs Atesor. The verified build stays the result. A
faster tune is an extra output.

```bash
python3 main.py --probe --target native     # what the machine supports
python3 main.py --target native --repo https://github.com/madler/zlib --optimize --package
```

The console shows the work as its own step after `finish_node`, only
with `--optimize`, and the summary at the end answers one question
first: did Atesor make a faster build?

```
[Step 9] optimize_node
   Status: OPTIMIZING
   Machine  : native, rv64gc+zba+zbb, core sifive-p550, misaligned access emulated (timing test)
   Package  : C/C++ library (cmake): build/libz.so.1.3.2.1, build/libz.a
   Baseline : rebuilding the verified build in a copy
   Workload : choosing one for a C/C++ library
   Tune-1   : core tuning = -mtune=sifive-p400-series -mstrict-align
              1.7% slower, inside the noise: time ratio 1.017 (95% CI 1.007-1.021)
   ...
   Result   : YES: Tune-6 (clang-19 -O3 + PGO) is 10.8% faster than the verified build
...
   Optimized: YES: Tune-6 (clang-19 -O3 + PGO) is 10.8% faster than the verified build
     Tune-1  core tuning              -mtune=sifive-p400-series ...      1.7% slower (inside the noise)
     Tune-6  profile-guided optimization PGO, trained on a second input  10.8% faster  <- accepted
   Optimized files: /workspace/repos/zlib.opt/accepted (with --package also in the zip under optimized/)
```

When no tune is faster, or none can be timed, the first line says
`Optimized: NO: ...`, followed by `Why:` (and `Fix:` when the machine
gives wrong results).

With a verified recipe of the package in the recipe cache, `--optimize`
builds that recipe again instead of planning: `cached_recipe_node`
turns the cached build plan (and its patches) into the plan, so the
analyst and the scout make no LLM call. A failed build still goes to
the fixer. `--force` plans again.

The steps:

1. **Machine check (native only).** Every CPU runs 10 zlib round trips
   of 8 MiB at the same time. Unchanged code must give the data back;
   when it does not (an unstable clock, bad memory), performance mode
   stops before it builds anything, because no timing on that machine
   can be trusted. The Megrez failed 3 of 40 such round trips at
   1.8 GHz and passed 160 of 160 at 1.4 GHz.
2. **Probe.** Atesor reads the machine before it builds anything: the ISA
   extensions (test instructions, `riscv_hwprobe`, `/proc/cpuinfo`), the
   core (device-tree name, or vendor and architecture ids), the
   misaligned-access speed (`riscv_hwprobe`, else a load-timing test),
   the CPUs, the governor, and the compilers with the flags they accept.
   `--probe` prints this, and the settings that follow from it.
3. **Package type.** The verified build's artifacts give the language
   and the type: a library, programs, or both. Test, example and
   benchmark programs do not count as the package's own programs.
4. **Baseline.** Atesor rebuilds the verified build in a copy
   (`<repo>.opt/base`), the same way as every tune, and checks that its
   code equals the verified build's code. Each tune is compared with this
   copy, so only the tune's own change counts.
5. **Workload, by package type.** It must do fixed work in every run:
   - Go: the module's own benchmarks, with a fixed iteration count.
   - Programs: the optimizer agent (an LLM) proposes one command on
     generated input; a strict grammar checks it.
   - A library: the optimizer agent writes a harness against the
     library's exported API (C/C++ for a C/C++ library, a Go benchmark, a
     Cargo example for a Rust crate). It must build, call the library,
     print the same `ATESOR-CHECK` line in two runs, and take 0.5-5 s per
     run. A C/C++ harness links the tree's own library, and `ldd` must
     show that copy. On an error the agent gets the error back (3 rounds
     at most).
   - Last, the package tests (a weak signal).
   Benchmark suites that run for a fixed time (Google Benchmark,
   criterion, most CTest/Meson benchmarks) are not used. Without a
   workload, a native run builds no tune and says why.
6. **Tunes.** C/C++: core tuning (`-mtune`), machine ISA (`-march` with
   only the probed extensions), vector ISA (only with V), `-O3`, LTO,
   then clang with the same kind of flags. Go: `GORISCV64`. Rust: the
   target CPU (only when rustc's feature list for it fits the machine)
   or the probed target features, then fat LTO. Then PGO on the best
   tune. A tune that the build already has (every source at `-O3`, or
   `-flto`) is skipped.
7. **Rebuild.** Each tune builds in a copy of the patched tree
   (`<repo>.opt/<id>`) with the commands of the verified build. Wrapper
   scripts add the tune's flags, and nothing else, to each compiler call. The
   harness builds in each tree too. Atesor finds each artifact right after
   the build, also in a folder named after a hash of the flags (lz4), and
   every later step uses that file.
8. **Gates.** The artifacts must be riscv64 and built with the tune's
   settings. The package tests must still pass. When the tests build an
   artifact again, Atesor puts back the tune's build. The workload output
   must equal the baseline output, and a run must not change its input
   file.
9. **Measurement (native only).** Baseline and tune runs alternate,
   pinned to one CPU. A tune is faster only when the 95% interval of
   its time ratio is below 0.98 (p < 0.05), and the gain holds again in a
   second code layout. The second check gives the layout flag
   (`-falign-functions=64`) to the baseline too, so only the tune's own
   change counts. A tune with the same code as an earlier one gets that
   result without a new measurement. Once a tune is faster, Atesor keeps
   enough of the time budget for its second check. A gain that got no
   second timing shows as "not confirmed", with the cause.

| | Native | QEMU |
|---|---|---|
| Probe | The board | The `--emulate-cpu` model, else QEMU's `max` CPU |
| ISA tunes | Yes | Only with `--emulate-cpu` |
| Gates | Yes | Yes, on the emulated CPU |
| Time measurement | Yes | No: emulation speed says nothing about hardware |
| Accepted tune | Yes | Never |

- On native hardware, one workload run must take at most 5 s, so one
  measurement (about 56 runs) fits the budget. A measurement that the
  remaining budget cannot finish does not start.
- Atesor warns when other work uses CPUs at the start of a measurement,
  and when other Atesor containers run on the machine.
- `ATESOR_OPTIMIZE_BUDGET` sets the time limit in seconds (default 3600).
  A batch run never optimizes.
- The report has a "Performance Mode" section with one row per tune and
  the harness source. With `--package`, the zip manifest has schema
  version 3 (`machine` and `optimization`), and the accepted artifacts
  are in `optimized/`.
- The clang tune needs clang-19 in the native image. Rebuild the image
  one time: `python3 main.py --target native --setup-only --rebuild`.
- Set the cpufreq governor of the board to `performance` for stable
  numbers. Atesor never changes a board setting; it warns instead.
- `--emulate-cpu` takes a QEMU CPU model (QEMU 9.0 or later), for example
  `rva22u64`, or the `QEMU model` line that `--probe --target native`
  prints for your board. `ATESOR_QEMU_CPU` is the env form (QEMU runs
  only). `max` counts as no model, and a model must start with a letter.

---


## Architecture (high-level)

Atesor AI is built in clear layers. A request enters at the top as a repo
URL and leaves at the bottom as verified `riscv64` artifacts plus a
replayable recipe. The middle is a **LangGraph** compiled `StateGraph` with
per-node conditional edges and an embedded build-fix subgraph - threading one
shared `AgentState` dataclass through every node.

```
   USER  --repo <url>  [--platform alpine | debian]
     |
     v
   +--------------------------------------------------------------+
   |  ORCHESTRATION   (main.py)                                   |
   |    (load .env -> verify keys -> provision riscv64 sandbox)   |
   +------------------------------+-------------------------------+
                  |
      recipe cache? --- HIT --->  render <pkg>_recipe.md  -->  EXIT
                  |
                  | MISS?   (carries one AgentState, mutated in place)
                  v
   +-------------------------------------------------------------+
   |  LANGGRAPH  (src/graph.py)   compiled StateGraph + subgraph |
   |                                                             |
   |   init_node                                                 |
   |     |  route_init_to_next(state)                            |
   |     v                                                       |
   |   planner_node  (LLM: TaskPlan + fallback)                  |
   |     |  route_planner_to_next(state)                         |
   |     v                                                       |
   |   scout_build_system  (0-LLM, reads build files)            |
   |     v                                                       |
   |   scout_deps  (0-LLM, reads deps)                           |
   |     v                                                       |
   |   scout_arch_issues  (0-LLM, counts arch patterns)          |
   |     v                                                       |
   |   scout_aggregator  (fan-in: default BuildPlan for          |
   |     |   well-known build systems, deps merged into setup)   |
   |     |  route_scout_aggregator_to_next(state)                |
   |     |----> scout_node  (LLM: BuildPlan, used when the       |
   |     |        aggregator defers: unknown/low-confidence      |
   |     |        build system or heavy arch-specific code)      |
   |     v                                                       |
   |   supervisor_node  (ZERO LLM - pure heuristic routing)      |
   |     |                                     |    |    |       |
   |     |  route_supervisor_to_next(state)    |    |    |       |
   |     |        |        |                   |    |    |       |
   |     |        |        |                   |    |    |       |
   |     v        v        v                   v    v    v       |
   |  planner  scout  build_fix_subgraph    finish  escalate     |
   |  (replan) (rescout)  (compiled subgraph)       (terminal)   |
   |                        |                                    |
   |                        v                                    |
   |                   build_node                                |
   |                     |  route_build_result(state)            |
   |                     v                                       |
   |               verify_node                                   |
   |                  |       route_verify_result(state)         |
   |                  |          |                   |           |
   |                  v          v                   v           |
   |              <verified>    fix_node          escalate       |
   |                  |          |  route_fix      (terminal)    |
   |                  |          |    |                          |
   |                  |          v    v                          |
   |                  | <fixed>  <can't fix>                     |
   |                  |    |         |                           |
   |                  v    v         v                           |
   |                 back to supervisor                          |
   |                                                             |
   +-------------------------------------------------------------+
       |                      |                      |
       v                      v                      v
   +-----------+        +----------+           +-----------+
   | FINISH    |        | ESCALATE |           | OUTPUTS   |
   | (LLM:     |        | (0-LLM)  |           | recipe.md |
   |  recipe)  |        |          |           | report    |
   +-----------+        +----------+           | state.json|
       |                                       | patches   |
       | self-learning: save few-shot examples +-----------+
       | + recipe cache
       v
   +----------------------------+
   |  MEMORY   (src/memory.py)  |
   |  examples + recipe_cache   |
   +----------------------------+
       |
       |-- few-shot prompts -->  PLANNER / SCOUT / FIXER
       |
       +-- fast-path -------->  next run's recipe-cache check  (top)
```

### Graph topology (LangGraph edges)

Every state has **per-node routing functions**. Each is unique to its source node and inspects real state fields:

- `route_init_to_next`: checks `state.build_status` → `planner_node` /
  `escalate_node`
- `route_planner_to_next`: checks `state.task_plan` → `scout_build_system`
  (entering the scout chain) / `escalate_node`
- `route_scout_aggregator_to_next`: checks `state.build_plan` →
  `supervisor_node` (heuristic plan built) / `scout_node` (defer to the
  LLM scout)
- `route_supervisor_to_next`: checks `state.build_status`, `state.task_plan`,
  `state.last_error_category`, `state.attempt_count` → 5 possible destinations
  (`planner_node`, `scout_node`, `build_fix_subgraph`, `finish_node`,
  `escalate_node`). A `SUCCESS` build always routes to `finish_node`,
  even at the attempt/cost ceiling.
- `route_finish_to_next`: checks `state.optimize_requested` →
  `optimize_node` (performance mode) / `__end__`

The build-fix cycle is a compiled **subgraph** (`create_build_fix_subgraph()`)
with its own internal routing:

```
build_node ──> verify_node ──> __end__  (success, exits subgraph)
    ↑              │              ↑
    │              v              │
    └────── fix_node ─────────────┘  (retry, or exit if unfixable)
```

Routing inside the subgraph uses `route_build_result`, `route_verify_result`,
and `route_fix_result` - each inspecting build/error state directly.

### Three pillars

- **Scripted Operations Layer** (`src/scripted_ops.py`) - deterministic,
  zero-LLM repo inspection. Handles ~70% of analysis at zero cost.
- **LangGraph state machine** (`src/graph.py`) - compiled `StateGraph` with
  per-node routing, 13 nodes, a build-fix subgraph, and `@agent_node`-wrapped
  uniform error handling.
- **Platform abstraction** (`src/platforms.py`) - one `PlatformProfile` per
  distro. Adding a sandbox is a single `PROFILES` entry; the rest of the
  code stays distro-agnostic.

## Configuration

Edit `.env` (template in `.env-example`):

| Variable | Required when | Purpose |
|---|---|---|
| `LLM_PROVIDER` | always | `gemini` (default), `openai`, `openrouter`, `anthropic` |
| `GOOGLE_API_KEY` | provider = `gemini` | |
| `OPENAI_API_KEY` | provider = `openai` | |
| `OPENROUTER_API_KEY` | provider = `openrouter` | |
| `ANTHROPIC_API_KEY` | provider = `anthropic` | Claude Haiku 5.5 for most roles, Claude Sonnet 5.5 for the fixer and the optimizer harness |
| `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2` | optional | LangSmith tracing |
| `ATESOR_PLATFORM` | optional | `alpine` / `debian` (overridden by `--platform`) |
| `ATESOR_TARGET` | optional | `qemu` (default) / `native` (overridden by `--target`). See [Native RISC-V machine](#native-risc-v-machine) for the SSH keys. |
| `ATESOR_CONTAINER` | optional | Override container name (overridden by `--container`) |
| `ATESOR_HOME` | optional | Base dir for runtime state (default `~/.local/share/atesor-ai`) |
| `ATESOR_OPTIMIZE_BUDGET` | optional | Time limit of `--optimize` in seconds (default 3600) |
| `ATESOR_QEMU_CPU` | optional | QEMU CPU model for `--optimize` and `--probe` under QEMU (overridden by `--emulate-cpu`) |

NOTE: When using the pre-build binary (atesor-ai) the:
`.env` is loaded from the current directory first, then
`$ATESOR_HOME/.env` and `~/.config/atesor-ai/.env`, so an installed CLI
finds your keys regardless of the working directory.

Models are selected per agent role in `src/models.py` (`MODEL_CONFIG`). Each role has its own temperature - deterministic for Builder/Supervisor, slightly hotter for Fixer.

---

## Outputs

Runtime state is written under a **state home** - `$ATESOR_HOME` if set,
otherwise `~/.local/share/atesor-ai` for an installed CLI, or the repo's
`./workspace` when running from a source checkout. Below, `$WS` is that
`…/workspace` directory.

NOTE: When using the pre-build binary (atesor-ai) the state home is `$ATESOR_HOME` (default `~/.local/share/atesor-ai`) and does not include a `workspace` subdir. The CLI writes directly to `$ATESOR_HOME/output/`, `$ATESOR_HOME/logs/`, etc.

| Path | Content |
|---|---|
| `$WS/output/{repo}_recipe.md` | Final Markdown porting **recipe** (replayable) |
| `$WS/output/{repo}_report_*.md` | Detailed build report (per run) |
| `$WS/output/{repo}_state_*.json` | Full `AgentState` snapshot |
| `$WS/output/{repo}_patches_*/` | Patches applied during the run |
| `$WS/packages/{owner}-{repo}-*-{platform}.zip` | Packaged artifact (with `--package`); `manifest.json` holds the verification verdict, the riscv64 outputs with SHA-256, the source commit and the package test result |
| `$WS/repos/{repo}.opt/accepted/` | The accepted tune's riscv64 artifacts (with `--optimize`) |
| `$WS/output/machine_profile_{target}.json` | The machine probe (with `--probe`) |
| `$WS/logs/agent_{repo}.log` | Per-repo DEBUG log |
| `$WS/logs/agent-call_{repo}.log` | Full LLM call audit trail (prompt + response + cost) |
| `data/recipe_cache.json` | Successful builds, keyed by `{package: {sandbox: recipe}}` |
| `data/examples/*.json` | Few-shot examples per agent (auto-learning enabled) |

A cache hit short-circuits the pipeline - it re-renders `{repo}_recipe.md`
from the cached recipe and skips all LLM and Docker work - unless `--force`
is set. The run prints `CACHED (not rebuilt)`. Cache entries are
per-sandbox; Alpine and Debian builds populate separate keys, and an entry
counts only when its repository URL matches. A new entry keeps the
verification verdict and the source commit of its build. An entry without
a verified verdict (every entry written before verification existed)
prints `NOT VERIFIED` and exits with code 3.

`main.py --repo` exit codes:

| Code | Meaning |
|---|---|
| 0 | Success: riscv64 output verified by its ELF headers |
| 1 | Failure (build, verification, setup, or keys) |
| 2 | The build succeeded but packaging (`--package`) failed |
| 3 | The build commands succeeded but no riscv64 output was proven (for example a header-only library, or a scan that stopped at its file limit); not cached or learned. Also a cache hit whose entry has no verified verdict |
| 130 | Interrupted |

`batch_test.py` reports PASS (0), UNVERIFIED (3) and FAIL (anything else)
separately. The pass rate counts verified passes only.

The package test phase uses at most 600 s. Set
`ATESOR_PACKAGE_TEST_TIMEOUT` to change the limit. `batch_test.py` gives
each run its kill time in `ATESOR_RUN_DEADLINE` (epoch seconds); the phase
then keeps 420 s of that time for the report and the zip. A limit under
60 s skips the phase, so `ATESOR_PACKAGE_TEST_TIMEOUT=0` turns it off.

---

## Key features

- **Native RISC-V builds** - no cross-compilation, no surprises at deploy time.
- **Multiple Build Systems** - Make, CMake, Meson, autotools, Cargo, Go modules. More are coming.
- **Parallel batch runs** - `batch_test.py` allocates one container per worker (`atesor-ai-sandbox-w0..wN`) to avoid `apk`/`apt` lock contention.
- **Few-shot memory** - agents learn from past successes; up to 100 examples per agent, retrieved by keyword/regex.
- **Recipe cache** - successful builds are replayable and skip the LLM entirely.
- **ELF verification** - the ELF header of every build output (`EM_RISCV`, 64-bit class) is read in the sandbox; committed prebuilt binaries are reported but do not decide the verdict.
- **Performance mode** - `--optimize` probes the machine, finds the package type, has the optimizer agent design a workload for it, builds tunes for the machine, and keeps a tune only when it is measurably faster on native hardware.
- **Cost-aware** - every LLM call is logged with token estimate and cost; hard cap at $1.00 per package.
- **Safe execution** - every shell command goes through a regex whitelist and runs inside the sandbox.

---

## Development

```bash
# Run the full test suite (PYTHONPATH=. is required)
PYTHONPATH=. pytest

# Single file or test case
PYTHONPATH=. pytest tests/test_graph_routing.py
PYTHONPATH=. pytest tests/test_state.py::TestState::test_add_error
```

Style: PEP 8, 79-char lines, Google-style docstrings, type hints on public APIs. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

---

## Contributing

Contributions are welcome - especially new platform profiles, additional few-shot examples from real porting runs, and bug reports for packages that fail in interesting ways. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding standards, and how to extend the system.

- [Open an issue](https://github.com/akifejaz/atesor-ai/issues)
- [License: MIT](LICENSE)

Built for the RISC-V community. Making the ecosystem catch up, one package at a time.
