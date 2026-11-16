# Few-Shot Example Store

The scout, fixer and builder files are v2.0 few-shot stores loaded by
`src.memory.AgentMemory`. Runtime writes happen under a file lock and use
atomic JSON replacement. Prompts use the scout and fixer examples. Builder
examples are learned, but no prompt uses them yet.

`supervisor_examples.json` is a legacy v1.0 file (`context` and
`expected_output` fields). No code reads it: the supervisor routes by
rules and makes no LLM call.

## Files

```text
data/examples/
├── scout_examples.json
├── fixer_examples.json
├── builder_examples.json
└── supervisor_examples.json
```

## Common Fields

Each v2.0 file has:

```json
{
  "version": "2.0",
  "description": "Examples for one agent role",
  "examples": [
    {
      "id": "scout-001",
      "name": "Human-readable name",
      "tags": ["go", "cgo"],
      "build_system": "go",
      "source": "manual",
      "repo_name": "example",
      "sandbox": "alpine-riscv64",
      "timestamp": "2026-09-30",
      "reasoning": "Why this pattern applies"
    }
  ]
}
```

`sandbox` is important. Retrieval hard-filters out examples from a
different sandbox before scoring, so Alpine `apk` plans do not appear in
Debian/Ubuntu prompts and vice versa. Legacy examples without `sandbox`
are treated as `alpine-riscv64`. `"sandbox": "any"` marks an example
whose commands run on every distro (no package manager); it passes the
filter everywhere and gets no sandbox bonus.

## Scout Examples

Scout examples use `trigger` and `plan`:

```json
{
  "id": "scout-001",
  "name": "Go command in subdirectory",
  "tags": ["go", "module-dir"],
  "build_system": "go",
  "sandbox": "alpine-riscv64",
  "trigger": {
    "build_system": "go",
    "has_main": true,
    "main_path": "cmd/app",
    "module_dir": "",
    "has_cgo": false
  },
  "plan": {
    "phases": [
      {
        "name": "build",
        "commands": ["go build -buildvcs=false -o app ./cmd/app"]
      }
    ]
  },
  "reasoning": "Build the actual main package and disable Go VCS stamping."
}
```

The trigger holds the layout facts that tell two packages of one build
system apart: for Go `main_path`, `module_dir` and `has_cgo`; for other
build systems `nested_root` (a build file in a subdirectory); and
`build_env` (the `NAME=value` settings in front of the build command).
Learned examples read these from the final plan of a verified run.

## Fixer Examples

Fixer examples use `error_pattern` and `fix`:

```json
{
  "id": "fixer-001",
  "name": "Missing pkg-config module",
  "tags": ["dependency", "pkgconfig"],
  "build_system": "cmake",
  "sandbox": "debian-riscv64",
  "error_pattern": "No package 'zlib' found",
  "error_context": {"error_message": "No package 'zlib' found"},
  "fix": {
    "analysis": "The development package is missing.",
    "strategy": "Install the canonical distro package.",
    "actions": [
      {"type": "command", "command": "apt-get install -y zlib1g-dev"}
    ]
  },
  "reasoning": "zlib1g-dev provides zlib.pc on Debian/Ubuntu."
}
```

The key is the decisive error line, never the failed command and never
Go's `go: downloading` progress lines (both sides drop those before
matching). `error_context.error_message` stores that line. An example
with a valid `error_pattern` regex matches only when the regex finds
the error: a word match brought another error's fix (meson's
`C header 'zlib.h' not found` got the libcap example). An older auto
example, whose pattern starts with `Build\ command\ failed:`, matches
when its stored line (24 characters or more, with an error word) occurs
in the error, or when 60% of the distinctive words of the stored line
do. A fix that runs the Go tool or names Go files (`go.mod`, `*.go`)
reaches a non-Go build only when the error names Go. Actions are
`command`, `plan_command` (`replaces` and `command`: a change of a
build plan command), `create_file` (`path`, `content`) and `patch`
(`file`, `content`).

## Builder Examples

Builder examples use compact `phases` and may include
`timeout_recommendation`:

```json
{
  "id": "builder-001",
  "name": "Simple make build",
  "tags": ["make"],
  "build_system": "make",
  "sandbox": "alpine-riscv64",
  "phases": [
    {"name": "build", "commands": ["make -j$(nproc)"]}
  ],
  "timeout_recommendation": "10m",
  "reasoning": "The project has a root Makefile with a default target."
}
```

## Relevance Scoring

After sandbox filtering (and, for the fixer, the error match), examples
are scored by:

- Build system match: `+0.5`
- Fixer `error_pattern` regex match: `+0.4`
- Each tag found in the error message: `+0.15`
- `has_main` match: `+0.1`
- `main_path` match: `+0.3` (`./cmd/x` and `cmd/x` are equal); another
  main package of the same shape (both at the root, or both under
  `cmd/`): `+0.15`
- `module_dir` match: `+0.2` (both empty means both at the root)
- `nested_root` match: `+0.2`
- `has_cgo` match: `+0.1`; an example without `has_cgo` and a `cgo`
  tag gets `+0.2` when the context has cgo
- Same sandbox bonus: `+0.25`; `any` gets no bonus
- Different sandbox penalty: `-0.35` (normally unreachable because of
  the hard filter)

A layout term counts only when the context and the example both name
the fact. The ranking puts an example of the same package first, then
the higher score, then the newer `timestamp`. Only positive-scoring
examples are shown. The prompt section names the examples it holds,
and the audit log records their ids.

## Auto-Learned Examples

Auto-learned examples set `source: "auto"` and receive ids like
`scout-auto-001`. When an agent file exceeds 100 examples, manual
examples are kept and the oldest auto-learned examples are pruned first.

Only a verified port teaches. Every fix of the run that worked becomes a
fixer example; read-only fixes and stub files do not. A fix worked when
the build passed after it, or when one of its error lines is gone
afterwards (make's summary lines do not count). Its key is a line it
removed: a fix that removed the second of two errors is not keyed on
the first one. The fixer store
dedups on the error pattern, so one package can teach several fixes. A
newer verified run of a package replaces its auto scout and builder
examples (same id) when the plan or the trigger changed. The CI merge
gives a new id to an auto example whose id another shard used for a
different package.

## Failure Memory

`data/failures.json` records fixes after which the error stayed:
`packages[repo][sandbox][error_key]` holds up to five records
(`strategy`, `commands`, `error_after`, `date`), one per strategy. The
fixer and escalation nodes write it; the scout and fixer prompts of a
later run of the package show it as "Tried before". The CI merge keeps
the newest record per strategy.
