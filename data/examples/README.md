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
are treated as `alpine-riscv64`.

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
    "module_dir": ""
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

After sandbox filtering, examples are scored by:

- Build system match: `+0.5`
- Fixer `error_pattern` regex match: `+0.4`
- Each tag found in the error message: `+0.15`
- `has_main` match: `+0.1`
- Exact `module_dir` match: `+0.15`
- Both module dirs present but different: `+0.05`
- CGO context plus `cgo` tag: `+0.2`
- Same sandbox bonus: `+0.25`
- Different sandbox penalty: `-0.35` (normally unreachable because of
  the hard filter)

Scores are capped at `1.0`; only positive-scoring examples are shown.

## Auto-Learned Examples

Auto-learned examples set `source: "auto"` and receive ids like
`scout-auto-001`. When an agent file exceeds 100 examples, manual
examples are kept and the oldest auto-learned examples are pruned first.
