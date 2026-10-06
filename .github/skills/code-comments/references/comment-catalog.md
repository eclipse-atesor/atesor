# Comment Catalog

This catalog defines common comment categories. Local repository rules take
priority over this catalog.

## Category Summary

| Category | Purpose | Typical form |
|---|---|---|
| Legal header | State license or ownership requirements | Exact file header |
| File overview | Define the file's responsibility and boundaries | Module doc |
| Public contract | Document behavior visible to callers | Doc comment |
| Rationale | Explain a non-obvious design choice | Block comment |
| Invariant | State a condition that must remain true | Block comment |
| Security | Explain a trust boundary or protection | Block comment |
| Compatibility | Record a platform or version workaround | Block comment |
| Concurrency | Explain ownership, ordering, or synchronization | Block comment |
| Performance | Preserve an intentional measured tradeoff | Block comment |
| Section heading | Divide a long file into stable concepts | Short heading |
| Data grouping | Label related constants, fields, or mappings | Short label |
| Inline exception | Explain a small local exception or unit | Inline comment |
| Task marker | Record traceable future work or a known defect | TODO or FIXME |
| Provenance | Identify an external rule, source, or checksum | Source comment |
| Test intent | Explain a regression or non-obvious fixture | Test comment |
| Generated marker | Prevent edits and identify the generator | File header |
| Tool directive | Control a linter, type checker, or parser | Exact directive |

## 1. Legal and License Headers

Use a legal header only when the repository or organization requires it.
Preserve approved wording, punctuation, dates, and SPDX identifiers exactly.
Do not invent a copyright owner or add a new header without a requirement.

```python
# SPDX-License-Identifier: MIT
```

Place a required header where the language and toolchain permit it. A shebang,
encoding declaration, or parser directive can require the first line.

Avoid decorative borders unless the approved template includes them.

## 2. File and Module Overviews

Use a file overview to define responsibility, boundaries, and important
interactions. Do not list every class or function.

```python
"""Collect build evidence for planner and fixer prompts.

This module reads repository files but does not execute build commands.
"""
```

A useful overview answers one or more of these questions:

- What responsibility does this file own?
- What important responsibility does it not own?
- Which boundary or lifecycle does it control?
- Which design constraint affects the whole file?

Do not duplicate package documentation or architecture documents.

## 3. Public API Documentation

Use the language's documentation-comment format for behavior that callers
must know. Document the contract, not the implementation sequence.

Include only applicable information:

- Purpose and observable behavior
- Parameter meaning, accepted forms, units, and defaults
- Return meaning and sentinel values
- Raised errors and failure conditions
- Side effects, I/O, mutation, and resource ownership
- Thread, process, or async safety
- Security-sensitive expectations
- A short example when normal use is not obvious

Do not repeat types that the signature already states. Explain their meaning.

```python
def execute_build(command: str, timeout: int) -> CommandResult:
    """Run a validated command inside the active sandbox.

    Args:
        command: The validated shell command.
        timeout: Maximum run time in seconds.

    Returns:
        The captured exit status and output.

    Raises:
        TimeoutError: If the command exceeds ``timeout``.
    """
```

Do not add empty sections such as `Returns:` for a function that returns
`None`. Follow the repository's documentation tool when it requires them.

## 4. Rationale and Design Decisions

Use a rationale comment when a reasonable maintainer could simplify the code
and accidentally restore a known failure.

State the constraint first. Then state the consequence.

```python
# Let the provider timeout close the socket first. Otherwise, the wall
# timeout leaves the worker connection open for the rest of the request.
wall_timeout = provider_timeout + 15
```

Weak:

```python
# Add 15 seconds to the timeout.
wall_timeout = provider_timeout + 15
```

The weak comment repeats the expression and omits the reason.

## 5. Invariants, Preconditions, and Postconditions

Use an invariant comment for a condition that several operations depend on.
Enforce the condition in code when practical. A comment alone is not a guard.

```python
# All graph nodes mutate this shared state object. Replacing it breaks the
# audit trail that later nodes read.
state.phase = TaskPhase.BUILDING
```

Include the affected scope and the failure caused by a violation. Do not state
an invariant that tests or code contradict.

Use assertions for cheap internal checks. Use validation and explicit errors
for user-controlled input.

## 6. Security and Trust Boundaries

Use a security comment to explain a threat, trust boundary, or required
defense. Place it next to the protection.

```python
# Resolve links inside the sandbox before the write. A repository symlink
# must not redirect output to a host or container system path.
destination = resolve_in_sandbox(target)
```

A security comment should identify:

- The untrusted input or actor
- The protected resource or boundary
- The attack or failure that the code prevents
- Any assumption that another layer enforces

Do not include exploit secrets, credentials, or unnecessary attack steps.
Do not label normal validation as "security" without a specific boundary.

## 7. Compatibility and Workarounds

Use a workaround comment when behavior differs by platform, provider,
dependency, compiler, or version.

Include:

- The affected component and version range when known
- The observed failure
- Why the selected workaround is safe
- An issue, specification, or upstream reference when useful
- The condition that permits removal

```python
# Podman 5.2 returns 255 when a container exists but is stopped. Check the
# error text before treating 255 as an SSH transport failure.
```

Avoid indefinite comments such as `# Temporary workaround`. Define what
"temporary" means.

## 8. Concurrency and Resource Lifecycle

Use these comments for lock order, thread-local state, ownership transfer,
cleanup order, and process boundaries.

```python
# Keep authentication state thread-local. Concurrent clone workers can use
# different repository credentials.
self._auth = threading.local()
```

State:

- Who owns the resource
- When ownership starts and ends
- Which lock protects the state
- Required lock order
- What cleanup must occur after failure

Do not use a comment to excuse a race. Fix the synchronization when possible.

## 9. Performance and Resource Tradeoffs

Use a performance comment only for intentional behavior that looks wasteful
or for an optimization that reduces clarity.

```python
# Read only the first 64 KiB. Build logs can exceed 100 MiB, and error
# signatures occur near the start of the captured failure block.
sample = stream.read(64 * 1024)
```

Include a measurement, bound, workload, or complexity reason when available.
Do not write unsupported claims such as `# Faster` or `# Optimized`.

## 10. Section Headings

Use headings only when a long file contains stable conceptual regions. Prefer
a short heading over decorative banners.

```python
# Routing rules


def route_supervisor_to_next(...):
    ...
```

Avoid:

```python
# ===========================================================================
# ROUTING RULES
# ===========================================================================
```

Do not use headings to compensate for unrelated responsibilities in one file.
Consider a module split during a separate refactor.

## 11. Data and Configuration Grouping

Use a short label to group fields, constants, packages, or mapping entries.

```python
package_map = {
    # Compression libraries
    "zstd": "zstd-dev",
    "lz4": "lz4-dev",
}
```

Use explanatory comments for surprising omissions or mappings.

```python
# Do not map Rust to the distro package. The image provides a newer rustup
# toolchain, and the distro package would shadow it on PATH.
```

Avoid one comment per obvious entry.

## 12. Inline and Trailing Comments

Use an inline comment only when it is short and applies to one expression.
Follow the language's spacing rules.

```python
retry_delay = 0.25  # Seconds; keep provider retries below the CLI timeout.
```

Good inline uses include:

- Units that the type cannot express
- A narrow compatibility exception
- A short reason for an unusual constant
- A narrowly scoped tool directive

Move longer explanations above the statement. Avoid aligning comments into
columns because routine edits create formatting churn.

## 13. TODO, FIXME, and HACK Markers

Use task markers only when the repository permits them.

- `TODO`: Known future work that does not make current behavior incorrect.
- `FIXME`: Known incorrect or unsafe behavior that needs correction.
- `HACK`: A deliberate workaround with no clean local solution.

Use one repository-wide format. A good default is:

```python
# TODO(#184): Remove the fallback after provider v2 becomes mandatory.
# FIXME(#207): Reject links that resolve outside the extraction root.
# HACK(#233): Keep one I/O thread until rustup fixes archive issue 2926.
```

Each marker must contain:

- A tracking issue or owner
- A specific action or defect
- A removal or completion condition

Do not use `TODO: Improve this`, `XXX`, or an untracked personal reminder.

## 14. Source and Provenance Comments

Use a provenance comment for a checksum, protocol value, generated table,
external algorithm, specification rule, or borrowed compatibility value.

```dockerfile
# SHA256 source: https://go.dev/dl/?mode=json&include=all
ARG GO_SHA256=...
```

Prefer an authoritative and stable source. Include a section, issue number, or
version when the source can change. Do not cite a search result or secondary
summary when a primary source exists.

## 15. Test Comments

Use descriptive test names for normal cases. Add a comment for a regression,
non-obvious fixture, unusual mock, or assertion that protects an invariant.

```python
# Regression: a stopped Podman container also returns 255. The message must
# distinguish it from an SSH transport failure.
result = classify_exit(255, stopped_container_error)
```

Avoid `# Arrange`, `# Act`, and `# Assert` when the blocks are already clear.
Avoid comments such as `# Check the result` before an assertion.

If local rules require test docstrings, state the behavior under test. Do not
repeat the method name with spaces.

## 16. Generated Files

Put an exact generated-file marker near the top of the file. Identify the
generator and source when practical.

```text
# Generated by scripts/build_package_index.py from packages/full.json.
# Do not edit this file directly.
```

Edit the generator or source instead of the output. Preserve markers that a
tool uses to detect generated content.

## 17. Tool Directives and Suppressions

Treat a tool directive as code. Use the narrowest scope and exact rule code.
State why the code cannot satisfy the rule normally.

```python
from src import sandbox  # noqa: E402 - PATH setup must run before import.
```

```python
except ImportError:  # pragma: no cover - dependency is required in release builds
    ...
```

```yaml
# shellcheck disable=SC2086  # Intentional word splitting for package flags.
run: build $PACKAGE_FLAGS
```

Prefer a local directive over a file-wide suppression. Remove the directive
when the exception no longer exists. Confirm the tool accepts trailing reason
text before you add it.

## 18. Dockerfile, Shell, and Workflow Comments

### Dockerfiles

Explain why a layer exists, why a version is pinned, why retries are safe, and
how cache or cleanup behavior affects the image.

```dockerfile
# Install only shared build tools. The agent installs package-specific
# libraries on demand to keep the base image stable.
RUN apt-get update && apt-get install -y ...
```

Do not list every package again in prose.

### Shell scripts

Document destructive effects, privilege expectations, non-obvious quoting,
cleanup traps, and intentional ShellCheck exceptions.

```sh
# Preserve the command status before cleanup so the trap returns the original
# failure to the caller.
status=$?
```

### YAML workflows

Explain permission choices, event filters, concurrency keys, artifact
retention, and shell behavior that the YAML syntax does not make clear.

```yaml
# Cancel older runs because only the newest package index can be published.
cancel-in-progress: true
```

## 19. Language Documentation Forms

Use the repository's documentation generator when one exists.

### Python

- Use module, class, function, and method docstrings.
- Follow PEP 257 and the configured style, such as Google style.
- Put the summary on the first line.
- Do not document `self` or `cls` as parameters.

### C and C++

- Use `///` or `/** ... */` for documented declarations.
- Follow the configured Doxygen or Sphinx syntax.
- Document ownership, lifetime, nullability, units, and error codes.

### Go

- Put the doc comment directly before the exported declaration.
- Start it with the declared identifier when repository lint requires it.
- Document concurrency behavior and ownership of channels or resources.

### Rust

- Use `///` for item documentation and `//!` for module documentation.
- Document `# Errors`, `# Panics`, `# Safety`, and examples when applicable.
- Explain every `unsafe` block's safety conditions next to the block.

### JavaScript and TypeScript

- Use JSDoc or TSDoc for public APIs when the project generates API docs.
- Do not repeat TypeScript types in prose.
- Document mutation, rejected promises, units, and browser or runtime limits.

## 20. Comments to Remove

Remove a comment when it:

- Repeats the code
- Describes old behavior
- Refers to a symbol that no longer exists
- Contains commented-out code
- Records a completed change or historical edit
- Makes an unsupported performance or security claim
- Uses a heading for a single obvious statement
- Gives instructions that tests or tools contradict

Before removal, check whether a tool parses the comment.

## 21. Review Checklist

For each comment, check:

- **Correctness**: Does the code prove the claim?
- **Need**: Does the comment add information that code cannot express?
- **Category**: Does it use the correct comment or documentation form?
- **Scope**: Is it next to the smallest code block it governs?
- **Contract**: Does public documentation include important side effects and
  failure conditions?
- **Traceability**: Does temporary work have an issue and removal condition?
- **Tool behavior**: Is every directive narrow, valid, and justified?
- **Maintenance**: Will a nearby code change reveal that the comment is stale?
- **Safety**: Does the comment avoid secrets and private data?
- **Style**: Does it follow local syntax, line length, vocabulary, and tone?

## 22. External Standards

Use these primary references when local rules do not answer a question:

- PEP 8, Comments:
  https://peps.python.org/pep-0008/#comments
- PEP 257, Docstring Conventions:
  https://peps.python.org/pep-0257/
- Google Python Style Guide, Comments and Docstrings:
  https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- Dockerfile reference, comments and parser directives:
  https://docs.docker.com/reference/dockerfile/
- ShellCheck directives:
  https://www.shellcheck.net/wiki/Directive
- Doxygen documentation blocks:
  https://www.doxygen.nl/manual/docblocks.html
- Go documentation comments:
  https://go.dev/doc/comment
- Rust documentation comments:
  https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html
- Rust API Guidelines, documentation:
  https://rust-lang.github.io/api-guidelines/documentation.html
- TSDoc specification:
  https://tsdoc.org/

Repository instructions and configured tools remain authoritative.