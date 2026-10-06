---
name: code-comments
description: >
  Review a repository and write, revise, or audit code comments and
  documentation comments. Use when asked to add comments, improve comments,
  explain non-obvious code, standardize docstrings, review TODO or FIXME
  markers, remove stale comments, or define a comment policy. Read local
  guidance and representative code first. Then apply language-specific syntax
  and minimal comments that explain contracts, constraints, and reasons.
argument-hint: "[path or scope] [write, review, or policy]"
---

# Code Comments

Use comments to preserve information that the code cannot express clearly.
Do not use comments as a substitute for clear names or simple control flow.

## Modes

Select one mode from the request:

- **Write**: Add or revise comments in a specified code area.
- **Review**: Find stale, missing, redundant, or misleading comments.
- **Policy**: Derive and document a repository-wide comment standard.

If the request does not specify a mode, use **Write** for an edit request and
**Review** for an assessment request.

## Required Workflow

### 1. Set the scope

Identify the target files, language, and requested mode. Do not expand a
comment-only request into a code refactor unless incorrect code blocks an
accurate comment.

### 2. Read the repository rules

Read the nearest applicable guidance before you inspect comment style:

- `AGENTS.md`, `CLAUDE.md`, and `.github/copilot-instructions.md`
- `CONTRIBUTING.md`, `README.md`, and architecture documents
- Formatter, linter, and documentation-tool configuration
- Language-specific skill or instruction files

Do not read secret files such as `.env`, credentials, keys, or tokens.

Use this order when rules conflict:

1. Explicit repository instructions
2. Formatter, linter, and documentation-tool configuration
3. Language or framework standards
4. The dominant pattern in maintained neighboring code
5. The defaults in this skill

### 3. Inspect a small representative sample

Read the complete target file and its nearest related test or caller. For a
broad review, sample one maintained file from each important file category.

Search for:

- File headers and module documentation
- Public API documentation
- Rationale and invariant comments
- Section headings and data-group labels
- `TODO`, `FIXME`, `HACK`, and suppression directives
- Generated-file markers and tool directives
- Commented-out code and change-history comments

Do not copy a pattern only because it occurs often. Check whether the pattern
agrees with the explicit repository rules.

### 4. Classify the information

Load [the comment catalog](./references/comment-catalog.md). Select the
category before you write the comment.

Use this decision sequence:

1. Can a clearer name or expression communicate the information?
   Prefer code when the task permits code changes.
2. Does the information define a public contract?
   Use the language's documentation-comment format.
3. Does it explain a hidden reason, constraint, invariant, or risk?
   Use a nearby block comment.
4. Does it suppress or control a tool?
   Use the narrowest directive and include the reason.
5. Does it record future work or a known defect?
   Use a traceable task marker with a removal condition.
6. Does it only repeat the next statement?
   Do not add it. Remove it during cleanup work.

### 5. Write the comment

Apply these rules unless the repository has a different explicit standard:

- Explain **why**, the contract, or the hidden constraint.
- Put the comment immediately before the smallest affected code block.
- Use an inline comment only for a short local exception or unit.
- Use active voice and the simple present tense.
- Use complete sentences for explanatory comments.
- Use the repository's line length and comment syntax.
- Name exact symbols, versions, units, and failure modes when relevant.
- State a workaround's removal condition.
- Keep one term for one concept throughout the file.
- Preserve required license and generated-file headers exactly.

Do not add:

- Narration of visible operations
- Speculation that the code cannot verify
- Personal notes, blame, jokes, or temporary debugging text
- Change history that belongs in version control
- Commented-out code
- Broad tool suppressions without a reason
- Secrets, private URLs, credentials, or user data

### 6. Check the comment against the code

Confirm each technical claim from the implementation, test, configuration, or
linked source. Check names, units, versions, error conditions, lock behavior,
path behavior, and cleanup behavior.

Treat these comments as executable behavior:

- Linter and type-checker suppressions
- Coverage exclusions
- ShellCheck directives
- Parser directives and encoding declarations
- Generated-file markers
- Documentation examples and doctests

### 7. Validate the change

Run the narrowest available formatter, linter, documentation check, syntax
check, or test for the touched file. A comment can change parsing, doctests,
coverage, generated output, and tool behavior.

For a comment-only change with no applicable executable check, inspect the
diff and verify every claim against nearby code.

## Review Output

For review mode, report findings in this order:

1. Misleading or stale comments
2. Unsafe or unexplained tool directives
3. Missing public contracts or critical constraints
4. Redundant narration and excessive headings
5. Style inconsistencies

Give the file location, the problem, and a replacement or removal action.
Do not report a preference as a defect when local rules permit both styles.

## Atesor AI Defaults

Use these defaults in this repository unless a nearer instruction overrides
them:

- Use PEP 8 comments and Google-style Python docstrings.
- Keep Python lines at 79 characters.
- Document each public Python module, class, function, and method.
- Document a private helper only when its contract is not clear from its name.
- Keep explanatory comments minimal and focused on hidden reasons.
- Preserve existing source-file license headers.
- Use short semantic headings in long modules. Avoid decorative banners.
- Use brief category labels in large mappings and state definitions.
- Explain security boundaries, concurrency rules, resource ownership, and
  provider or platform workarounds.
- In Dockerfiles, explain layer purpose, version pins, retry reasons, cache
  choices, and packages that must not be installed.
- Give each suppression directive a narrow scope and a concrete reason.

## Resources

Read [the comment catalog](./references/comment-catalog.md) for category
rules, language examples, anti-patterns, and external standards.

For one local edit, load only the relevant catalog sections. For a broad
review or policy task, read the complete catalog.