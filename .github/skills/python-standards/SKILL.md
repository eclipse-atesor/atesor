---
name: python-standards
description: >
  Enforce consistent Python coding standards across the repository. Use this
  skill when writing new Python code, reviewing or refactoring existing Python
  files, auditing a file or module for style, fixing naming conventions,
  standardizing docstrings or comments, reformatting code, or cleaning up
  file and folder structure. Trigger for any Python task including casual
  requests like "clean up this file", "write a Python function",
  "review my code", or "make the repo consistent".
  Keywords: python, pep8, pep 8, refactor, review, docstring, naming,
  imports, formatting, clean code, code style, type hints, standards.
allowed-tools:
  - read_file
  - write_file
  - list_directory
---

# Python Coding Standards

Apply these standards to **all Python code** — new and existing. When in
doubt, refer to PEP 8. The rules below are the agreed house style for this
repository.

## Quick Reference

| Rule             | Value                                     |
|------------------|-------------------------------------------|
| Base standard    | PEP 8                                     |
| Line length      | **79 characters** (hard limit)            |
| Docstring format | **Google style**                          |
| Type hints       | Public functions & methods only           |
| Import order     | isort: stdlib → third-party → local       |
| Indentation      | 4 spaces (no tabs)                        |
| String quotes    | Double quotes `"` (single inside f-strings) |
| Trailing spaces  | Never                                     |
| Blank lines      | 2 between top-level defs, 1 inside classes|

---

## 1. File & Folder Structure

```
project/
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── module_name.py      # snake_case filenames
│       └── subpackage/
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_module_name.py    # mirrors src structure
├── docs/
├── scripts/                   # one-off / CLI scripts
├── pyproject.toml
├── requirements.txt
└── README.md
```

- All filenames: `snake_case.py` — no spaces, hyphens, or CamelCase
- Every package directory must have an `__init__.py`
- Test files mirror `src/` with a `test_` prefix
- Scripts not part of the package go in `scripts/`

---

## 2. Naming Conventions

| Entity              | Style                        | Example                  |
|---------------------|------------------------------|--------------------------|
| Variables           | `snake_case`                 | `user_count`             |
| Functions           | `snake_case`                 | `get_user_by_id()`       |
| Methods             | `snake_case`                 | `self.parse_response()`  |
| Classes             | `PascalCase`                 | `UserAuthManager`        |
| Constants           | `UPPER_SNAKE`                | `MAX_RETRIES = 3`        |
| Private attrs/funcs | `_single_leading_underscore` | `_validate()`            |
| Modules             | `snake_case`                 | `auth_utils.py`          |
| Packages            | `snake_case`                 | `data_pipeline/`         |
| Type aliases        | `PascalCase`                 | `UserId = int`           |

**Never:**
- Single-letter names except loop counters (`i`, `j`, `k`) or math (`x`, `y`)
- Abbreviations that aren't universally known (`usr` → `user`, `cfg` → `config`)
- Shadowing builtins (`list`, `id`, `type`, `input`)

---

## 3. Imports

Three groups, each separated by a blank line:

```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Optional

# 2. Third-party
import requests
from pydantic import BaseModel

# 3. Local / project
from mypackage.utils import format_date
from mypackage.models import User
```

- Absolute imports preferred over relative
- No wildcard imports (`from module import *`)
- One import per line for top-level modules
- No unused imports — remove them

---

## 4. Docstrings — Google Style

Every public module, class, function, and method must have a docstring.

### Module
```python
"""Short summary of what this module does.

Optional longer description explaining design decisions or usage.
"""
```

### Function / method
```python
def fetch_user(user_id: int, active_only: bool = True) -> Optional[User]:
    """Fetch a user record from the database by ID.

    Args:
        user_id: The unique identifier of the user.
        active_only: If True, only returns the user if their
            account is active. Defaults to True.

    Returns:
        A User object if found, or None if no matching record exists.

    Raises:
        DatabaseError: If the database connection fails.
        ValueError: If user_id is not a positive integer.

    Example:
        >>> user = fetch_user(42)
        >>> print(user.name)
        'Alice'
    """
```

### Class
```python
class PaymentProcessor:
    """Handles payment transactions via the Stripe API.

    Attributes:
        api_key: The Stripe secret key used for authentication.
        max_retries: Maximum number of retry attempts on failure.
    """
```

- First line: one sentence, imperative mood, ends with period
- Blank line between summary and body sections
- `__init__` gets no docstring — put it on the class instead
- Private helpers may use a single-line docstring

---

## 5. Type Hints

Apply to **all public** functions and methods (those without a `_` prefix):

```python
from typing import Optional, List, Dict, Tuple

# Public — must have type hints
def process_orders(
    orders: List[Dict[str, int]],
    dry_run: bool = False,
) -> Tuple[int, int]:
    ...

# Private — type hints optional
def _build_query(filters):
    ...
```

- Use `Optional[X]` for nullable params (not `X | None`)
- Always annotate return type on public functions, including `-> None`
- Avoid `Any`; if used, add a comment explaining why

---

## 6. Formatting

### Line length — 79 chars hard limit
Use parentheses to break long lines (not backslash):
```python
result = (
    some_long_variable_name
    + another_long_variable_name
    + yet_another_variable
)

response = requests.post(
    url=endpoint,
    headers=auth_headers,
    json=payload,
    timeout=30,
)
```

### Blank lines
- 2 blank lines before/after top-level class or function definitions
- 1 blank line between methods inside a class
- Use blank lines inside functions sparingly — only to separate logical steps

### String quotes
```python
message = "Hello, world"           # double quotes standard
query = f"SELECT * FROM {table}"   # double in f-strings
escaped = 'She said "hello"'       # single only to avoid escaping
```

### Trailing commas
Add in multi-line collections and function signatures:
```python
ALLOWED_METHODS = [
    "GET",
    "POST",
    "DELETE",    # trailing comma
]
```

---

## 7. Comments

### Inline
```python
x = x + 1  # Compensate for border offset
```
- 2 spaces before `#`, one space after
- Explain *why*, not *what*

### Block
```python
# Retry up to MAX_RETRIES times with exponential backoff.
# The first attempt is immediate; subsequent ones double the wait.
for attempt in range(MAX_RETRIES):
    ...
```
- Full sentences, capitalised, ending with period
- Same indentation as the code below

### TODO / FIXME
```python
# TODO(username): Refactor once API v2 is stable.
# FIXME: This breaks when offset exceeds list length.
```

---

## 8. Auditing Existing Code

When asked to review or refactor an existing file:

1. Read the full file before making changes
2. Report violations grouped by: Naming · Imports · Docstrings · Type hints · Formatting · Structure
3. Rewrite in one pass — apply all fixes together
4. Preserve logic — never change behaviour during a reformat
5. Flag ambiguous names and ask before renaming if intent is unclear

---

## 9. Common Anti-Patterns to Fix

```python
# ❌ Mutable default argument
def add_item(item, lst=[]):  ...
# ✅
def add_item(item, lst=None):
    if lst is None:
        lst = []

# ❌ Bare except
try: ...
except: pass
# ✅
try: ...
except ValueError as e:
    logger.warning("Invalid value: %s", e)

# ❌ String concatenation in loop
result = ""
for item in items:
    result += str(item)
# ✅
result = "".join(str(item) for item in items)

# ❌ Type check with ==
if type(x) == list:  ...
# ✅
if isinstance(x, list):  ...
```

---

## 10. Reference Files

For extended detail, load on demand:
- `references/pep8_exceptions.md` — Agreed deviations and `# noqa` rules
- `references/project_patterns.md` — Module layout, class, error handling, and test templates
