# Agreed PEP 8 Exceptions & Edge Cases

These are deliberate, documented deviations from strict PEP 8 for this project.

---

## Accepted Exceptions

### E501 — Line too long
- URLs in comments/docstrings may exceed 79 chars if breaking them would make
  them unfindable. Put the URL alone on its line with a `# noqa: E501` comment.

```python
# See full spec at:
# https://very-long-domain.example.com/path/to/really/long/documentation/page  # noqa: E501
```

### W503 / W504 — Line break before/after binary operator
- Line break **before** the operator is preferred (W504 style):

```python
# Preferred
total = (
    first_value
    + second_value
    - discount
)
```

### E203 — Whitespace before ':'
- Allowed in slice notation when aligning columns in NumPy-style arrays.

---

## When to Use `# noqa`

Use sparingly. Always add the specific code, never bare `# noqa`.

```python
from module import *  # noqa: F401  -- re-exported intentionally
```

Document *why* the suppression is needed in the same comment.

---

## `__init__.py` Public API Pattern

It's acceptable (and encouraged) to re-export names in `__init__.py` to define
the package's public API, even if linters flag unused imports:

```python
# mypackage/__init__.py
from mypackage.core import MyClass  # noqa: F401
from mypackage.utils import helper  # noqa: F401
```
