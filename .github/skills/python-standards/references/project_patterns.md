# Project-Specific Patterns & Examples

Concrete templates for common code patterns in this codebase.

---

## Standard Module Layout

Every non-trivial module should follow this structure order:

```python
"""Module docstring."""

# 1. Imports (isort order)
import os
from typing import Optional

import requests

from mypackage.config import settings

# 2. Module-level constants
MAX_RETRIES: int = 3
DEFAULT_TIMEOUT: int = 30

# 3. Module-level type aliases (if any)
UserId = int

# 4. Classes (most important first)
class MyClass:
    ...

# 5. Top-level functions (public first, private last)
def public_function() -> None:
    ...

def _private_helper() -> str:
    ...

# 6. Script entry point (if applicable)
if __name__ == "__main__":
    main()
```

---

## Class Template

```python
class DataProcessor:
    """Process and transform raw data records.

    Applies configurable validation and normalisation rules
    to incoming data before storage.

    Attributes:
        schema: Validation schema applied to each record.
        strict: If True, raises on first validation failure.
    """

    DEFAULT_BATCH_SIZE: int = 100  # Class-level constant

    def __init__(self, schema: dict, strict: bool = False) -> None:
        self.schema = schema
        self.strict = strict
        self._errors: list[str] = []  # Private state

    def process(self, records: list[dict]) -> list[dict]:
        """Process a batch of records and return valid ones.

        Args:
            records: Raw data records to validate and transform.

        Returns:
            List of records that passed validation.
        """
        ...

    def _validate_record(self, record: dict) -> bool:
        """Return True if the record passes schema validation."""
        ...
```

---

## Error Handling Pattern

```python
import logging

logger = logging.getLogger(__name__)


def fetch_data(url: str) -> Optional[dict]:
    """Fetch JSON data from a URL.

    Args:
        url: The endpoint to request.

    Returns:
        Parsed JSON as a dict, or None on failure.

    Raises:
        ValueError: If url is empty.
    """
    if not url:
        raise ValueError("url must not be empty")

    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        logger.error("HTTP error fetching %s: %s", url, e)
        return None
    except requests.RequestException as e:
        logger.error("Network error fetching %s: %s", url, e)
        return None
```

---

## Test File Template

```python
"""Tests for mypackage.module_name."""

import pytest

from mypackage.module_name import MyClass, public_function


class TestMyClass:
    """Tests for MyClass."""

    def test_init_sets_attributes(self) -> None:
        """Constructor should set schema and strict attributes."""
        obj = MyClass(schema={"key": "value"}, strict=True)
        assert obj.schema == {"key": "value"}
        assert obj.strict is True

    def test_process_returns_valid_records(self) -> None:
        """process() should filter out invalid records."""
        ...


class TestPublicFunction:
    """Tests for public_function."""

    def test_returns_expected_value(self) -> None:
        ...

    @pytest.mark.parametrize("input,expected", [
        ("a", 1),
        ("b", 2),
    ])
    def test_parametrized(self, input: str, expected: int) -> None:
        assert public_function(input) == expected
```
