# Python Coding Style Rules

These rules MUST be followed for all Python code in this project.

## Core Principles

1. **Readability First**: Code is read more than written
2. **Simplicity**: Prefer the simplest working solution
3. **DRY**: Don't Repeat Yourself - extract common logic
4. **YAGNI**: You Aren't Gonna Need It - only build what's needed

## File Organization

### Maximum Sizes
- **Functions**: 50 lines maximum
- **Files**: 500 lines maximum (800 absolute limit)
- **Nesting**: 4 levels maximum

### Directory Structure
```
src/
├── __init__.py
├── main.py          # Entry point
├── config/          # Configuration
├── domain/          # Business logic
├── adapters/        # External interfaces
└── utils/           # Shared utilities
```

## Naming Conventions

### Variables and Functions
```python
# Use snake_case
user_name = "John"
def calculate_total(items: list) -> float: ...

# Descriptive names
# BAD
x = get_d()
# GOOD
user_data = get_user_details()
```

### Classes
```python
# Use PascalCase
class UserRepository:
    pass

class OrderProcessor:
    pass
```

### Constants
```python
# Use UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"
```

## Type Hints (Required)

All functions MUST have type hints.

```python
# REQUIRED
def process_user(user_id: int, options: dict[str, Any] | None = None) -> User:
    ...

# For complex types, use type aliases
UserData = dict[str, str | int | list[str]]
def parse_user(data: UserData) -> User:
    ...
```

## Docstrings

### Public APIs (Required)
```python
def calculate_discount(price: Decimal, rate: float) -> Decimal:
    """Calculate discounted price.

    Args:
        price: Original price
        rate: Discount rate (0.0 to 1.0)

    Returns:
        Discounted price

    Raises:
        ValueError: If rate is not between 0 and 1
    """
    if not 0 <= rate <= 1:
        raise ValueError("Rate must be between 0 and 1")
    return price * Decimal(1 - rate)
```

### Internal Functions (Optional)
For internal functions, docstrings are optional if the function name and type hints are self-explanatory.

## Error Handling

### Always Catch Specific Exceptions
```python
# BAD
try:
    result = process()
except:
    pass

# GOOD
try:
    result = process()
except ValidationError as e:
    logger.warning(f"Validation failed: {e}")
    raise
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    raise ServiceError("Failed to process") from e
```

### Use Custom Exceptions
```python
class DomainError(Exception):
    """Base exception for domain errors."""
    pass

class NotFoundError(DomainError):
    """Resource not found."""
    pass
```

## Imports

Managed by Ruff (isort rules).

```python
# Standard library
import os
from collections.abc import Callable
from pathlib import Path

# Third-party
import httpx
from pydantic import BaseModel

# Local
from src.config import settings
from src.domain.models import User
```

## Code Quality Checklist

Before committing, verify:

- [ ] Functions under 50 lines
- [ ] Files under 500 lines
- [ ] Nesting depth <= 4
- [ ] All public functions have type hints
- [ ] No `print()` or `breakpoint()` statements
- [ ] No hardcoded values (use constants)
- [ ] Error handling is specific (no bare `except:`)
- [ ] Tests exist for new code

## Ruff Configuration

These rules are enforced by Ruff in `pyproject.toml`:

```toml
[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
]
```

## Quick Reference

| Rule | Limit |
|------|-------|
| Line length | 88 characters |
| Function length | 50 lines |
| File length | 500 lines |
| Nesting depth | 4 levels |
| Test coverage | 80% minimum |
