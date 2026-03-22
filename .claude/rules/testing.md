# Testing Rules

These rules MUST be followed for all testing in this project.

## Coverage Requirements

| Code Type | Minimum Coverage |
|-----------|-----------------|
| General code | 80% |
| Critical business logic | 100% |
| Security-related code | 100% |
| Data validation | 100% |

## Test-Driven Development (TDD)

### Required Workflow: RED-GREEN-REFACTOR

```
1. RED    → Write failing test first
2. GREEN  → Write minimal code to pass
3. REFACTOR → Improve code quality
```

### TDD is Required For:
- New features
- New functions/classes
- Bug fixes (write reproducing test FIRST)
- Critical business logic

## Test Structure

### File Organization
```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_models.py
│   └── test_services.py
├── integration/
│   ├── __init__.py
│   └── test_api.py
└── e2e/                 # End-to-end tests
    └── test_workflows.py
```

### Test Naming
```python
# Pattern: test_<what>_<condition>_<expected>
def test_calculate_total_empty_cart_returns_zero():
    ...

def test_authenticate_invalid_password_raises_error():
    ...
```

### Test Structure (AAA Pattern)
```python
def test_user_creation():
    # Arrange
    user_data = {"name": "John", "email": "john@example.com"}

    # Act
    user = User.create(user_data)

    # Assert
    assert user.name == "John"
    assert user.email == "john@example.com"
```

## Pytest Best Practices

### Use Fixtures
```python
# conftest.py
import pytest

@pytest.fixture
def sample_user():
    return User(id=1, name="Test User")

@pytest.fixture
def mock_database(mocker):
    return mocker.patch("src.adapters.database.Database")

# test_user.py
def test_user_service(sample_user, mock_database):
    mock_database.get_user.return_value = sample_user
    result = user_service.get(1)
    assert result == sample_user
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input_value,expected", [
    (0, "zero"),
    (1, "one"),
    (-1, "negative"),
])
def test_number_to_word(input_value, expected):
    assert number_to_word(input_value) == expected
```

### Testing Exceptions
```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="Input cannot be empty"):
        process_data("")
```

## Mock Guidelines

### When to Mock
- External APIs
- Databases
- File system operations
- Time-dependent operations
- Third-party services

### When NOT to Mock
- Pure functions
- Simple data transformations
- Your own code under test

### Mock Example
```python
from unittest.mock import MagicMock, patch

def test_api_client():
    with patch("src.adapters.api.httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_client.return_value.get.return_value = mock_response

        result = api_client.fetch_data()

        assert result["status"] == "ok"
```

## Test Categories

### Unit Tests
- Test single functions/classes
- Fast execution (milliseconds)
- No external dependencies
- Run with: `make test`

### Integration Tests
- Test component interactions
- May use real dependencies
- Marked with: `@pytest.mark.integration`
- Run with: `uv run pytest -m integration`

### End-to-End Tests
- Test complete workflows
- Full system behavior
- Marked with: `@pytest.mark.e2e`
- Run with: `uv run pytest -m e2e`

## Test Commands

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific file
uv run pytest tests/unit/test_models.py

# Run specific test
uv run pytest -k "test_calculate_total"

# Run with verbose output
uv run pytest -v

# Run only unit tests
uv run pytest tests/unit/

# Run marked tests
uv run pytest -m "not slow"
```

## When Tests Fail

1. **DO NOT modify tests to pass** (unless test is wrong)
2. **Fix the implementation** instead
3. If tests are incorrect:
   - Document why
   - Get review approval
   - Then modify test

## Pre-commit Test Verification

Before committing:
```bash
make test       # All tests pass
make lint       # Ruff passes
make typecheck  # Ty passes
```

## Test Quality Checklist

- [ ] Tests are independent (no shared state)
- [ ] Tests are deterministic (same result every run)
- [ ] Tests are fast (unit tests < 100ms each)
- [ ] Tests have descriptive names
- [ ] Edge cases are covered
- [ ] Error conditions are tested
- [ ] No `print()` statements in tests
