# UV, Ruff, Ty Workflow

Domain knowledge for Astral's modern Python toolchain.

## UV - Package Manager

### Key Concepts
- **Rust-based**: 10-100x faster than pip
- **Automatic venv**: No manual activation needed
- **Lock files**: Reproducible builds with `uv.lock`

### Common Commands

```bash
# Project setup
uv sync                     # Install all dependencies
uv sync --all-extras --dev  # Install with all extras and dev deps

# Add dependencies
uv add httpx                # Add production dependency
uv add --dev pytest         # Add development dependency
uv add "pydantic>=2.0"      # Add with version constraint

# Remove dependencies
uv remove httpx

# Update dependencies
uv lock --upgrade           # Update lock file
uv lock --upgrade-package httpx  # Update specific package

# Run commands
uv run python src/main.py   # Run with managed venv
uv run pytest               # Run tests
uv run ruff check src/      # Run linting
```

### pyproject.toml Structure

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "Project description"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.7.0",
    "ty>=0.0.1a1",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pre-commit>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Ruff - Linter & Formatter

### Key Concepts
- **Replaces**: Black, isort, flake8, pylint
- **Rust-based**: 10-100x faster than Python alternatives
- **Auto-fix**: Can automatically fix many issues

### Configuration

```toml
[tool.ruff]
line-length = 88
target-version = "py312"
fix = true

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
ignore = []

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert in tests

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

### Commands

```bash
# Linting
uv run ruff check src/          # Check for issues
uv run ruff check src/ --fix    # Auto-fix issues

# Formatting
uv run ruff format src/         # Format code
uv run ruff format src/ --check # Check format without changing

# Combined
uv run ruff check src/ --fix && uv run ruff format src/
```

### Common Rules

| Rule | Description |
|------|-------------|
| E/W | pycodestyle (PEP 8) |
| F | pyflakes (errors) |
| I | isort (import sorting) |
| B | flake8-bugbear (common bugs) |
| UP | pyupgrade (modern Python) |
| SIM | simplify (code simplification) |
| S | flake8-bandit (security) |

### Inline Ignores

```python
# Ignore specific rule on line
from module import *  # noqa: F403

# Ignore multiple rules
value = eval(user_input)  # noqa: S307, B307

# Ignore for entire file (at top)
# ruff: noqa: F401
```

## Ty - Type Checker

### Key Concepts
- **Rust-based**: 10-20x faster than mypy
- **Gradual typing**: Works with partial type hints
- **Astral-made**: Part of the unified toolchain

### Configuration

```toml
[tool.ty]
# Currently using defaults (alpha version)
# Configuration options will expand as Ty matures
```

### Commands

```bash
# Type check entire project
uv run ty check

# Type check specific path
uv run ty check src/
```

### Inline Ignores

```python
# Ignore type error on line
result = dynamic_function()  # type: ignore

# Ignore specific error
value: int = get_value()  # type: ignore[assignment]
```

### Type Stub Files

For libraries without type hints, create `.pyi` stub files:

```python
# src/vendor.pyi
def external_function(data: dict[str, str]) -> list[int]: ...

class ExternalClass:
    def method(self, value: int) -> str: ...
```

## Makefile Integration

```makefile
# Development commands
install:
	uv sync --all-extras --dev

format:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check src/ tests/ --fix

typecheck:
	uv run ty check

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=term-missing

# Combined checks
check: lint typecheck test

# Pre-commit
pre-commit:
	uv run pre-commit run --all-files
```

## Pre-commit Integration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: ty
        name: ty
        entry: uv run ty check
        language: system
        types: [python]
        pass_filenames: false
```

## Workflow Summary

### Daily Development
```bash
# 1. Start work
uv sync                    # Ensure dependencies

# 2. Write code
# ... edit files ...

# 3. Check quality
make lint-fix              # Fix linting issues
make format                # Format code
make typecheck             # Check types

# 4. Test
make test                  # Run tests

# 5. Commit
git add .
git commit                 # Pre-commit hooks run automatically
```

### Dependency Updates
```bash
# Update specific package
uv add "httpx>=0.26"

# Update all dependencies
make update                # uv lock --upgrade

# Verify nothing broke
make check                 # lint + typecheck + test
```

## Troubleshooting

### UV Issues
```bash
# Clear cache
uv cache clean

# Force reinstall
rm -rf .venv && uv sync
```

### Ruff Conflicts
```bash
# Check which rules conflict
uv run ruff check src/ --show-fixes

# See rule explanation
uv run ruff rule E501
```

### Ty Limitations (Alpha)
- Some edge cases may not work yet
- Fall back to pyright if needed:
  ```bash
  uv add --dev pyright
  uv run pyright src/
  ```
