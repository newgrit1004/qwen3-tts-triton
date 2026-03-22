# Python Project Structure

Domain knowledge for organizing Python projects.

## Standard Structure

```
project-name/
├── .claude/                    # Claude Code configuration
│   ├── agents/                 # Agent definitions
│   ├── commands/               # Slash commands
│   ├── rules/                  # Project rules
│   ├── skills/                 # Domain knowledge
│   └── hooks/                  # Automation hooks
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # GitHub Actions
│   └── dependabot.yml          # Dependency updates
├── src/                        # Source code
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config/                 # Configuration
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── domain/                 # Business logic
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── services.py
│   ├── adapters/               # External interfaces
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── database.py
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       └── helpers.py
├── tests/                      # Test directory
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures
│   ├── unit/
│   │   └── test_services.py
│   └── integration/
│       └── test_api.py
├── docs/                       # Documentation
│   └── api.md
├── .env.example                # Environment template
├── .gitignore
├── .gitmessage.txt             # Commit template
├── .pre-commit-config.yaml     # Pre-commit hooks
├── pyproject.toml              # Project configuration
├── uv.lock                     # Locked dependencies
├── Makefile                    # Development commands
├── CLAUDE.md                   # Claude Code instructions
└── README.md                   # Project documentation
```

## Directory Purposes

### `src/` - Source Code

| Directory | Purpose |
|-----------|---------|
| `config/` | Configuration and settings |
| `domain/` | Business logic, models, services |
| `adapters/` | External interfaces (API, DB, etc.) |
| `utils/` | Shared helper functions |

### `tests/` - Test Code

| Directory | Purpose |
|-----------|---------|
| `unit/` | Unit tests (isolated, fast) |
| `integration/` | Integration tests (real deps) |
| `e2e/` | End-to-end tests (full system) |
| `conftest.py` | Shared fixtures |

### `.claude/` - Claude Code Configuration

| Directory | Purpose |
|-----------|---------|
| `agents/` | Specialized AI agents |
| `commands/` | `/command` definitions |
| `rules/` | Mandatory guidelines |
| `skills/` | Domain knowledge |
| `hooks/` | Event-driven automation |

## Configuration Files

### pyproject.toml
```toml
[project]
name = "project-name"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["ruff", "ty", "pytest", "pytest-cov", "pre-commit"]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.pytest.ini_options]
pythonpath = "src"
testpaths = ["tests"]
```

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environment
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# Environment
.env
.env.local
*.local

# Coverage
htmlcov/
.coverage
coverage.xml

# Cache
.pytest_cache/
.ruff_cache/
.mypy_cache/
```

### Makefile
```makefile
.PHONY: install format lint lint-fix typecheck test test-cov clean

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
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

check: lint typecheck test

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
```

## Naming Conventions

### Files
| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `user_service.py` |
| Tests | test_ prefix | `test_user_service.py` |
| Config | lowercase | `settings.py` |

### Code
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `UserService` |
| Functions | snake_case | `get_user_by_id` |
| Variables | snake_case | `user_name` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |

## Module Organization

### __init__.py Pattern
```python
# src/domain/__init__.py
from .models import User, Order
from .services import UserService, OrderService

__all__ = ["User", "Order", "UserService", "OrderService"]
```

### Circular Import Prevention
```python
# Use TYPE_CHECKING for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import OtherClass

def function(obj: "OtherClass") -> None:
    ...
```

## Configuration Patterns

### Environment-Based Config
```python
# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False
    database_url: str
    api_key: str

    class Config:
        env_file = ".env"

settings = Settings()
```

### Usage
```python
from src.config import settings

if settings.debug:
    print("Debug mode enabled")
```
