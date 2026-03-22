# /learn Command

Extract and document patterns discovered during the current session.

## Usage
```
/learn
/learn <specific pattern or concept>
```

## What This Command Does

1. **Analyzes**: Current session's code changes and discussions
2. **Extracts**: Reusable patterns and learnings
3. **Documents**: Adds to project knowledge base

## When to Use

- After solving a complex problem
- When discovering a new pattern
- After debugging a tricky issue
- When implementing a novel solution
- Before ending a long session

## Learning Categories

### 1. Code Patterns

```python
# Pattern: Retry with exponential backoff
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        return wrapper
    return decorator
```

### 2. Bug Patterns

```markdown
## Bug: Circular Import Error

### Symptom
ImportError: cannot import name 'X' from partially initialized module

### Root Cause
Two modules importing each other at module level

### Solution
Use TYPE_CHECKING for type hints:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import OtherClass
```

### Prevention
- Avoid circular dependencies in design
- Use dependency injection
- Move shared types to separate module
```

### 3. Configuration Patterns

```markdown
## Pattern: Environment-based Configuration

### Implementation
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False
    database_url: str

    class Config:
        env_file = ".env"
```

### When to Use
- Multiple deployment environments
- Sensitive configuration values
- 12-factor app compliance
```

### 4. Testing Patterns

```markdown
## Pattern: Parametrized Fixture

### Implementation
```python
@pytest.fixture(params=["sqlite", "postgres"])
def database(request):
    if request.param == "sqlite":
        return SQLiteDatabase(":memory:")
    return PostgresDatabase(TEST_DB_URL)
```

### Benefit
Run same tests against multiple backends
```

## Output Format

```markdown
## Session Learning Report

### Session Summary
- Duration: 2 hours
- Focus: User authentication implementation
- Files modified: 5

### Patterns Extracted

#### 1. JWT Token Validation Pattern
**Category**: Security
**Location**: `src/auth/jwt.py`
**Description**: Reusable JWT validation with proper error handling
**Code**:
```python
def validate_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid token")
```

#### 2. Database Transaction Context Manager
**Category**: Database
**Location**: `src/db/transaction.py`
**Description**: Safe transaction handling with automatic rollback
**Code**:
```python
@contextmanager
def transaction(session):
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
```

### Bugs Resolved

#### Bug: Race Condition in Cache Update
**Symptom**: Stale data returned intermittently
**Root Cause**: Cache update not atomic
**Solution**: Use Redis MULTI/EXEC for atomic updates
**Prevention**: Always use atomic operations for shared state

### Tools/Techniques Discovered

1. **UV cache clean**: Resolves dependency resolution issues
2. **Ruff rule lookup**: `uv run ruff rule E501` explains rules
3. **Pytest --lf**: Reruns only last failed tests

### Recommendations for Future

1. Add retry logic to all external API calls
2. Consider using Redis for session storage
3. Add structured logging for auth events
```

## Workflow

### Step 1: Trigger Learning

```
/learn
```

### Step 2: Review Extracted Patterns

Claude will analyze:
- Code changes in session
- Problem-solving approaches used
- New techniques discovered

### Step 3: Confirm and Save

Patterns are documented to:
- `LEARNINGS.md` (if exists)
- Session notes
- Relevant skill files

## Integration

### With Skills

Learned patterns can enhance existing skills:

```markdown
# .claude/skills/python-patterns.md

## Learned Patterns

### Session 2024-01-15: Retry Pattern
[Pattern details added automatically]
```

### With Rules

Bug patterns can become new rules:

```markdown
# .claude/rules/common-bugs.md

## Circular Import Prevention
[Rule added from learned bug pattern]
```

## Quick Reference

```bash
# Extract learnings from current session
/learn

# Extract specific pattern
/learn "JWT authentication flow"

# Extract bug pattern
/learn "circular import fix"
```

## Related Commands

- `/plan` - Plan implementation
- `/code-review` - Review code
- `/refactor-clean` - Clean up code
