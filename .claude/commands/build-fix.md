# /build-fix Command

Diagnose and fix build errors in the project.

## Usage
```
/build-fix
/build-fix <specific error message>
```

## What This Command Does

1. **Invokes**: `build-error-resolver` agent
2. **Analyzes**: Build output and error messages
3. **Fixes**: Dependency, import, type, and lint errors

## Workflow

### Step 1: Run Diagnostics

```bash
# Run full check to identify issues
make check
```

### Step 2: Identify Error Type

| Error Type | Indicator |
|------------|-----------|
| Dependency | `ModuleNotFoundError`, UV resolution error |
| Import | `ImportError`, circular import |
| Type | Ty error messages |
| Lint | Ruff error codes (E, W, F) |

### Step 3: Apply Fix

```bash
# Dependency issues
uv sync
uv add missing-package

# Lint issues
make lint-fix

# Type issues
# Manual fix based on Ty output

# Format issues
make format
```

### Step 4: Verify

```bash
make check  # Should pass
```

## Common Scenarios

### Scenario 1: Missing Dependency

```
Error: ModuleNotFoundError: No module named 'httpx'

Fix:
uv add httpx
```

### Scenario 2: Version Conflict

```
Error: Could not resolve dependencies

Fix:
uv lock --upgrade
# or relax version constraints in pyproject.toml
```

### Scenario 3: Pre-commit Failure

```
Error: pre-commit hook 'ruff' failed

Fix:
make lint-fix
make format
```

### Scenario 4: Type Error

```
Error: Ty: Incompatible return type

Fix:
# Update function signature or add type cast
```

## Quick Reference

```bash
# Full diagnostics
make check

# Individual checks
make lint        # Ruff lint
make lint-fix    # Auto-fix lint
make format      # Ruff format
make typecheck   # Ty check
make test        # pytest

# Dependency management
uv sync          # Install dependencies
uv lock          # Regenerate lock file
uv add package   # Add new dependency
```

## Output Format

```markdown
## Build Fix Report

### Errors Found
| Type | Location | Message |
|------|----------|---------|
| Lint | src/main.py:15 | E501: Line too long |
| Type | src/utils.py:30 | Incompatible return |

### Fixes Applied
1. `make lint-fix` - Auto-fixed 3 lint issues
2. Updated return type in `src/utils.py:30`

### Verification
- `make check`: PASSED
```

## Related Commands

- `/code-review` - Review code quality
- `/tdd` - Test-driven development
- `/refactor-clean` - Clean up code
