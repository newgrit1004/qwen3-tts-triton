# /code-review Command

Perform automated code review on uncommitted changes.

## Usage
```
/code-review
```

## What This Command Does

1. **Invokes**: `code-reviewer` agent
2. **Analyzes**: All uncommitted changes
3. **Reports**: Issues by severity level

## Review Process

### Step 1: Detect Changes
```bash
git diff          # Unstaged changes
git diff --staged # Staged changes
```

### Step 2: Analyze Each File
For each modified file:
- Security vulnerabilities
- Code quality issues
- Best practice violations

### Step 3: Generate Report

```markdown
## Code Review Report

### Summary
- Files reviewed: 5
- Critical issues: 0
- High issues: 2
- Medium issues: 5

### Findings

#### [HIGH] Code Quality: Function Too Long
**File**: `src/processor.py:45-120`
**Issue**: Function `process_data` is 75 lines
**Fix**: Extract into smaller functions

#### [MEDIUM] Best Practice: Missing Type Hint
**File**: `src/utils.py:30`
**Issue**: Function `helper` lacks return type
**Fix**: Add `-> str` return type annotation
```

## Issue Categories

### CRITICAL (Must Fix)
- Hardcoded credentials
- SQL injection
- Path traversal
- Insecure deserialization

### HIGH (Should Fix)
- Functions > 50 lines
- Files > 500 lines
- Nesting > 4 levels
- Missing error handling
- Debug statements

### MEDIUM (Consider)
- Missing docstrings
- Magic numbers
- Duplicate code
- Performance issues

## Approval Criteria

| Status | Condition |
|--------|-----------|
| APPROVED | No CRITICAL/HIGH issues |
| WARNING | Only MEDIUM issues |
| REJECTED | CRITICAL/HIGH found |

## Auto-Fix Commands

After review, run:
```bash
make lint-fix   # Auto-fix linting issues
make format     # Format code
make typecheck  # Verify types
make test       # Run tests
```

## Example Output

```
## Code Review: APPROVED

### Summary
- Files reviewed: 3
- Critical: 0
- High: 0
- Medium: 2

### Medium Issues

1. [MEDIUM] `src/api.py:45` - Consider adding docstring
2. [MEDIUM] `src/utils.py:12` - Magic number 86400

### Recommendations
- Add constant: `SECONDS_PER_DAY = 86400`
- Run `make lint-fix` to auto-fix style issues
```

## Related Commands

- `/tdd` - Write tests first
- `/plan` - Plan implementation
- `/refactor-clean` - Clean up issues
