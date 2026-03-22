# /dead-code Command

Find unused functions, classes, and variables using skylos.

## Usage
```
/dead-code [path]
```

**Arguments:**
- `path`: Directory to analyze (default: `src/`)

## What This Command Does

1. **Runs**: skylos static analysis on specified path
2. **Identifies**: Unused functions, classes, variables
3. **Provides**: Confidence scores for each finding
4. **Optionally**: Scans for security issues (hardcoded secrets)

## Confidence Levels

| Score | Meaning | Recommended Action |
|-------|---------|-------------------|
| 90-100 | Definitely unused | Safe to delete |
| 70-89 | Highly likely unused | Review briefly, then delete |
| 50-69 | Probably unused | Review carefully |
| 30-49 | Possibly unused | Likely false positive |
| 0-29 | Uncertain | Skip, probably used |

## Execution Steps

### Step 1: Run Dead Code Analysis
```bash
uv run skylos [path] --confidence 60 --gate
```

### Step 2: Security Scan (Optional)
```bash
uv run skylos [path] --danger --secrets
```

### Step 3: Generate Report

```markdown
## Dead Code Analysis Report

### Summary
- Files scanned: 15
- Unused functions: 5
- Unused classes: 2
- Unused variables: 8
- Security issues: 1

### High Confidence (Safe to Delete)

| Item | Type | File:Line | Confidence | Reason |
|------|------|-----------|------------|--------|
| old_helper | function | src/utils.py:45 | 98% | No calls found |
| LegacyClass | class | src/models.py:120 | 95% | No instantiations |
| UNUSED_CONST | variable | src/config.py:10 | 92% | Never referenced |

### Medium Confidence (Review Required)

| Item | Type | File:Line | Confidence | Note |
|------|------|-----------|------------|------|
| _internal_func | function | src/api.py:80 | 72% | May be called dynamically |
| temp_data | variable | src/main.py:25 | 65% | Check usage in tests |

### Security Findings

| Severity | File:Line | Issue |
|----------|-----------|-------|
| HIGH | src/config.py:15 | Hardcoded API key pattern |
```

## Framework-Aware Analysis

Skylos recognizes patterns from popular frameworks:

| Framework | Auto-Excluded Patterns |
|-----------|------------------------|
| Django | `models.Model`, views, migrations |
| Flask | Route handlers, decorators |
| FastAPI | Endpoint functions, dependencies |
| Pytest | Test functions, fixtures |

## Deletion Workflow

### Step 1: Review High Confidence Items
```python
# src/utils.py:45 - Confidence: 98%
def old_helper():  # <- Safe to delete
    pass
```

### Step 2: Verify Before Deletion
- Check for dynamic calls: `getattr()`, `importlib`
- Check for test-only usage
- Check for plugin/extension patterns

### Step 3: Delete and Test
```bash
# After deletion
make test        # All tests must pass
make typecheck   # No type errors
```

## Security Scanning

### Enable Security Features
```bash
uv run skylos . --danger --secrets
```

### What It Detects
- Hardcoded passwords/tokens
- API keys in source code
- Private keys
- Database credentials

### Security Report Format
```markdown
### Security Issues Found

#### HIGH: Hardcoded API Key
**File**: src/config.py:15
**Pattern**: `API_KEY = "sk-..."`
**Fix**: Move to environment variable

#### MEDIUM: Possible Password
**File**: src/db.py:30
**Pattern**: `password = "admin123"`
**Fix**: Use secrets manager
```

## Integration with Other Tools

1. **Run Dead Code First**: Reduces codebase before complexity analysis
2. **Then Complexity**: `/complexity` on remaining code
3. **Finally Tests**: `make test` to verify no regressions

## Output Formats

```bash
# JSON output for CI/CD
uv run skylos . --output json > dead-code.json

# SARIF for GitHub integration
uv run skylos . --output sarif > dead-code.sarif

# CSV for spreadsheet analysis
uv run skylos . --output csv > dead-code.csv
```

## Related Commands

- `/complexity` - Analyze code complexity
- `/code-quality` - Run full quality analysis
- `/security-check` - Full security review
