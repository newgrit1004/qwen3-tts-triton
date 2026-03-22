# /code-quality Command

Run comprehensive code quality analysis combining dead code detection and complexity analysis.

## Usage
```
/code-quality [path]
```

**Arguments:**
- `path`: Directory to analyze (default: `src/`)

## What This Command Does

1. **Dead Code Analysis**: Find unused code (skylos)
2. **Complexity Analysis**: Find hard-to-understand code (complexipy)
3. **Security Scan**: Detect hardcoded secrets (optional)
4. **Unified Report**: Prioritized improvement recommendations

## Improvement Cycle

```
┌─────────────────────────────────────────────────────────┐
│  1. DETECT                                              │
│     ├── skylos: Find dead code                          │
│     └── complexipy: Find complex functions              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. PRIORITIZE                                          │
│     ├── Delete dead code first (reduces codebase)       │
│     └── Then tackle high-complexity functions           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  3. FIX                                                 │
│     ├── Remove confirmed dead code                      │
│     ├── Extract functions (reduce complexity)           │
│     ├── Simplify conditionals                           │
│     └── Apply refactoring patterns                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  4. VERIFY                                              │
│     ├── Re-run tools to confirm improvement             │
│     ├── Run tests (make test)                           │
│     └── Pre-commit hooks prevent new issues             │
└─────────────────────────────────────────────────────────┘
```

## Execution Steps

### Step 1: Run Dead Code Analysis
```bash
uv run skylos [path] --confidence 60 --gate
```

### Step 2: Run Complexity Analysis
```bash
uv run complexipy [path] --max-complexity-allowed 15
```

### Step 3: Security Scan (Optional)
```bash
uv run skylos [path] --danger --secrets
```

### Step 4: Generate Unified Report

```markdown
## Code Quality Report

### Summary
| Category | Count | Action |
|----------|-------|--------|
| Dead code (high confidence) | 5 | Delete |
| Dead code (review needed) | 3 | Review |
| Complex functions | 4 | Refactor |
| Security issues | 1 | Fix immediately |

### Priority Actions

#### 1. Security Issues (Immediate)
| File:Line | Issue | Fix |
|-----------|-------|-----|
| src/config.py:15 | Hardcoded API key | Move to env var |

#### 2. Dead Code (Delete First)
| Item | File:Line | Confidence |
|------|-----------|------------|
| old_helper | src/utils.py:45 | 98% |
| LegacyClass | src/models.py:120 | 95% |

#### 3. High Complexity (Refactor)
| Function | File:Line | Score | Suggestion |
|----------|-----------|-------|------------|
| process_data | src/main.py:25 | 18 | Extract helper functions |
| validate_all | src/api.py:80 | 16 | Use early returns |

### Metrics Improvement
| Metric | Before | Target |
|--------|--------|--------|
| Lines of code | 2,500 | 2,300 (-8%) |
| Avg complexity | 12 | 8 (-33%) |
| Dead code items | 8 | 0 |
```

## Quick Commands

```bash
# Full analysis
make code-quality

# Just complexity
make complexity

# Just dead code
make dead-code
```

## Threshold Configuration

### Environment Variables
```bash
export COMPLEXIPY_MAX_COMPLEXITY=15    # 10=strict, 15=moderate, 20=lenient
export SKYLOS_CONFIDENCE=60            # 0-100 (higher = more certain)
```

### pyproject.toml
```toml
[tool.complexipy]
max-complexity-allowed = 15

[tool.skylos]
complexity = 10
nesting = 4
```

## Integration with CI/CD

```yaml
# .github/workflows/quality.yml
- name: Code Quality Check
  run: |
    uv run skylos src/ --gate --confidence 80
    uv run complexipy src/ --max-complexity-allowed 15
```

## Related Commands

- `/complexity` - Detailed complexity analysis
- `/dead-code` - Detailed dead code analysis
- `/refactor-clean` - Auto-fix code issues
- `/security-check` - Full security review
