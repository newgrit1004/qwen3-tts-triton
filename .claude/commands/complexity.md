# /complexity Command

Analyze cognitive complexity of Python functions using complexipy.

## Usage
```
/complexity [path]
```

**Arguments:**
- `path`: Directory or file to analyze (default: `src/`)

## What This Command Does

1. **Runs**: complexipy on specified path
2. **Identifies**: Functions exceeding complexity threshold
3. **Suggests**: Refactoring strategies for each

## Complexity Thresholds

| Score | Level | Action Required |
|-------|-------|-----------------|
| 1-10 | Low | Acceptable |
| 11-15 | Medium | Consider refactoring |
| 16-20 | High | Should refactor |
| 21+ | Critical | Must refactor |

## Execution Steps

### Step 1: Run Complexity Analysis
```bash
uv run complexipy [path] --max-complexity-allowed 15
```

### Step 2: Analyze Results
For each function exceeding threshold:
- Identify complexity sources
- Suggest specific fixes
- Prioritize by complexity score

### Step 3: Generate Report

```markdown
## Complexity Analysis Report

### Summary
- Files analyzed: 10
- Functions analyzed: 45
- Functions exceeding threshold: 3

### High Complexity Functions

| Function | File:Line | Score | Issue |
|----------|-----------|-------|-------|
| process_data | src/main.py:25 | 18 | Nested loops + conditionals |
| validate_input | src/utils.py:50 | 15 | Multiple early returns |
| transform_result | src/api.py:100 | 12 | Deep nesting |

### Recommended Fixes

#### 1. `process_data` (Score: 18)
**Location**: src/main.py:25
**Issues**:
- 3 levels of nested loops
- Multiple conditional branches

**Suggestions**:
- Extract inner loop to separate function
- Use early return pattern
- Consider list comprehension

#### 2. `validate_input` (Score: 15)
**Location**: src/utils.py:50
**Suggestions**:
- Use validation library (pydantic)
- Extract validation rules to separate functions
```

## Cognitive Complexity Factors

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Nested if/else | +1 per level | Flatten with early returns |
| Loops (for/while) | +1 per level | Extract to functions |
| try/except | +1 | Consolidate handlers |
| Logical operators (and/or) | +1 per operator | Extract to boolean functions |
| Recursion | +1 | Consider iteration |

## Quick Fixes

### Pattern: Deep Nesting → Early Return
```python
# BEFORE (complexity: 5)
def process(data):
    if data:
        if data.is_valid:
            if data.has_items:
                return data.items
    return []

# AFTER (complexity: 2)
def process(data):
    if not data:
        return []
    if not data.is_valid:
        return []
    if not data.has_items:
        return []
    return data.items
```

### Pattern: Complex Conditionals → Helper Functions
```python
# BEFORE (complexity: 4)
if user.age >= 18 and user.is_verified and user.has_subscription:
    process(user)

# AFTER (complexity: 1)
def is_eligible(user):
    return user.age >= 18 and user.is_verified and user.has_subscription

if is_eligible(user):
    process(user)
```

## Integration with Other Tools

After running `/complexity`:

1. **Fix Issues**: Use `/refactor-clean` for auto-fixable issues
2. **Test Changes**: Run `make test` after refactoring
3. **Verify**: Re-run `/complexity` to confirm improvement

## Related Commands

- `/dead-code` - Find unused code to remove
- `/code-quality` - Run full quality analysis
- `/refactor-clean` - Auto-fix code issues
