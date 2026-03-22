# /refactor-clean Command

Systematically improve code quality.

## Usage
```
/refactor-clean [file or directory]
```

## What This Command Does

1. **Invokes**: `refactor-cleaner` agent
2. **Analyzes**: Code for improvement opportunities
3. **Refactors**: While maintaining functionality

## Analysis Targets

- Functions > 50 lines
- Files > 500 lines
- Nesting depth > 4
- Duplicate code
- Code smells

## Refactoring Process

### Step 1: Ensure Test Coverage
```bash
make test-cov
# Verify 80%+ coverage before refactoring
```

### Step 2: Identify Issues
- Run static analysis
- Check complexity metrics
- Find code smells

### Step 3: Refactor Incrementally
- One change at a time
- Run tests after each change
- Maintain git history

### Step 4: Verify
```bash
make test      # Tests still pass
make lint      # No new warnings
make typecheck # Types valid
```

## Common Refactorings

### Extract Function
```python
# Before
def process(data):
    # 75 lines of code
    ...

# After
def process(data):
    validated = validate_input(data)
    transformed = transform_data(validated)
    return format_output(transformed)
```

### Early Return
```python
# Before
def check(user):
    if user:
        if user.active:
            if user.verified:
                return True
    return False

# After
def check(user):
    if not user:
        return False
    if not user.active:
        return False
    if not user.verified:
        return False
    return True
```

### Replace Magic Numbers
```python
# Before
if age >= 18:
    discount = price * 0.1

# After
ADULT_AGE = 18
DISCOUNT_RATE = Decimal("0.10")

if age >= ADULT_AGE:
    discount = price * DISCOUNT_RATE
```

## Output Report

```markdown
## Refactoring Report

### Files Analyzed
- `src/processor.py` (450 lines)

### Issues Found
| Issue | Location | Action |
|-------|----------|--------|
| Long function | process:45-120 | Extracted 3 functions |
| Deep nesting | validate:30-50 | Applied early returns |

### Changes Made
1. Extracted `validate_input()` from `process()`
2. Extracted `transform_data()` from `process()`
3. Applied early return pattern in `validate()`

### Metrics
| Metric | Before | After |
|--------|--------|-------|
| Max function lines | 75 | 20 |
| Max nesting | 5 | 2 |
| Complexity | 15 | 5 |
```

## Verification

```bash
make test      # All tests pass
make lint      # Clean
make typecheck # No errors
make test-cov  # Coverage maintained
```

## Related Commands

- `/plan` - Plan before major refactoring
- `/tdd` - Add tests before refactoring
- `/code-review` - Review after refactoring
