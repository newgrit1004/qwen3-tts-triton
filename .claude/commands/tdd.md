# /tdd Command

Invoke the TDD workflow for test-driven development.

## Usage
```
/tdd <feature description>
```

## What This Command Does

1. **Invokes**: `tdd-guide` agent
2. **Enforces**: RED-GREEN-REFACTOR cycle
3. **Ensures**: 80%+ test coverage

## Workflow

### Step 1: Understand the Feature
Analyze the feature requirements and identify:
- Input/output specifications
- Edge cases
- Error conditions

### Step 2: RED Phase - Write Failing Test
```python
# tests/test_feature.py
def test_feature_basic_case():
    result = feature_function(valid_input)
    assert result == expected_output

def test_feature_edge_case():
    with pytest.raises(ValueError):
        feature_function(invalid_input)
```

### Step 3: Verify Test Fails
```bash
make test
# Expected: FAILED
```

### Step 4: GREEN Phase - Minimal Implementation
```python
# src/feature.py
def feature_function(input_data):
    # Minimal code to pass the test
    return expected_output
```

### Step 5: Verify Test Passes
```bash
make test
# Expected: PASSED
```

### Step 6: REFACTOR Phase
- Improve code quality
- Add type hints
- Add docstrings
- Run linting

```bash
make lint-fix
make format
make typecheck
```

### Step 7: Repeat
Continue with next test case.

## Coverage Requirements

| Code Type | Coverage |
|-----------|----------|
| General | 80% |
| Business Logic | 100% |
| Security Code | 100% |

## Verification Commands

```bash
make test       # Run tests
make test-cov   # Run with coverage
make lint       # Check linting
make typecheck  # Check types
```

## Example Session

```
User: /tdd Add a function to calculate order total with discounts

Claude:
1. Let me first write a failing test...
   [Creates test_order.py with test cases]

2. Running test - should fail...
   [make test - FAILED as expected]

3. Now implementing minimal solution...
   [Creates order.py with calculate_total function]

4. Running test - should pass...
   [make test - PASSED]

5. Refactoring and adding type hints...
   [Updates code with types, docstrings]

6. Final verification...
   [make lint, make typecheck, make test-cov]
```

## Related Commands

- `/plan` - Plan before implementing
- `/code-review` - Review after implementing
- `/refactor-clean` - Clean up code
