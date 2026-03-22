# /plan Command

Create a comprehensive implementation plan before coding.

## Usage
```
/plan <feature or task description>
```

## What This Command Does

1. **Invokes**: `planner` agent
2. **Analyzes**: Requirements and codebase
3. **Creates**: Step-by-step implementation plan

## Planning Process

### Step 1: Requirements Analysis
- What needs to be built?
- What are the success criteria?
- What are the constraints?

### Step 2: Architecture Review
- Analyze existing code structure
- Identify affected files
- Map dependencies

### Step 3: Risk Assessment
- Identify potential issues
- Assign risk levels (HIGH/MEDIUM/LOW)
- Plan mitigations

### Step 4: Step-by-Step Plan
- Break into testable units
- Define implementation order
- Include verification steps

## Plan Document Structure

```markdown
## Overview
Brief description of what will be implemented

## Requirements
- [ ] Requirement 1
- [ ] Requirement 2
- [ ] Requirement 3

## Files to Modify
- `src/module.py` - Add new function
- `tests/test_module.py` - Add tests

## New Files
- `src/new_feature.py` - Feature implementation

## Implementation Steps

### Step 1: Create Test File [RISK: LOW]
- Create `tests/test_feature.py`
- Add failing test cases
- Verify: `make test` fails as expected

### Step 2: Implement Core Logic [RISK: MEDIUM]
- Create `src/feature.py`
- Implement main function
- Verify: Tests pass

### Step 3: Add Error Handling [RISK: LOW]
- Add validation
- Add custom exceptions
- Verify: Edge case tests pass

## Test Strategy
- Unit tests: 80% coverage minimum
- Integration tests: Key workflows
- Edge cases: Invalid inputs, errors

## Risk Mitigation
| Risk | Mitigation |
|------|------------|
| Breaking existing code | Add tests first |
| Performance impact | Profile after implementation |

## Success Criteria
- [ ] All tests pass
- [ ] Ruff lint passes
- [ ] Ty typecheck passes
- [ ] Coverage >= 80%
```

## Important Rules

1. **No Code Until Approval**
   - Planner does NOT write implementation code
   - Wait for explicit user confirmation

2. **Modification Options**
   - "modify: [changes]" - Adjust the plan
   - "alternative: [approach]" - Different strategy

3. **Approval**
   - "proceed" or "approved" - Start implementation
   - Then use `/tdd` for test-driven development

## Example Session

```
User: /plan Add user authentication with JWT

Claude:
## Implementation Plan: JWT Authentication

### Overview
Add JWT-based authentication to the API...

### Requirements
- [ ] User login endpoint
- [ ] Token generation
- [ ] Token validation middleware
...

[Full plan document]

---
Would you like to:
- Modify the plan (say "modify: [changes]")
- Try alternative approach (say "alternative: [approach]")
- Proceed with implementation (say "proceed")
```

## After Planning

1. User approves plan
2. `/tdd` - Implement with TDD
3. `/code-review` - Review changes
4. Commit changes

## Related Commands

- `/tdd` - Start test-driven development
- `/code-review` - Review implementation
- `/refactor-clean` - Clean up code
