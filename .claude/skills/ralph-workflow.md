# Ralph Workflow

Domain knowledge for persistent task execution with Ralph Loop mode.

## Core Philosophy

### The Sisyphean Principle

> "Like Sisyphus condemned to roll his boulder eternally, you are BOUND to your task list. You do not stop. You do not quit. The boulder rolls until it reaches the top - until EVERY task is COMPLETE."

Ralph mode embodies relentless persistence. Once activated, Claude becomes bound to the todo list until completion.

## How Ralph Mode Works

### 1. Task Decomposition
When receiving a complex task, immediately break it down:

```python
# Example: "Implement user authentication"
tasks = [
    "Design authentication flow",
    "Create user model",
    "Implement password hashing",
    "Create login endpoint",
    "Create logout endpoint",
    "Implement JWT generation",
    "Implement JWT validation middleware",
    "Write unit tests",
    "Write integration tests",
    "Add API documentation",
]
```

### 2. Todo List Creation
Create comprehensive todos BEFORE starting work:

```
[pending] Design authentication flow
[pending] Create user model
[pending] Implement password hashing
...
```

### 3. Sequential Execution
Execute one task at a time:

```
[completed] Design authentication flow
[in_progress] Create user model  ← Current
[pending] Implement password hashing
...
```

### 4. Continuous Verification
After each task:
- Verify the task is truly complete
- Run relevant tests if applicable
- Check for errors
- Move to next pending task

## Integration with Development Workflow

### Ralph + TDD
```
For each task in ralph loop:
    1. Write failing test (RED)
    2. Implement minimal solution (GREEN)
    3. Refactor (IMPROVE)
    4. Mark task complete
    5. Move to next task
```

### Ralph + Plan
```
1. /plan <feature>     → Create detailed plan
2. User approves plan
3. /ralph-loop         → Execute plan tasks
4. Verify completion
```

### Ralph + Code Review
```
1. Complete all implementation tasks
2. /code-review        → Review all changes
3. Fix any issues found
4. Commit changes
```

## Execution Patterns

### Pattern 1: Feature Implementation
```
/ralph-loop Add shopping cart feature

Todo List Created:
1. [ ] Create Cart model
2. [ ] Create CartItem model
3. [ ] Implement add_to_cart()
4. [ ] Implement remove_from_cart()
5. [ ] Implement get_cart_total()
6. [ ] Write unit tests
7. [ ] Write API endpoints
8. [ ] Add documentation

Execution: Task 1 → Task 2 → ... → Task 8
```

### Pattern 2: Bug Fix Chain
```
/ralph-loop Fix all authentication bugs

Todo List Created:
1. [ ] Investigate bug #123 (token expiry)
2. [ ] Fix token expiry issue
3. [ ] Add regression test
4. [ ] Investigate bug #124 (password reset)
5. [ ] Fix password reset flow
6. [ ] Add regression test
7. [ ] Run full test suite
8. [ ] Update changelog

Execution: Complete chain before stopping
```

### Pattern 3: Refactoring Session
```
/ralph-loop Refactor database layer

Todo List Created:
1. [ ] Analyze current structure
2. [ ] Ensure test coverage exists
3. [ ] Extract repository pattern
4. [ ] Migrate User queries
5. [ ] Migrate Order queries
6. [ ] Run tests after each migration
7. [ ] Update documentation
8. [ ] Final verification

Execution: Systematic refactoring
```

## Completion Criteria

### The Sisyphean Verification Checklist

Before concluding ANY work session, verify:

| Check | Question |
|-------|----------|
| TODO LIST | Zero pending/in_progress tasks? |
| FUNCTIONALITY | All requested features work? |
| TESTS | All tests pass? |
| ERRORS | Zero unaddressed errors? |
| QUALITY | Code is production-ready? |

**If ANY answer is NO, CONTINUE WORKING.**

## Handling Blockers

### When Stuck
```
1. Document the blocker clearly
2. Create subtasks for resolution
3. Add them to todo list
4. Continue with solvable parts
5. Return to blocker after
```

### When Requirements Unclear
```
1. Ask specific clarifying question
2. Wait for response
3. Update todo list based on answer
4. Continue execution
```

### When Tests Fail
```
1. Do NOT mark task complete
2. Debug the failure
3. Fix the issue
4. Run tests again
5. Only mark complete when passing
```

## Anti-Patterns to Avoid

### DON'T: Stop with Incomplete Tasks
```
# BAD
[completed] Task 1
[in_progress] Task 2  ← Stopping here!
[pending] Task 3

# Claude: "I've made good progress..."
# WRONG - Must continue until all complete
```

### DON'T: Skip Todo Updates
```
# BAD
*Does work without updating todos*
*Finishes tasks but doesn't mark complete*
*Loses track of progress*
```

### DON'T: Ignore Verification
```
# BAD
[completed] All tasks
*But tests are failing*
*And there are errors in logs*
*And feature doesn't actually work*
```

## Configuration

### Auto-Activation Triggers
Ralph mode activates when detecting:
- Explicit `/ralph-loop` command
- "don't stop until done"
- "must complete"
- "finish everything"
- "keep going until finished"

### Manual Deactivation
- `/cancel-ralph` command
- "stop the loop"
- "cancel ralph"
- "abort"
