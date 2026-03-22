# /ralph-loop Command

Start a self-referential execution loop that continues until task completion.

## Usage
```
/ralph-loop <task description>
```

## What This Command Does

1. **Activates**: Persistent execution mode
2. **Enforces**: Task completion before stopping
3. **Maintains**: Todo list tracking throughout

## The Ralph Philosophy

> "The boulder never stops until it reaches the top."

Like Sisyphus condemned to roll his boulder eternally, Ralph mode binds Claude to the task list. Work continues until EVERY task is COMPLETE.

## When to Use

- Complex multi-step implementations
- Tasks that must not be abandoned
- When you say "don't stop until done"
- Critical features requiring complete delivery

## Auto-Activation Signals

Ralph mode activates automatically when detecting:
- "don't stop until done"
- "must complete"
- "finish everything"
- "until all tasks are done"

## Execution Flow

```
1. Parse task → Create comprehensive todo list
2. Start first task → Mark as in_progress
3. Complete task → Mark as completed
4. Check remaining → Continue if pending tasks exist
5. Verify completion → All tasks done? Stop : Continue
```

## Todo List Management

### Required Behaviors
- Create todos BEFORE starting work
- Mark exactly ONE task as `in_progress` at a time
- Mark tasks `completed` IMMEDIATELY when done
- NEVER stop with pending tasks

### Task States
| State | Meaning |
|-------|---------|
| `pending` | Not yet started |
| `in_progress` | Currently working on |
| `completed` | Finished successfully |

## Verification Checklist

Before concluding ANY ralph-loop session:

- [ ] **TODO LIST**: Zero pending/in_progress tasks
- [ ] **FUNCTIONALITY**: All requested features work
- [ ] **TESTS**: All tests pass (if applicable)
- [ ] **ERRORS**: Zero unaddressed errors
- [ ] **QUALITY**: Code is production-ready

**If ANY checkbox is unchecked, CONTINUE WORKING.**

## Continuation Enforcement

If you attempt to stop with incomplete tasks:

```
[SYSTEM REMINDER - TODO CONTINUATION]
Incomplete tasks remain in your todo list.
Continue working on the next pending task.
Proceed without asking for permission.
Mark each task complete when finished.
Do not stop until all tasks are done.
```

## Example Session

```
User: /ralph-loop Implement user authentication with JWT, tests, and documentation

Claude:
Creating todo list for complete implementation:
1. [ ] Design authentication architecture
2. [ ] Create user model and schema
3. [ ] Implement JWT token generation
4. [ ] Implement token validation middleware
5. [ ] Add login/logout endpoints
6. [ ] Write unit tests (80%+ coverage)
7. [ ] Write integration tests
8. [ ] Add API documentation

Starting task 1: Design authentication architecture...
[Continues until ALL 8 tasks are completed]

Final verification:
✓ All todos completed
✓ Tests passing
✓ No errors
✓ Documentation complete

Ralph loop completed successfully.
```

## Cancellation

To cancel an active ralph loop:
```
/cancel-ralph
```

Or explicitly say:
- "stop the loop"
- "cancel ralph"
- "abort task"

## Integration with Other Commands

Ralph mode works with:
- `/plan` - Create detailed plan first
- `/tdd` - Use TDD within each task
- `/code-review` - Review after implementation

## Best Practices

1. **Be Specific**: Detailed task descriptions help
2. **Break Down**: Large tasks → smaller subtasks
3. **Trust the Loop**: Let it complete naturally
4. **Review at End**: Check final output quality

## Related Commands

- `/cancel-ralph` - Stop the current loop
- `/plan` - Pre-plan before ralph execution
- `/tdd` - Test-driven within ralph tasks
