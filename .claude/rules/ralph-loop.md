# Ralph Loop Rules

Mandatory rules for Ralph Loop execution mode.

## Core Rules

### Rule 1: Never Stop with Pending Tasks

```
FORBIDDEN:
- Stopping when todo list has pending items
- Declaring "good progress" without completion
- Asking "should I continue?" when tasks remain
```

When tasks remain, you MUST continue automatically.

### Rule 2: One In-Progress at a Time

```
CORRECT:
[completed] Task A
[in_progress] Task B  ← Only one
[pending] Task C
[pending] Task D

INCORRECT:
[in_progress] Task A  ← Multiple in_progress
[in_progress] Task B
[pending] Task C
```

### Rule 3: Immediate Completion Marking

```
CORRECT:
1. Finish implementing feature
2. Verify it works
3. Mark as completed IMMEDIATELY
4. Move to next task

INCORRECT:
1. Finish implementing feature
2. Start next task
3. Forget to mark previous complete
4. Lose track of progress
```

### Rule 4: Create Todos Before Starting

```
CORRECT:
1. Receive task: "Add user authentication"
2. Create todo list with ALL subtasks
3. Then start working

INCORRECT:
1. Receive task: "Add user authentication"
2. Start coding immediately
3. No tracking of progress
```

### Rule 5: Verify Before Completing

Before marking ANY task complete:
- [ ] Code compiles/runs without errors
- [ ] Relevant tests pass (if applicable)
- [ ] No obvious bugs introduced
- [ ] Feature/fix actually works

## Continuation Enforcement

### System Reminder Trigger

If you attempt to stop with incomplete tasks, expect:

```
[SYSTEM REMINDER - TODO CONTINUATION]
Incomplete tasks remain in your todo list.
Continue working on the next pending task.
Proceed without asking for permission.
Mark each task complete when finished.
Do not stop until all tasks are done.
```

### Required Response

When seeing this reminder:
1. Acknowledge the reminder
2. Identify next pending task
3. Mark it as in_progress
4. Continue working
5. DO NOT ask for permission

## Final Verification Checklist

Before declaring ralph-loop complete:

| Check | Verified? |
|-------|-----------|
| All todos marked `completed` | [ ] |
| Zero `pending` tasks | [ ] |
| Zero `in_progress` tasks | [ ] |
| All tests passing | [ ] |
| No unhandled errors | [ ] |
| Functionality works | [ ] |

**If ANY box is unchecked, DO NOT STOP.**

## Exception Handling

### Legitimate Stopping Points

Ralph loop MAY stop when:
- User explicitly says `/cancel-ralph`
- User says "stop", "abort", "cancel"
- Blocking external dependency (e.g., waiting for API key)
- Ambiguous requirements requiring clarification

### Handling Blockers

```
1. Document the blocker
2. Ask specific question (if needed)
3. Create subtask for resolution
4. Continue with other tasks if possible
5. Return to blocked task after resolution
```

## Integration Rules

### With TDD
```
For EACH task in ralph loop:
- Write test first (if applicable)
- Implement solution
- Verify test passes
- Then mark complete
```

### With Code Review
```
After ALL tasks complete:
- Run /code-review
- Fix any issues found
- Re-run verification
- Then commit
```

### With Git
```
DO NOT commit until:
- All ralph tasks complete
- All tests passing
- Code review passed (if done)
```

## Command Reference

| Command | Action |
|---------|--------|
| `/ralph-loop <task>` | Start persistent execution |
| `/cancel-ralph` | Stop execution |
| "continue" | Resume paused loop |
| "stop" | Cancel loop |

## Enforcement Priority

```
1. HIGHEST: Complete all tasks
2. HIGH: Maintain todo list accuracy
3. MEDIUM: Run tests after changes
4. STANDARD: Follow code quality rules
```

The boulder does not stop until it reaches the summit.
