# /cancel-ralph Command

Cancel an active Ralph Loop execution.

## Usage
```
/cancel-ralph
```

## What This Command Does

1. **Stops**: Current ralph-loop execution
2. **Preserves**: Work completed so far
3. **Reports**: Progress summary

## When to Use

- Task requirements have changed
- Discovered blocking issue
- Need to pivot to different approach
- Emergency interruption needed

## Cancellation Triggers

The loop also cancels on these phrases:
- "stop the loop"
- "cancel ralph"
- "abort task"
- "halt execution"

## Output After Cancellation

```markdown
## Ralph Loop Cancelled

### Progress Summary
- Tasks completed: 5/8
- Current task: "Implement token validation" (in_progress)
- Remaining tasks: 3

### Completed Work
1. ✓ Design authentication architecture
2. ✓ Create user model and schema
3. ✓ Implement JWT token generation
4. ✓ Add login endpoint
5. ✓ Add logout endpoint

### Pending Work
6. ○ Implement token validation middleware
7. ○ Write unit tests
8. ○ Add API documentation

### To Resume
Use `/ralph-loop` with remaining tasks, or manually complete them.
```

## Important Notes

- Completed work is preserved
- Uncommitted changes remain staged
- Todo list shows current state
- Can resume with new `/ralph-loop`

## Related Commands

- `/ralph-loop` - Start or resume execution
- `/plan` - Replan remaining work
