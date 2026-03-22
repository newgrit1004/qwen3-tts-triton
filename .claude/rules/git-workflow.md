# Git Workflow Rules

These rules MUST be followed for all Git operations in this project.

## Commit Message Format

```
<type>: <subject>

[body]

[footer]
```

### Types
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code refactoring |
| `docs` | Documentation |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |

### Subject Rules
- 50 characters or less
- Imperative mood ("Add feature" not "Added feature")
- No period at the end
- Capitalize first letter

### Examples
```bash
# Good
feat: Add user authentication endpoint
fix: Resolve race condition in order processing
refactor: Extract validation logic to separate module
docs: Update API documentation for v2 endpoints

# Bad
added new feature
Fix bug.
REFACTOR: Updated code
```

## Pre-Commit Hooks

All commits must pass pre-commit checks:

1. **File checks**
   - No large files (> 500KB)
   - Valid JSON/YAML/TOML syntax
   - No merge conflicts

2. **Code quality**
   - Ruff linting (auto-fix enabled)
   - Ruff formatting
   - Ty type checking

### Manual Pre-commit Run
```bash
make pre-commit
```

## Branch Strategy

### Main Branches
- `main` - Production-ready code
- `develop` - Integration branch (if used)

### Feature Branches
```bash
# Pattern: <type>/<description>
feature/user-authentication
fix/order-validation-bug
refactor/database-layer
```

### Branch Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Work and commit
git add .
git commit  # Uses commit template

# Push and create PR
git push -u origin feature/new-feature
```

## Pull Request Process

### Before Creating PR

1. **Ensure all checks pass**
   ```bash
   make lint        # Ruff lint
   make format      # Ruff format
   make typecheck   # Ty type check
   make test        # All tests pass
   ```

2. **Review all changes**
   ```bash
   git diff main...HEAD
   git log main...HEAD
   ```

### PR Description Template
```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Test Plan
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] No security vulnerabilities introduced
```

## Development Workflow

### 1. Planning Phase
```bash
/plan  # Invoke planner agent
```

### 2. Test-Driven Development
```bash
/tdd  # Follow RED-GREEN-REFACTOR
```

### 3. Code Review
```bash
/code-review  # Get automated review
```

### 4. Commit
```bash
git add .
git commit  # Opens template editor, runs pre-commit
```

### 5. Push
```bash
git push -u origin <branch>
```

## Commit Checklist

Before committing:

- [ ] Code compiles/runs without errors
- [ ] All tests pass (`make test`)
- [ ] Ruff lint passes (`make lint`)
- [ ] Ty typecheck passes (`make typecheck`)
- [ ] No debug statements (`print()`, `breakpoint()`)
- [ ] No hardcoded secrets
- [ ] Commit message follows format

## Git Commands Reference

```bash
# Status and diff
git status
git diff
git diff --staged

# Staging
git add <file>
git add -p  # Interactive staging

# Committing
git commit  # Opens template
git commit -m "type: message"

# Branches
git branch -a
git checkout -b <branch>
git switch <branch>

# Syncing
git fetch origin
git pull origin main
git push -u origin <branch>

# History
git log --oneline -10
git log --graph --oneline

# Undo
git restore <file>          # Discard changes
git restore --staged <file> # Unstage
git reset --soft HEAD~1     # Undo last commit (keep changes)
```

## Protected Operations

### Require Extra Caution
- `git push --force` - Use `--force-with-lease` instead
- `git reset --hard` - Can lose work
- `git rebase` on shared branches

### Never Do on Main
```bash
# DON'T
git push --force origin main
git reset --hard on main
git rebase main (when others are using it)
```

## Conflict Resolution

1. **Fetch latest changes**
   ```bash
   git fetch origin
   ```

2. **Rebase on main**
   ```bash
   git rebase origin/main
   ```

3. **Resolve conflicts**
   - Edit conflicted files
   - Remove conflict markers
   - Stage resolved files

4. **Continue rebase**
   ```bash
   git rebase --continue
   ```

5. **Verify**
   ```bash
   make test
   make lint
   ```
