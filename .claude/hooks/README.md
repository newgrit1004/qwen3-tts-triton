# Claude Code Hooks

This directory contains hook configurations for Claude Code automation.

## Hook Types

### PreToolUse
Runs before a tool is executed. Can:
- `allow` - Permit the action
- `pause` - Ask for confirmation
- `block` - Prevent the action
- `warn` - Show warning but continue

### PostToolUse
Runs after a tool completes. Can:
- `run` - Execute a command
- `log` - Log a message
- `notify` - Send notification

### Stop
Runs when Claude Code session ends. Can:
- `check` - Run verification checks
- `remind` - Show reminders

## Configuration

### hooks.json Structure

```json
{
  "hooks": {
    "PreToolUse": [...],
    "PostToolUse": [...],
    "Stop": [...]
  },
  "settings": {
    "auto_format": true,
    ...
  }
}
```

### Matchers

```json
{
  "matcher": {
    "tool": "Bash|Write|Edit|...",
    "command_pattern": "regex pattern",
    "file_pattern": "regex pattern"
  }
}
```

## Current Hooks

### PreToolUse
1. **Allow UV commands** - Permits `uv run` commands
2. **Confirm git push** - Pauses before pushing
3. **Warn on markdown creation** - Alerts when creating docs

### PostToolUse
1. **Auto-format Python** - Runs Ruff after Python file changes
2. **Log PR creation** - Reminds to request review

### Stop
1. **Check for print()** - Warns about debug statements
2. **Check for breakpoint()** - Warns about debugger calls
3. **Check for secrets** - Alerts on hardcoded credentials

## Usage

To apply these hooks, copy the configuration to your Claude Code settings:

```bash
# Copy to global settings
cp hooks.json ~/.claude/settings.json
```

Or reference in project-level settings.

## Customization

Edit `hooks.json` to add or modify hooks for your specific workflow needs.
