# Autonomous Agent Workflow

Domain knowledge for long-running autonomous task execution across multiple sessions.

## Core Concept

The Autonomous Agent system overcomes context window limitations by running multiple sessions with persistent state.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Session-Based Execution                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Session 1          Session 2          Session 3               │
│   ┌────────┐        ┌────────┐        ┌────────┐               │
│   │ Task 1 │   →    │ Task 4 │   →    │ Task 7 │               │
│   │ Task 2 │        │ Task 5 │        │ Task 8 │               │
│   │ Task 3 │        │ Task 6 │        │ ...    │               │
│   └───┬────┘        └───┬────┘        └───┬────┘               │
│       │                 │                 │                     │
│       ▼                 ▼                 ▼                     │
│   ┌─────────────────────────────────────────────┐               │
│   │           progress.json (persistence)        │               │
│   │   - Completed task list                      │               │
│   │   - Next task to execute                     │               │
│   │   - Metadata (start time, session count)     │               │
│   └─────────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Task Definition (tasks.json)
```json
{
  "tasks": [
    {"id": "task_001", "description": "Research trending topics", "agent": "research"},
    {"id": "task_002", "description": "Generate script", "agent": "script"},
    {"id": "task_003", "description": "Create audio", "agent": "audio"}
  ]
}
```

### 2. Session Execution Loop
```python
while not all_tasks_complete:
    session = new_claude_session()  # Fresh context
    load_progress()                 # What's already done?
    next_task = get_next_pending()

    execute_task(session, next_task)
    save_progress()                 # Persist state

    session.close()
    wait(delay)                     # Rate limiting
```

### 3. Progress Persistence
```json
{
  "created_at": "2025-01-20T22:00:00",
  "session_count": 5,
  "tasks": [...],
  "completed_tasks": ["task_001", "task_002"],
  "current_task_index": 2
}
```

## Relationship with Ralph Mode

| Aspect | Ralph Mode | Autonomous Agent |
|--------|------------|------------------|
| Scope | **Intra-session** | **Inter-session** |
| Context | Single window | Multiple fresh contexts |
| Persistence | In-memory todo | File-based (JSON) |
| Duration Limit | Context window | **None** (36h+ possible) |
| Use Case | Complex single tasks | Long automation pipelines |

### Complementary Usage
Ralph Mode operates **inside** each Autonomous Agent session:

```
┌─────────────────────────────────────────┐
│ Autonomous Agent Session                 │
│ ┌─────────────────────────────────────┐ │
│ │ Ralph Mode (within session)          │ │
│ │ - Task breakdown                     │ │
│ │ - Todo tracking                      │ │
│ │ - Completion verification            │ │
│ └─────────────────────────────────────┘ │
│ → Save to progress.json                  │
│ → Start next session                     │
└─────────────────────────────────────────┘
```

## Running the Agent

### Prerequisites
```bash
# Install Claude Code SDK
pip install claude-code-sdk

# Set OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### Basic Execution
```bash
# Run with tasks file
python -m src.autonomous_agent.run \
    --project-dir . \
    --tasks-file tasks.json

# Test with limited iterations
python -m src.autonomous_agent.run \
    --project-dir . \
    --tasks-file tasks.json \
    --max-iterations 3
```

### Options
| Option | Default | Description |
|--------|---------|-------------|
| `--project-dir` | `./project` | Working directory |
| `--tasks-file` | None | Task definition JSON |
| `--max-iterations` | Unlimited | Session limit |
| `--model` | claude-sonnet-4-5 | Model to use |
| `--delay` | 3 | Seconds between sessions |

## Task Definition Format

### Basic Structure
```json
{
  "project_name": "My Project",
  "config": {
    "model": "claude-sonnet-4-5-20250929"
  },
  "tasks": [
    {
      "id": "unique_task_id",
      "description": "What needs to be done",
      "agent": "agent_type",
      "status": "pending",
      "dependencies": ["other_task_id"]
    }
  ]
}
```

### Task Status Values
- `pending` - Not started
- `in_progress` - Currently executing
- `complete` - Successfully finished
- `failed` - Encountered error

### Dependencies
Tasks can depend on other tasks:
```json
{
  "id": "visual_003",
  "description": "Generate subtitles",
  "dependencies": ["audio_002", "visual_002"]
}
```

## Security

### Command Allowlist
The security module restricts bash commands:

```python
ALLOWED_COMMANDS = {
    "ls", "cat", "grep", "find",  # Read-only
    "mkdir", "cp", "mv",          # File ops
    "npm", "python", "uv",        # Package managers
    "git",                        # Version control
}
```

### Dangerous Pattern Blocking
```python
DANGEROUS_PATTERNS = [
    "sudo", "rm -rf /",
    "> /dev/", "nc -l",
]
```

## Best Practices

### 1. Task Granularity
- Break large tasks into smaller, verifiable units
- Each task should be completable in one session
- Include clear completion criteria

### 2. Error Handling
```json
{
  "id": "task_005",
  "retry_count": 3,
  "on_failure": "skip"  // or "stop", "retry"
}
```

### 3. Progress Monitoring
```bash
# Watch progress in real-time
watch -n 5 cat progress.json | jq '.completed_tasks | length'
```

### 4. Interrupt & Resume
- `Ctrl+C` to gracefully stop
- Run same command to resume from last checkpoint

## Integration with Project Structure

```
project/
├── src/
│   └── autonomous_agent/
│       ├── __init__.py
│       ├── run.py          # Entry point
│       ├── agent.py        # Session logic
│       ├── progress.py     # State persistence
│       ├── client.py       # SDK configuration
│       └── security.py     # Command allowlist
├── prompts/
│   ├── system.md           # Agent personality
│   └── task.md             # Task execution template
├── tasks.json              # Task definitions
└── progress.json           # Runtime state (auto-generated)
```

## Example: YouTube Shorts Pipeline

```json
{
  "tasks": [
    {"id": "research", "description": "Topic research"},
    {"id": "script", "description": "Script generation"},
    {"id": "audio", "description": "TTS + BGM mixing"},
    {"id": "visual", "description": "Image generation + video compositing"},
    {"id": "qa", "description": "Quality validation"},
    {"id": "publish", "description": "Upload to YouTube"}
  ]
}
```

This pipeline runs automatically across multiple sessions until all 6 stages complete.
