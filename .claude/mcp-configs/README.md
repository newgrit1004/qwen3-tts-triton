# MCP Configurations

Model Context Protocol (MCP) server configurations for enhanced Claude Code integration.

## Overview

MCP servers extend Claude Code's capabilities by providing access to external tools and services. This directory contains sample configurations for common integrations.

## Important Guidelines

From [everything-claude-code](https://github.com/affaan-m/everything-claude-code):

> "Don't enable all MCPs at once. Your 200k context window can shrink to 70k with too many tools enabled."

**Recommended limits:**
- 20-30 configured MCPs total
- Under 10 enabled per project
- Fewer than 80 active tools

## Configuration Location

MCP servers are configured in `~/.claude.json`:

```json
{
  "mcpServers": {
    "github": { ... },
    "database": { ... }
  }
}
```

## Available Configurations

### 1. GitHub (`github.json`)

Access GitHub repositories, issues, and PRs.

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
```

**Capabilities:**
- Create/read/update issues
- Create/review PRs
- Access repository content
- Search code

### 2. Database (`database.json`)

Direct database access for queries.

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

**Capabilities:**
- Execute SQL queries
- Inspect schema
- Read/write data

### 3. Filesystem (`filesystem.json`)

Enhanced filesystem operations.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

**Capabilities:**
- Read/write files
- Directory operations
- Search files

### 4. Memory (`memory.json`)

Persistent memory across sessions.

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

**Capabilities:**
- Store key-value pairs
- Persist across sessions
- Knowledge graph support

## Setup Instructions

### Step 1: Install Node.js

MCP servers require Node.js:

```bash
# macOS
brew install node

# Ubuntu
sudo apt install nodejs npm

# Or use nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
```

### Step 2: Create Configuration

Create or edit `~/.claude.json`:

```bash
# Create if doesn't exist
touch ~/.claude.json

# Add configuration
cat > ~/.claude.json << 'EOF'
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
EOF
```

### Step 3: Set Environment Variables

For sensitive values, use environment variables:

```bash
# .bashrc or .zshrc
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_xxxxxxxxxxxx"
export DATABASE_URL="postgresql://..."
```

### Step 4: Restart Claude Code

After configuration changes:

```bash
# Restart Claude Code to load new MCPs
```

## Security Considerations

1. **Never commit tokens**: Use environment variables
2. **Limit permissions**: Use minimal required scopes
3. **Audit regularly**: Review which MCPs are enabled
4. **Rotate tokens**: Change tokens periodically

## Troubleshooting

### MCP Not Loading

```bash
# Check if npx is available
which npx

# Test MCP server manually
npx -y @modelcontextprotocol/server-github --help
```

### Context Window Issues

If context is limited:
1. Disable unused MCPs
2. Reduce number of active tools
3. Use project-specific configurations

### Connection Errors

```bash
# Check environment variables
echo $GITHUB_PERSONAL_ACCESS_TOKEN

# Verify network connectivity
curl -I https://api.github.com
```

## Project-Specific Configuration

For project-specific MCPs, create `.claude.json` in project root:

```json
{
  "mcpServers": {
    "project-db": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "${PROJECT_DATABASE_URL}"
      }
    }
  }
}
```

## References

- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Server List](https://github.com/modelcontextprotocol/servers)
- [everything-claude-code](https://github.com/affaan-m/everything-claude-code)
