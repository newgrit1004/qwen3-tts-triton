# /security-check Command

Perform comprehensive security analysis.

## Usage
```
/security-check [file or directory]
```

## What This Command Does

1. **Invokes**: `security-reviewer` agent
2. **Scans**: Code for vulnerabilities
3. **Reports**: Security issues with remediation

## Security Checks

### Code Analysis
- Hardcoded credentials
- SQL injection vectors
- Path traversal risks
- Command injection
- Insecure deserialization
- SSRF vulnerabilities

### Dependency Analysis
- Known vulnerabilities (CVEs)
- Outdated packages
- Insecure dependencies

## Scan Process

### Step 1: Static Analysis
```bash
uv run bandit -r src/
```

### Step 2: Dependency Audit
```bash
uv run safety check
uv run pip-audit
```

### Step 3: Secret Detection
```bash
uv run detect-secrets scan
```

## Output Report

```markdown
## Security Scan Report

### Summary
- Files scanned: 15
- Critical: 0
- High: 1
- Medium: 2
- Low: 3

### Critical Findings
None

### High Findings

#### [HIGH] SQL Injection Risk
**File**: `src/db.py:45`
**Code**: `query = f"SELECT * FROM users WHERE id = {user_id}"`
**Fix**: Use parameterized query
```python
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

### Medium Findings
...

### Recommendations
1. Enable Bandit in pre-commit hooks
2. Add safety check to CI pipeline
3. Update vulnerable dependencies
```

## Severity Levels

| Level | Description | Response |
|-------|-------------|----------|
| CRITICAL | Active exploitation possible | Immediate fix |
| HIGH | Significant risk | Fix before commit |
| MEDIUM | Moderate risk | Fix soon |
| LOW | Minor risk | Fix when convenient |

## Required Tools

```bash
# Install security tools
uv add --dev bandit safety pip-audit detect-secrets
```

## Integration with CI

Add to `.github/workflows/ci.yml`:
```yaml
- name: Security scan
  run: |
    uv run bandit -r src/
    uv run safety check
```

## Related Commands

- `/code-review` - General code review
- `/plan` - Plan security improvements
