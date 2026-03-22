# Security Rules

These rules MUST be followed for all code in this project.

## Pre-Commit Security Checklist

Before EVERY commit, verify:

- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All user inputs are validated
- [ ] SQL queries use parameterized statements
- [ ] File paths are sanitized (no path traversal)
- [ ] Error messages don't expose sensitive info
- [ ] Dependencies are up to date

## Critical: No Hardcoded Secrets

### NEVER Do This
```python
# CRITICAL VIOLATION
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "supersecret123"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

### Always Do This
```python
import os

# From environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")

# Using pydantic-settings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

## Input Validation

### Validate All External Input
```python
from pydantic import BaseModel, Field, validator

class UserInput(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int = Field(ge=0, le=150)

    @validator("username")
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v
```

## SQL Injection Prevention

### NEVER Do This
```python
# CRITICAL VULNERABILITY
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
```

### Always Do This
```python
# Parameterized query
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))

# Using ORM (SQLAlchemy)
user = session.query(User).filter(User.id == user_id).first()
```

## Path Traversal Prevention

### NEVER Do This
```python
# VULNERABILITY: User can access any file
file_path = f"/data/{user_input}"
with open(file_path) as f:
    return f.read()
```

### Always Do This
```python
from pathlib import Path

SAFE_BASE_DIR = Path("/data")

def safe_read_file(filename: str) -> str:
    # Resolve to absolute path
    requested = (SAFE_BASE_DIR / filename).resolve()

    # Verify it's within the allowed directory
    if not requested.is_relative_to(SAFE_BASE_DIR):
        raise ValueError("Access denied: Invalid path")

    if not requested.exists():
        raise FileNotFoundError("File not found")

    return requested.read_text()
```

## Secure Error Handling

### NEVER Do This
```python
# VULNERABILITY: Exposes internal details
try:
    result = database.query(sql)
except Exception as e:
    return f"Database error: {e}"  # Exposes SQL, connection info
```

### Always Do This
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = database.query(sql)
except DatabaseError as e:
    logger.error(f"Database query failed: {e}", exc_info=True)
    raise ServiceError("An error occurred processing your request")
```

## Dependency Security

### Regular Vulnerability Checks
```bash
# Check for vulnerabilities
uv add --dev safety
uv run safety check

# Alternative: pip-audit
uv add --dev pip-audit
uv run pip-audit
```

### Keep Dependencies Updated
```bash
# Update all dependencies
make update

# Check for outdated packages
uv pip list --outdated
```

## Logging Security

### NEVER Log Sensitive Data
```python
# VIOLATION
logger.info(f"User {username} logged in with password {password}")
logger.debug(f"API response: {api_key}")
```

### Safe Logging
```python
# CORRECT
logger.info(f"User {username} logged in successfully")
logger.debug("API call completed", extra={"status": response.status_code})
```

## Environment File Rules

### .env.example (Commit this)
```env
# Required environment variables
API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:pass@localhost/db
SECRET_KEY=generate_a_secure_key
```

### .env (NEVER commit)
```env
API_KEY=sk-actual-production-key
DATABASE_URL=postgresql://prod:realpass@prod-db/production
SECRET_KEY=super-secret-production-key
```

### .gitignore (Required entries)
```gitignore
# Environment files
.env
.env.local
.env.*.local

# Credentials
*.pem
*.key
credentials.json
secrets/
*_secret*
*_credentials*
```

## Response Protocol for Security Issues

When a security vulnerability is discovered:

1. **STOP** all current work
2. **ASSESS** severity (Critical/High/Medium/Low)
3. **REPORT** finding with exact location
4. **FIX** critical issues immediately
5. **INVALIDATE** any exposed credentials
6. **SCAN** codebase for similar issues
7. **DOCUMENT** the incident and fix

## Security Tools

```bash
# Static security analysis
uv add --dev bandit
uv run bandit -r src/

# Secret detection
uv add --dev detect-secrets
uv run detect-secrets scan

# Dependency vulnerabilities
uv run safety check
uv run pip-audit
```

## Security Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| CRITICAL | Active exploitation possible | Immediate |
| HIGH | Significant risk | Within hours |
| MEDIUM | Moderate risk | Within days |
| LOW | Minor risk | Next release |
