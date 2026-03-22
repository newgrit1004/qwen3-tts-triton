# /e2e Command

Generate and run end-to-end tests for user workflows.

## Usage
```
/e2e <workflow description>
/e2e
```

## What This Command Does

1. **Invokes**: `e2e-runner` agent
2. **Generates**: E2E test cases for workflows
3. **Runs**: Tests and reports results

## Workflow

### Step 1: Identify Workflow

Analyze the workflow to test:
- User journey steps
- API endpoints involved
- Expected outcomes

### Step 2: Generate Tests

```python
# tests/e2e/test_workflow.py
import pytest
import httpx

@pytest.mark.e2e
async def test_user_registration_workflow(client):
    """Test complete user registration flow."""
    # Step 1: Register
    response = await client.post("/register", json={
        "email": "test@example.com",
        "password": "secure123"
    })
    assert response.status_code == 201

    # Step 2: Verify email (simulated)
    user_id = response.json()["id"]

    # Step 3: Login
    response = await client.post("/login", json={
        "email": "test@example.com",
        "password": "secure123"
    })
    assert response.status_code == 200
    assert "token" in response.json()
```

### Step 3: Run Tests

```bash
# Run E2E tests
uv run pytest tests/e2e/ -v -m e2e

# Run with detailed output
uv run pytest tests/e2e/ -v --tb=long
```

### Step 4: Review Results

```markdown
## E2E Test Results

### Summary
- Total: 5
- Passed: 4
- Failed: 1

### Failed Test
- test_checkout_workflow: Timeout on payment API
```

## Test Categories

### API Workflows

```python
@pytest.mark.e2e
async def test_crud_workflow(client):
    # Create
    response = await client.post("/items", json={"name": "Test"})
    item_id = response.json()["id"]

    # Read
    response = await client.get(f"/items/{item_id}")
    assert response.json()["name"] == "Test"

    # Update
    response = await client.patch(f"/items/{item_id}", json={"name": "Updated"})
    assert response.json()["name"] == "Updated"

    # Delete
    response = await client.delete(f"/items/{item_id}")
    assert response.status_code == 204
```

### CLI Workflows

```python
from click.testing import CliRunner

@pytest.mark.e2e
def test_cli_workflow():
    runner = CliRunner()

    # Initialize
    result = runner.invoke(main, ["init", "--name", "myproject"])
    assert result.exit_code == 0

    # Process
    result = runner.invoke(main, ["process", "--input", "data.json"])
    assert result.exit_code == 0
    assert "Success" in result.output
```

### Web Workflows (Playwright)

```python
@pytest.mark.e2e
def test_login_workflow(page):
    page.goto("/login")
    page.fill("[name=email]", "user@example.com")
    page.fill("[name=password]", "password")
    page.click("button[type=submit]")
    page.wait_for_url("/dashboard")
    assert page.title() == "Dashboard"
```

## Configuration

### pyproject.toml

```toml
[tool.pytest.ini_options]
markers = [
    "e2e: End-to-end tests (may be slow)",
]
asyncio_mode = "auto"
```

### Install Dependencies

```bash
# For API testing
uv add --dev httpx pytest-asyncio

# For web testing
uv add --dev playwright
uv run playwright install
```

## Output Format

```markdown
## E2E Test Report

### Workflow: User Registration
| Step | Description | Status | Duration |
|------|-------------|--------|----------|
| 1 | POST /register | PASS | 120ms |
| 2 | Verify email | PASS | 50ms |
| 3 | POST /login | PASS | 80ms |
| 4 | GET /profile | PASS | 45ms |

### Overall
- **Status**: PASSED
- **Duration**: 295ms
- **Coverage**: User registration, login, profile access
```

## Quick Reference

```bash
# Run all E2E tests
uv run pytest tests/e2e/ -v

# Run specific workflow
uv run pytest tests/e2e/test_user_workflows.py -v

# Run with markers
uv run pytest -m e2e -v

# Run with coverage
uv run pytest tests/e2e/ --cov=src -v
```

## Related Commands

- `/tdd` - Test-driven development
- `/code-review` - Review code quality
- `/build-fix` - Fix build errors
