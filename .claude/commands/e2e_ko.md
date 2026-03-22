# /e2e 명령어

사용자 워크플로우를 위한 엔드투엔드 테스트를 생성하고 실행합니다.

## 사용법
```
/e2e <워크플로우 설명>
/e2e
```

## 이 명령어의 기능

1. **호출**: `e2e-runner` 에이전트
2. **생성**: 워크플로우용 E2E 테스트 케이스
3. **실행**: 테스트 실행 및 결과 리포트

## 워크플로우

### 단계 1: 워크플로우 식별

테스트할 워크플로우 분석:
- 사용자 여정 단계
- 관련된 API 엔드포인트
- 예상 결과

### 단계 2: 테스트 생성

```python
# tests/e2e/test_workflow.py
import pytest
import httpx

@pytest.mark.e2e
async def test_user_registration_workflow(client):
    """전체 사용자 등록 흐름 테스트."""
    # 단계 1: 등록
    response = await client.post("/register", json={
        "email": "test@example.com",
        "password": "secure123"
    })
    assert response.status_code == 201

    # 단계 2: 이메일 인증 (시뮬레이션)
    user_id = response.json()["id"]

    # 단계 3: 로그인
    response = await client.post("/login", json={
        "email": "test@example.com",
        "password": "secure123"
    })
    assert response.status_code == 200
    assert "token" in response.json()
```

### 단계 3: 테스트 실행

```bash
# E2E 테스트 실행
uv run pytest tests/e2e/ -v -m e2e

# 상세 출력과 함께 실행
uv run pytest tests/e2e/ -v --tb=long
```

### 단계 4: 결과 검토

```markdown
## E2E 테스트 결과

### 요약
- 총: 5
- 통과: 4
- 실패: 1

### 실패한 테스트
- test_checkout_workflow: 결제 API 타임아웃
```

## 테스트 카테고리

### API 워크플로우

```python
@pytest.mark.e2e
async def test_crud_workflow(client):
    # 생성
    response = await client.post("/items", json={"name": "Test"})
    item_id = response.json()["id"]

    # 읽기
    response = await client.get(f"/items/{item_id}")
    assert response.json()["name"] == "Test"

    # 업데이트
    response = await client.patch(f"/items/{item_id}", json={"name": "Updated"})
    assert response.json()["name"] == "Updated"

    # 삭제
    response = await client.delete(f"/items/{item_id}")
    assert response.status_code == 204
```

### CLI 워크플로우

```python
from click.testing import CliRunner

@pytest.mark.e2e
def test_cli_workflow():
    runner = CliRunner()

    # 초기화
    result = runner.invoke(main, ["init", "--name", "myproject"])
    assert result.exit_code == 0

    # 처리
    result = runner.invoke(main, ["process", "--input", "data.json"])
    assert result.exit_code == 0
    assert "성공" in result.output
```

### 웹 워크플로우 (Playwright)

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

## 설정

### pyproject.toml

```toml
[tool.pytest.ini_options]
markers = [
    "e2e: 엔드투엔드 테스트 (느릴 수 있음)",
]
asyncio_mode = "auto"
```

### 의존성 설치

```bash
# API 테스트용
uv add --dev httpx pytest-asyncio

# 웹 테스트용
uv add --dev playwright
uv run playwright install
```

## 출력 형식

```markdown
## E2E 테스트 보고서

### 워크플로우: 사용자 등록
| 단계 | 설명 | 상태 | 소요 시간 |
|------|------|------|----------|
| 1 | POST /register | 통과 | 120ms |
| 2 | 이메일 인증 | 통과 | 50ms |
| 3 | POST /login | 통과 | 80ms |
| 4 | GET /profile | 통과 | 45ms |

### 전체
- **상태**: 통과
- **소요 시간**: 295ms
- **커버리지**: 사용자 등록, 로그인, 프로필 접근
```

## 빠른 참조

```bash
# 모든 E2E 테스트 실행
uv run pytest tests/e2e/ -v

# 특정 워크플로우 실행
uv run pytest tests/e2e/test_user_workflows.py -v

# 마커로 실행
uv run pytest -m e2e -v

# 커버리지와 함께 실행
uv run pytest tests/e2e/ --cov=src -v
```

## 관련 명령어

- `/tdd` - 테스트 주도 개발
- `/code-review` - 코드 품질 검토
- `/build-fix` - 빌드 오류 수정
